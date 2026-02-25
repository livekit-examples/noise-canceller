#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
import sys
import wave
from pathlib import Path
import numpy as np
import soundfile as sf

# Rich imports for beautiful CLI
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

from livekit import rtc, api
from livekit.agents import AgentServer, AutoSubscribe, JobContext
from livekit.agents.job import JobExecutorType
from livekit.agents.voice import AgentSession, Agent, room_io, io as voice_io
from livekit.plugins import noise_cancellation, ai_coustics
from livekit.plugins.ai_coustics import EnhancerModel
from dotenv import load_dotenv

SAMPLERATE = 48000
CHUNK_DURATION_MS = 10  # 10ms chunks
SAMPLES_PER_CHUNK = int(SAMPLERATE * CHUNK_DURATION_MS / 1000)
CHANNELS = 1

load_dotenv()

# Initialize Rich console (will be updated based on silent mode)
console = Console()

# Set up logger with Rich
logger = logging.getLogger("noise-canceller")


class NullConsole:
    """A console that suppresses all output for silent mode"""
    def print(self, *args, **kwargs):
        pass
    
    def status(self, *args, **kwargs):
        return NullContext()
    
    def print_exception(self, *args, **kwargs):
        pass


class NullContext:
    """A context manager that does nothing"""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class NullProgress:
    """A progress tracker that does nothing for silent mode"""
    def __init__(self, *args, **kwargs):
        # Accept any arguments but ignore them
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def add_task(self, *args, **kwargs):
        return 0
    
    def update(self, *args, **kwargs):
        pass


class CapturingAudioInput(voice_io.AudioInput):
    """Wraps an AudioInput to capture processed frames as they pass through.

    The AgentSession's internal forward task still consumes frames normally,
    but we record each one before handing it off.
    """

    def __init__(self, source: voice_io.AudioInput, expected_frames: int) -> None:
        super().__init__(label="Capture", source=source)
        self.frames: list[bytes] = []
        self.expected_frames = expected_frames
        self.done = asyncio.Event()

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await super().__anext__()
        self.frames.append(frame.data)
        if len(self.frames) >= self.expected_frames:
            self.done.set()
        return frame


_config: dict = {}

server = AgentServer(job_executor_type=JobExecutorType.THREAD)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Agent entrypoint — processes the file then exits the process."""
    exit_code = 0
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

        filters = _config["filters"]
        silent = _config["silent"]
        input_file = Path(_config["input_file"])
        outputs: list[tuple[str, str]] = []  # (display_name, output_path)

        for fc in filters:
            processor = AudioFileProcessor(
                room=ctx.room,
                noise_filter=fc["noise_filter"],
                filter_key=fc["filter"],
                use_webrtc=fc["use_webrtc"],
                silent=silent,
            )
            await processor.process_file(input_file, Path(fc["output"]))
            outputs.append((_filter_display_name(fc["filter"]), fc["output"]))

        if not silent:
            if len(outputs) == 1:
                name, path = outputs[0]
                body = (
                    "🎉 [bold green]All Done![/bold green]\n"
                    f"[dim]Your {name} noise-cancelled audio is ready at:[/dim]\n"
                    f"[cyan]{path}[/cyan]"
                )
            else:
                lines = ["🎉 [bold green]All Done![/bold green]"]
                for name, path in outputs:
                    lines.append(f"  [dim]{name}:[/dim] [cyan]{path}[/cyan]")
                body = "\n".join(lines)
            console.print()
            console.print(Panel.fit(body, style="green"))

    except Exception as e:
        exit_code = 1
        if not _config.get("silent"):
            error_panel = Panel.fit(
                f"💥 [bold red]Processing Failed[/bold red]\n\n"
                f"[dim]Error details:[/dim]\n"
                f"[red]{e}[/red]",
                style="red"
            )
            console.print(error_panel)
        else:
            sys.stderr.write(f"ERROR: Processing failed - {e}\n")
    finally:
        ctx.shutdown("processing complete")
        os._exit(exit_code)


def _filter_display_name(filter_key: str) -> str:
    """Human-readable name for a --filter value."""
    return {
        "NC": "Krisp Noise Cancellation",
        "BVC": "Krisp Background Voice Cancellation",
        "BVCTelephony": "Krisp BVC (Telephony)",
        "WebRTC": "WebRTC Noise Suppression",
        "aic-quail-l": "Ai-Coustics QUAIL-L",
        "aic-quail-vfl": "Ai-Coustics QUAIL-VF-L",
    }.get(filter_key, filter_key)


class AudioFileProcessor:
    def __init__(self, room: rtc.Room, noise_filter, filter_key: str, use_webrtc=False, silent=False):
        self.room = room
        self.noise_filter = noise_filter
        self.filter_key = filter_key
        self.use_webrtc = use_webrtc
        self.processed_frames: list[bytes] = []
        self.silent = silent

    async def process_file(self, input_path: Path, output_path: Path):
        """Process an audio file with LiveKit noise cancellation or WebRTC noise suppression"""
        if not self.silent:
            # Create beautiful header panel
            display_name = _filter_display_name(self.filter_key)
            header = Panel.fit(
                f"🎵 [bold cyan]{display_name}[/bold cyan] 🎵\n"
                f"[dim]Powered by LiveKit Cloud[/dim]",
                style="cyan"
            )
            console.print(header)
            console.print()

            # Show file info table
            file_info = Table(title="📁 File Information", show_header=True, header_style="bold magenta")
            file_info.add_column("Property", style="cyan")
            file_info.add_column("Value", style="green")
            
            file_info.add_row("Input File", str(input_path))
            file_info.add_row("Output File", str(output_path))
            if self.use_webrtc:
                file_info.add_row("Processing Type", "WebRTC AudioProcessingModule")
                file_info.add_row("Features", "Noise Suppression + Echo Cancellation + High-pass Filter")
            else:
                file_info.add_row("Processing Type", display_name)
            
            console.print(file_info)
            console.print()
        
        # Load the input audio file
        with console.status("[bold green]Loading audio file...", spinner="dots"):
            audio_data = self._load_audio_file(input_path)

        if self.use_webrtc:
            # Process with WebRTC AudioProcessingModule
            await self._process_with_webrtc_apm(audio_data)
        else:
            # Process with LiveKit enhanced noise cancellation
            await self._process_with_noise_cancellation(audio_data)
        
        # Save the processed audio
        with console.status("[bold green]Saving processed audio...", spinner="dots"):
            self._save_output(output_path)
        
        if not self.silent:
            # Success message
            success_panel = Panel.fit(
                f"✅ [bold green]Processing Complete![/bold green]\n"
                f"[dim]Clean audio saved to: {output_path}[/dim]",
                style="green"
            )
            console.print(success_panel)

    async def _process_with_webrtc_apm(self, audio_data):
        """Process audio data using WebRTC AudioProcessingModule"""
        chunk_count = len(audio_data) // SAMPLES_PER_CHUNK
        if len(audio_data) % SAMPLES_PER_CHUNK != 0:
            chunk_count += 1
        
        if not self.silent:
            console.print("🔧 [yellow]Initializing WebRTC AudioProcessingModule...[/yellow]")
        
        # Create WebRTC AudioProcessingModule with noise suppression enabled
        apm = rtc.AudioProcessingModule(
            noise_suppression=True,
            echo_cancellation=True,  # Also enable echo cancellation for better results
            high_pass_filter=True,   # High-pass filter removes low-frequency noise
            auto_gain_control=False  # Keep gain control disabled for file processing
        )
        
        # Process audio in 10ms chunks (required by WebRTC APM)
        progress_class = NullProgress if self.silent else Progress
        
        with progress_class(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            process_task = progress.add_task("🎙️ Processing with WebRTC APM", total=chunk_count)
            
            for i in range(chunk_count):
                start_idx = i * SAMPLES_PER_CHUNK
                end_idx = min(start_idx + SAMPLES_PER_CHUNK, len(audio_data))
                chunk = audio_data[start_idx:end_idx]
                
                # Pad last chunk if necessary with silence
                if len(chunk) < SAMPLES_PER_CHUNK:
                    chunk = np.concatenate([chunk, np.zeros(SAMPLES_PER_CHUNK - len(chunk), dtype=np.int16)])
                
                # Create audio frame (WebRTC APM requires exactly 10ms frames)
                audio_frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=SAMPLERATE,
                    num_channels=CHANNELS,
                    samples_per_channel=len(chunk)
                )
                
                # Process frame in-place with WebRTC APM
                apm.process_stream(audio_frame)
                # Store processed frame
                self.processed_frames.append(audio_frame.data.tobytes())
                # Update progress
                progress.update(process_task, advance=1)
            
            logger.info(f"Successfully processed {len(self.processed_frames)} frames with WebRTC APM")

    async def _process_with_noise_cancellation(self, audio_data):
        """Process audio data through the LiveKit Agents pipeline with noise cancellation.

        Uses a two-connection architecture so that the agents SDK's RoomIO
        handles credential passing to FrameProcessor-based filters (e.g.
        Ai-Coustics) automatically — no private method calls required.

        Publisher connection  --[audio]--> LiveKit SFU --[subscribe]--> Agent RoomIO
                                                                          |
                                                           noise cancellation applied
                                                                          |
                                                                  CapturingAudioInput
        """
        chunk_count = len(audio_data) // SAMPLES_PER_CHUNK
        if len(audio_data) % SAMPLES_PER_CHUNK != 0:
            chunk_count += 1

        publisher_room: rtc.Room | None = None
        session: AgentSession | None = None

        try:
            # Step 1: Create a second "publisher" room connection that will
            # appear as a remote participant to the agent's room.
            with console.status("[bold yellow]Connecting publisher to LiveKit room...", spinner="dots"):
                publisher_token = (
                    api.AccessToken(
                        os.environ["LIVEKIT_API_KEY"],
                        os.environ["LIVEKIT_API_SECRET"],
                    )
                    .with_identity("file-publisher")
                    .with_grants(api.VideoGrants(room_join=True, room=self.room.name))
                    .to_jwt()
                )
                publisher_room = rtc.Room()
                await publisher_room.connect(os.environ["LIVEKIT_URL"], publisher_token)
                logger.debug("Publisher connected to room %s", self.room.name)

            # Step 2: Publish the raw audio track from the publisher connection.
            with console.status("[bold yellow]Publishing audio track...", spinner="dots"):
                file_source = FileAudioSource(audio_data, SAMPLERATE, CHANNELS)
                input_track = rtc.LocalAudioTrack.create_audio_track("raw-input", file_source)
                pub_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
                publication = await publisher_room.local_participant.publish_track(
                    input_track, pub_options,
                )
                await asyncio.sleep(0.5)

            if not self.silent:
                if publication:
                    console.print(
                        f"✅ [green]Track published: [bold]{publication.name}[/bold]"
                        f" (SID: {publication.sid})[/green]"
                    )
                    console.print(f"🏠 [cyan]Room: {self.room.name}[/cyan]")
                    console.print()

            # Step 3: Start an AgentSession whose RoomIO receives the
            # publisher's audio through the noise-cancellation pipeline.
            session = AgentSession()
            await session.start(
                agent=Agent(instructions=""),
                room=self.room,
                room_options=room_io.RoomOptions(
                    audio_input=room_io.AudioInputOptions(
                        noise_cancellation=self.noise_filter,
                        sample_rate=SAMPLERATE,
                        num_channels=CHANNELS,
                        frame_size_ms=CHUNK_DURATION_MS,
                    ),
                    audio_output=False,
                    text_input=False,
                    text_output=False,
                    close_on_disconnect=False,
                ),
            )

            # Step 4: Wrap the RoomIO audio input with our capturing wrapper.
            # _forward_audio_task was just created via asyncio.create_task and
            # hasn't executed yet — safe to swap before yielding.
            raw_audio_input = session.input.audio
            if raw_audio_input is None:
                raise RuntimeError("AgentSession did not create an audio input")
            capturing = CapturingAudioInput(raw_audio_input, expected_frames=chunk_count)
            session.input.audio = capturing

            # Step 5: Feed audio data from the publisher and wait for capture.
            progress_class = NullProgress if self.silent else Progress
            with progress_class(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                feed_task = progress.add_task("🎤 Feeding audio chunks", total=chunk_count)
                capture_task = progress.add_task("🔊 Capturing processed audio", total=chunk_count)

                async def _feed():
                    await self._feed_audio_data_with_progress(
                        file_source, audio_data, chunk_count, progress, feed_task,
                    )

                async def _wait_capture():
                    last_count = 0
                    while not capturing.done.is_set():
                        await asyncio.sleep(0.05)
                        new_count = len(capturing.frames)
                        if new_count > last_count:
                            progress.update(capture_task, advance=new_count - last_count)
                            last_count = new_count
                    # Final update
                    if last_count < len(capturing.frames):
                        progress.update(capture_task, advance=len(capturing.frames) - last_count)

                try:
                    await asyncio.wait_for(
                        asyncio.gather(_feed(), _wait_capture()), timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    if not self.silent:
                        console.print("⚠️  [yellow]Processing timed out[/yellow]")

            self.processed_frames = capturing.frames
            logger.info("Successfully processed %d frames", len(self.processed_frames))

        except Exception as e:
            if not self.silent:
                console.print(f"❌ [red]Error during noise cancellation: {e}[/red]")
            raise
        finally:
            if session:
                try:
                    await session.aclose()
                except Exception as e:
                    logger.debug("AgentSession cleanup: %s", e)
            if publisher_room:
                try:
                    await publisher_room.disconnect()
                except Exception as e:
                    logger.debug("Publisher disconnect: %s", e)

    async def _feed_audio_data_with_progress(self, file_source, audio_data, chunk_count, progress, task_id):
        """Feed audio data to the source with precise timing and progress updates"""
        chunk_duration = SAMPLES_PER_CHUNK / SAMPLERATE
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        
        for i in range(chunk_count):
            start_idx = i * SAMPLES_PER_CHUNK
            end_idx = min(start_idx + SAMPLES_PER_CHUNK, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            
            if len(chunk) < SAMPLES_PER_CHUNK:
                chunk = np.concatenate([chunk, np.zeros(SAMPLES_PER_CHUNK - len(chunk), dtype=np.int16)])
            
            audio_frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=SAMPLERATE,
                num_channels=CHANNELS,
                samples_per_channel=len(chunk)
            )
            
            await file_source.capture_frame(audio_frame)
            progress.update(task_id, advance=1)
            
            target_time = start_time + (i + 1) * chunk_duration
            current_time = loop.time()
            delay = max(0, target_time - current_time)
            
            if delay > 0:
                await asyncio.sleep(delay)

    def _load_audio_file(self, input_path: Path):
        """Load and preprocess audio file"""
        try:
            audio_data, sample_rate = sf.read(str(input_path), dtype='int16')
            
            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[1]
            
            duration_s = len(audio_data) / sample_rate
            
            if not self.silent:
                audio_info = Table(title="🎵 Audio Properties", show_header=True, header_style="bold blue")
                audio_info.add_column("Property", style="cyan")
                audio_info.add_column("Value", style="green")
                
                audio_info.add_row("Sample Rate", f"{sample_rate:,} Hz")
                audio_info.add_row("Channels", str(channels))
                audio_info.add_row("Duration", f"{duration_s:.2f} seconds")
                audio_info.add_row("Format", input_path.suffix.upper())
                
                console.print(audio_info)
                console.print()
            
            audio_array = audio_data

            # Resample to 48kHz mono if needed
            if sample_rate != SAMPLERATE or channels != CHANNELS:
                audio_array = self._resample_audio(audio_array, sample_rate, channels)
                if not self.silent:
                    console.print(f"🔄 [yellow]Resampled to: {SAMPLERATE}Hz, {CHANNELS} channel(s)[/yellow]")
                    console.print()
            
            return audio_array
            
        except Exception as e:
            if not self.silent:
                console.print(f"❌ [red]Error loading audio file: {e}[/red]")
                console.print("[dim]Supported formats: WAV, FLAC, OGG, MP3 (with ffmpeg), M4A, and more[/dim]")
                console.print("[dim]Make sure you have ffmpeg installed for MP3/M4A support[/dim]")
            else:
                # In silent mode, still show critical errors to stderr
                sys.stderr.write(f"ERROR: Failed to load audio file - {str(e)}\n")
            raise

    def _resample_audio(self, audio_array, original_rate, original_channels):
        """High-quality resampling using LiveKit's AudioResampler"""
        # Convert to mono if stereo
        if original_channels == 2:
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1).astype(np.int16)
            else:
                stereo = audio_array.reshape(-1, 2)
                audio_array = stereo.mean(axis=1).astype(np.int16)
        
        # Resample if needed
        if original_rate != SAMPLERATE:
            resampler = rtc.AudioResampler(
                input_rate=original_rate,
                output_rate=SAMPLERATE,
                num_channels=1,
                quality=rtc.AudioResamplerQuality.VERY_HIGH
            )
            
            input_frame = rtc.AudioFrame(
                data=audio_array.tobytes(),
                sample_rate=original_rate,
                num_channels=1,
                samples_per_channel=len(audio_array)
            )
            
            output_frames = resampler.push(input_frame)
            output_frames.extend(resampler.flush())
            
            if len(output_frames) > 0:
                resampled_data = b''.join(frame.data for frame in output_frames)
                audio_array = np.frombuffer(resampled_data, dtype=np.int16)
            else:
                if not self.silent:
                    console.print("⚠️  [yellow]Warning: No output from AudioResampler, using original data[/yellow]")
        
        return audio_array

    def _save_output(self, output_path: Path):
        """Save processed audio frames to output file"""
        if not self.processed_frames:
            if not self.silent:
                console.print("⚠️  [yellow]Warning: No processed frames to save[/yellow]")
            return
            
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLERATE)
            
            for frame_data in self.processed_frames:
                wav_file.writeframes(frame_data)


class FileAudioSource(rtc.AudioSource):
    """Custom audio source that streams from file data"""
    def __init__(self, audio_data, sample_rate=SAMPLERATE, num_channels=CHANNELS):
        super().__init__(sample_rate, num_channels)
        self.audio_data = audio_data


def setup_logging(log_level: str, silent: bool = False):
    """Setup beautiful Rich logging configuration"""
    level = getattr(logging, log_level.upper())
    
    if silent:
        # For silent mode, still allow WARNING and above to be logged to stderr
        # This helps with debugging while keeping output clean
        logging.basicConfig(
            level=logging.WARNING,  # Allow warnings and errors
            format="%(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],  # Send to stderr
            force=True
        )
    else:
        # Create Rich handler with beautiful formatting
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_suppress=[rtc, api, noise_cancellation, ai_coustics]
        )
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[rich_handler],
            force=True
        )

    # Suppress noisy agent framework / livekit SDK logs — our own logger
    # ("noise-canceller") already captures everything the user needs to see.
    for name in ("livekit", "livekit.agents", "livekit.plugins", "livekit.rtc"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # The livekit-rtc Room uses logging.info() on the root logger for
    # "ignoring text stream" messages — filter those out.
    class _IgnoreTextStreamFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "ignoring text stream" not in record.getMessage()

    logging.getLogger().addFilter(_IgnoreTextStreamFilter())


def main():
    global console

    parser = argparse.ArgumentParser(
        description="🎵 Process audio files with LiveKit noise cancellation",
        epilog="""
✨ Examples:
  uv run noise-canceller.py input.mp3
  uv run noise-canceller.py input.wav -o clean_audio.wav
  uv run noise-canceller.py song.flac --filter BVC
  uv run noise-canceller.py audio.m4a --filter WebRTC
  uv run noise-canceller.py audio.m4a --filter aic-quail-l
  uv run noise-canceller.py audio.m4a --filter aic-quail-vfl
  uv run noise-canceller.py audio.m4a --filter all
  uv run noise-canceller.py audio.m4a -o processed.wav --silent
  
📁 Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC, AIFF, and more
📝 Note: Some formats may require ffmpeg to be installed
📡 The tool will show track publication status automatically
  
🔧 Environment variables:
  LIVEKIT_URL: Your LiveKit Cloud server URL
  LIVEKIT_API_KEY: Your LiveKit API key  
  LIVEKIT_API_SECRET: Your LiveKit API secret
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input audio file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: output/<input-file-name>-processed.wav)"
    )
    parser.add_argument(
        "--filter",
        choices=["NC", "BVC", "BVCTelephony", "WebRTC", "aic-quail-l", "aic-quail-vfl", "all"],
        default="NC",
        help="Noise cancellation filter type (default: NC). 'all' runs every filter and saves separate output files."
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "-s", "--silent",
        action="store_true",
        help="Suppress all output (silent mode)"
    )

    args = parser.parse_args()
    
    # Setup console for silent mode
    if args.silent:
        console = NullConsole()
    
    # Setup beautiful logging
    setup_logging(args.log_level, args.silent)
    
    # Check environment with beautiful error messages
    if not os.getenv("LIVEKIT_URL"):
        if not args.silent:
            error_panel = Panel.fit(
                "❌ [bold red]Missing Environment Variable[/bold red]\n\n"
                "[dim]LIVEKIT_URL environment variable is required.[/dim]\n"
                "[dim]Set it to your LiveKit server URL, e.g.:[/dim]\n"
                "[cyan]export LIVEKIT_URL=wss://your-project.livekit.cloud[/cyan]",
                style="red"
            )
            console.print(error_panel)
        else:
            # In silent mode, still show critical errors to stderr
            sys.stderr.write("ERROR: LIVEKIT_URL environment variable is required\n")
        sys.exit(1)
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        if not args.silent:
            console.print(f"❌ [red]Input file '{input_path}' does not exist[/red]")
        else:
            sys.stderr.write(f"ERROR: Input file '{input_path}' does not exist\n")
        sys.exit(1)
    
    # Build filter config(s)
    filter_map = {
        "NC": lambda: noise_cancellation.NC(),
        "BVC": lambda: noise_cancellation.BVC(),
        "BVCTelephony": lambda: noise_cancellation.BVCTelephony(),
        "aic-quail-l": lambda: ai_coustics.audio_enhancement(model=EnhancerModel.QUAIL_L),
        "aic-quail-vfl": lambda: ai_coustics.audio_enhancement(model=EnhancerModel.QUAIL_VF_L),
    }
    ALL_FILTERS = ["NC", "BVC", "BVCTelephony", "WebRTC", "aic-quail-l", "aic-quail-vfl"]
    selected = ALL_FILTERS if args.filter == "all" else [args.filter]

    filter_configs: list[dict] = []
    for fk in selected:
        use_webrtc = fk == "WebRTC"
        nf = None if use_webrtc else filter_map[fk]()
        if args.output and len(selected) == 1:
            out = Path(args.output)
        else:
            out = Path(f"output/{input_path.stem}-{fk.lower()}-processed.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        filter_configs.append({
            "filter": fk,
            "noise_filter": nf,
            "use_webrtc": use_webrtc,
            "output": str(out),
        })

    _config.update({
        "input_file": str(input_path),
        "filters": filter_configs,
        "silent": args.silent,
    })

    # Replicate the agents CLI "connect" command: create a real room via the
    # API, then simulate_job(fake_job=False) so the entrypoint gets a genuine
    # room connection with proper agent credentials.
    room_name = f"noise-canceller-{os.getpid()}"

    @server.once("worker_started")
    def _on_started():
        async def _run_job():
            lk_api = api.LiveKitAPI()
            try:
                rooms = await lk_api.room.list_rooms(
                    api.ListRoomsRequest(names=[room_name])
                )
                if rooms.rooms:
                    room_info = rooms.rooms[0]
                else:
                    room_info = await lk_api.room.create_room(
                        api.CreateRoomRequest(name=room_name)
                    )
            finally:
                await lk_api.aclose()

            await server.simulate_job(
                room=room_name,
                fake_job=False,
                room_info=room_info,
                agent_identity="noise-canceller",
            )

        asyncio.ensure_future(_run_job())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server.run(devmode=True, unregistered=True))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
