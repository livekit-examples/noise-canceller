#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
import re
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import soundfile as sf

# Rich imports for beautiful CLI
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.table import Table

from livekit import rtc, api
from livekit.agents import AgentServer, AutoSubscribe, JobContext, inference
from livekit.agents.job import JobExecutorType
from livekit.agents.stt import SpeechEventType
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

    def __init__(
        self,
        source: voice_io.AudioInput,
        expected_frames: int,
        processed_stt: "SttStream | None" = None,
    ) -> None:
        super().__init__(label="Capture", source=source)
        self.frames: list[bytes] = []
        self.expected_frames = expected_frames
        self.done = asyncio.Event()
        self._processed_stt = processed_stt

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await super().__anext__()
        self.frames.append(frame.data)
        if not self.done.is_set():
            if self._processed_stt is not None:
                self._processed_stt.push_frame(frame)
            if len(self.frames) >= self.expected_frames:
                if self._processed_stt is not None:
                    self._processed_stt.end_input()
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
        ground_truth_file = _config.get("transcript")
        stt_model = _config.get("stt", "deepgram/nova-3:en")
        outputs: list[tuple[str, str]] = []  # (display_name, output_path)
        processors: list[AudioFileProcessor] = []

        # ----- Job summary -----
        if not silent:
            filter_names = [_filter_display_name(fc["filter"]) for fc in filters]
            summary = Table(
                title="🔊 Job Summary",
                show_header=False,
                title_style="bold cyan",
                border_style="cyan",
                padding=(0, 1),
            )
            summary.add_column("key", style="dim")
            summary.add_column("value")
            summary.add_row("Input", str(input_file))
            if len(filter_names) == 1:
                summary.add_row("Filter", filter_names[0])
            else:
                summary.add_row("Filters", ", ".join(filter_names))
            if ground_truth_file:
                summary.add_row("Transcript", ground_truth_file)
                summary.add_row("STT Model", stt_model)
            console.print(summary)
            console.print()

        # ----- Audio processing (+ streaming transcription) -----
        original_stt_stream: SttStream | None = None
        processed_stt_streams: list[tuple[dict, SttStream]] = []

        progress_class = NullProgress if silent else Progress
        with progress_class(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for i, fc in enumerate(filters):
                short_name = _filter_short_name(fc["filter"])
                processor = AudioFileProcessor(
                    room=ctx.room,
                    noise_filter=fc["noise_filter"],
                    filter_key=fc["filter"],
                    use_webrtc=fc["use_webrtc"],
                    silent=silent,
                    direct=_config.get("direct", False),
                )

                orig_stream: SttStream | None = None
                proc_stream: SttStream | None = None

                bar_ids: dict[str, int] = {}
                bar_ids["feed"] = progress.add_task(
                    "  📦 Chunking original audio",
                    total=0,
                )
                if ground_truth_file and i == 0:
                    orig_stream = SttStream(stt_model, "orig")
                    original_stt_stream = orig_stream
                    bar_ids["orig_stt"] = progress.add_task(
                        f"  📡 Sending to {stt_model}",
                        total=0,
                    )
                bar_ids["nc"] = progress.add_task(
                    f"  🎤 Sending to {short_name}",
                    total=0,
                )
                if ground_truth_file:
                    filter_name = _filter_display_name(fc["filter"])
                    proc_stream = SttStream(stt_model, filter_name)
                    processed_stt_streams.append((fc, proc_stream))
                    bar_ids["proc_stt"] = progress.add_task(
                        f"  📡 Sending {short_name} output to {stt_model}",
                        total=0,
                    )

                await processor.process_file(
                    input_file,
                    Path(fc["output"]),
                    progress=progress,
                    bar_ids=bar_ids,
                    original_stt=orig_stream,
                    processed_stt=proc_stream,
                )
                outputs.append((_filter_display_name(fc["filter"]), fc["output"]))
                processors.append(processor)

        # Wait for STT collection to finish (transcripts still arriving
        # after all chunks have been sent).
        if ground_truth_file:
            with console.status(
                "[bold green]Waiting for transcription results…", spinner="dots"
            ):
                if original_stt_stream is not None:
                    await original_stt_stream.result()
                for _fc, ps in processed_stt_streams:
                    await ps.result()

        # ----- Results -----
        if not silent:
            result_lines: list[str] = []
            for filter_name, path in outputs:
                result_lines.append(f"  [dim]{filter_name}:[/dim] [cyan]{path}[/cyan]")

            if ground_truth_file and original_stt_stream is not None:
                ground_truth = Path(ground_truth_file).read_text().strip()
                input_transcript = await original_stt_stream.result()

                for fc, proc_stream in processed_stt_streams:
                    output_transcript = await proc_stream.result()
                    report = generate_transcript_report(
                        ground_truth=ground_truth,
                        input_transcript=input_transcript,
                        output_transcript=output_transcript,
                        input_file=str(input_file),
                        output_file=fc["output"],
                        filter_name=_filter_display_name(fc["filter"]),
                        stt_model=stt_model,
                    )
                    report_path = Path(fc["output"]).with_suffix(".transcript.md")
                    report_path.write_text(report)
                    result_lines.append(
                        f"  [dim]Transcript:[/dim] [cyan]{report_path}[/cyan]"
                    )

            body = "🎉 [bold green]All Done![/bold green]\n" + "\n".join(result_lines)
            console.print()
            console.print(Panel.fit(body, style="green"))
        elif ground_truth_file and original_stt_stream is not None:
            ground_truth = Path(ground_truth_file).read_text().strip()
            input_transcript = await original_stt_stream.result()
            for fc, proc_stream in processed_stt_streams:
                output_transcript = await proc_stream.result()
                report = generate_transcript_report(
                    ground_truth=ground_truth,
                    input_transcript=input_transcript,
                    output_transcript=output_transcript,
                    input_file=str(input_file),
                    output_file=fc["output"],
                    filter_name=_filter_display_name(fc["filter"]),
                    stt_model=stt_model,
                )
                report_path = Path(fc["output"]).with_suffix(".transcript.md")
                report_path.write_text(report)

    except Exception as e:
        exit_code = 1
        if not _config.get("silent"):
            error_panel = Panel.fit(
                f"💥 [bold red]Processing Failed[/bold red]\n\n"
                f"[dim]Error details:[/dim]\n"
                f"[red]{e}[/red]",
                style="red",
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


def _filter_short_name(filter_key: str) -> str:
    """Short name for progress bar labels."""
    return {
        "NC": "Krisp NC",
        "BVC": "Krisp BVC",
        "BVCTelephony": "Krisp BVC-T",
        "WebRTC": "WebRTC",
        "aic-quail-l": "aic-quail-l",
        "aic-quail-vfl": "aic-quail-vfl",
    }.get(filter_key, filter_key)


class AudioFileProcessor:
    def __init__(
        self,
        room: rtc.Room,
        noise_filter,
        filter_key: str,
        use_webrtc=False,
        silent=False,
        direct=False,
    ):
        self.room = room
        self.noise_filter = noise_filter
        self.filter_key = filter_key
        self.use_webrtc = use_webrtc
        self.processed_frames: list[bytes] = []
        self.original_audio: np.ndarray | None = None
        self.silent = silent
        self.direct = direct

    async def process_file(
        self,
        input_path: Path,
        output_path: Path,
        progress=None,
        bar_ids: dict[str, int] | None = None,
        original_stt: "SttStream | None" = None,
        processed_stt: "SttStream | None" = None,
    ):
        """Process an audio file with LiveKit noise cancellation or WebRTC noise suppression.

        When *progress* is an active Rich Progress instance the method adds its
        tasks to that display instead of creating its own.  ``console.status()``
        spinners are skipped in that case because they conflict with the live
        progress display.

        *bar_ids*, when provided, is a dict mapping logical bar names
        (``"feed"``, ``"nc"``, ``"orig_stt"``, ``"proc_stt"``) to Rich task IDs
        that were pre-created by the caller.  The processing loops advance these
        bars directly.

        *original_stt* and *processed_stt*, when provided, receive audio frames
        in real-time as they flow through the processing pipeline.
        """
        if not self.silent:
            display_name = _filter_display_name(self.filter_key)
            header = Panel.fit(
                f"🎵 [bold cyan]{display_name}[/bold cyan] 🎵\n"
                f"[dim]Powered by LiveKit Cloud[/dim]",
                style="cyan",
            )
            console.print(header)
            console.print()

            file_info = Table(
                title="📁 File Information",
                show_header=True,
                header_style="bold magenta",
            )
            file_info.add_column("Property", style="cyan")
            file_info.add_column("Value", style="green")

            file_info.add_row("Input File", str(input_path))
            file_info.add_row("Output File", str(output_path))
            if self.use_webrtc:
                file_info.add_row("Processing Type", "WebRTC AudioProcessingModule")
                file_info.add_row(
                    "Features",
                    "Noise Suppression + Echo Cancellation + High-pass Filter",
                )
            else:
                file_info.add_row("Processing Type", display_name)

            console.print(file_info)
            console.print()

        if progress is not None:
            audio_data = self._load_audio_file(input_path)
        else:
            with console.status("[bold green]Loading audio file...", spinner="dots"):
                audio_data = self._load_audio_file(input_path)
        self.original_audio = audio_data

        if self.use_webrtc:
            await self._process_with_webrtc_apm(
                audio_data,
                progress=progress,
                bar_ids=bar_ids,
                original_stt=original_stt,
                processed_stt=processed_stt,
            )
        elif self.direct:
            await self._process_direct(
                audio_data,
                progress=progress,
                bar_ids=bar_ids,
            )
        else:
            await self._process_with_noise_cancellation(
                audio_data,
                progress=progress,
                bar_ids=bar_ids,
                original_stt=original_stt,
                processed_stt=processed_stt,
            )

        if progress is not None:
            self._save_output(output_path)
        else:
            with console.status(
                "[bold green]Saving processed audio...", spinner="dots"
            ):
                self._save_output(output_path)

    async def _process_with_webrtc_apm(
        self,
        audio_data,
        progress=None,
        bar_ids: dict[str, int] | None = None,
        original_stt: "SttStream | None" = None,
        processed_stt: "SttStream | None" = None,
    ):
        """Process audio data using WebRTC AudioProcessingModule"""
        chunk_count = len(audio_data) // SAMPLES_PER_CHUNK
        if len(audio_data) % SAMPLES_PER_CHUNK != 0:
            chunk_count += 1

        if not self.silent:
            console.print(
                "🔧 [yellow]Initializing WebRTC AudioProcessingModule...[/yellow]"
            )

        apm = rtc.AudioProcessingModule(
            noise_suppression=True,
            echo_cancellation=True,
            high_pass_filter=True,
            auto_gain_control=False,
        )

        if progress is None:
            progress_class = NullProgress if self.silent else Progress
            ctx = progress_class(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            )
        else:
            ctx = nullcontext(progress)

        ids = bar_ids or {}

        with ctx as prog:
            if not ids:
                ids["nc"] = prog.add_task(
                    "  🎙️ Processing with WebRTC APM", total=chunk_count
                )
            else:
                for tid in ids.values():
                    prog.update(tid, total=chunk_count)

            for i in range(chunk_count):
                start_idx = i * SAMPLES_PER_CHUNK
                end_idx = min(start_idx + SAMPLES_PER_CHUNK, len(audio_data))
                chunk = audio_data[start_idx:end_idx]

                if len(chunk) < SAMPLES_PER_CHUNK:
                    chunk = np.concatenate(
                        [
                            chunk,
                            np.zeros(SAMPLES_PER_CHUNK - len(chunk), dtype=np.int16),
                        ]
                    )

                audio_frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=SAMPLERATE,
                    num_channels=CHANNELS,
                    samples_per_channel=len(chunk),
                )

                if original_stt is not None:
                    original_stt.push_frame(audio_frame)

                apm.process_stream(audio_frame)
                processed_bytes = audio_frame.data.tobytes()
                self.processed_frames.append(processed_bytes)

                if processed_stt is not None:
                    processed_frame = rtc.AudioFrame(
                        data=processed_bytes,
                        sample_rate=SAMPLERATE,
                        num_channels=CHANNELS,
                        samples_per_channel=SAMPLES_PER_CHUNK,
                    )
                    processed_stt.push_frame(processed_frame)

                for tid in ids.values():
                    prog.update(tid, advance=1)

            if original_stt is not None:
                original_stt.end_input()
            if processed_stt is not None:
                processed_stt.end_input()

            logger.info(
                f"Successfully processed {len(self.processed_frames)} frames with WebRTC APM"
            )

    async def _process_direct(
        self,
        audio_data,
        progress=None,
        bar_ids: dict[str, int] | None = None,
    ):
        """Process audio directly through the FrameProcessor, bypassing the SFU.

        This avoids Opus encode/decode and produces output identical to direct
        plugin FFI processing.  Useful for bit-exact comparison testing.
        """
        chunk_count = len(audio_data) // SAMPLES_PER_CHUNK
        if len(audio_data) % SAMPLES_PER_CHUNK != 0:
            chunk_count += 1

        # Set up credentials on the FrameProcessor so the underlying Enhancer
        # can authenticate with the ai-coustics service.
        token = (
            api.AccessToken(
                os.environ["LIVEKIT_API_KEY"],
                os.environ["LIVEKIT_API_SECRET"],
            )
            .with_identity("noise-canceller-direct")
            .with_grants(api.VideoGrants(room_join=True, room=self.room.name))
            .to_jwt()
        )
        self.noise_filter._on_credentials_updated(
            token=token, url=os.environ["LIVEKIT_URL"]
        )
        self.noise_filter._on_stream_info_updated(
            room_name=self.room.name,
            participant_identity="direct-processing",
            publication_sid="direct-sid",
        )

        if progress is None:
            progress_class = NullProgress if self.silent else Progress
            ctx = progress_class(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            )
        else:
            ctx = nullcontext(progress)

        ids = bar_ids or {}

        with ctx as prog:
            if not ids:
                ids["nc"] = prog.add_task(
                    "  🎤 Processing directly (no SFU)", total=chunk_count
                )
            else:
                for tid in ids.values():
                    prog.update(tid, total=chunk_count)

            for i in range(chunk_count):
                start_idx = i * SAMPLES_PER_CHUNK
                end_idx = min(start_idx + SAMPLES_PER_CHUNK, len(audio_data))
                chunk = audio_data[start_idx:end_idx]

                if len(chunk) < SAMPLES_PER_CHUNK:
                    chunk = np.concatenate(
                        [
                            chunk,
                            np.zeros(SAMPLES_PER_CHUNK - len(chunk), dtype=np.int16),
                        ]
                    )

                audio_frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=SAMPLERATE,
                    num_channels=CHANNELS,
                    samples_per_channel=len(chunk),
                )

                processed_frame = self.noise_filter._process(audio_frame)
                self.processed_frames.append(processed_frame.data)

                for tid in ids.values():
                    prog.update(tid, advance=1)

        logger.info(
            "Direct processing: %d frames processed", len(self.processed_frames)
        )

    async def _process_with_noise_cancellation(
        self,
        audio_data,
        progress=None,
        bar_ids: dict[str, int] | None = None,
        original_stt: "SttStream | None" = None,
        processed_stt: "SttStream | None" = None,
    ):
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
        ids = bar_ids or {}

        try:
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

            file_source = FileAudioSource(audio_data, SAMPLERATE, CHANNELS)
            input_track = rtc.LocalAudioTrack.create_audio_track(
                "raw-input", file_source
            )
            pub_options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_MICROPHONE
            )
            publication = await publisher_room.local_participant.publish_track(
                input_track,
                pub_options,
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

            raw_audio_input = session.input.audio
            if raw_audio_input is None:
                raise RuntimeError("AgentSession did not create an audio input")
            capturing = CapturingAudioInput(
                raw_audio_input,
                expected_frames=chunk_count,
                processed_stt=processed_stt,
            )
            session.input.audio = capturing

            if progress is None:
                progress_class = NullProgress if self.silent else Progress
                ctx = progress_class(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=console,
                )
            else:
                ctx = nullcontext(progress)

            # feed_ids: bars that advance with each input chunk (feed, orig_stt, nc)
            feed_ids = [v for k, v in ids.items() if k in ("feed", "orig_stt", "nc")]
            # capture_ids: bars that advance with each processed chunk (proc_stt)
            capture_ids = [v for k, v in ids.items() if k == "proc_stt"]

            with ctx as prog:
                if ids:
                    for tid in ids.values():
                        prog.update(tid, total=chunk_count)
                else:
                    feed_ids = [
                        prog.add_task("  🎤 Feeding audio chunks", total=chunk_count)
                    ]
                    capture_ids = [
                        prog.add_task(
                            "  🔊 Capturing processed audio", total=chunk_count
                        )
                    ]

                async def _feed():
                    await self._feed_audio_data_with_progress(
                        file_source,
                        audio_data,
                        chunk_count,
                        prog,
                        feed_ids,
                        original_stt=original_stt,
                    )

                async def _wait_capture():
                    last_count = 0
                    while not capturing.done.is_set():
                        await asyncio.sleep(0.05)
                        new_count = len(capturing.frames)
                        if new_count > last_count:
                            delta = new_count - last_count
                            for tid in capture_ids:
                                prog.update(tid, advance=delta)
                            last_count = new_count
                    if last_count < len(capturing.frames):
                        delta = len(capturing.frames) - last_count
                        for tid in capture_ids:
                            prog.update(tid, advance=delta)

                try:
                    await asyncio.wait_for(
                        asyncio.gather(_feed(), _wait_capture()),
                        timeout=120.0,
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

    async def _feed_audio_data_with_progress(
        self,
        file_source,
        audio_data,
        chunk_count,
        progress,
        task_ids: list[int] | int,
        original_stt: "SttStream | None" = None,
    ):
        """Feed audio data to the source with precise timing and progress updates."""
        ids = task_ids if isinstance(task_ids, list) else [task_ids]
        chunk_duration = SAMPLES_PER_CHUNK / SAMPLERATE
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        for i in range(chunk_count):
            start_idx = i * SAMPLES_PER_CHUNK
            end_idx = min(start_idx + SAMPLES_PER_CHUNK, len(audio_data))
            chunk = audio_data[start_idx:end_idx]

            if len(chunk) < SAMPLES_PER_CHUNK:
                chunk = np.concatenate(
                    [chunk, np.zeros(SAMPLES_PER_CHUNK - len(chunk), dtype=np.int16)]
                )

            audio_frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=SAMPLERATE,
                num_channels=CHANNELS,
                samples_per_channel=len(chunk),
            )

            await file_source.capture_frame(audio_frame)
            if original_stt is not None:
                original_stt.push_frame(audio_frame)
            for tid in ids:
                progress.update(tid, advance=1)

            target_time = start_time + (i + 1) * chunk_duration
            current_time = loop.time()
            delay = max(0, target_time - current_time)

            if delay > 0:
                await asyncio.sleep(delay)

        if original_stt is not None:
            original_stt.end_input()

    def _load_audio_file(self, input_path: Path):
        """Load and preprocess audio file"""
        try:
            audio_data, sample_rate = sf.read(str(input_path), dtype="int16")

            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[1]

            duration_s = len(audio_data) / sample_rate

            if not self.silent:
                audio_info = Table(
                    title="🎵 Audio Properties",
                    show_header=True,
                    header_style="bold blue",
                )
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
                    console.print(
                        f"🔄 [yellow]Resampled to: {SAMPLERATE}Hz, {CHANNELS} channel(s)[/yellow]"
                    )
                    console.print()

            return audio_array

        except Exception as e:
            if not self.silent:
                console.print(f"❌ [red]Error loading audio file: {e}[/red]")
                console.print(
                    "[dim]Supported formats: WAV, FLAC, OGG, MP3 (with ffmpeg), M4A, and more[/dim]"
                )
                console.print(
                    "[dim]Make sure you have ffmpeg installed for MP3/M4A support[/dim]"
                )
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
                quality=rtc.AudioResamplerQuality.VERY_HIGH,
            )

            input_frame = rtc.AudioFrame(
                data=audio_array.tobytes(),
                sample_rate=original_rate,
                num_channels=1,
                samples_per_channel=len(audio_array),
            )

            output_frames = resampler.push(input_frame)
            output_frames.extend(resampler.flush())

            if len(output_frames) > 0:
                resampled_data = b"".join(frame.data for frame in output_frames)
                audio_array = np.frombuffer(resampled_data, dtype=np.int16)
            else:
                if not self.silent:
                    console.print(
                        "⚠️  [yellow]Warning: No output from AudioResampler, using original data[/yellow]"
                    )

        return audio_array

    def _save_output(self, output_path: Path):
        """Save processed audio frames to output file"""
        if not self.processed_frames:
            if not self.silent:
                console.print(
                    "⚠️  [yellow]Warning: No processed frames to save[/yellow]"
                )
            return

        with wave.open(str(output_path), "wb") as wav_file:
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


# ---------------------------------------------------------------------------
# Transcription & word-error-rate helpers
# ---------------------------------------------------------------------------


class SttStream:
    """Wraps a LiveKit ``inference.STT`` streaming session so that callers
    can push frames one at a time (from any chunk loop) rather than feeding
    a complete numpy array after the fact.

    Usage::

        stream = SttStream("deepgram/nova-3:en", label="Original audio → STT")
        # … in your chunk loop:
        stream.push_frame(audio_frame)
        # … when done:
        stream.end_input()
        text = await stream.result()
    """

    def __init__(self, stt_model: str, label: str, total_chunks: int = 0):
        self.label = label
        self.total_chunks = total_chunks
        self.chunks_sent = 0
        self.words = 0
        self.sending_done = False
        self.done = False

        self._stt = inference.STT(model=stt_model)
        self._stream = self._stt.stream()
        self._transcripts: list[str] = []
        self._collect_task = asyncio.create_task(self._collect())

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._stream.push_frame(frame)
        self.chunks_sent += 1

    def end_input(self) -> None:
        self._stream.end_input()
        self.sending_done = True

    async def result(self, timeout: float = 120.0) -> str:
        """Wait for collection to finish and return the full transcript."""
        try:
            await asyncio.wait_for(self._collect_task, timeout=timeout)
        except asyncio.TimeoutError:
            self._collect_task.cancel()
            try:
                await self._collect_task
            except asyncio.CancelledError:
                pass
        return " ".join(self._transcripts)

    async def _collect(self) -> None:
        try:
            idle_timeout = 10.0
            stream_iter = self._stream.__aiter__()
            while True:
                try:
                    event = await asyncio.wait_for(
                        stream_iter.__anext__(),
                        timeout=idle_timeout,
                    )
                except asyncio.TimeoutError:
                    break
                except StopAsyncIteration:
                    break
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    if event.alternatives and event.alternatives[0].text:
                        text = event.alternatives[0].text
                        self._transcripts.append(text)
                        self.words += len(text.split())
        finally:
            await self._stream.aclose()
            self.done = True


def _normalize_word(word: str) -> str:
    """Lowercase and strip non-alphanumeric characters for comparison."""
    return re.sub(r"[^\w]", "", word.lower())


def compute_word_alignment(
    reference: str,
    hypothesis: str,
) -> list[tuple[str, str | None, str | None]]:
    """Word-level alignment via minimum edit distance.

    Returns [(operation, ref_word, hyp_word), ...] where *operation* is one of
    ``'correct'``, ``'substitution'``, ``'insertion'``, or ``'deletion'``.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    n, m = len(ref_words), len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if _normalize_word(ref_words[i - 1]) == _normalize_word(hyp_words[j - 1]):
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j],  # deletion
                )

    # Backtrace
    alignment: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and _normalize_word(ref_words[i - 1]) == _normalize_word(hyp_words[j - 1])
        ):
            alignment.append(("correct", ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append(("substitution", ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append(("insertion", None, hyp_words[j - 1]))
            j -= 1
        elif i > 0:
            alignment.append(("deletion", ref_words[i - 1], None))
            i -= 1
        else:
            break

    alignment.reverse()
    return alignment


def format_annotated_transcript(
    alignment: list[tuple[str, str | None, str | None]],
) -> str:
    """Render an alignment as a Markdown string with error markers.

    * ~~word~~              — deletion  (in ground truth but not transcribed)
    * **word**              — insertion (transcribed but not in ground truth)
    * ~~expected~~**actual** — substitution (no space between)
    """
    parts: list[str] = []
    for op, ref, hyp in alignment:
        if op == "correct":
            parts.append(hyp)  # type: ignore[arg-type]
        elif op == "substitution":
            parts.append(f"~~{ref}~~**{hyp}**")
        elif op == "insertion":
            parts.append(f"**{hyp}**")
        elif op == "deletion":
            parts.append(f"~~{ref}~~")
    return " ".join(parts)


def _alignment_error_counts(
    alignment: list[tuple[str, str | None, str | None]],
) -> tuple[int, int, int]:
    """Return (substitutions, insertions, deletions) from an alignment."""
    subs = sum(1 for op, _, _ in alignment if op == "substitution")
    ins = sum(1 for op, _, _ in alignment if op == "insertion")
    dels = sum(1 for op, _, _ in alignment if op == "deletion")
    return subs, ins, dels


def generate_transcript_report(
    ground_truth: str,
    input_transcript: str,
    output_transcript: str,
    input_file: str,
    output_file: str,
    filter_name: str,
    stt_model: str,
) -> str:
    """Build a Markdown report comparing pre- and post-processed transcriptions."""
    in_align = compute_word_alignment(ground_truth, input_transcript)
    out_align = compute_word_alignment(ground_truth, output_transcript)

    ref_words = len(ground_truth.split())
    in_s, in_i, in_d = _alignment_error_counts(in_align)
    out_s, out_i, out_d = _alignment_error_counts(out_align)
    in_total = in_s + in_i + in_d
    out_total = out_s + out_i + out_d
    in_wer = (in_total / ref_words * 100) if ref_words else 0.0
    out_wer = (out_total / ref_words * 100) if ref_words else 0.0

    in_annotated = format_annotated_transcript(in_align)
    out_annotated = format_annotated_transcript(out_align)

    return f"""\
# Transcription Report

| | |
|---|---|
| **Input** | `{input_file}` |
| **Output** | `{output_file}` |
| **Filter** | {filter_name} |
| **STT Model** | `{stt_model}` |

## Metrics

| Metric | Original | After {filter_name} |
|--------|----------|------|
| Word Error Rate (WER) | {in_wer:.1f}% | {out_wer:.1f}% |
| Substitutions | {in_s} | {out_s} |
| Insertions | {in_i} | {out_i} |
| Deletions | {in_d} | {out_d} |
| Total Errors | {in_total} | {out_total} |
| Reference Words | {ref_words} | {ref_words} |

## Error Legend

| Syntax | Meaning |
|--------|---------|
| ~~word~~ | Missing word (in ground truth but not transcribed) |
| **word** | Extra word (transcribed but not in ground truth) |
| ~~expected~~**actual** | Wrong word (substitution) |

## Ground Truth

{ground_truth}

## Original Transcription

{input_transcript}

### Diff

{in_annotated}

## After {filter_name}

{output_transcript}

### Diff

{out_annotated}
"""


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
            force=True,
        )
    else:
        # Create Rich handler with beautiful formatting
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_suppress=[rtc, api, noise_cancellation, ai_coustics],
        )

        # Configure logging
        logging.basicConfig(
            level=level, format="%(message)s", handlers=[rich_handler], force=True
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
  uv run noise-canceller.py input.wav -t ground_truth.txt
  uv run noise-canceller.py input.wav -t ground_truth.txt --stt-model deepgram/nova-3:en
  
📁 Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC, AIFF, and more
📝 Note: Some formats may require ffmpeg to be installed
📡 The tool will show track publication status automatically
  
🔧 Environment variables:
  LIVEKIT_URL: Your LiveKit Cloud server URL
  LIVEKIT_API_KEY: Your LiveKit API key  
  LIVEKIT_API_SECRET: Your LiveKit API secret
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (default: output/<input-file-name>-processed.wav)",
    )
    parser.add_argument(
        "--filter",
        choices=[
            "NC",
            "BVC",
            "BVCTelephony",
            "WebRTC",
            "aic-quail-l",
            "aic-quail-vfl",
            "all",
        ],
        default="NC",
        help="Noise cancellation filter type (default: NC). 'all' runs every filter and saves separate output files.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "-s", "--silent", action="store_true", help="Suppress all output (silent mode)"
    )
    parser.add_argument(
        "-t",
        "--transcript",
        type=str,
        help="Path to ground-truth transcript file. When provided, the tool "
        "transcribes both original and processed audio, compares to the "
        "ground truth, and writes a Markdown report alongside each output.",
    )
    parser.add_argument(
        "--stt",
        type=str,
        default="deepgram/nova-3:en",
        help="LiveKit Inference STT model (default: deepgram/nova-3:en). "
        "Format: provider/model[:language]",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Process audio directly through the plugin's FrameProcessor "
        "without routing through the LiveKit SFU.  Bypasses Opus "
        "encode/decode so output is bit-exact with direct FFI processing.  "
        "Only compatible with ai-coustics filters (aic-quail-l, aic-quail-vfl).",
    )

    args = parser.parse_args()

    # --direct is only meaningful for ai-coustics FrameProcessor filters.
    _AIC_FILTERS = {"aic-quail-l", "aic-quail-vfl"}
    if args.direct:
        if args.filter == "all":
            parser.error(
                "--direct cannot be used with --filter all (it only supports "
                "ai-coustics filters: aic-quail-l, aic-quail-vfl)"
            )
        if args.filter not in _AIC_FILTERS:
            parser.error(
                f"--direct is only supported with ai-coustics filters "
                f"({', '.join(sorted(_AIC_FILTERS))}), not '{args.filter}'"
            )

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
                style="red",
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

    # Validate transcript file if provided
    if args.transcript:
        transcript_path = Path(args.transcript)
        if not transcript_path.exists():
            if not args.silent:
                console.print(
                    f"❌ [red]Transcript file '{transcript_path}' does not exist[/red]"
                )
            else:
                sys.stderr.write(
                    f"ERROR: Transcript file '{transcript_path}' does not exist\n"
                )
            sys.exit(1)

    # Build filter config(s)
    filter_map = {
        "NC": lambda: noise_cancellation.NC(),
        "BVC": lambda: noise_cancellation.BVC(),
        "BVCTelephony": lambda: noise_cancellation.BVCTelephony(),
        "aic-quail-l": lambda: ai_coustics.audio_enhancement(
            model=EnhancerModel.QUAIL_L
        ),
        "aic-quail-vfl": lambda: ai_coustics.audio_enhancement(
            model=EnhancerModel.QUAIL_VF_L
        ),
    }
    ALL_FILTERS = [
        "NC",
        "BVC",
        "BVCTelephony",
        "WebRTC",
        "aic-quail-l",
        "aic-quail-vfl",
    ]
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
        filter_configs.append(
            {
                "filter": fk,
                "noise_filter": nf,
                "use_webrtc": use_webrtc,
                "output": str(out),
            }
        )

    _config.update(
        {
            "input_file": str(input_path),
            "filters": filter_configs,
            "silent": args.silent,
            "transcript": args.transcript,
            "stt": args.stt,
            "direct": args.direct,
        }
    )

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
