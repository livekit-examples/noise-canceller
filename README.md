# LiveKit Audio Noise Canceller

A command-line tool that processes audio files with the LiveKit [enhanced noise cancellation](https://docs.livekit.io/cloud/noise-cancellation/) feature. Useful for testing, verification, or offline use.

## Limitations

- **Requires LiveKit Cloud**: As noise cancellation is a feature of paid LiveKit Cloud accounts, this tool consumes real connection minutes while in use (even though it runs locally).
- **Realtime output**: This tool outputs in realtime speed, so a 5 minute audio file will take 5 minutes to process.

## Installation

1. **Install dependencies:**
```bash
uv sync
```

2. **Set up LiveKit credentials:**

Add your LiveKit Cloud credentials to `.env`:

```bash
LIVEKIT_URL="wss://your-project.livekit.cloud"
LIVEKIT_API_KEY="your-api-key"
LIVEKIT_API_SECRET="your-api-secret"
```

## Usage

### Basic Usage
```bash
# Process input.mp3 and save to output/input-processed.wav
uv run noise-canceller.py input.mp3

# Specify custom output file
uv run noise-canceller.py input.wav -o clean_audio.wav

# Use different noise cancellation filter
uv run noise-canceller.py input.flac --filter BVC

# Use WebRTC built-in noise suppression (faster, local processing)
uv run noise-canceller.py input.wav --filter WebRTC

# Run all filters and save separate output files
uv run noise-canceller.py input.mp3 --filter all
```

### Filter Types

- **NC**: Standard enhanced noise cancellation (default)
- **BVC**: Background voice cancellation (removes background voices + noise)
- **BVCTelephony**: BVC optimized for telephony applications
- **aic-quail-l**: Ai-Coustics QUAIL-L speech enhancement
- **aic-quail-vfl**: Ai-Coustics QUAIL-VF-L speech enhancement
- **WebRTC**: For comparison purposes, apply WebRTC built-in `noise_suppression` to the audio

### Transcription & WER Analysis

When a ground-truth transcript is provided via `-t`, the tool transcribes both the original and processed audio using [LiveKit Inference STT](https://docs.livekit.io/agents/integrations/stt/) and generates a Markdown report comparing word error rates.

Transcription runs in parallel with audio processing — original audio chunks are streamed to both the noise cancellation pipeline and the STT service simultaneously, and processed chunks are sent to a second STT stream as they arrive from the pipeline.

```bash
# Transcribe and compare against ground truth
uv run noise-canceller.py input.mp3 --filter NC -t transcript.txt

# Use a different STT model
uv run noise-canceller.py input.mp3 --filter BVC -t transcript.txt --stt deepgram/nova-3:en

# Run all filters with transcription
uv run noise-canceller.py input.mp3 --filter all -t transcript.txt
```

The report is saved as a `.transcript.md` file alongside each output file and includes:

- **Metrics table** with WER, substitutions, insertions, and deletions for both original and processed audio
- **Raw transcripts** for both original and processed audio
- **Diff view** with errors annotated inline:
  - ~~word~~ — missing word (in ground truth but not transcribed)
  - **word** — extra word (transcribed but not in ground truth)
  - ~~expected~~**actual** — wrong word (substitution)

## License

This tool is provided as-is under the MIT License. See [LICENSE](LICENSE) for details.
