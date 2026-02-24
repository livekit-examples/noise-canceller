# LiveKit Audio Noise Canceller

A command-line tool that processes audio files with the LiveKit [enhanced noise cancellation](https://docs.livekit.io/cloud/noise-cancellation/) feature. Useful for testing, verification, or offline use.

## Limitations

- **Requires LiveKit Cloud**: Krisp-based filters (NC, BVC, BVCTelephony) are a feature of paid LiveKit Cloud accounts and consume real connection minutes while in use (even though the tool runs locally).
- **Ai-coustics billing**: `aic-quail-l` is included with LiveKit Cloud. `aic-quail-vfl` incurs additional usage-based cost.
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

# Use an Ai-coustics enhancement model
uv run noise-canceller.py input.wav --filter aic-quail-l
uv run noise-canceller.py input.wav --filter aic-quail-vfl

# Use WebRTC built-in noise suppression (faster, local processing)
uv run noise-canceller.py input.wav --filter WebRTC
```

### Filter types

**Krisp (LiveKit Cloud)**

- **NC**: Standard enhanced noise cancellation (default)
- **BVC**: Background voice cancellation — removes background voices and noise
- **BVCTelephony**: BVC optimized for telephony applications

**Ai-coustics**

- **aic-quail-l**: Tuned for voice AI and STT performance — optimizes downstream transcription accuracy rather than general audio quality
- **aic-quail-vfl**: Voice Focus variant — isolates the foreground speaker while suppressing background voices and noise; higher quality than `aic-quail-l` but incurs additional cost

**Other**

- **WebRTC**: Applies the WebRTC built-in `noise_suppression` locally for comparison purposes

## License

This tool is provided as-is under the MIT License. See [LICENSE](LICENSE) for details.