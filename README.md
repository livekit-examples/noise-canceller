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
- **aic-quail-l**: Ai-Coustics QUAIL-L speech enhancement
- **aic-quail-vfl**: Ai-Coustics QUAIL-VF-L speech enhancement
- **aic-quail-vfs**: Ai-Coustics QUAIL-VF-S speech enhancement
- **viva-voice-isolation**: Krisp Viva in voice isolation mode
- **viva-voice-isolation-telephony**: Krisp Viva in voice isolation mode optimized for telephony applications
- **BVC**: Background voice cancellation (removes background voices + noise)
- **BVCTelephony**: BVC optimized for telephony applications
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

## Benchmarks

`benchmark.py` runs the WER analysis above over full datasets instead of single
files, and aggregates the per-clip metrics into paired statistics per filter.

The suites come from ai-coustics and Krisp on Hugging Face (CC BY-NC 4.0,
downloaded at run time, never committed):

- `voice-focus-examples` (n=10) — ai-coustics' demo set (VF 2.3 version).
  A smoke test, too small for conclusions.
- `dawn-chorus-en` (n=450) — ai-coustics' competing-talker benchmark
  (foreground speaker + background speech + noise, human transcripts).
  The dataset behind their published WER numbers. ai-coustics no longer uses
  it internally: the clips are short, and Voice Focus 2.1+ waits out a
  warm-up period before suppressing a background speaker who talks first.
- `aic-test-calls-en` (n=110) — ai-coustics' recommended replacement,
  [`aic_test_calls_en`](https://huggingface.co/datasets/ai-coustics/aic_test_calls_en):
  real recorded calls. Every clip is
  [trimmed](https://huggingface.co/datasets/ai-coustics/aic_test_calls_en#trimming)
  to start 0.15 s before the caller's first word. ai-coustics recommends
  enhancement level 0.8 (`--enhancement-level 0.8`) for benchmarks.
- `aic-test-calls-en-untrimmed` (n=83) — the earlier, untrimmed release of
  the same set (tag `v1-83-untrimmed`). Some clips have a background talker
  before the caller speaks, so this suite also tests which voice a model
  locks on to. Compare it with the trimmed suite to see that effect.
- `krisp-vi-call-center` (n=106), `krisp-vi-work` (n=96) and
  `krisp-vi-phone-calls` (n=63) — Krisp's
  [Voice Isolation benchmark](https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset),
  32 kHz real recordings. In the call center and work sets, a scripted
  secondary talker speaks alone for about 30% of the time, and the correct
  transcript for that stretch is empty. The phone calls set is mostly a
  single caller on band-limited audio.

The suites are reported separately on purpose: averaging a 10-clip demo set
into a 450-clip benchmark would skew the distribution.

Each suite is pinned to a Hugging Face commit, and `benchmark_data/<suite>/`
records the revision it was fetched at. `run` stops if the local data or
`results.jsonl` comes from a different revision. Delete the data directory to
fetch the pinned revision, and move the old results away.

```bash
# 1. Run filters over a suite (fetches the dataset automatically on first
#    use; resumable, skips already-scored clips)
uv run benchmark.py run --suite dawn-chorus-en \
    --filters aic-quail-l,aic-quail-vfl,aic-quail-vfs --jobs 4

# 2. Aggregate into a Markdown report
uv run benchmark.py report benchmark_results/dawn-chorus-en/results.jsonl
```

The report shows, per filter: corpus (micro) WER on original vs. processed
audio, i.e. errors and reference words summed over all clips before dividing,
with the paired delta and a bootstrap 95% CI over clips; the mean of per-clip
WER with its own CI; better/tie/worse clip counts; and the
substitution/insertion/deletion decomposition. Corpus WER is the headline
because the per-clip mean weighs a five-word clip like a two-minute call, so a
single inserted phrase on a short clip (well over 100% WER) can move it by
several points. Raw per-clip metrics and
ASR hypotheses are kept in `results.jsonl` for re-analysis without
re-transcribing.

**Cost and time:** each clip is processed in real time through LiveKit Cloud
and consumes connection minutes. Dawn Chorus is ~2.4 hours of audio per
filter; `--jobs` runs clips concurrently. For ai-coustics filters,
`--direct` bypasses the SFU and runs faster than real time.

### Benchmarking unreleased ai-coustics plugin builds

The ai-coustics models run locally inside the installed
`livekit-plugins-ai-coustics` wheel (the cloud connection only supplies
license credentials), so the benchmark can gate an SDK or model upgrade
*before* release: build a wheel from the plugin branch and point uv at it,
then run the same benchmark against both wheels.

```toml
# pyproject.toml
[tool.uv.sources]
livekit-plugins-ai-coustics = { path = "../plugins-ai-coustics-internal/dist/livekit_plugins_ai_coustics-X.Y.Z-....whl" }
```

Same clips, same STT, same report — the only variable is the wheel.

## License

This tool is provided as-is under the MIT License. See [LICENSE](LICENSE) for details.
