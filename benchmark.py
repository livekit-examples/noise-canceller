#!/usr/bin/env python3
"""WER benchmark harness built on top of noise-canceller.py.

Runs a filter set over a full dataset of noisy clips with human transcripts,
collects per-clip WER metrics, and aggregates them into paired statistics
(processed vs. original) per filter.

Suites (kept separate on purpose — never average across them):

- voice-focus-examples  (n=10): ai-coustics' demo set (VF 2.3 version). Good
  as a smoke test, too small for conclusions.
- dawn-chorus-en        (n=450): ai-coustics' competing-talker benchmark
  (foreground speaker + background speech + noise, human transcripts).
  This is the dataset behind their published WER numbers. ai-coustics has
  since moved away from it internally: the clips are short, and Voice Focus
  2.1+ deliberately waits out a warm-up period before suppressing a
  background speaker who talks first, which penalizes short clips.
- aic-test-calls-en     (n=110): ai-coustics' recommended replacement — real
  recorded calls, each trimmed to start 0.15 s before the caller's first
  word. They recommend enhancement level 0.8 for benchmarks.
- aic-test-calls-en-untrimmed (n=83): the earlier, untrimmed release of the
  same set. A background talker may speak before the caller, so this suite
  also measures which voice a model locks on to.
- krisp-vi-call-center  (n=106), krisp-vi-work (n=96), krisp-vi-phone-calls
  (n=63): Krisp's Voice Isolation benchmark. Real recordings at 32 kHz.
  Call center and work clips script a secondary talker who speaks alone for
  about 30% of the time (the correct transcript there is empty); phone calls
  are mostly a single caller on band-limited audio.

Every suite is pinned to a Hugging Face revision, so its clips cannot change
under existing results. All datasets are CC BY-NC 4.0, fetched from Hugging
Face at run time and never committed to this repo.

Usage:
    uv run benchmark.py run --suite dawn-chorus-en \
        --filters aic-quail-l,aic-quail-vfl,aic-quail-vfs --jobs 4
    uv run benchmark.py report benchmark_results/dawn-chorus-en/results.jsonl

`run` fetches the suite from Hugging Face automatically on first use; the
`fetch` subcommand exists only to pre-download.

`run` shells out to noise-canceller.py per clip, so each job consumes
LiveKit Cloud connection minutes and processes in real time — use --jobs to
run clips concurrently, or --direct (ai-coustics and Krisp Viva filters) to
bypass the SFU and run faster than real time. Runs are resumable: already-scored
(clip, filter) pairs are skipped.
"""

import argparse
import asyncio
import io
import json
import random
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "benchmark_data"
RESULTS_DIR = ROOT / "benchmark_results"

AIC_SUITE = {"audio_column": "mix", "transcript_column": "transcript"}
KRISP_SUITE = {
    "hf_dataset": "Krisp-AI/VoiceIsolation-Benchmark-Dataset",
    "revision": "626bac8ca07fb9b4846d4f62d27c321753160b36",
    "transcript_column": "full_transcript",
}

# Revisions are commit SHAs, not tags or branches: a dataset owner can move
# either, and resumed runs would then mix clips from two releases.
SUITES: dict[str, dict[str, str]] = {
    "voice-focus-examples": {
        **AIC_SUITE,
        "hf_dataset": "ai-coustics/voice-focus-examples",
        "revision": "499760ccd7655e3df626a357ad32282ff171e562",
    },
    "dawn-chorus-en": {
        **AIC_SUITE,
        "hf_dataset": "ai-coustics/dawn_chorus_en",
        "revision": "d0294c5f5f8a76e657fa3caef6408267ef931e4a",
    },
    "aic-test-calls-en": {
        **AIC_SUITE,
        "hf_dataset": "ai-coustics/aic_test_calls_en",
        "revision": "ddc6f123102bc08ee477e375e2f65c95189e93fc",
    },
    "aic-test-calls-en-untrimmed": {
        **AIC_SUITE,
        "hf_dataset": "ai-coustics/aic_test_calls_en",
        # tag v1-83-untrimmed
        "revision": "3825a2dc1a1146c17627fb92a022a8d43c0dd71c",
    },
    "krisp-vi-call-center": {**KRISP_SUITE, "config": "call_center"},
    "krisp-vi-work": {**KRISP_SUITE, "config": "work"},
    "krisp-vi-phone-calls": {**KRISP_SUITE, "config": "phone_calls"},
}
REVISION_FILE = "REVISION"


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def load_parquet_rows(suite: dict[str, str]) -> list[tuple[dict, bytes]]:
    """(row, audio bytes) for a suite stored as parquet (ai-coustics)."""
    from datasets import Audio, load_dataset

    dataset = load_dataset(suite["hf_dataset"], revision=suite["revision"])
    split = dataset[next(iter(dataset))]
    # decode=False keeps the raw bytes; soundfile decodes them without
    # needing torchcodec. Cast every audio column (dawn-chorus-en also has
    # a clean 'speech' reference we don't use).
    for name, feature in split.features.items():
        if isinstance(feature, Audio):
            split = split.cast_column(name, Audio(decode=False))
    return [(row, row[suite["audio_column"]]["bytes"]) for row in split]


def load_audiofolder_rows(suite: dict[str, str]) -> list[tuple[dict, bytes]]:
    """(row, audio bytes) for a suite stored as WAVs + metadata.jsonl (Krisp).

    load_dataset would re-encode this layout and needs torchcodec for that,
    so download the files directly instead.
    """
    from huggingface_hub import snapshot_download

    config = suite["config"]
    root = Path(
        snapshot_download(
            suite["hf_dataset"],
            repo_type="dataset",
            revision=suite["revision"],
            allow_patterns=f"{config}/*",
        )
    ) / config
    rows = load_results(root / "metadata.jsonl")
    return [(row, (root / row["file_name"]).read_bytes()) for row in rows]


def cmd_fetch(args: argparse.Namespace) -> None:
    # Imported here so `run` and `report` work without the benchmark group.
    try:
        import soundfile as sf
        import datasets  # noqa: F401
    except ImportError:
        sys.exit(
            "fetch needs the 'benchmark' dependency group: "
            "uv run --group benchmark benchmark.py fetch ..."
        )

    suite = SUITES[args.suite]
    out_dir = DATA_DIR / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)

    if "config" in suite:
        rows = load_audiofolder_rows(suite)
    else:
        rows = load_parquet_rows(suite)
    print(f"{args.suite}: {len(rows)} clips, columns: {list(rows[0][0])}")

    with open(out_dir / "manifest.jsonl", "w") as manifest:
        for i, (row, audio) in enumerate(rows):
            clip_id = str(row.get("id") or f"clip{i:04d}")
            samples, sr = sf.read(io.BytesIO(audio), dtype="float32")
            sf.write(out_dir / f"{clip_id}.wav", samples, sr)
            transcript = str(row[suite["transcript_column"]]).strip()
            (out_dir / f"{clip_id}.txt").write_text(transcript + "\n")
            condition = {
                k: v
                for k, v in row.items()
                if k != suite["transcript_column"] and not isinstance(v, dict)
            }
            manifest.write(
                json.dumps({"id": clip_id, "condition": condition}) + "\n"
            )
    # Written last, so an interrupted fetch leaves no revision marker.
    (out_dir / REVISION_FILE).write_text(suite["revision"] + "\n")
    print(f"written to {out_dir}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def load_results(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


async def cmd_run(args: argparse.Namespace) -> None:
    revision = SUITES[args.suite]["revision"]
    data_dir = DATA_DIR / args.suite
    clips = sorted(data_dir.glob("*.wav"))
    revision_path = data_dir / REVISION_FILE
    fetched = revision_path.read_text().strip() if revision_path.exists() else None
    if clips and fetched != revision:
        sys.exit(
            f"{data_dir} holds revision {fetched or 'unknown'}, but "
            f"{args.suite} is pinned to {revision}. Delete that directory "
            "to fetch the pinned revision."
        )
    if not clips:
        # First run: fetch the suite automatically. The subprocess pulls in
        # the 'benchmark' dependency group so this venv stays light.
        print(f"{args.suite}: no local data, fetching from Hugging Face...")
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "--group", "benchmark",
            "benchmark.py", "fetch", args.suite,
            cwd=ROOT,
        )
        await proc.communicate()
        if proc.returncode != 0:
            sys.exit(f"fetch failed (exit {proc.returncode})")
        clips = sorted(data_dir.glob("*.wav"))
        if not clips:
            sys.exit(f"fetch produced no clips in {data_dir}")
    if args.limit:
        clips = clips[: args.limit]

    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    results_dir = RESULTS_DIR / args.suite
    audio_dir = results_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"

    existing = load_results(results_path)
    stale = {row.get("dataset_revision") for row in existing} - {revision}
    if stale:
        sys.exit(
            f"{results_path} has rows from another dataset revision "
            f"({', '.join(sorted(r or 'unknown' for r in stale))}). "
            "Move it away before running against the pinned revision."
        )
    done = {(row["clip"], row["filter"]) for row in existing}
    todo: list[tuple[Path, list[str]]] = []
    for clip in clips:
        missing = [f for f in filters if (clip.stem, f) not in done]
        if missing:
            todo.append((clip, missing))
    print(
        f"{args.suite}: {len(clips)} clips x {len(filters)} filters, "
        f"{len(done)} already scored, {len(todo)} clips to run"
    )
    if not todo:
        return

    semaphore = asyncio.Semaphore(args.jobs)
    write_lock = asyncio.Lock()
    counters = {"ok": 0, "failed": 0}
    total_pairs = sum(len(fs) for _, fs in todo)

    async def run_clip(clip: Path, clip_filters: list[str]) -> None:
        transcript = clip.with_suffix(".txt")
        if not transcript.exists():
            print(f"SKIP {clip.stem}: no transcript", file=sys.stderr)
            return
        cmd = [
            "uv",
            "run",
            "noise-canceller.py",
            str(clip),
            "--filter",
            ",".join(clip_filters),
            "-t",
            str(transcript),
            "--silent",
            "--json",
            "--stt",
            args.stt,
            "--sample-rate",
            str(args.sample_rate),
            "--output-dir",
            str(audio_dir),
        ]
        if args.direct:
            cmd.append("--direct")
        if args.enhancement_level is not None:
            cmd += ["--ai-coustics-enhancement-level", str(args.enhancement_level)]

        async with semaphore:
            # One retry: jobs can fail transiently (room setup, STT stream).
            for _ in range(2):
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=ROOT,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode == 0:
                    break

        rows = []
        for fk in clip_filters:
            metrics_path = audio_dir / (
                f"{clip.stem}-{fk.lower()}-processed.metrics.json"
            )
            if not metrics_path.exists():
                counters["failed"] += 1
                print(
                    f"FAIL {clip.stem} [{fk}]: no metrics "
                    f"(exit {proc.returncode}): "
                    f"{stderr.decode(errors='replace').strip()[-300:]}",
                    file=sys.stderr,
                )
                continue
            metrics = json.loads(metrics_path.read_text())
            metrics["suite"] = args.suite
            metrics["dataset_revision"] = revision
            metrics["clip"] = clip.stem
            rows.append(metrics)
            counters["ok"] += 1

        async with write_lock:
            with open(results_path, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        total = counters["ok"] + counters["failed"]
        print(f"[{total}/{total_pairs}] {clip.stem}: {', '.join(clip_filters)}")

    await asyncio.gather(*(run_clip(clip, fs) for clip, fs in todo))
    print(
        f"done: {counters['ok']} scored, {counters['failed']} failed, "
        f"results in {results_path}"
    )
    if counters["ok"]:
        print(f"next: uv run benchmark.py report {results_path}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def bootstrap_ci(
    samples: list[Any],
    statistic: Callable[[list[Any]], float] = statistics.fmean,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% bootstrap CI of `statistic` over `samples` resampled with replacement.

    The default statistic is the mean, i.e. the CI of the mean paired delta.
    """
    rng = random.Random(seed)
    n = len(samples)
    values = sorted(
        statistic(rng.choices(samples, k=n)) for _ in range(iterations)
    )
    return values[int(0.025 * iterations)], values[int(0.975 * iterations) - 1]


def micro_wer(scores: list[dict]) -> float:
    """Corpus WER: total errors / total reference words (sclite-style)."""
    words = sum(s["ref_words"] for s in scores)
    return sum(s["total_errors"] for s in scores) / words if words else 0.0


def micro_wer_delta(frows: list[dict]) -> float:
    """Paired corpus-WER delta, processed - original, over the same clips."""
    return micro_wer([r["processed"] for r in frows]) - micro_wer(
        [r["original"] for r in frows]
    )


def format_ci(lo: float, hi: float) -> tuple[bool, str]:
    """(CI excludes zero, CI formatted in percentage points)."""
    significant = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    return significant, f"[{lo * 100:+.2f}, {hi * 100:+.2f}]"


def cmd_report(args: argparse.Namespace) -> None:
    results_path = Path(args.results)
    rows = load_results(results_path)
    if not rows:
        sys.exit(f"no rows in {results_path}")

    suites = sorted({row.get("suite", "?") for row in rows})
    stt_models = sorted({row.get("stt_model", "?") for row in rows})
    revisions = sorted({row.get("dataset_revision") or "unknown" for row in rows})
    lines = [
        "# WER benchmark report",
        "",
        f"- results: `{results_path}`",
        f"- suite(s): {', '.join(suites)}",
        f"- dataset revision(s): {', '.join(revisions)}",
        f"- STT model(s): {', '.join(stt_models)}",
        "",
        "WER convention matches noise-canceller.py reports: "
        "(S+I+D) / reference words, punctuation-insensitive, case-insensitive. "
        "Headline WER is corpus (micro) WER: errors and reference words are "
        "summed over all clips before dividing, so a long call counts more than "
        "a five-word clip. The mean of per-clip WER is shown alongside; it "
        "weighs every clip equally and is dominated by short clips, where a "
        "single inserted phrase can exceed 100% WER.",
        "",
    ]

    for suite in suites:
        suite_rows = [r for r in rows if r.get("suite", "?") == suite]
        filters = sorted({r["filter"] for r in suite_rows})
        lines += [
            f"## {suite}",
            "",
            "| filter | n | WER orig | WER proc | Δ (pp) | 95% CI (pp) | rel. "
            "| mean clip WER orig | mean clip WER proc | mean clip Δ (pp) "
            "| mean clip 95% CI (pp) | better/tie/worse | S/I/D proc |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for fk in filters:
            frows = [r for r in suite_rows if r["filter"] == fk]
            # Headline: corpus (micro) WER, every reference word weighs the same.
            wer_orig = micro_wer([r["original"] for r in frows])
            wer_proc = micro_wer([r["processed"] for r in frows])
            wer_delta = micro_wer_delta(frows)
            # Secondary: mean of per-clip WER (macro), every clip weighs the same.
            orig = [r["original"]["wer"] for r in frows]
            proc = [r["processed"]["wer"] for r in frows]
            deltas = [p - o for o, p in zip(orig, proc)]
            mean_orig = statistics.fmean(orig)
            mean_proc = statistics.fmean(proc)
            mean_delta = statistics.fmean(deltas)
            if len(deltas) > 1:
                significant, ci = format_ci(*bootstrap_ci(frows, micro_wer_delta))
                mean_significant, mean_ci = format_ci(*bootstrap_ci(deltas))
            else:
                significant, ci = False, "n/a"
                mean_significant, mean_ci = False, "n/a"
            rel = (wer_delta / wer_orig * 100) if wer_orig else 0.0
            better = sum(1 for d in deltas if d < 0)
            worse = sum(1 for d in deltas if d > 0)
            tie = len(deltas) - better - worse
            s = sum(r["processed"]["substitutions"] for r in frows)
            i_ = sum(r["processed"]["insertions"] for r in frows)
            d = sum(r["processed"]["deletions"] for r in frows)
            bold = "**" if significant else ""
            mean_bold = "**" if mean_significant else ""
            lines.append(
                f"| {fk} | {len(frows)} | {wer_orig * 100:.1f}% "
                f"| {wer_proc * 100:.1f}% "
                f"| {bold}{wer_delta * 100:+.2f}{bold} | {ci} | {rel:+.1f}% "
                f"| {mean_orig * 100:.1f}% | {mean_proc * 100:.1f}% "
                f"| {mean_bold}{mean_delta * 100:+.2f}{mean_bold} | {mean_ci} "
                f"| {better}/{tie}/{worse} | {s}/{i_}/{d} |"
            )
        # Original-audio error decomposition, once per suite (identical
        # across filters up to STT nondeterminism, so take the first).
        first = {r["clip"]: r for r in suite_rows}
        s = sum(r["original"]["substitutions"] for r in first.values())
        i_ = sum(r["original"]["insertions"] for r in first.values())
        d = sum(r["original"]["deletions"] for r in first.values())
        lines += [
            "",
            f"Original audio S/I/D: {s}/{i_}/{d} over {len(first)} clips. "
            "Δ is processed − original WER in percentage points; negative is "
            "better. Bold Δ means the 95% CI (bootstrap over clips) excludes "
            "zero.",
            "",
        ]

    report = "\n".join(lines)
    report_path = results_path.with_name("report.md")
    report_path.write_text(report)
    print(report)
    print(f"\nwritten to {report_path}", file=sys.stderr)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download a suite from Hugging Face")
    p_fetch.add_argument("suite", choices=sorted(SUITES))

    p_run = sub.add_parser("run", help="run filters over a suite")
    p_run.add_argument("--suite", choices=sorted(SUITES), required=True)
    p_run.add_argument(
        "--filters",
        default="aic-quail-l,aic-quail-vfl,aic-quail-vfs",
        help="comma-separated noise-canceller.py filter keys "
        "(default: the ai-coustics filters)",
    )
    p_run.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="concurrent clips; each job is its own room + STT streams (default: 4)",
    )
    p_run.add_argument(
        "--direct",
        action="store_true",
        help="bypass the SFU (ai-coustics and Krisp Viva filters, faster than real time)",
    )
    p_run.add_argument("--limit", type=int, help="only run the first N clips")
    p_run.add_argument("--stt", default="deepgram/nova-3:en")
    p_run.add_argument("--sample-rate", type=int, default=48_000)
    p_run.add_argument(
        "--enhancement-level",
        type=float,
        help="forwarded to --ai-coustics-enhancement-level",
    )

    p_report = sub.add_parser("report", help="aggregate a results.jsonl")
    p_report.add_argument("results")

    args = parser.parse_args()
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "run":
        asyncio.run(cmd_run(args))
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
