#!/usr/bin/env python3
"""WER benchmark harness built on top of noise-canceller.py.

Runs a filter set over a full dataset of noisy clips with human transcripts,
collects per-clip WER metrics, and aggregates them into paired statistics
(processed vs. original) per filter.

Suites (kept separate on purpose — never average across them):

- voice-focus-examples  (n=10): ai-coustics' demo set. The docs samples come
  from these clips. Good as a smoke test, too small for conclusions.
- dawn-chorus-en        (n=450): ai-coustics' competing-talker benchmark
  (foreground speaker + background speech + noise, human transcripts).
  This is the dataset behind their published WER numbers. ai-coustics has
  since moved away from it internally: the clips are short, and Voice Focus
  2.1+ deliberately waits out a warm-up period before suppressing a
  background speaker who talks first, which penalizes short clips.
- aic-test-calls-en     (n=83): ai-coustics' recommended replacement — a
  published subset of their internal aic_calls set with longer, more
  realistic voice-agent recordings. voice-focus-examples is a small subset
  of the same source. They recommend enhancement level 0.8 for benchmarks.

All three datasets are CC BY-NC 4.0, fetched from Hugging Face at run time and
never committed to this repo.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "benchmark_data"
RESULTS_DIR = ROOT / "benchmark_results"

SUITES = {
    "voice-focus-examples": {
        "hf_dataset": "ai-coustics/voice-focus-examples",
        "audio_column": "mix",
    },
    "dawn-chorus-en": {
        "hf_dataset": "ai-coustics/dawn_chorus_en",
        "audio_column": "mix",
    },
    "aic-test-calls-en": {
        "hf_dataset": "ai-coustics/aic_test_calls_en",
        "audio_column": "mix",
    },
}


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> None:
    # Imported here so `run` and `report` work without the benchmark group.
    try:
        import soundfile as sf
        from datasets import Audio, load_dataset
    except ImportError:
        sys.exit(
            "fetch needs the 'benchmark' dependency group: "
            "uv run --group benchmark benchmark.py fetch ..."
        )

    suite = SUITES[args.suite]
    out_dir = DATA_DIR / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(suite["hf_dataset"])
    split = dataset[next(iter(dataset))]
    audio_column = suite["audio_column"]
    # decode=False keeps the raw bytes; soundfile decodes them without
    # needing torchcodec. Cast every audio column (dawn-chorus-en also has
    # a clean 'speech' reference we don't use).
    for name, feature in split.features.items():
        if isinstance(feature, Audio):
            split = split.cast_column(name, Audio(decode=False))
    print(f"{args.suite}: {len(split)} clips, columns: {split.column_names}")

    with open(out_dir / "manifest.jsonl", "w") as manifest:
        for i, row in enumerate(split):
            clip_id = str(row.get("id") or f"clip{i:04d}")
            samples, sr = sf.read(
                io.BytesIO(row[audio_column]["bytes"]), dtype="float32"
            )
            sf.write(out_dir / f"{clip_id}.wav", samples, sr)
            transcript = str(row["transcript"]).strip()
            (out_dir / f"{clip_id}.txt").write_text(transcript + "\n")
            condition = {
                k: v
                for k, v in row.items()
                if k != "transcript" and not isinstance(v, dict)
            }
            manifest.write(
                json.dumps({"id": clip_id, "condition": condition}) + "\n"
            )
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
    data_dir = DATA_DIR / args.suite
    clips = sorted(data_dir.glob("*.wav"))
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

    done = {
        (row["clip"], row["filter"]) for row in load_results(results_path)
    }
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
    deltas: list[float], iterations: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap CI of the mean paired delta."""
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        statistics.fmean(rng.choices(deltas, k=n)) for _ in range(iterations)
    )
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def cmd_report(args: argparse.Namespace) -> None:
    results_path = Path(args.results)
    rows = load_results(results_path)
    if not rows:
        sys.exit(f"no rows in {results_path}")

    suites = sorted({row.get("suite", "?") for row in rows})
    stt_models = sorted({row.get("stt_model", "?") for row in rows})
    lines = [
        "# WER benchmark report",
        "",
        f"- results: `{results_path}`",
        f"- suite(s): {', '.join(suites)}",
        f"- STT model(s): {', '.join(stt_models)}",
        "",
        "WER convention matches noise-canceller.py reports: "
        "(S+I+D) / reference words, punctuation-insensitive, case-insensitive.",
        "",
    ]

    for suite in suites:
        suite_rows = [r for r in rows if r.get("suite", "?") == suite]
        filters = sorted({r["filter"] for r in suite_rows})
        lines += [
            f"## {suite}",
            "",
            "| filter | n | WER orig | WER proc | Δ (pp) | 95% CI (pp) | rel. | better/tie/worse | S/I/D proc |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for fk in filters:
            frows = [r for r in suite_rows if r["filter"] == fk]
            orig = [r["original"]["wer"] for r in frows]
            proc = [r["processed"]["wer"] for r in frows]
            deltas = [p - o for o, p in zip(orig, proc)]
            mean_orig = statistics.fmean(orig)
            mean_proc = statistics.fmean(proc)
            mean_delta = statistics.fmean(deltas)
            if len(deltas) > 1:
                lo, hi = bootstrap_ci(deltas)
                significant = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
                ci = f"[{lo * 100:+.2f}, {hi * 100:+.2f}]"
            else:
                significant, ci = False, "n/a"
            rel = (mean_delta / mean_orig * 100) if mean_orig else 0.0
            better = sum(1 for d in deltas if d < 0)
            worse = sum(1 for d in deltas if d > 0)
            tie = len(deltas) - better - worse
            s = sum(r["processed"]["substitutions"] for r in frows)
            i_ = sum(r["processed"]["insertions"] for r in frows)
            d = sum(r["processed"]["deletions"] for r in frows)
            bold = "**" if significant else ""
            lines.append(
                f"| {fk} | {len(frows)} | {mean_orig * 100:.1f}% "
                f"| {mean_proc * 100:.1f}% "
                f"| {bold}{mean_delta * 100:+.2f}{bold} | {ci} | {rel:+.1f}% "
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
            "better. Bold Δ means the 95% CI excludes zero.",
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
