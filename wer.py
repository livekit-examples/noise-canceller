"""Word-error-rate helpers shared by noise-canceller.py and benchmark.py.

Keeping the alignment and report code in one module guarantees that the
single-file reports and the benchmark aggregates use the same WER definition.
"""

import re


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


def alignment_error_counts(
    alignment: list[tuple[str, str | None, str | None]],
) -> tuple[int, int, int]:
    """Return (substitutions, insertions, deletions) from an alignment."""
    subs = sum(1 for op, _, _ in alignment if op == "substitution")
    ins = sum(1 for op, _, _ in alignment if op == "insertion")
    dels = sum(1 for op, _, _ in alignment if op == "deletion")
    return subs, ins, dels


def score_transcript(ground_truth: str, hypothesis: str) -> dict:
    """Score a hypothesis against the ground truth.

    Returns a dict with the error counts, WER (fraction of reference words),
    and the raw hypothesis so downstream tooling can re-analyze without
    re-transcribing.
    """
    alignment = compute_word_alignment(ground_truth, hypothesis)
    subs, ins, dels = alignment_error_counts(alignment)
    ref_words = len(ground_truth.split())
    total = subs + ins + dels
    return {
        "wer": (total / ref_words) if ref_words else 0.0,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "total_errors": total,
        "ref_words": ref_words,
        "transcript": hypothesis,
    }


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
    in_s, in_i, in_d = alignment_error_counts(in_align)
    out_s, out_i, out_d = alignment_error_counts(out_align)
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
