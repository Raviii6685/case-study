"""Reproducibility check: N untouched runs in fresh processes, then diff.

Graded columns (route, week_of, numbers, flagged, matched_note_id) must be
identical. reason is allowed to vary in wording when an LLM writes it, so it is
compared and reported separately.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import config

STRICT_COLUMNS = [c for c in config.OUTPUT_COLUMNS if c != "reason"]


def reproduce(runs: int = 3, provider: str | None = None, base_dir: Path = config.OUTPUT_DIR / "repro") -> str:
    paths = []
    for i in range(1, runs + 1):
        run_dir = base_dir / f"run_{i}"
        cmd = [sys.executable, "-m", "freight_watch", "run", "--output-dir", str(run_dir), "--no-cache", "--quiet"]
        if provider:
            cmd += ["--llm", provider]
        subprocess.run(cmd, check=True, cwd=config.ROOT)
        paths.append(run_dir / "flagged_routes.csv")

    frames = [pd.read_csv(p, dtype=str).fillna("") for p in paths]
    lines = ["# Reproducibility report", "", f"{runs} runs, each in a fresh process with the LLM cache disabled.", "",
             "| Run | Rows | SHA-256 (graded columns) | SHA-256 (full file) | LLM | live calls | failed calls |",
             "|---|---|---|---|---|---|---|"]
    llm_live = []
    for i, (p, f) in enumerate(zip(paths, frames), start=1):
        summary = json.loads((p.parent / "run_summary.json").read_text())
        usage = summary["token_usage"]
        llm_live.append(usage["llm_calls_live"])
        lines.append(f"| {i} | {len(f)} | `{_digest(f[STRICT_COLUMNS])}` | `{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}` "
                     f"| {summary['llm_provider']} {summary['llm_model'] or ''} | {usage['llm_calls_live']} "
                     f"| {usage['llm_calls_failed']} |")

    strict_same = all(f[STRICT_COLUMNS].equals(frames[0][STRICT_COLUMNS]) for f in frames[1:])
    reason_same = all(f["reason"].equals(frames[0]["reason"]) for f in frames[1:] if len(f) == len(frames[0]))
    lines += [
        "",
        f"- Flags, numbers, verdicts and cited notes identical across runs: **{'YES' if strict_same else 'NO'}**",
        f"- Explanation wording identical across runs: **{'YES' if reason_same else 'NO (allowed by the brief)'}**",
    ]
    if provider and provider != "offline" and not all(llm_live):
        lines.append("- **WARNING: the LLM did not answer in every run**, so these runs only demonstrate the "
                     "deterministic fallback path, not LLM reproducibility.")
    if not strict_same:
        for i, f in enumerate(frames[1:], start=2):
            diff = frames[0][STRICT_COLUMNS].compare(f[STRICT_COLUMNS]) if len(f) == len(frames[0]) else "row count differs"
            lines += ["", f"## Run 1 vs run {i}", "```", str(diff), "```"]
    report = "\n".join(lines) + "\n"
    (base_dir / "repro_report.md").write_text(report)
    return report


def _digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()[:16]
