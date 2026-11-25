# Freight Watch — a cost-anomaly assistant that refuses to guess

Flags route-weeks where freight cost per tonne-km is rising out of the ordinary,
checks the context notes for a genuine reason, and explains each verdict in plain
English — citing the exact note, or saying plainly that it found none.

**Result:** 19 flagged route-weeks: **16 unexplained, 3 justified**
(Ahmedabad-Mumbai festival week → N002; Chennai-Bangalore flood weeks of Feb 24 and Mar 3 → N001).
Full reasoning: **[docs/DESIGN.md](docs/DESIGN.md)**.

## Run it

```bash
make install                                   # venv + requirements
.venv/bin/python -m freight_watch doctor       # pre-flight check; warms the embedding cache
.venv/bin/python -m freight_watch run          # -> output/flagged_routes.csv (works offline)
.venv/bin/python -m freight_watch eval         # -> output/eval_report.md
.venv/bin/python -m freight_watch repro        # 3 fresh-process runs -> output/repro/repro_report.md
.venv/bin/python -m pytest -q                  # 78 tests
```

With an LLM (Groq free tier, open-weight `openai/gpt-oss-120b`; or `--llm ollama` locally):

```bash
export GROQ_API_KEY=...
.venv/bin/python -m freight_watch run --llm groq --no-cache
.venv/bin/python -m freight_watch ask "Why did Chennai-Bangalore get pricier in March 2025?"
```

Options: `--own-threshold` / `--peer-threshold` change the flag rule, and
`--all-weeks` also outputs the 709 unflagged weeks. `mcp` serves the Q&A tools to
any MCP client.

## How it works

1. **Numbers come from code, never the LLM.** Weeks run Monday–Sunday. Weekly cost per tonne-km is **Σ cost ÷ Σ(tonnes × km)**, which reproduces all three sample rows exactly.
   - **vs own history** = mean of the 8 calendar weeks strictly before. Fewer than 8 → all available: the first 8 weeks of each route, 56 of 728 route-weeks in Jan–Feb 2024. The first week has no history and is never flagged.
   - **vs similar routes** = mean of the *other* routes of the same type that week.
2. **Flag rule.** A week is flagged if it's rising vs its own history and either ≥15% above that history or ≥20% above peers. Both thresholds come from the data's noise (~3.5σ and ~2.2σ). The spike episodes and all 3 justified verdicts stay the same at every threshold tried; only the length of the slow Mumbai-Pune creep changes.
3. **Retrieval.** Notes are first filtered by route and date window (±4 weeks), then ranked with hybrid search: local `bge-small` embeddings (0.65) plus BM25 (0.35). Retrieval only proposes candidates.
4. **Five hard checks in code decide "justified".** The note must:
   - apply to the **route**
   - overlap the **week** (windows are read from the note text: "from Feb 24 to Mar 8", "until Mar 15", "for two weeks"…)
   - describe a **cost increase** ("costs were not affected", "waived" and "returned to normal" don't count)
   - not be an **all-routes** change used to explain a gap vs peers, which were hit too
   - not state a **size** far smaller than the rise ("5-7%" can't explain +30%)

   If no note passes, the row is unexplained, `matched_note_id` is blank, and the reason names the closest near-miss and why it failed.
5. **The LLM does two narrow jobs.** It cross-checks each note's cost impact (once per note; it can only *veto*) and writes the 3 justified explanations, each validated to cite the right note and date or replaced with a template. Its text is treated as data, so an injected instruction can't change a verdict.
6. **Stretch goal:** a bounded ReAct agent (6 steps / 60 s / 40K tokens) over 4 read-only tools. It screens questions for prompt injection and only answers with note IDs and dates that appeared in its tool results.

## How I know it's right

| Check (`freight_watch eval`) | Result |
|---|---|
| Output format identical to the sample contract | 20/20 |
| The 3 sample rows reproduce exactly | 3/3 |
| Golden set: 21 hand-labelled route-weeks, mostly traps | 21/21 |
| Citation audit: every justified row vs hand-labelled note windows, scope and impact | 3/3 |
| Note cost-impact: rule **and** Groq LLM vs hand labels | 16/16 |
| 12 unseen probe notes (new phrasings, negations, injection) | 12/12 |
| **Label-free** metamorphic checks (no hand labels involved) | 9/9 |

The golden labels are my own, so on their own they only show the system does what
I intended. The metamorphic checks don't depend on labels. For example: removing a
justifying note must unexplain exactly its rows, and tripling every cost must change
no percentage or verdict. Deliberately breaking the code makes them fail. Details in
[docs/DESIGN.md](docs/DESIGN.md#how-i-convinced-myself-it-is-right).

## Reproducibility

3 untouched runs in fresh processes, with the LLM on (cache disabled), gave identical
flags, numbers, verdicts and cited notes. At temperature 0 the wording was identical
too: [output/repro/repro_report.md](output/repro/repro_report.md). Every LLM call uses
`temperature=0`, `seed=7`. Even if a hosted model drifts, verdicts can't move,
because they're decided in code before the LLM runs and the LLM can only veto.

## Cost (one full run, measured)

| LLM calls | Input tokens | Output tokens | Cost (Groq `gpt-oss-120b`, $0.15 / $0.60 per 1M) |
|---|---|---|---|
| 9 (6 note checks + 3 explanations) | 2,280 | 554 | **$0.00067** ($0 on the free tier) |

The 16 unexplained rows need no LLM call. Embeddings run locally for $0. Per-call log:
[output/token_log.json](output/token_log.json). Offline mode makes 0 calls and gives
the same verdicts.

## Assumptions to know

- **Rows:** the output lists flagged weeks only, because the sample reads as a findings report (`--all-weeks` for every week).
- **"Rising"** means above the route's own trailing average.
- **Open-ended notes** ("starting this week") can justify a rise for 8 weeks, until the baseline absorbs it.
- **Sample file:** the provided `sample_output_format_v2.csv` has an unquoted comma in row 3's `reason`. Our output is properly quoted.

## Deliverables

`output/flagged_routes.csv` (the answer) · `output/token_log.json` · `output/repro/repro_report.md` ·
`output/eval_report.md` · `output/run_trace.jsonl` (why each verdict was reached) · code in `freight_watch/` ·
labels in `evals/` · tests in `tests/` · design notes in `docs/DESIGN.md`
