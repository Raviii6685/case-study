# Freight Watch — design notes

The full reasoning behind the [README](../README.md). Read the README first.

Spots route-weeks where freight cost per tonne-km is rising out of the ordinary,
checks the context notes for a genuine reason, and explains each verdict in plain
English — citing the exact note, or saying plainly that it found none.

---

## How it works

```
shipment_records.csv ─► data.py ─► metrics.py ─► flagging.py ──┐            (deterministic, no LLM)
                        validate    weekly cost    rule:        │
                        + week_of   + 2 baselines  rising AND   │
                                                   out of range │
context_notes.csv ─► notes.py ─► retrieval.py ◄────────────────┘
                     windows,     1. metadata pre-filter (route scope, ±4 weeks)
                     impact       2. hybrid rank: bge-small dense (0.65) + BM25 (0.35)
                     rule         3. top-3 candidates
                                        │
                                        ▼
                                  verdict.py  ── hard checks in code: ROUTE · WINDOW · IMPACT · MARKET · MAGNITUDE
                                        │        (IMPACT optionally cross-checked by LLM, which can only veto)
                                        ▼
                                  pipeline.py ── justified → LLM explanation, validated, else template
                                        │        unexplained → template naming the closest near-miss note
                                        ▼
                    flagged_routes.csv · run_trace.jsonl · token_log.json · weekly_metrics.csv

output/ ─► tools.py (4 read-only tools) ─► agent.py (bounded ReAct Q&A)  and  mcp_server.py (MCP)
```

**Where it sits on the autonomy spectrum.** The batch job is a *fixed pipeline*:
the steps never change with the input, so a deterministic chain is the most
reliable, testable choice and a free-roaming agent would add risk without
benefit. The LLM is used only at narrow, validated points. Autonomy (a ReAct loop)
is reserved for the Q&A stretch goal, where the number of steps genuinely depends
on the question. One agent, not several: four tools and one job don't justify
multi-agent coordination overhead.

### Core logic (graded numbers)

| Step | Definition used | Notes |
|---|---|---|
| Route | `origin-destination`, grouped with the provided `route_type` | Each route has exactly one type (validated). |
| Week | Monday–Sunday; `week_of` = that Monday | Time-of-day stripped before bucketing. |
| Weekly cost / tonne-km | **Σ cost ÷ Σ(tonnes × km)** over the route's shipments that week | Volume-weighted. Reproduces all three sample rows exactly (4.17 / +35.5% / +21.0% etc.). The mean of per-shipment ratios gives 4.16 / +35.2% and does not. |
| vs own history | Mean of the route's weekly values in the **8 calendar weeks strictly before** this week | No look-ahead (unit-tested). Calendar-based, so a missing week is not silently replaced by an older one. |
| vs similar routes | Mean of **other** routes of the same `route_type`, same week | Route excluded from its own peer average (unit-tested). |

**Fewer than 8 prior weeks.** The first 8 weeks of each route (56 of 728 route-weeks,
all in Jan–Feb 2024) use all prior weeks available — 1 to 7 — with no padding or
extrapolation. The very first week has no history, so it cannot be "rising" and is
never flagged. `prior_weeks_used` is recorded per week in `output/weekly_metrics.csv`.
None of these early weeks end up flagged.

### The flag rule and choosing the thresholds

A week is flagged when it is **rising** (above its own trailing average) **and**
either **≥ 15% above its own history** or **≥ 20% above similar routes**.

The thresholds are set from the data's own noise, not picked by feel:

| Signal | Std. dev. across 728 route-weeks | Threshold | Meaning |
|---|---|---|---|
| vs own history | 4.2 pp | 15% | ~3.5 σ |
| vs similar routes | 9.0 pp | 20% | ~2.2 σ, above the 98th percentile |

The peer signal is noisier because Short and Long routes have **only one peer
route each**, so "peer average" is a single route. Hence the higher bar.

**Sensitivity.** The three spike episodes (Delhi-Jaipur Nov 2024, Ahmedabad-Mumbai
festival, Chennai-Bangalore flood period) and all three justified verdicts are
identical at every setting tried. Only the number of weeks in the slow Mumbai-Pune
creep changes:

| own / peer threshold | 10/15 | 15/15 | **15/20** | 20/20 | 15/25 |
|---|---|---|---|---|---|
| rows flagged | 27 | 26 | **19** | 19 | 13 |
| of which Mumbai-Pune creep | 19 | 19 | **12** | 12 | 6 |

The rule also reproduces the sample's flag decisions, including Mumbai-Pune
2025-09-15, which is only +9.2% vs history but +23.6% vs peers. A creeping route
drags its own baseline up with it, so the peer comparison is what catches it.
Rows are output only for anomalous weeks. Unremarkable weeks are in `weekly_metrics.csv`.

### The AI layer: retrieval, verification, explanation

**1. Notes become checkable facts** (`notes.py`). Each note gets:
- an **effective window** read from its text. The first matching rule wins:
  1. explicit range: "from Feb 24 to Mar 8", "between Mar 3 and Mar 9", "Dec 29 to Jan 4", which crosses the year boundary
  2. end date: "until March 15th"
  3. start date: "effective from 7 April", which is open-ended
  4. stated duration: "for two weeks"
  5. "starting this week" or "onwards", which is open-ended
  6. otherwise the issue week

  Dates can be "Mar 8", "8th of March", "March 8, 2025" or ISO. Open-ended changes count for 8 weeks, which is when the rolling baseline absorbs them.
- a **cost-impact** judgement. It is vetoed by whole-note statements ("not affected", "waived", "did not increase", "costs fell", "unchanged"). An increase phrase ("surcharge", "rates climbed", "pushed rates up", "toll hike") counts only if its own sentence isn't negated, so "No surcharge was applied" is not an increase.
- any **stated size** ("5-7%").

**2. Retrieval** (`retrieval.py`). The brief suggests a vector DB with RAG. With
10 one-sentence notes, each note is one chunk. There's no external database; a
numpy matrix is the vector index. The steps:
- Metadata pre-filter: route is this route or "All Routes", and the window is within ±4 weeks.
- Hybrid rank: local open-source `BAAI/bge-small-en-v1.5` embeddings (ONNX, deterministic on CPU, $0) blended 0.65/0.35 with BM25 keyword scores, top 3. If the embedding model is unavailable, it falls back to BM25 only.

Retrieval only *proposes* candidates. It never decides.

**3. Hard checks in code** (`verdict.py`). The LLM decides *what*; code decides *whether*.
A note justifies a flag only if all of these pass:

| Check | Why it exists | Trap in this dataset it catches |
|---|---|---|
| ROUTE | Wrong-route citations count as fabricated | N002 (Ahmedabad-Mumbai) near any other Jan 2025 spike |
| WINDOW | Wrong-window citations count as fabricated | Chennai-Bangalore wk of Mar 10: flood note N001 ends Mar 8 |
| IMPACT | "A note that says costs weren't affected does not count" | N005, N006, N008, N009 ("returned to normal"), N010 ("without a rate change"), N004 (routes not in dataset), N007 (roads *improved*) |
| MARKET | An all-routes change hits the peers too, so it can't explain a gap vs peers | Mumbai-Pune wk of Jun 23: diesel N003 is in window but the flag is +20.6% vs peers |
| MAGNITUDE | A note stating "5-7%" cannot explain +30% | Diesel N003 against any large own-history jump |

If no candidate passes, the verdict is **unexplained** and `matched_note_id` is blank.
The reason names the closest near-miss and why it failed, for example *"The closest
note (N001, 2025-02-24) covers 2025-02-24 to 2025-03-08, outside this week…"*. This
mirrors the sample's third row, and a reviewer can see the system looked rather than gave up.

**4. Where the LLM is used** (`llm_tasks.py`). There are two narrow, separate calls,
each auditable on its own:

| Call | When | Guardrails |
|---|---|---|
| `impact_check` | Once per distinct candidate note (6 per run), run in parallel | JSON validated with Pydantic. The evidence must be a **verbatim quote** from the note. One corrective retry, then fall back to the rule. Final = rule **AND** LLM, so the LLM can only make a verdict *more* conservative. Disagreements are logged in `run_summary.json`. |
| `explain` | Once per **justified** row (3 per run), run in parallel | Checked against concrete criteria: must cite the matched note ID and its date, no other note IDs, one short paragraph. One rewrite with the specific problems, then a deterministic template. |

Unexplained rows need no LLM call: there is nothing to explain, so nothing to invent.
Note text is passed as data inside `<note>` tags with an instruction to ignore embedded
instructions. Even if a note *did* contain an injected instruction, it cannot change
a verdict, because verdicts are code (tested in `test_verdict.py`).

### Stretch goal: ask questions about the findings

`python -m freight_watch ask "…"` runs a **bounded ReAct loop** (`agent.py`) over
four read-only tools (`tools.py`):

| Tool | Use |
|---|---|
| `list_flags(route?, month?, status?)` | which weeks were flagged / justified / unexplained |
| `get_route_weeks(route, month)` | every week of a route in a month, with both baselines, verdict and evidence trace |
| `search_notes(query, route?)` | hybrid search over the notes |
| `get_note(note_id)` | one note with its effective window and impact |

"Why did X get pricier in March?" resolves by matching `week_of` Mondays in that
month. Without a year, the agent is told to check both 2024 and 2025.

The loop is bounded in several ways:
- **Budgets:** 6 steps, 60 s, 40K tokens. On exhaustion it gives an honest fallback, never a guess.
- **Parallel tool calls** when the model asks for several at once.
- **Tool errors come back as `{"error": …}`**, so the model knows a call failed.
- **Input screening** for prompt-injection patterns.
- **Output grounding check:** every note ID and ISO date in the answer must have appeared in a tool result this session, or the answer is withheld.
- **Full step trace** with `--trace`.

The same four tools are served over **MCP** (`mcp_server.py`, MCP Python SDK v2).
Any MCP client can use them without custom integration code.

---

## How I convinced myself it is right

**1. Eval harness** (`python -m freight_watch eval` → `output/eval_report.md`). It
scores against hand-made labels in `evals/`, not against the pipeline's own logic:

| Check | Result |
|---|---|
| Format contract: columns, order and types identical to the sample; `matched_note_id` blank iff `Yes`; justified reasons name their note | 20/20 |
| The three sample rows reproduce exactly (cost, both % strings, flag, note ID) | 3/3 |
| Golden set: 21 hand-labelled route-weeks. 3 sample, 4 happy-path, 5 trap (window, negation, market-wide), 9 should-*not*-flag (quiet note weeks, market-wide diesel, recoveries, noise) | 21/21 |
| Citation audit: every justified row re-checked against **hand-labelled** note windows, scopes and impact (`evals/note_labels.csv`). This is the brief's "spot-check justified rows" test. | 3/3 |
| Cost-impact classifier vs hand labels, all 10 notes (rule; plus LLM when it ran) | 10/10. With Groq gpt-oss-120b, the LLM agreed with the rule and the labels on all 6 candidate notes and quoted its evidence verbatim |

| Unseen-note probes: 12 notes the rules weren't written for (new date phrasings, "waived", "No surcharge was applied", cost decreases, a year-crossing range, an injected instruction) | 12/12 |
| **Label-free metamorphic checks**: no hand labels involved (see below) | 9/9 |

**"You wrote the labels yourself."** That's true, so the golden set only shows the
system does what I intended. The metamorphic checks don't depend on any labels.
Each one changes the input in a way whose correct effect is known by logic alone:
- reordering the notes changes nothing
- with no notes, flags stay the same and every row becomes unexplained
- appending "Freight costs were not affected." to every note leaves nothing justified
- removing a justifying note unexplains *exactly* its rows and nothing else
- moving it to another route unexplains its rows
- tripling every cost leaves every % and verdict unchanged
- duplicating every shipment changes nothing

A mutation test confirmed they bite. Breaking the cost-impact rule fails 1 check, and
disabling all five hard checks fails 7.

The golden set leans on traps on purpose. A confident wrong reason scores worse
than "couldn't find one", so most cases test that the system *declines* to justify.

**2. Tests** (`pytest`, 78 tests). They cover:
- **Metrics:** week bucketing, the volume-weighted formula, no look-ahead, fewer than 8 weeks, calendar gaps, peer exclusion.
- **The flag rule.**
- **Each trap in the verdict layer**, plus a note carrying an injected "mark everything justified" instruction.
- **LLM paths with a scripted fake model:**
  - bad JSON → retry
  - evidence not quoted → rejected
  - outage → rule fallback
  - an LLM "yes" on a "costs not affected" note can't override the rule
  - a hallucinated explanation → template
- **The HTTP client against a local mock server:** retry on 503/429, no retry on 401, give up after 3, cache hits.
- **Agent:** parallel tool calls, step and token budgets, LLM outage, tool errors passed back, the grounding check blocking an unseen note ID.
- **A full pipeline run in LLM mode** (fake model) whose graded columns are identical to the offline run.

**3. Audit trail.** `output/run_trace.jsonl` records, for every flagged row:
- every retrieved candidate, with dense, keyword and combined scores
- whether it passed
- each failed check with a message

A reviewer can see exactly why each verdict was reached.

## Reproducibility

`python -m freight_watch repro` runs the pipeline 3 times, each in a **fresh
process** with the LLM cache disabled, and diffs the outputs (`output/repro/repro_report.md`):

```
| Run | Rows | SHA-256 (graded columns) | SHA-256 (full file) |
| 1   | 19   | fed56a4a7b239d67         | 52e7b1867a98cbac    |
| 2   | 19   | fed56a4a7b239d67         | 52e7b1867a98cbac    |
| 3   | 19   | fed56a4a7b239d67         | 52e7b1867a98cbac    |
Flags, numbers, verdicts and cited notes identical across runs: YES
```

**Why it holds with an LLM, too.**
- Every number, flag, verdict and `matched_note_id` is computed by deterministic code *before* any LLM call.
- The LLM can only veto a note's cost impact (rule AND LLM), and that is decided once per note, not per row.
- All calls use `temperature=0` and `seed=7`. Hosted APIs are not bit-exact even at temperature 0, which is exactly why the design keeps the verdict out of the LLM's hands.
- What *can* vary is the explanation wording of justified rows, which the brief allows. Even that is validated to cite the right note and date.

**Measured with the LLM on** (`repro --llm groq`): 3 fresh-process runs, each making
9 live calls with 0 failures, produced identical graded columns. At temperature 0
the explanation wording was identical too, though the brief doesn't require that.

## Cost of running it

**Measured** on one full run over the entire shipment file (Groq, `openai/gpt-oss-120b`,
`--no-cache`), taken from `output/token_log.json`:

| | LLM calls | Input tokens | Output tokens | Cost at published rate ($0.15 / $0.60 per 1M) |
|---|---|---|---|---|
| `impact_check` (one per distinct candidate note) | 6 | 1,435 | 289 | |
| `explain` (one per justified row) | 3 | 845 | 265 | |
| **Total** | **9** | **2,280** | **554** | **$0.00067** (free tier: $0) |

The 16 unexplained rows need no LLM call. Output tokens include the model's
low-effort hidden reasoning. Each call averaged about 1 s. Embeddings run locally
(bge-small, ~130 MB one-off download) at $0. The Q&A agent is per question and not
part of the batch run. A typical question used 2 steps and about 3.2K tokens.

| Mode | LLM calls | Cost |
|---|---|---|
| Offline (rule + templates) | 0 | $0 |
| Groq `openai/gpt-oss-120b` (open-weight) | 9 | $0.00067 |
| Ollama local (`qwen2.5:7b-instruct`) | 9 | $0 |

Rates live in `freight_watch/config.py`, taken from console.groq.com/docs/models on 2026-10-14.

## Trade-offs and limitations

- **Rows emitted:** only anomalous route-weeks by default. The sample suggests a report of findings. `run --all-weeks` emits all 728 weeks (unflagged ones as `flagged = No`), and `--own-threshold` / `--peer-threshold` change the flag rule, so either grading convention is a flag away.
- **"Rising" means above the route's own trailing average.** Mumbai-Pune 2025-11-10 is +21.4% vs peers but −0.1% vs its own history, so it is not flagged that week. It is still inside a long, flagged run of weeks.
- **Open-ended notes** ("starting this week") are treated as able to justify a rise for 8 weeks, the time the rolling baseline needs to absorb a step change. This is a judgement call, stated here.
- **Rule-based impact check.** It is a phrase list with sentence-level negation. It matches all 10 hand labels and 12 unseen probe notes (`evals/probe_notes.csv`), but a phrase list will always miss some wordings. That's where the LLM cross-check earns its place. Because the check works as rule AND LLM, a rule miss makes the system *more* cautious, never less. Offline mode relies on the rule alone.
- **Peer groups are tiny** (one peer for Short and Long routes), so peer comparisons are noisy. The 20% bar reflects that.
- **The provided `sample_output_format_v2.csv` is not strictly valid CSV.** Its third row has an unquoted comma inside `reason`. The eval reader joins overflow fields back into `reason`, which is the last column. Our output is properly quoted.
- **Failures are reported, not hidden.** Failed LLM calls are counted in `token_log.json` (`llm_calls_failed`). `run` prints a warning, `repro` shows live and failed calls per run, and `eval` says whether LLM judgements were actually scored. Groq's default model is `openai/gpt-oss-120b`, an open-weight reasoning model. It is called with `reasoning_effort=low` and `include_reasoning=false`, and with output limits raised so hidden reasoning can't use up the budget before the answer. `llama-3.3-70b-versatile` returned `model_not_found` for a free-tier key. A run where the LLM was unreachable still produces a valid CSV through the rule and template fallback, but it can't pass as an LLM run. The first real Groq attempt hit exactly this: Cloudflare returned `403 / 1010` to Python's default User-Agent. The client now sends its own, and a regression test covers it.
- **Typographic characters in model output.** gpt-oss writes non-breaking hyphens and narrow spaces ("Chennai‑Bangalore", "2025‑03‑03"). All LLM text is normalised to ASCII when it arrives, and the agent's grounding check normalises again before matching dates. Without that, the first real agent answer passed the date check only because the check silently found nothing to compare. A regression test covers it.
- **Not production-scale by design:** no persistent vector DB, queue or long-term memory. Ten notes and one batch job don't need them. The index is a class, so swapping in pgvector or Qdrant would be a local change.

## Brief checklist

| Requirement | Status |
|---|---|
| Group by route + provided route_type | ✅ `data.py` |
| Cost per tonne per km, week by week (Mon–Sun, `week_of` = Monday) | ✅ `metrics.py`, matches sample exactly |
| vs own history: trailing 8 weeks, strictly prior, fewer → all available | ✅ unit-tested; 56 early weeks documented above |
| vs similar routes: same type, same week, self excluded | ✅ unit-tested |
| Flag rising + out-of-ordinary vs history **or** peers | ✅ `flagging.py`, thresholds justified above |
| Check notes before flagging (RAG) | ✅ pre-filter + hybrid dense/BM25 retrieval |
| Plain-English explanation grounded in notes | ✅ validated LLM text or template quoting the note |
| Q&A over findings (stretch) | ✅ bounded ReAct agent + MCP server |
| Justified ⇒ exact note, right route, right window, actually raises cost | ✅ five hard checks + independent citation audit |
| No supporting note ⇒ unexplained, blank `matched_note_id` | ✅ |
| Eval harness / labelled set | ✅ `evals/` + `evaluate.py`, all passing |
| 3 identical runs; randomness stated | ✅ `repro`; temperature 0, seed 7, LLM cannot move verdicts |
| Token / cost log; open-source / free tier preferred | ✅ ledger → `token_log.json`; Groq gpt-oss-120b (open-weight) / Ollama / offline |
| Output CSV in exact sample format | ✅ column contract checked by eval |

## Project layout

```
freight_watch/
  config.py        every threshold, path and provider setting, in one place
  data.py          load + validate shipments, week bucketing
  metrics.py       weekly cost per tonne-km and both baselines
  flagging.py      anomaly rule
  notes.py         note parsing: windows, impact rule, stated magnitude
  retrieval.py     metadata pre-filter + hybrid dense/BM25 index
  verdict.py       deterministic justification checks
  llm.py           OpenAI-compatible client: retries, cache, token ledger
  llm_tasks.py     impact_check + explain prompts with validation
  pipeline.py      orchestration and output writing
  tools.py         4 read-only tools shared by the agent and MCP
  agent.py         bounded ReAct Q&A with guardrails
  mcp_server.py    MCP server
  evaluate.py      eval harness
  repro.py         reproducibility runner
  doctor.py        pre-demo health check
evals/             golden_labels.csv, note_labels.csv, probe_notes.csv
tests/             78 tests
output/            committed results of the offline run
```
