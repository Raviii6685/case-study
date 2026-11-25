"""Read-only tools over the pipeline's own results (Section 4: few, well-described tools).

The same four tools back both the Q&A agent (OpenAI-style function calling) and
the MCP server, so there is one implementation and one set of schemas. Every
tool is read-only -> naturally idempotent, safe to retry and to run in parallel.
Errors are returned as {"error": ...} objects, never raised into the agent loop,
so the model always *knows* a call failed instead of guessing a result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from . import config
from .notes import Note, load_notes
from .retrieval import NoteIndex

Status = Literal["all", "unexplained", "justified"]


class FindingsStore:
    """Loads a finished run (output/) so questions are answered from *our* results only."""

    def __init__(self, output_dir: Path = config.OUTPUT_DIR, notes_path: Path = config.NOTES_CSV, use_dense: bool = True):
        flagged_path = output_dir / "flagged_routes.csv"
        if not flagged_path.exists():
            raise FileNotFoundError(f"{flagged_path} not found -- run `python -m freight_watch run` first")
        self.flags = pd.read_csv(flagged_path, dtype=str).fillna("")
        self.weekly = pd.read_csv(output_dir / "weekly_metrics.csv", dtype={"week_of": str})
        self.trace = {
            (r["route"], r["week_of"]): r
            for r in map(json.loads, (output_dir / "run_trace.jsonl").read_text().splitlines())
        }
        self.notes: list[Note] = load_notes(notes_path)
        self.index = NoteIndex(self.notes, use_dense=use_dense)
        self.routes = sorted(self.weekly["route"].unique())

    def resolve_route(self, route: str) -> str:
        wanted = re.sub(r"\s*(?:-|to|→|–)\s*", "-", route.strip(), flags=re.IGNORECASE).lower()
        for known in self.routes:
            if known.lower() == wanted:
                return known
        raise ValueError(f"Unknown route '{route}'. Known routes: {', '.join(self.routes)}")


# ---- argument models (validated before execution) ------------------------------

class ListFlagsArgs(BaseModel):
    route: str | None = Field(None, description="Route as Origin-Destination, e.g. 'Mumbai-Pune'. Omit for all routes.")
    month: str | None = Field(None, pattern=r"^\d{4}-\d{2}$", description="Calendar month YYYY-MM; matches weeks whose week_of Monday falls in it.")
    status: Status = Field("all", description="'unexplained' (flagged=Yes), 'justified' (No (justified)) or 'all'.")


class RouteWeeksArgs(BaseModel):
    route: str = Field(..., description="Route as Origin-Destination, e.g. 'Chennai-Bangalore'.")
    month: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="Calendar month YYYY-MM.")


class SearchNotesArgs(BaseModel):
    query: str = Field(..., min_length=2, description="Natural-language description of the cost driver being looked for.")
    route: str | None = Field(None, description="Optional route to restrict to notes for that route or All Routes.")


class GetNoteArgs(BaseModel):
    note_id: str = Field(..., pattern=r"^N\d{3}$", description="Note id such as 'N001'.")


# ---- implementations ------------------------------------------------------------

def list_flags(store: FindingsStore, args: ListFlagsArgs) -> dict:
    df = store.flags
    if args.route:
        df = df[df["route"] == store.resolve_route(args.route)]
    if args.month:
        df = df[df["week_of"].str.startswith(args.month)]
    if args.status == "unexplained":
        df = df[df["flagged"] == "Yes"]
    elif args.status == "justified":
        df = df[df["flagged"] == "No (justified)"]
    return {"count": len(df), "rows": df.to_dict(orient="records")}


def get_route_weeks(store: FindingsStore, args: RouteWeeksArgs) -> dict:
    route = store.resolve_route(args.route)
    weeks = store.weekly[(store.weekly["route"] == route) & store.weekly["week_of"].str.startswith(args.month)]
    if weeks.empty:
        return {"route": route, "month": args.month, "weeks": [], "note": "No shipments for this route in that month."}
    out = []
    for w in weeks.itertuples():
        flag = store.flags[(store.flags["route"] == route) & (store.flags["week_of"] == w.week_of)]
        out.append({
            "week_of": w.week_of,
            "cost_per_tonne_km": round(w.cost_per_tonne_km, 3),
            "own_history_avg": None if pd.isna(w.own_history_avg) else round(w.own_history_avg, 3),
            "vs_own_pct": None if pd.isna(w.vs_own_pct) else round(w.vs_own_pct, 1),
            "peer_avg": None if pd.isna(w.peer_avg) else round(w.peer_avg, 3),
            "vs_peer_pct": None if pd.isna(w.vs_peer_pct) else round(w.vs_peer_pct, 1),
            "flagged": flag.iloc[0]["flagged"] if len(flag) else "No (within normal range)",
            "matched_note_id": flag.iloc[0]["matched_note_id"] if len(flag) else "",
            "reason": flag.iloc[0]["reason"] if len(flag) else "",
            "evidence_trace": store.trace.get((route, w.week_of)),
        })
    return {"route": route, "month": args.month, "weeks": out}


def search_notes(store: FindingsStore, args: SearchNotesArgs) -> dict:
    pool = None
    if args.route:
        route = store.resolve_route(args.route)
        pool = [n for n in store.notes if n.applies_to_route(route)]
    hits = store.index.search(args.query, candidates=pool, top_k=3)
    return {"results": [_note_view(h.note) | {"score": round(h.score, 3)} for h in hits]}


def get_note(store: FindingsStore, args: GetNoteArgs) -> dict:
    for note in store.notes:
        if note.note_id == args.note_id:
            return _note_view(note)
    return {"error": f"No note with id {args.note_id}"}


def _note_view(note: Note) -> dict:
    return {
        "note_id": note.note_id,
        "date": note.date.isoformat(),
        "applies_to": note.applies_to,
        "text": note.text,
        "effective_window": f"{note.window_start} to {note.window_end}",
        "describes_cost_increase": note.raises_cost,
    }


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[FindingsStore, BaseModel], dict]

    def schema(self) -> dict:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": params}}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool(
        "list_flags",
        "List flagged route-weeks from the cost report. Use for questions like 'which routes were flagged', "
        "'what was unexplained in March 2025', or 'which flags were justified'. Returns rows with cost, "
        "comparisons, verdict, matched_note_id and reason.",
        ListFlagsArgs, list_flags,
    ),
    Tool(
        "get_route_weeks",
        "Get every week of one route in one calendar month with cost per tonne-km, both baselines, the "
        "flag verdict and the retrieval evidence trace. Use for 'why did <route> get pricier in <month>'. "
        "If the year is not given, call it for each year in the data (2024 and 2025).",
        RouteWeeksArgs, get_route_weeks,
    ),
    Tool(
        "search_notes",
        "Semantic + keyword search over the context notes (fuel, tolls, weather, festivals). Use to see "
        "what real-world reasons exist. Do NOT use it to decide verdicts -- the report already did that.",
        SearchNotesArgs, search_notes,
    ),
    Tool(
        "get_note",
        "Fetch one context note by id, with its effective date window and whether it describes a cost increase.",
        GetNoteArgs, get_note,
    ),
]}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS.values()]


def execute_tool(store: FindingsStore, name: str, raw_args: str | dict) -> dict:
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}"}
    try:
        payload = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        args = tool.args_model.model_validate(payload or {})
        return tool.fn(store, args)
    except (json.JSONDecodeError, ValidationError) as exc:
        return {"error": f"Invalid arguments for {name}: {exc}"}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # tool bug: report it, don't crash the loop
        return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}

