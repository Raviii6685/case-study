"""CLI: python -m freight_watch {run, ask, eval, repro, mcp}"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

from . import config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freight_watch", description="Freight cost anomaly assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full pipeline and write output/flagged_routes.csv")
    run_p.add_argument("--llm", choices=["offline", *config.PROVIDERS], help="LLM provider (default: env / auto)")
    run_p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    run_p.add_argument("--no-cache", action="store_true", help="Disable the LLM response cache")
    run_p.add_argument("--no-dense", action="store_true", help="BM25-only retrieval (no embedding model)")
    run_p.add_argument("--quiet", action="store_true")
    run_p.add_argument("--own-threshold", type=float, default=config.OWN_HISTORY_THRESHOLD_PCT,
                       help="Flag when cost is at least this %% above its own 8-week history (default %(default)s)")
    run_p.add_argument("--peer-threshold", type=float, default=config.PEER_THRESHOLD_PCT,
                       help="Flag when cost is at least this %% above similar routes (default %(default)s)")
    run_p.add_argument("--all-weeks", action="store_true",
                       help="Also output unflagged route-weeks (flagged = No). Default: flagged weeks only")

    ask_p = sub.add_parser("ask", help="Ask a plain-English question about the findings")
    ask_p.add_argument("question")
    ask_p.add_argument("--llm", choices=list(config.PROVIDERS))
    ask_p.add_argument("--trace", action="store_true", help="Print the agent's step trace")

    sub.add_parser("eval", help="Score output/ against the hand-labelled eval set")

    repro_p = sub.add_parser("repro", help="Run the pipeline N times in fresh processes and diff")
    repro_p.add_argument("--runs", type=int, default=3)
    repro_p.add_argument("--llm", choices=["offline", *config.PROVIDERS])

    sub.add_parser("mcp", help="Serve the findings tools over MCP (stdio)")

    doctor_p = sub.add_parser("doctor", help="Pre-demo health check; warms the embedding cache")
    doctor_p.add_argument("--llm", choices=["offline", *config.PROVIDERS])
    doctor_p.add_argument("--no-llm-call", action="store_true", help="Skip the one-token LLM round trip")

    models_p = sub.add_parser("models", help="List the model ids your LLM key can access")
    models_p.add_argument("--llm", choices=list(config.PROVIDERS), default="groq")

    args = parser.parse_args(argv)
    quiet = getattr(args, "quiet", False)
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore")  # third-party deprecation noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "run":
        from .pipeline import run

        result = run(config.llm_settings_from_env(args.llm, use_cache=not args.no_cache),
                     output_dir=args.output_dir, use_dense=not args.no_dense,
                     own_threshold=args.own_threshold, peer_threshold=args.peer_threshold, all_weeks=args.all_weeks)
        if not quiet:
            s = result.summary
            print(f"\n{s['flagged_rows']} flagged route-weeks: {s['unexplained']} unexplained, {s['justified']} justified")
            print(f"LLM: {s['llm_provider']} {s['llm_model'] or ''} | retrieval: {s['retrieval']}")
            print("Token usage:", json.dumps({k: v for k, v in s["token_usage"].items() if k != "by_purpose"}))
            print(f"Output: {args.output_dir / 'flagged_routes.csv'}")
            usage = s["token_usage"]
            if s["llm_provider"] != "offline" and usage["llm_calls_failed"]:
                print(f"\n!! {usage['llm_calls_failed']} LLM call(s) failed; affected steps fell back to the "
                      "rule/template path. Output is valid, but this is NOT a successful LLM run.", file=sys.stderr)
        return 0

    if args.command == "ask":
        from .agent import ask
        from .llm import LLMClient
        from .tools import FindingsStore

        settings = config.llm_settings_from_env(args.llm)
        if not settings.enabled:
            print("Q&A needs an LLM. Set GROQ_API_KEY (free tier) or run Ollama and pass --llm ollama.", file=sys.stderr)
            return 2
        answer = ask(args.question, LLMClient(settings), FindingsStore())
        print(answer.answer)
        print(f"\n[{answer.status} in {answer.steps} steps, {answer.input_tokens}+{answer.output_tokens} tokens, "
              f"{answer.elapsed_s}s]", file=sys.stderr)
        if args.trace:
            print(json.dumps(answer.trace, indent=2), file=sys.stderr)
        return 0

    if args.command == "eval":
        from .evaluate import evaluate, render

        checks = evaluate()
        report = render(checks)
        (config.OUTPUT_DIR / "eval_report.md").write_text(report)
        print(report)
        return 0 if all(c.ok for c in checks) else 1

    if args.command == "repro":
        from .repro import reproduce

        print(reproduce(args.runs, args.llm))
        return 0

    if args.command == "doctor":
        from .doctor import render, run_checks

        results = run_checks(args.llm, call_llm=not args.no_llm_call)
        print(render(results))
        return 0 if all(ok for ok, _, _ in results) else 1

    if args.command == "models":
        from .llm import list_models

        settings = config.llm_settings_from_env(args.llm)
        print(f"Models available to your {args.llm} key (current default: {settings.model}):")
        for model_id in list_models(settings):
            print(" ", model_id)
        print("Use one with: export LLM_MODEL=<id>")
        return 0

    if args.command == "mcp":
        from .mcp_server import main as serve

        serve()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
