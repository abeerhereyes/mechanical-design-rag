"""Command-line interface for one-shot and interactive questions."""
from __future__ import annotations

import argparse
import inspect
import json
from typing import Any, Optional

from src.agent import MechanicalDesignAgent
from src.llm import OllamaError


def _print_result(result: dict) -> None:
    course_answer = result.get("course_answer") or result.get("answer") or ""
    general_answer = result.get("general_answer")
    print("Course notes:")
    print(course_answer)
    if general_answer:
        print("\nGeneral knowledge:")
        print(general_answer)
    if result.get("citations"):
        print("\nSources:")
        for citation in result["citations"]:
            print(
                f"- {citation['source']} — Section {citation['section']} "
                f"(p. {citation['page']})"
            )
    for label, key in (("Assumptions", "assumptions"), ("Warnings", "warnings")):
        if result.get(key):
            print(f"\n{label}:")
            for item in result[key]:
                print(f"- {item}")


def _ask(
    agent: MechanicalDesignAgent,
    query: str,
    course_id: str,
    include_general: bool,
    params: Optional[dict[str, Any]] = None,
) -> dict:
    """Bridge the current and intended multi-course agent signatures."""
    parameters = inspect.signature(agent.ask).parameters
    accepts_kwargs = any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if "params" in parameters or accepts_kwargs:
        kwargs["params"] = params or {}
    if "course_id" in parameters or accepts_kwargs:
        kwargs["course_id"] = course_id
    if "include_general" in parameters or accepts_kwargs:
        kwargs["include_general"] = include_general
    return agent.ask(query, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-course notes assistant")
    parser.add_argument("query", nargs="*", help="question to answer")
    parser.add_argument("--json", action="store_true", help="print the complete JSON result")
    parser.add_argument(
        "--course",
        default="mechanical-design",
        help="course ID (default: mechanical-design)",
    )
    parser.add_argument(
        "--no-general",
        action="store_true",
        help="exclude the separate general-knowledge answer",
    )
    args = parser.parse_args()

    agent = MechanicalDesignAgent()
    try:
        if args.query:
            result = _ask(
                agent,
                " ".join(args.query),
                args.course,
                not args.no_general,
            )
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                _print_result(result)
            return 0

        print(f"Course Notes Assistant [{args.course}]. Type 'quit' to exit.")
        while True:
            query = input("\nYou: ").strip()
            if query.lower() in {"quit", "exit"}:
                return 0
            if query:
                _print_result(
                    _ask(agent, query, args.course, not args.no_general)
                )
    except (OllamaError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
