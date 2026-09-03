"""Command-line interface for one-shot and interactive questions."""
from __future__ import annotations

import argparse
import json

from src.agent import MechanicalDesignAgent
from src.llm import OllamaError


def _print_result(result: dict) -> None:
    print(result["answer"])
    if result.get("citations"):
        print("\nSources:")
        for citation in result["citations"]:
            print(
                f"- {citation['source']} — Section {citation['section']} "
                f"(p. {citation['page']})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Mechanical Design RAG assistant")
    parser.add_argument("query", nargs="*", help="question to answer")
    parser.add_argument("--json", action="store_true", help="print the complete JSON result")
    args = parser.parse_args()

    agent = MechanicalDesignAgent()
    try:
        if args.query:
            result = agent.ask(" ".join(args.query))
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                _print_result(result)
            return 0

        print("Mechanical Design RAG. Type 'quit' to exit.")
        while True:
            query = input("\nYou: ").strip()
            if query.lower() in {"quit", "exit"}:
                return 0
            if query:
                _print_result(agent.ask(query))
    except (OllamaError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
