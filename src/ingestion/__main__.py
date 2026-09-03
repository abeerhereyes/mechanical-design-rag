"""Command line interface for course PDF ingestion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import discover_courses
from .pdf import ExtractionConfig
from .schemas import ContentKind, ReviewStatus
from .workflow import export_jsonl, ingest_course, read_document, review_document


DEFAULT_COURSES_DIR = Path(__file__).resolve().parents[2] / "data" / "courses"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.ingestion")
    parser.add_argument(
        "--courses-dir", type=Path, default=DEFAULT_COURSES_DIR,
        help="course registry root",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("courses", help="list registered courses")

    ingest = commands.add_parser("ingest", help="extract one course's PDFs")
    ingest.add_argument("course")
    ingest.add_argument("--source", action="append", dest="sources")
    ingest.add_argument("--ocr", action="store_true")
    ingest.add_argument("--vision", action="store_true")
    ingest.add_argument("--render-dpi", type=int, default=160)
    ingest.add_argument("--vision-model")
    ingest.add_argument("--ollama-base-url")
    ingest.add_argument("--duplicate-threshold", type=float, default=0.88)

    status = commands.add_parser("status", help="show review counts")
    status.add_argument("course", nargs="?")

    for name in ("approve", "reject"):
        review = commands.add_parser(name, help=f"{name} extracted units")
        review.add_argument("course")
        review.add_argument("document_id")
        review.add_argument("--unit", action="append", dest="units")
        review.add_argument(
            "--kind",
            action="append",
            choices=[kind.value for kind in ContentKind],
            dest="kinds",
            help="limit approval/rejection to one or more content kinds",
        )

    export = commands.add_parser("export", help="write approved indexable JSONL")
    export.add_argument("output", type=Path)
    export.add_argument("--course")
    export.add_argument("--include-duplicates", action="store_true")
    return parser


def _course(registry, course_id):
    try:
        return registry[course_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown course {course_id!r}; available: "
            f"{', '.join(sorted(registry)) or '(none)'}"
        ) from exc


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        registry = discover_courses(args.courses_dir)
        if args.command == "courses":
            for course_id, (manifest, _) in registry.items():
                print(f"{course_id}\t{manifest.title}\t{len(manifest.sources)} sources")
            return 0
        if args.command == "ingest":
            manifest, course_dir = _course(registry, args.course)
            config = ExtractionConfig(
                render_dpi=args.render_dpi,
                ocr=args.ocr,
                vision=args.vision,
            )
            if args.vision_model:
                config.vision_model = args.vision_model
            if args.ollama_base_url:
                config.ollama_base_url = args.ollama_base_url
            paths = ingest_course(
                manifest,
                course_dir,
                args.sources,
                config,
                args.duplicate_threshold,
            )
            for path in paths:
                print(path)
            print("Extraction is not indexable until explicitly approved.")
            return 0
        if args.command == "status":
            selected = [_course(registry, args.course)] if args.course else registry.values()
            for manifest, course_dir in selected:
                counts = {status.value: 0 for status in ReviewStatus}
                documents = 0
                for path in sorted((course_dir / "raw").glob("*.json")):
                    documents += 1
                    for unit in read_document(path).units:
                        counts[unit.review_status.value] += 1
                print(json.dumps(
                    {"course": manifest.id, "documents": documents, "units": counts},
                    sort_keys=True,
                ))
            return 0
        if args.command in {"approve", "reject"}:
            _, course_dir = _course(registry, args.course)
            action = (
                ReviewStatus.APPROVED
                if args.command == "approve"
                else ReviewStatus.REJECTED
            )
            kinds = [ContentKind(kind) for kind in (args.kinds or [])]
            print(
                review_document(
                    course_dir,
                    args.document_id,
                    action,
                    args.units,
                    kinds,
                )
            )
            return 0
        if args.command == "export":
            count = export_jsonl(
                args.courses_dir,
                args.output,
                args.course,
                args.include_duplicates,
            )
            print(f"Exported {count} approved units to {args.output}")
            return 0
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
