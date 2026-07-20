"""Command-line smoke demo for the ingestion module."""

import sys
from argparse import ArgumentParser, Namespace

from rag_agent_platform.ingestion.loaders import build_default_loader_registry
from rag_agent_platform.ingestion.pipeline import RealIngestionPipeline


def build_parser() -> ArgumentParser:
    """Build the ingestion demo command parser."""
    parser = ArgumentParser(description="Run small ingestion module smoke checks.")
    parser.add_argument("file_path", nargs="?", help="Document path to ingest in later steps.")
    parser.add_argument("--list", action="store_true", help="List ingested documents.")
    parser.add_argument("--delete", metavar="DOCUMENT_ID", help="Delete one ingested document.")
    parser.add_argument(
        "--show-clean", action="store_true", help="Show cleaned text in later steps."
    )
    parser.add_argument(
        "--show-chunks", action="store_true", help="Show generated chunks in later steps."
    )
    parser.add_argument(
        "--index", action="store_true", help="Write chunks to vector storage in later steps."
    )
    parser.add_argument(
        "--repo-smoke", action="store_true", help="Run repository smoke checks later."
    )
    return parser


def main() -> None:
    """Run the ingestion demo."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    run(args)


def run(args: Namespace) -> None:
    """Dispatch the selected smoke action."""
    pipeline = RealIngestionPipeline()
    if args.list:
        documents = pipeline.list_documents()
        print(f"documents: {len(documents)}")
        return
    if args.delete:
        pipeline.delete_document(args.delete)
        print(f"deleted document: {args.delete}")
        return
    if args.file_path:
        loaded = build_default_loader_registry().load(args.file_path)
        preview = loaded.content.strip().replace("\n", " ")[:80]
        print(f"filename: {loaded.metadata['filename']}")
        print(f"file_type: {loaded.metadata['file_type']}")
        print(f"encoding: {loaded.metadata['encoding']}")
        print(f"characters: {len(loaded.content)}")
        print(f"preview: {preview}")
        return
    if args.show_clean or args.show_chunks or args.index or args.repo_smoke:
        print("selected smoke option is reserved for a later implementation step")
        return
    print("ingestion demo is ready; use --help to see available commands")


if __name__ == "__main__":
    main()
