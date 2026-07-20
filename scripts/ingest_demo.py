"""Command-line smoke demo for the ingestion module."""

import sys
from argparse import ArgumentParser, Namespace
from tempfile import TemporaryDirectory

from rag_agent_platform.ingestion.chunker import ChunkingConfig, ParentChildChunker
from rag_agent_platform.ingestion.cleaner import TextCleaner
from rag_agent_platform.ingestion.loaders import build_default_loader_registry
from rag_agent_platform.ingestion.pipeline import RealIngestionPipeline
from rag_agent_platform.models import ChildChunk, DocumentRecord, ParentChunk
from rag_agent_platform.storage.file_repository import FileDocumentRepository


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
    if args.repo_smoke:
        run_repository_smoke()
        return
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
        should_clean = args.show_clean or args.show_chunks
        content = TextCleaner().clean(loaded.content) if should_clean else loaded.content
        if args.show_chunks:
            chunker = ParentChildChunker(
                ChunkingConfig(parent_chunk_size=80, child_chunk_size=32, child_overlap=8)
            )
            parents, children = chunker.split(
                document_id="demo-document",
                content=content,
                source=loaded.metadata["filename"],
                file_type=loaded.metadata["file_type"],
            )
            print(f"filename: {loaded.metadata['filename']}")
            print(f"parent_chunks: {len(parents)}")
            print(f"child_chunks: {len(children)}")
            if children:
                first_child = children[0]
                print(f"first_child_id: {first_child.chunk_id}")
                print(f"first_child_parent_id: {first_child.parent_id}")
                print(f"first_child_preview: {first_child.content[:80]}")
            return
        preview = content.strip().replace("\n", " ")[:80]
        print(f"filename: {loaded.metadata['filename']}")
        print(f"file_type: {loaded.metadata['file_type']}")
        print(f"encoding: {loaded.metadata['encoding']}")
        if args.show_clean:
            print(f"raw_characters: {len(loaded.content)}")
            print(f"cleaned_characters: {len(content)}")
        else:
            print(f"characters: {len(loaded.content)}")
        print(f"preview: {preview}")
        return
    if args.show_clean or args.show_chunks or args.index:
        print("selected smoke option is reserved for a later implementation step")
        return
    print("ingestion demo is ready; use --help to see available commands")


def run_repository_smoke() -> None:
    """Run a minimal persistent repository check."""
    with TemporaryDirectory() as temp_dir:
        repository = FileDocumentRepository(f"{temp_dir}/documents.json")
        document = DocumentRecord(
            document_id="demo-document",
            filename="leave_policy.txt",
            file_type="txt",
            source_path="tests/fixtures/leave_policy.txt",
            status="ready",
        )
        parent = ParentChunk(
            chunk_id="demo-parent",
            document_id=document.document_id,
            content="员工请假制度父块",
            metadata={"source": document.filename},
        )
        child = ChildChunk(
            chunk_id="demo-child",
            document_id=document.document_id,
            parent_id=parent.chunk_id,
            content="员工请假制度子块",
            metadata={"source": document.filename, "parent_id": parent.chunk_id},
        )
        repository.save_document(document)
        repository.save_parent_chunks([parent])
        repository.save_child_chunks([child])
        print(f"documents_after_save: {len(repository.list_documents())}")
        print(f"parent_chunks_after_save: {len(repository.list_parent_chunks())}")
        print(f"child_chunks_after_save: {len(repository.list_child_chunks())}")

        reloaded = FileDocumentRepository(f"{temp_dir}/documents.json")
        print(f"documents_after_reload: {len(reloaded.list_documents())}")
        reloaded.delete_document(document.document_id)
        print(f"documents_after_delete: {len(reloaded.list_documents())}")
        print(f"child_chunks_after_delete: {len(reloaded.list_child_chunks())}")


if __name__ == "__main__":
    main()
