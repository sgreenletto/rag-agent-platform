# Member 1 Ingestion And Storage Handoff

## Scope

Member 1 provides the real document ingestion and storage MVP for the RAG platform.
The module turns local documents into normalized records, parent chunks, child chunks,
persistent metadata, and Chroma vector entries.

Implemented formats:

- TXT
- Markdown
- PDF
- DOCX

Implemented storage:

- `FileDocumentRepository` for document and chunk metadata
- `ChromaVectorStore` for child chunk vectors
- `RepositoryChunkCorpus` for member-two sparse retrieval input
- `ChromaDenseSearchBackend` for member-two dense retrieval input

## Main Imports

```python
from rag_agent_platform.embeddings import HashEmbeddingModel
from rag_agent_platform.ingestion.pipeline import RealIngestionPipeline
from rag_agent_platform.storage import ChromaDenseSearchBackend, RepositoryChunkCorpus
from rag_agent_platform.storage.chroma_store import ChromaVectorStore
from rag_agent_platform.storage.file_repository import FileDocumentRepository
```

## Build A Local Pipeline

```python
repository = FileDocumentRepository("data/metadata/documents.json")
vector_store = ChromaVectorStore(
    persist_directory="data/chroma",
    collection_name="rag_child_chunks",
    embedding_model=HashEmbeddingModel(dimensions=16),
)
pipeline = RealIngestionPipeline(repository=repository, vector_store=vector_store)
```

The demo script uses the same default local layout.

Runtime data is written under:

- `data/metadata/`
- `data/chroma/`

These directories are ignored by Git.

## Ingest And Delete

```python
result = pipeline.ingest("tests/fixtures/sample.pdf")
print(result.document.document_id)
print(result.parent_chunk_count)
print(result.child_chunk_count)

pipeline.delete_document(result.document.document_id)
```

`delete_document` removes both repository metadata/chunks and Chroma vectors for the
given document id.

## Member Two Integration

Sparse retrieval can read child chunks through:

```python
corpus = RepositoryChunkCorpus(repository)
chunks = corpus.list_chunks(document_ids=None)
```

Dense retrieval can search Chroma through:

```python
dense_backend = ChromaDenseSearchBackend(vector_store)
hits = dense_backend.search("policy approval", document_ids=None, limit=5)
```

Each dense hit exposes the child chunk, normalized score, source, document id, parent id,
and page metadata needed by retrieval fusion and citation display.

## Manual Acceptance Commands

Run from the project root:

```powershell
uv run python -m compileall src tests scripts
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run pytest -q
uv run pytest tests/test_member1_integration_ready.py -q
```

Expected result:

- compileall completes without errors
- Ruff reports `All checks passed!`
- pytest passes all tests
- integration-ready test passes

## Demo Commands

```powershell
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt
uv run python scripts/ingest_demo.py tests/fixtures/sample.pdf --show-clean
uv run python scripts/ingest_demo.py tests/fixtures/sample.docx --show-clean
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt --adapter-smoke
```

The normal ingest command prints `document_id`, chunk counts, document count, and vector
count. Use that `document_id` with:

```powershell
uv run python scripts/ingest_demo.py --delete <DOCUMENT_ID>
```

## Notes

The current embedding model is deterministic hash-based for local tests and classroom
demo stability. It avoids external model downloads while preserving the vector-store
contract. A production embedding model can replace `HashEmbeddingModel` without changing
the ingestion pipeline or member-two retrieval adapter interfaces.
