# 成员一文档入库与存储模块实施步骤计划

> **模块名称**：文档入库与存储  
> **开发分支**：`feature/ingestion-storage`  
> **正式项目目录**：`rag-agent-platform/`  
> **日期**：2026-07-20  
> **核心原则**：每一步都产出可运行、可验证、可提交的 MVP  
> **工程规范**：需求分析 -> 设定边界 -> 开发步骤 -> 分层设计 -> Python package -> 依赖注入 -> 基类/接口 + 实现 -> TDD -> commit/tag/push

---

## 一、开发目标

成员一负责把用户上传的文档转化为项目统一可用的知识库数据，为 Naive RAG、Advanced RAG、GraphRAG 和 Agentic RAG 提供稳定的数据入口。

完整流程：

```text
本地文件 / WebUI 上传文件
        ↓
文件类型识别
        ↓
文档解析
        ↓
文本清洗
        ↓
父子切块
        ↓
文档与块元数据持久化
        ↓
子块向量写入 Chroma
        ↓
文档列表、删除、检索过滤可用
```

最终交付给全组的核心能力：

```python
IngestionPipeline.ingest(file_path)
IngestionPipeline.list_documents()
IngestionPipeline.delete_document(document_id)
```

---

## 二、功能划分与边界

| 功能 | 内容 |
|------|------|
| 文档解析 | 支持 TXT、Markdown，逐步扩展 PDF、DOCX |
| 文本清洗 | 清理空白、不可见字符、异常换行，保留段落结构 |
| 父子切块 | 父块保留完整上下文，子块用于向量检索 |
| 文档存储 | 保存文档记录、父块、子块和 metadata |
| 向量入库 | 使用 embedding 生成子块向量，写入 ChromaDB |
| 删除清理 | 删除文档时同步删除文档记录、块数据和向量 |
| 对接支持 | 为检索模块提供 `document_id` 过滤和 `parent_id` 回溯 |

---

## 三、现有仓库基线

正式仓库已经具备：

```text
app.py
pyproject.toml
README.md
docs/
tests/
src/rag_agent_platform/
  ingestion/
  storage/
  retrieval/
  graph/
  agent/
  generation/
  evaluation/
  ui/
```

当前成员一开发应放入：

```text
src/rag_agent_platform/ingestion/
src/rag_agent_platform/storage/
src/rag_agent_platform/models/
scripts/
tests/
docs/
```

---

## 四、接口契约

成员一必须遵守当前仓库 `docs/interfaces.md` 中的公共接口。

### IngestionPipeline

路径：

```text
src/rag_agent_platform/ingestion/base.py
```

接口：

```python
class IngestionPipeline:
    def ingest(self, file_path: str | Path) -> IngestionResult:
        ...

    def list_documents(self) -> list[DocumentRecord]:
        ...

    def delete_document(self, document_id: str) -> None:
        ...
```

### DocumentRepository

路径：

```text
src/rag_agent_platform/storage/base.py
```

接口：

```python
class DocumentRepository:
    def save_document(self, document: DocumentRecord) -> None:
        ...

    def get_document(self, document_id: str) -> DocumentRecord | None:
        ...

    def list_documents(self) -> list[DocumentRecord]:
        ...

    def delete_document(self, document_id: str) -> None:
        ...

    def save_parent_chunks(self, chunks: list[ParentChunk]) -> None:
        ...

    def save_child_chunks(self, chunks: list[ChildChunk]) -> None:
        ...
```

### 关键 metadata

每个 `ChildChunk` 的 `metadata` 至少包含：

```python
{
    "document_id": "...",
    "parent_id": "...",
    "chunk_id": "...",
    "source": "文件名",
    "file_type": "txt/md/pdf/docx",
    "page": 1,
}
```

### 成员二对接基线

成员二已完成 Naive RAG、Advanced RAG 和离线评估模块，成员一的真实入库结果必须能直接接入成员二的检索代码。

必须满足：

```text
Repository 能提供 ChildChunk 快照
Chroma 后端能返回 DenseSearchHit
BM25Retriever 能基于成员一的 ChunkCorpus 构建索引
DenseRetriever 能基于成员一的 DenseSearchBackend 返回 RetrievedChunk
document_ids=None 表示全部文档
document_ids=[] 表示无可检索文档
删除文档后 Repository、Chroma、BM25 索引状态保持一致
```

成员一最终联调命令至少包含：

```powershell
uv run pytest tests/test_retrieval_adapters.py
uv run pytest tests/test_bm25_retriever.py tests/test_dense_retriever.py
```

---

## 五、步骤总览

| Step | 内容 | MVP 验证点 | Tag |
|------|------|------------|-----|
| **1** | 补齐公共模型与成员一基线 | `uv run pytest tests/test_schemas.py tests/test_contracts.py` 通过 | `member1-step-1-contract-baseline` |
| **2** | 成员一工程骨架与 demo 入口 | `uv run python scripts/ingest_demo.py --help` 可运行 | `member1-step-2-ingestion-skeleton` |
| **3** | TXT / Markdown 文档解析 | demo 能读取 TXT/MD 并输出文档基本信息 | `member1-step-3-text-loaders` |
| **4** | 文本清洗 | demo 能展示清洗前后长度，测试覆盖清洗规则 | `member1-step-4-cleaner` |
| **5** | 父子切块 | demo 能输出 parent/child 数量和 parent_id 关系 | `member1-step-5-parent-child-chunking` |
| **6** | 本地文档 Repository | 文档可保存、列表展示、按 id 删除 | `member1-step-6-document-repository` |
| **7** | Chroma 向量存储 | 子块可写入 Chroma，能按 document_id 删除 | `member1-step-7-chroma-store` |
| **8** | 成员二检索适配器 | 提供 `ChunkCorpus` 与 `DenseSearchBackend` 适配 | `member1-step-8-retrieval-adapters` |
| **9** | IngestionPipeline 闭环 | `ingest/list/delete` 完整可运行 | `member1-step-9-ingestion-mvp` |
| **10** | PDF / DOCX 扩展 | 至少 PDF 或 DOCX 一种格式可入库 | `member1-step-10-pdf-docx` |
| **11** | 测试、文档、交付 | compileall、Ruff、pytest 和成员二检索联调通过 | `member1-step-11-integration-ready` |

---

## 六、开发前置步骤

### Step 0：创建成员一分支

| 项目 | 内容 |
|------|------|
| **目标** | 从 `develop` 创建成员一功能分支 |
| **涉及文件** | Git 分支 |
| **MVP 效果** | `git status --short --branch` 显示当前分支为 `feature/ingestion-storage` |
| **Git** | 本步只创建分支，不打 tag |

执行命令：

```powershell
git switch develop
git pull origin develop
git switch -c feature/ingestion-storage
git push -u origin feature/ingestion-storage
```

---

## 七、第一阶段：公共模型与成员一骨架

> **目标：Step 2 完成后，项目测试基线恢复，成员一模块有自己的可执行入口。**

### Step 1：补齐公共模型与成员一基线

| 项目 | 内容 |
|------|------|
| **目标** | 补齐 `rag_agent_platform.models`，让现有接口和测试能正常导入 |
| **涉及文件** | `src/rag_agent_platform/models/__init__.py`、`src/rag_agent_platform/models/schemas.py` |
| **MVP 效果** | `uv run pytest tests/test_schemas.py tests/test_contracts.py` 通过 |
| **Git** | commit + tag `member1-step-1-contract-baseline` |

#### 本步涉及的技术与设计

| 技术/设计 | What | Why | Which | How |
|-----------|------|-----|-------|-----|
| 公共数据模型 | 全组模块共用的数据结构 | 没有模型包时 ingestion、retrieval、agent 都无法导入 | dataclass 轻量稳定，适合当前骨架 | 定义 `DocumentRecord`、`ParentChunk`、`ChildChunk`、`IngestionResult` 等 |
| 契约测试 | 验证公共接口是否可用 | 保证后续每一步从绿色基线开始 | 先修基线再开发功能 | 运行 schema 和 contract 测试 |

验证命令：

```powershell
uv run pytest tests/test_schemas.py tests/test_contracts.py
```

提交命令：

```powershell
git add src/rag_agent_platform/models tests
git commit -m "feat: add public data schemas"
git tag member1-step-1-contract-baseline
git push origin feature/ingestion-storage --tags
```

---

### Step 2：成员一工程骨架与 demo 入口

| 项目 | 内容 |
|------|------|
| **目标** | 建立真实 ingestion/storage 实现文件和命令行 demo |
| **涉及文件** | `src/rag_agent_platform/ingestion/loaders.py`、`src/rag_agent_platform/ingestion/cleaner.py`、`src/rag_agent_platform/ingestion/chunker.py`、`src/rag_agent_platform/ingestion/pipeline.py`、`scripts/ingest_demo.py` |
| **MVP 效果** | `uv run python scripts/ingest_demo.py --help` 可运行 |
| **Git** | commit + tag `member1-step-2-ingestion-skeleton` |

#### 本步涉及的技术与设计

| 技术/设计 | What | Why | Which | How |
|-----------|------|-----|-------|-----|
| Python package | 用正式包组织代码 | 避免散脚本无法测试和复用 | 代码统一放入 `src/rag_agent_platform/` | 新建 `loaders/cleaner/chunker/pipeline` |
| demo 入口 | 可运行的最小入口 | 每一步都能看到效果 | 用 CLI demo 先验证核心逻辑 | `scripts/ingest_demo.py` 调用 pipeline |

验证命令：

```powershell
uv run python scripts/ingest_demo.py --help
```

提交命令：

```powershell
git add src/rag_agent_platform/ingestion scripts/ingest_demo.py
git commit -m "feat: scaffold ingestion pipeline"
git tag member1-step-2-ingestion-skeleton
git push origin feature/ingestion-storage --tags
```

---

## 八、第二阶段：文档解析与文本处理

> **目标：Step 5 完成后，TXT/Markdown 可以进入系统，并形成父子块。**

### Step 3：TXT / Markdown 文档解析

| 项目 | 内容 |
|------|------|
| **目标** | 支持最小稳定文档读取 |
| **涉及文件** | `src/rag_agent_platform/ingestion/loaders.py`、`tests/test_ingestion_loaders.py`、`tests/fixtures/*.txt`、`tests/fixtures/*.md` |
| **MVP 效果** | demo 能读取 TXT/MD 并输出文件名、类型、文本长度 |
| **Git** | commit + tag `member1-step-3-text-loaders` |

实现要点：

```text
支持 .txt / .md
自动尝试 utf-8 / gbk / gb18030
空文件抛出明确异常
不支持格式抛出明确异常
metadata 记录 filename、source_path、file_type
```

验证命令：

```powershell
uv run pytest tests/test_ingestion_loaders.py
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt
```

提交命令：

```powershell
git add src/rag_agent_platform/ingestion tests scripts/ingest_demo.py
git commit -m "feat: implement txt markdown loaders"
git tag member1-step-3-text-loaders
git push origin feature/ingestion-storage --tags
```

---

### Step 4：文本清洗

| 项目 | 内容 |
|------|------|
| **目标** | 将原始文本转成适合 RAG 切分和检索的干净文本 |
| **涉及文件** | `src/rag_agent_platform/ingestion/cleaner.py`、`tests/test_ingestion_cleaner.py` |
| **MVP 效果** | demo 能展示清洗前后长度，测试覆盖主要清洗规则 |
| **Git** | commit + tag `member1-step-4-cleaner` |

#### 本步涉及的技术与设计

| 技术/设计 | What | Why | Which | How |
|-----------|------|-----|-------|-----|
| 文本清洗 | 规范化原始文档文本 | 提升切块和检索质量 | 温和清洗优先，避免破坏语义 | 正则处理换行、空格、控制字符 |
| 段落保留 | 保留自然段落边界 | 段落边界有助于后续父子切块 | 不把所有换行压成空格 | 连续空行压缩为最多两个换行 |

验证命令：

```powershell
uv run pytest tests/test_ingestion_cleaner.py
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt --show-clean
```

提交命令：

```powershell
git add src/rag_agent_platform/ingestion/cleaner.py tests/test_ingestion_cleaner.py
git commit -m "feat: add document text cleaner"
git tag member1-step-4-cleaner
git push origin feature/ingestion-storage --tags
```

---

### Step 5：父子切块

| 项目 | 内容 |
|------|------|
| **目标** | 生成父块和子块，支持后续精准检索和父块回溯 |
| **涉及文件** | `src/rag_agent_platform/ingestion/chunker.py`、`tests/test_ingestion_chunker.py` |
| **MVP 效果** | demo 能输出 parent/child 数量，并证明每个 child 都有 `parent_id` |
| **Git** | commit + tag `member1-step-5-parent-child-chunking` |

实现要点：

```text
父块：保留较完整上下文
子块：进入向量检索
子块支持 overlap
每个 child 记录 document_id、parent_id、chunk_id、source、page
```

验证命令：

```powershell
uv run pytest tests/test_ingestion_chunker.py
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt --show-chunks
```

提交命令：

```powershell
git add src/rag_agent_platform/ingestion/chunker.py tests/test_ingestion_chunker.py
git commit -m "feat: implement parent child chunking"
git tag member1-step-5-parent-child-chunking
git push origin feature/ingestion-storage --tags
```

---

## 九、第三阶段：存储与向量入库

> **目标：Step 9 完成后，成员一完整 MVP 可以被 Streamlit 和检索模块调用。**

### Step 6：本地文档 Repository

| 项目 | 内容 |
|------|------|
| **目标** | 持久化保存文档记录、父块、子块 |
| **涉及文件** | `src/rag_agent_platform/storage/file_repository.py`、`tests/test_document_repository.py` |
| **MVP 效果** | 文档可保存、列表展示、按 id 删除，重启后数据仍在 |
| **Git** | commit + tag `member1-step-6-document-repository` |

#### 本步涉及的技术与设计

| 技术/设计 | What | Why | Which | How |
|-----------|------|-----|-------|-----|
| Repository 模式 | 封装数据读写 | UI 和 pipeline 不直接操作文件或数据库 | 先用文件/SQLite 快速完成 MVP | 实现 `DocumentRepository` 抽象方法 |
| 持久化元数据 | 保存文档和块结构 | 支持文档列表、删除和父块回溯 | JSONL/SQLite 都可，接口保持稳定 | 数据放入 `data/metadata/` 或 SQLite |

验证命令：

```powershell
uv run pytest tests/test_document_repository.py
uv run python scripts/ingest_demo.py --repo-smoke
```

提交命令：

```powershell
git add src/rag_agent_platform/storage tests/test_document_repository.py
git commit -m "feat: implement persistent document repository"
git tag member1-step-6-document-repository
git push origin feature/ingestion-storage --tags
```

---

### Step 7：Chroma 向量存储

| 项目 | 内容 |
|------|------|
| **目标** | 将子块向量写入 ChromaDB，并支持按文档删除 |
| **涉及文件** | `src/rag_agent_platform/storage/chroma_store.py`、`src/rag_agent_platform/embeddings/`、`tests/test_chroma_store.py` |
| **MVP 效果** | 子块可写入 Chroma，能按 `document_id` 删除 |
| **Git** | commit + tag `member1-step-7-chroma-store` |

实现要点：

```text
Chroma 持久化目录使用 data/chroma/
embedding 模型通过构造函数注入
upsert 子块时写入 documents、ids、metadatas、embeddings
delete_document(document_id) 清理对应向量
```

依赖建议：

```powershell
uv add chromadb sentence-transformers
```

验证命令：

```powershell
uv run pytest tests/test_chroma_store.py
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt --index
```

提交命令：

```powershell
git add pyproject.toml uv.lock src/rag_agent_platform/storage src/rag_agent_platform/embeddings tests/test_chroma_store.py
git commit -m "feat: persist child chunks to chroma"
git tag member1-step-7-chroma-store
git push origin feature/ingestion-storage --tags
```

---

### Step 8：成员二检索适配器

| 项目 | 内容 |
|------|------|
| **目标** | 为成员二的 Naive/Advanced Retriever 提供真实数据适配 |
| **涉及文件** | `src/rag_agent_platform/storage/chunk_corpus.py`、`src/rag_agent_platform/storage/chroma_backend.py`、`tests/test_retrieval_adapters.py` |
| **MVP 效果** | `ChunkCorpus.list_chunks(document_ids)` 和 `DenseSearchBackend.search()` 可被成员二检索模块调用，并能喂给 `BM25Retriever` / `DenseRetriever` |
| **Git** | commit + tag `member1-step-8-retrieval-adapters` |

成员二已完成的检索模块需要两个适配器：

```python
class ChunkCorpus:
    def list_chunks(
        self,
        document_ids: list[str] | None = None,
    ) -> list[ChildChunk]:
        ...
```

```python
class DenseSearchBackend:
    def search(
        self,
        query: str,
        document_ids: list[str] | None,
        limit: int,
    ) -> list[DenseSearchHit]:
        ...
```

实现要点：

```text
ChunkCorpus 从 Repository 返回 ChildChunk 快照
document_ids=None 表示全部文档
document_ids=[] 表示无可检索文档
DenseSearchBackend 调用 Chroma 查询
明确 Chroma 返回 distance 还是 similarity
将 Chroma 命中结果映射为 DenseSearchHit
保留 chunk_id、document_id、parent_id、source、page
删除文档后 Repository 与 Chroma 必须同步清理
文档新增/删除后需要通知 BM25Retriever.refresh()
```

验证命令：

```powershell
uv run pytest tests/test_retrieval_adapters.py
uv run pytest tests/test_bm25_retriever.py tests/test_dense_retriever.py
```

提交命令：

```powershell
git add src/rag_agent_platform/storage tests/test_retrieval_adapters.py
git commit -m "feat: add retrieval storage adapters"
git tag member1-step-8-retrieval-adapters
git push origin feature/ingestion-storage --tags
```

---

### Step 9：IngestionPipeline 闭环

| 项目 | 内容 |
|------|------|
| **目标** | 串联加载、清洗、切块、Repository、Chroma，替换 Mock 入库能力 |
| **涉及文件** | `src/rag_agent_platform/ingestion/pipeline.py`、`src/rag_agent_platform/ingestion/__init__.py`、`scripts/ingest_demo.py`、`tests/test_ingestion_pipeline.py` |
| **MVP 效果** | `ingest/list/delete` 三个接口完整可运行 |
| **Git** | commit + tag `member1-step-9-ingestion-mvp` |

依赖注入结构：

```python
class RealIngestionPipeline(IngestionPipeline):
    def __init__(
        self,
        loader_registry,
        cleaner,
        chunker,
        repository,
        vector_store,
    ) -> None:
        ...
```

MVP 验证流程：

```powershell
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt
uv run python scripts/ingest_demo.py --list
uv run python scripts/ingest_demo.py --delete 文档ID
uv run pytest tests/test_ingestion_pipeline.py
```

提交命令：

```powershell
git add src/rag_agent_platform/ingestion scripts/ingest_demo.py tests/test_ingestion_pipeline.py
git commit -m "feat: complete ingestion pipeline mvp"
git tag member1-step-9-ingestion-mvp
git push origin feature/ingestion-storage --tags
```

---

## 十、第四阶段：格式扩展与最终交付

> **目标：Step 11 完成后，成员一功能可以合并到 `develop` 并交给全组联调。**

### Step 10：PDF / DOCX 扩展

| 项目 | 内容 |
|------|------|
| **目标** | 在 TXT/Markdown 稳定后，增加 PDF 和 DOCX 解析 |
| **涉及文件** | `src/rag_agent_platform/ingestion/loaders.py`、`tests/test_ingestion_loaders.py`、`tests/fixtures/*.pdf`、`tests/fixtures/*.docx` |
| **MVP 效果** | 至少 PDF 或 DOCX 一种格式可完整入库 |
| **Git** | commit + tag `member1-step-10-pdf-docx` |

建议优先级：

```text
TXT 必须稳定
Markdown 必须稳定
PDF 优先做
DOCX 时间允许再做
```

依赖建议：

```powershell
uv add pypdf python-docx
```

验证命令：

```powershell
uv run pytest tests/test_ingestion_loaders.py
uv run python scripts/ingest_demo.py tests/fixtures/sample.pdf
```

提交命令：

```powershell
git add pyproject.toml uv.lock src/rag_agent_platform/ingestion tests
git commit -m "feat: add pdf docx loaders"
git tag member1-step-10-pdf-docx
git push origin feature/ingestion-storage --tags
```

---

### Step 11：测试、文档、交付

| 项目 | 内容 |
|------|------|
| **目标** | 完成成员一模块质量检查、文档和交接说明 |
| **涉及文件** | `docs/member1-ingestion-storage-plan.md`、`README.md`、`tests/`、`scripts/ingest_demo.py` |
| **MVP 效果** | compileall、Ruff、pytest 通过，成员二 BM25/Dense 检索可基于成员一数据运行 |
| **Git** | commit + tag `member1-step-11-integration-ready` |

最终验证命令：

```powershell
uv run python -m compileall src tests app.py
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run pytest tests/test_retrieval_adapters.py tests/test_bm25_retriever.py tests/test_dense_retriever.py
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt
uv run python scripts/ingest_demo.py --list
```

提交命令：

```powershell
git add .
git commit -m "docs: finalize ingestion storage handoff"
git tag member1-step-11-integration-ready
git push origin feature/ingestion-storage --tags
```

---

## 十一、TDD 测试计划

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_schemas.py` | 公共模型字段和校验 |
| `tests/test_contracts.py` | 公共导出和接口契约 |
| `tests/test_ingestion_loaders.py` | TXT/MD/PDF/DOCX 解析 |
| `tests/test_ingestion_cleaner.py` | 文本清洗规则 |
| `tests/test_ingestion_chunker.py` | 父子切块和 metadata |
| `tests/test_document_repository.py` | 文档记录、父块、子块持久化 |
| `tests/test_chroma_store.py` | 子块向量入库和删除 |
| `tests/test_retrieval_adapters.py` | 成员二 `ChunkCorpus` 和 `DenseSearchBackend` 对接 |
| `tests/test_ingestion_pipeline.py` | `ingest/list/delete` 闭环 |

最低用例：

```text
正常 TXT 可以入库
正常 Markdown 可以入库
空文件抛出明确异常
不支持格式抛出明确异常
清洗后文本不为空
父子块关系正确
子块 metadata 完整
文档列表可查询
删除文档会清理 Repository 和 Chroma
重复入库不会造成不可控脏数据
```

---

## 十二、成员一与其他模块的交接内容

交给成员二：

```text
Chroma collection 名称
Chroma 持久化目录
子块 metadata 字段说明
document_id 过滤方式
parent_id 回溯父块方式
demo 文档和测试问题
```

交给成员三：

```text
如何遍历文档块
每个块的 document_id/chunk_id/source/page 字段
删除文档后如何清理相关图数据
```

交给成员四：

```text
如何在 UI 中调用 ingest
如何展示 list_documents
如何调用 delete_document
异常信息如何展示给用户
```

---

## 十三、Git 工作流

成员一开发分支：

```text
feature/ingestion-storage
```

每一步完成后：

```powershell
uv run pytest 对应测试文件
git add .
git commit -m "feat: implement xxx"
git tag member1-step-x-xxx
git push origin feature/ingestion-storage --tags
```

阶段完成后创建 Pull Request：

```text
base: develop
compare: feature/ingestion-storage
```

最终由组长合并到 `develop`，再合并到 `main` 并打总版本 tag。

---

## 十四、最终验收标准

成员一完成后，应能演示：

```text
上传 / 指定一个文档
    ↓
解析文本
    ↓
清洗文本
    ↓
生成父块和子块
    ↓
保存文档记录和块数据
    ↓
子块写入 Chroma
    ↓
list_documents 能看到文档
    ↓
delete_document 能删除文档、块和向量
    ↓
成员二能按 document_id 检索该文档
```

最终必须满足：

```text
1. 正式代码位于 rag-agent-platform/
2. 公共模型和接口可导入
3. TXT/Markdown 稳定入库
4. PDF/DOCX 至少完成一个可用版本
5. 父子块关系正确
6. metadata 字段完整
7. Repository 持久化可用
8. Chroma 入库和删除可用
9. demo 脚本可运行
10. 每一步都有 commit + tag + 可执行 MVP
```
