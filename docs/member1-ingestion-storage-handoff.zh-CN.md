# 成员一：文档入库与存储模块交接说明

## 一、我完成了什么

成员一负责的是 RAG 系统最底层的“文档入库与存储”模块。简单说，就是把用户上传或指定的文档，转换成后续 Naive RAG、Advanced RAG、GraphRAG 和 Agentic RAG 都能使用的知识库数据。

目前已经完成：

- TXT / Markdown / PDF / DOCX 文档解析
- 文本清洗
- 父子分块
- 文档记录、父块、子块持久化保存
- 子块向量写入 Chroma
- 文档列表查询
- 根据 `document_id` 删除文档、块数据和向量
- 给成员二检索模块提供真实数据适配器
- 最终联调测试和命令行 demo

## 二、核心调用入口

成员四做 UI 或组长集成时，主要调用这个类：

```python
from rag_agent_platform.ingestion.pipeline import RealIngestionPipeline
```

核心方法：

```python
pipeline.ingest(file_path)
pipeline.list_documents()
pipeline.delete_document(document_id)
```

返回值和数据结构使用项目统一模型：

- `DocumentRecord`
- `ParentChunk`
- `ChildChunk`
- `IngestionResult`

## 三、本地默认存储位置

命令行 demo 默认使用：

```text
data/metadata/documents.json
data/chroma/
```

说明：

- `data/metadata/` 保存文档记录、父块、子块元数据
- `data/chroma/` 保存 Chroma 向量库
- 这些目录是运行时数据，已经加入 `.gitignore`，不会提交到仓库

## 四、给成员二的对接内容

成员二做 Naive RAG / Advanced RAG 时，可以直接使用成员一提供的两个适配器。

稀疏检索 / BM25 使用：

```python
from rag_agent_platform.storage import RepositoryChunkCorpus

corpus = RepositoryChunkCorpus(repository)
chunks = corpus.list_chunks(document_ids=None)
```

向量检索 / Dense Search 使用：

```python
from rag_agent_platform.storage import ChromaDenseSearchBackend

dense_backend = ChromaDenseSearchBackend(vector_store)
hits = dense_backend.search("policy approval", document_ids=None, limit=5)
```

`document_ids` 规则：

```text
None        表示检索全部文档
[]          表示不检索任何文档
["doc_xxx"] 表示只检索指定文档
```

检索结果中会保留：

- `chunk_id`
- `document_id`
- `parent_id`
- `source`
- `page`
- `score`
- `metadata`

这些字段可以用于结果融合、父块回溯、来源引用和 UI 展示。

## 五、metadata 字段说明

子块写入 Chroma 时，主要 metadata 包括：

```python
{
    "chunk_id": "...",
    "document_id": "...",
    "parent_id": "...",
    "source": "原始文件名",
    "file_type": "txt/md/pdf/docx",
    "page": -1,
}
```

说明：

- `source` 默认是原始文件名，方便 UI 展示引用来源
- `parent_id` 用于从子块回溯到父块
- `document_id` 用于按用户选择的文档过滤检索范围
- 当前 TXT / Markdown / DOCX 没有稳定页码，`page` 会用 `-1` 存储，读取时转换为 `None`
- PDF 已记录 `page_count`，如果后面要逐页切分，可以继续扩展

## 六、命令行验证方式

在项目根目录执行：

```powershell
uv run python -m compileall src tests scripts
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run pytest -q
```

当前验证结果：

```text
67 passed
```

成员一和成员二对接测试：

```powershell
uv run pytest tests/test_member1_integration_ready.py -q
```

期望结果：

```text
1 passed
```

## 七、Demo 命令

TXT 真实入库：

```powershell
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt
```

PDF 解析展示：

```powershell
uv run python scripts/ingest_demo.py tests/fixtures/sample.pdf --show-clean
```

DOCX 解析展示：

```powershell
uv run python scripts/ingest_demo.py tests/fixtures/sample.docx --show-clean
```

成员二检索适配烟测：

```powershell
uv run python scripts/ingest_demo.py tests/fixtures/leave_policy.txt --adapter-smoke
```

期望看到类似：

```text
corpus_chunks: 2
dense_hits: 2
first_hit_document_id: demo-document
first_hit_source: leave_policy.txt
```

## 八、删除文档

普通入库命令会输出 `document_id`：

```text
document_id: doc-xxxx
```

删除时执行：

```powershell
uv run python scripts/ingest_demo.py --delete doc-xxxx
```

删除会同时清理：

- 文档记录
- 父块
- 子块
- Chroma 向量

## 九、当前 embedding 说明

当前使用的是：

```python
HashEmbeddingModel
```

这是一个本地确定性哈希 embedding，主要用于课程项目和本地测试稳定运行。

这样做的原因：

- 不需要下载大模型
- 不依赖外部 API
- 测试稳定
- 不影响 Chroma 和检索模块接口

后续如果要换成 BGE-M3、OpenAI Embedding 或其他真实 embedding，只需要替换注入给 `ChromaVectorStore` 的 `embedding_model`，不用改入库管线和成员二检索适配器。

## 十、交付结论

成员一模块已经可以交给全组集成：

- 可以入库 TXT / Markdown / PDF / DOCX
- 可以清洗文本并生成父子块
- 可以持久化文档和块数据
- 可以写入和删除 Chroma 向量
- 可以按 `document_id` 过滤检索范围
- 可以给成员二 BM25 / Dense / Advanced RAG 提供真实数据
- 可以给成员四 UI 调用 `ingest / list_documents / delete_document`
