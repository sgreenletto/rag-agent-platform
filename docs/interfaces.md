# 公共接口约定

所有路径均相对于正式 Python 包 `src/rag_agent_platform/`。`src` 只是源码根目录，不是导入包；
项目内部统一使用 `rag_agent_platform...` 绝对导入。

## 公共数据模型

定义于 `rag_agent_platform.models.schemas`，并由 `rag_agent_platform.models` 导出。

### QueryType

`StrEnum`：`SIMPLE="simple"`、`COMPLEX="complex"`、`RELATION="relation"`、
`CHAT="chat"`。

### RetrievalStrategy

`StrEnum`：`NONE="none"`、`NAIVE="naive"`、`ADVANCED="advanced"`、
`GRAPH="graph"`。

### DocumentRecord

| 字段 | 类型 |
|---|---|
| `document_id` | `str` |
| `filename` | `str` |
| `file_type` | `str` |
| `source_path` | `str` |
| `status` | `str` |
| `metadata` | `dict[str, Any]` |

### ParentChunk

字段为 `chunk_id: str`、`document_id: str`、`content: str`、`page: int | None`、
`metadata: dict[str, Any]`。

### ChildChunk

字段为 `chunk_id: str`、`document_id: str`、`parent_id: str`、`content: str`、
`page: int | None`、`metadata: dict[str, Any]`。

### RetrievedChunk

字段为 `chunk_id: str`、`content: str`、`normalized_score: float`、`source: str`、
`document_id: str | None`、`parent_id: str | None`、`page: int | None`、
`retrieval_method: str`、`metadata: dict[str, Any]`。

约束：内容和来源不得为空；`normalized_score` 必须在 `[0.0, 1.0]`，且全项目统一规定分数
越高越相关。

### Citation

字段为 `index: int`、`source: str`、`page: int | None`、`chunk_id: str | None`。

### IngestionResult

字段为 `document: DocumentRecord`、`parent_chunk_count: int`、`child_chunk_count: int`、
`warnings: list[str]`。

### AgentResult

字段为 `answer: str`、`citations: list[Citation]`、`strategy: RetrievalStrategy`、
`query_type: QueryType`、`retry_count: int`、`execution_trace: list[str]`。

## IngestionPipeline

路径：`rag_agent_platform.ingestion.base.IngestionPipeline`。

```python
ingest(file_path: str | Path) -> IngestionResult
list_documents() -> list[DocumentRecord]
delete_document(document_id: str) -> None
```

真实实现负责加载、清洗、切块、保存和索引。删除必须清理该文档关联数据。

## DocumentRepository

路径：`rag_agent_platform.storage.base.DocumentRepository`。

```python
save_document(document: DocumentRecord) -> None
get_document(document_id: str) -> DocumentRecord | None
list_documents() -> list[DocumentRecord]
delete_document(document_id: str) -> None
save_parent_chunks(chunks: list[ParentChunk]) -> None
save_child_chunks(chunks: list[ChildChunk]) -> None
```

当前 `MockDocumentRepository` 使用内存字典；真实数据库实现不得改变这些签名。

## BaseRetriever

路径：`rag_agent_platform.retrieval.base.BaseRetriever`。

```python
retrieve(
    query: str,
    document_ids: list[str] | None = None,
    top_k: int = 5,
) -> list[RetrievedChunk]
```

所有实现只能返回 `list[RetrievedChunk]`，不得返回 LangChain Document、元组、字典或裸字符串。
结果必须按 `normalized_score` 降序；`document_ids` 非空时只返回指定文档；`top_k` 必须大于 0。

## GraphService

路径：`rag_agent_platform.graph.base.GraphService`。

```python
build(chunks: list[ChildChunk]) -> None
delete_document(document_id: str) -> None
retrieve(
    query: str,
    document_ids: list[str] | None = None,
    top_k: int = 5,
) -> list[RetrievedChunk]
```

图构建输入是子块；图检索证据必须转换为相同的 `RetrievedChunk` 契约。

## AnswerGenerator

路径：`rag_agent_platform.generation.base.AnswerGenerator`。

```python
generate(
    query: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, list[Citation]]
```

返回回答文本及与证据对应的结构化引用。

## AnswerEvaluator 与 EvaluationResult

路径：`rag_agent_platform.evaluation.base`。

`EvaluationResult` 字段为 `passed: bool`、`reason: str`、
`suggested_query: str | None`。

```python
evaluate(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
) -> EvaluationResult
```

未来重写循环仅在评估未通过且未超过重试上限时使用 `suggested_query`。

## AgentService

路径：`rag_agent_platform.agent.base.AgentService`。

```python
invoke(
    query: str,
    document_ids: list[str] | None = None,
    mode: str = "agent",
) -> AgentResult
```

`mode` 只允许 `agent`、`naive`、`advanced`、`graph`。当前
`rag_agent_platform.agent.mock.MockAgentService` 只进行规则路由和 Mock 回答，不是完整
LangGraph Agent。

## AgentState

路径：`rag_agent_platform.agent.state.AgentState`，为 `TypedDict`，包含：

- `original_query: str`、`current_query: str`、`document_ids: list[str]`；
- `query_type: QueryType`、`retrieval_strategy: RetrievalStrategy`；
- `retrieved_chunks: list[RetrievedChunk]`、`retrieval_sufficient: bool`；
- `answer: str`、`citations: list[Citation]`；
- `answer_passed: bool`、`evaluation_reason: str`；
- `retry_count: int`、`max_retries: int`、`execution_trace: list[str]`；
- `error: str | None`。

本阶段只定义 State，不构建真实 LangGraph。

## 异常处理

空问题、非法 `top_k`、不存在或空文件、不支持格式等必须抛出信息明确的异常。UI 层负责捕获
预期异常并展示；任何模块都不得静默吞掉异常，也不得用伪造成功结果掩盖错误。
