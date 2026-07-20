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
`document_ids=None` 表示不限制文档，`document_ids=[]` 表示没有可检索文档并必须返回空列表；
所有正式实现还必须按 `chunk_id` 去重。同分结果统一以 `chunk_id` 升序作为确定性次序。

## ChunkCorpus

路径：`rag_agent_platform.retrieval.corpus.ChunkCorpus`。

```python
list_chunks(document_ids: list[str] | None = None) -> list[ChildChunk]
```

该只读协议用于 BM25 等需要语料快照的 Retriever，并将检索算法与 MySQL、Chroma 或内存
存储解耦。成员一可以通过适配器实现该协议，无需修改现有 `DocumentRepository` 公共接口。

检索实现应复用 `rag_agent_platform.retrieval.validation` 中的公共规则：校验空查询和
`top_k`、将原始分数归一化到 `[0.0, 1.0]`，并在文档过滤后按分数降序截取结果。

## 阶段二检索实现

`rag_agent_platform.retrieval.bm25.BM25Retriever` 使用 `jieba` 和 `rank-bm25` 建立可刷新
的内存稀疏索引。构造时接收 `ChunkCorpus`，`refresh()` 用于文档新增或删除后重建快照；结果
在 `metadata["bm25_score"]` 中保留原始分数。

`rag_agent_platform.retrieval.dense.DenseRetriever` 不直接依赖 Chroma。成员一提供的向量存储
适配器只需实现以下协议：

```python
DenseSearchBackend.search(
    query: str,
    document_ids: list[str] | None,
    limit: int,
) -> list[DenseSearchHit]
```

`DenseSearchHit` 包含 `ChildChunk`、后端原始分数、来源和可选元数据。`DenseRetriever` 支持
`similarity`（越大越相关）和 `distance`（越小越相关）两类后端分数，负责归一化、阈值过滤
并转换为统一的 `RetrievedChunk`。真实 Embedding 与 Chroma 适配器由存储实现接入。

当前 Dense 分数采用单次候选集合内的 Min-Max 归一化；只有一个候选或全部同分时统一记为
`1.0`。因此阈值是模式内、查询内的相对分数阈值，不应将 BM25、Dense、RRF 与 Reranker 的
`normalized_score` 作为可跨模式直接比较的绝对置信度。真实 Chroma Backend 接入前，团队需
确认距离/相似度类型以及是否提供可校准的绝对分数映射。

## RRF 与混合检索

`rag_agent_platform.retrieval.fusion.reciprocal_rank_fusion` 按 `chunk_id` 合并多个命名结果
列表，使用加权 Reciprocal Rank Fusion：

```text
fused_score(chunk) = Σ weight(retriever) / (rrf_k + rank)
```

融合结果的 `retrieval_method` 为 `hybrid_rrf`，并在 metadata 中保留 `rrf_score` 以及每一路
的原排名、归一化分数和 RRF 贡献。

`rag_agent_platform.retrieval.hybrid.HybridRetriever` 组合 Dense 与 Sparse Retriever。两路各
召回 `top_k * candidate_multiplier` 个候选，再通过 RRF 去重融合。默认 `failure_mode="fallback"`：
单路失败时继续使用另一路，并在结果的 `metadata["retrieval_warnings"]` 中记录错误；两路均
失败时抛出 `RuntimeError`。使用 `failure_mode="raise"` 可以启用严格模式，直接传播单路异常。

## Query Rewrite 与 Multi-Query

查询改写服务实现 `rag_agent_platform.retrieval.multi_query.QueryTransformer`：

```python
transform(query: str) -> list[str]
```

`MultiQueryRetriever` 始终保留原问题，对改写结果去空、去重并应用 `max_queries` 上限，然后
扩大每个查询的候选集并使用 RRF 合并。`IdentityQueryTransformer` 是未配置 LLM 时的安全
默认值。改写服务异常时默认退回原问题，并在结果 metadata 中写入
`query_transform_warning`；关闭 `fallback_on_transform_error` 后会直接传播异常。

## Reranker

所有重排器实现 `rag_agent_platform.retrieval.reranker.BaseReranker`：

```python
rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]
```

`TokenOverlapReranker` 是无需模型服务即可运行的确定性实现，将查询词覆盖率与召回分数加权
组合，并在 metadata 中保留 `pre_rerank_score`、`token_overlap_score` 和 `rerank_score`。
`RerankingRetriever` 用于包装任意 Retriever，先扩大候选集，再调用可插拔 Reranker 并截取
Top-K。未来 Cross-Encoder 或外部 Rerank API 只需实现 `BaseReranker`，无需修改上层接口。

## 上下文压缩

压缩器实现 `rag_agent_platform.retrieval.compression.ContextCompressor`：

```python
compress(
    query: str,
    chunks: list[RetrievedChunk],
    max_chars: int,
) -> list[RetrievedChunk]
```

`SentenceContextCompressor` 按查询词覆盖率选择句子，并保证所有返回内容的总字符数不超过
`max_chars`。压缩不会改变 `chunk_id`、文档/父块关联、来源、页码或分数；metadata 会记录压缩
方法及压缩前后长度。`CompressionRetriever` 可包装任意 Retriever，在统一接口内执行压缩。
当前实现适合自然语言段落；代码块、Markdown 表格和结构化文本可能被截断，接入真实文档时
应依据 Chunk 类型配置跳过策略或专用压缩器。全局预算按检索排名依次分配，不保证 Chunk 间
平均分配。

## 无答案阈值

`rag_agent_platform.retrieval.threshold.RelevanceThreshold` 支持四项组合规则：单块最低分、
第一名最低分、最少有效结果数，以及第一名与第二名的最低分差。任何条件不满足时返回空列表，
即当前公共 Retriever 契约中的“无答案”信号。

`ThresholdRetriever` 会扩大底层候选集以便判断最少证据数，再应用策略并返回 Top-K。阈值均
作用于 `[0.0, 1.0]` 的 `normalized_score`；具体值必须通过离线测试集校准，不能直接将默认值
视为生产配置。
只有一个有效结果时不应用 `min_score_gap`，因为不存在第二名。当前空列表同时表示“无相关
答案”和“底层 Retriever 正常返回空结果”；系统异常仍应抛出，不得转换为空列表。若成员四
需要展示拒绝原因，需要团队确认新的诊断接口，不能改变现有 `list[RetrievedChunk]` 契约。

## 检索离线评估

`rag_agent_platform.evaluation` 导出 `RetrievalEvaluationCase`、数据集加载器、排名指标和
`RetrievalEvaluationRunner`。Runner 对命名 Retriever 使用相同问题集和 Top-K，输出逐题结果
及 Recall@K、Precision@K、Hit Rate@K、MRR、nDCG@K、无答案准确率和平均耗时。

`relevant_document_ids` 仅是相关性标注，不会用于过滤检索结果；只有独立的 `document_ids`
表示用户选择的检索范围。详细格式与复现命令见 `docs/retrieval-evaluation.md`。

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
