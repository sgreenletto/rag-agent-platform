# 成员二检索模块：Embedding、Chroma 与参数审计

## 1. 当前实现情况

- `bootstrap.py` 通过 `build_embedding_model(settings)` 创建唯一 EmbeddingModel，并把同一实例注入 `ChromaVectorStore`；入库和查询均经该 VectorStore 使用它。Retriever 内没有创建第二个 Embedding 客户端。
- 查询向量由 `ChromaVectorStore.search()` 使用所注入的 EmbeddingModel 生成；DenseRetriever 只接收 `DenseSearchBackend`，不依赖模型供应商细节。
- Chroma 持久化目录和 Collection 名称均由统一 `Settings` 的 `CHROMA_PERSIST_DIRECTORY`、`CHROMA_COLLECTION_NAME` 提供，默认分别为 `data/chroma`、`rag_child_chunks`。Retriever 不创建 Collection，也没有第二套默认值。
- Collection 使用 cosine 空间。Chroma 返回 distance；Bootstrap 明确以 `score_kind="distance"` 构造 DenseRetriever。DenseRetriever 先用负距离建立“越大越相关”的排序分数，再做 Min-Max 归一化。相同分数统一归一为 1.0，空结果不参与除法。
- Collection metadata 已记录 `embedding_provider`、`embedding_model`、`embedding_base_url` 和已知时的 `embedding_dimensions`。存储适配层在查询前检查当前 Embedding identity 与 Collection metadata；模型或维度不一致会拒绝查询并提示重建索引。

## 2. 本次修改

| 文件 | 内容与原因 | 对其他成员接口影响 |
| --- | --- | --- |
| `.env.example`、`config.py` | 在现有 Settings 中增加 Dense/BM25 候选数、Rerank 保留数和归一化阈值；没有新建 Settings | 仅新增可选字段，兼容 |
| `retrieval/config.py` | 新增 `RetrievalParameters.from_settings()` 和跨参数校验，供成员四后续注入 | 新增公共类型，不改变旧调用 |
| `retrieval/dense.py` | 增加可选固定 `candidate_k`；保留 multiplier 默认行为；补充标准化分数 metadata | 构造参数仅增量可选 |
| `retrieval/hybrid.py` | 增加可选、相互独立的 Dense/BM25 candidate K，并校验不小于请求 top_k | 构造参数仅增量可选 |
| `retrieval/reranker.py` | 增加可选 `rerank_top_k` 输出上限，保持默认行为 | 构造参数仅增量可选 |
| `retrieval/__init__.py` | 导出 RetrievalParameters | 新增导出 |
| `tests/test_retrieval_config.py` 及三个既有测试文件 | 覆盖配置映射、边界、显式覆盖、独立候选数、分数 metadata 和 Rerank 上限 | 无运行时影响 |

`retrieve()` 签名、`list[RetrievedChunk]` 返回契约、已有 metadata 字段和所有旧构造方式均保持不变。

## 3. 一致性验证

- 模型一致性：已验证。Collection metadata 中 provider/model/base URL 与当前 Embedding identity 冲突时，现有 ChromaVectorStore 会快速失败。
- 维度一致性：已验证已记录维度的 Collection；Hash 模型建库与查询维度不一致时会在查询前失败。
- 空 Collection：不会误报维度冲突；缺 metadata 的空 Collection 由成员一现有实现初始化 identity metadata。
- 更换模型：只要旧 Collection 有 identity metadata，查询会被阻止，并要求重建向量索引。
- Retriever 不直接读取 Collection 私有实现，也不自动删除、迁移或重建数据。

## 4. 检索参数

| 环境变量 | 默认值 | 语义与作用位置 | 调用方可覆盖 |
| --- | ---: | --- | --- |
| `RETRIEVAL_TOP_K` | 5 | 最终返回上层的最大块数；当前由成员四注入调用 | 是，`retrieve(top_k=...)` |
| `DENSE_CANDIDATE_K` | 20 | Dense 初始召回上限；必须不小于最终 top_k | 是，Dense/Hybrid 可选构造参数 |
| `BM25_CANDIDATE_K` | 20 | Hybrid 调用 BM25 的初始召回上限；必须不小于最终 top_k | 是，Hybrid 可选构造参数 |
| `RERANK_TOP_K` | 5 | Reranker 重排后最多保留数，必须大于 0 | 是，RerankingRetriever 可选构造参数 |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.0 | 对 `[0,1]` 的统一归一化相关性生效，不直接比较 Chroma distance、BM25 或 RRF 原始分数 | 是，现有阈值构造参数 |

显式构造参数优先，Settings 不会暗中覆盖单元测试或调用方传值。Dense metadata 现在同时保留旧字段及 `raw_score`、`score_type`、`normalized_score`。

## 5. 测试结果

- 新增测试：12 个。
- 收集总数：309 个。
- 隔离真实外部服务后：308 passed，1 skipped，0 failed。
- 格式检查：121 files already formatted。
- Ruff：All checks passed。
- Compileall：通过。
- 覆盖率：项目未配置覆盖率命令，本次未生成覆盖率数字。
- 外部服务：未调用真实 Embedding API；隔离执行时未调用真实 MySQL。

本机 `.env` 启用了真实 MySQL 且凭据被拒绝时，全套测试表现为 308 passed、1 failed；失败项是成员一的真实数据库契约测试，与检索修改无关。将 `DOCUMENT_REPOSITORY_PROVIDER=file` 且清空 `MYSQL_USER` 后，该测试按设计跳过，全套通过。

## 6. 仅记录、不跨模块修改的问题

### 成员一

- 非空旧 Collection 若缺少 `embedding_dimensions`，现有实现会补写当前查询模型维度，未通过一条样本向量先确认旧索引维度。建议存储层提供只读的 Collection identity/dimension 描述接口，或在建库/首次 upsert 时强制写全 metadata。
- `ChromaVectorStore.search()` 直接取 `embed_texts([query])[0]`。外部实现若返回 `None`、空列表、NaN/Inf 或错误嵌套结构，错误不一定统一清晰。建议在 VectorStore 的 query-embedding 边界复用严格数值向量验证；该逻辑属于生成查询向量的存储边界，本次未跨模块修改。
- 建议统一兼容 metadata 键 `embedding_dimension` 与当前 `embedding_dimensions`，并保持向后兼容。
- 相对 Chroma 路径当前由运行进程工作目录解释；建议配置/Bootstrap 层提供项目根目录解析后的绝对路径，再统一注入入库和查询侧。

### 成员三

- 本次未发现需要图检索模块修改的问题。若图检索参与统一阈值处理，应只输出 `[0,1]` 且越大越相关的 `normalized_score`，原始图分数保存在 metadata。

### 成员四

- Bootstrap 当前仍只使用 `RETRIEVAL_TOP_K`，尚未把 `RetrievalParameters.from_settings(settings)` 中的 Dense/BM25 candidate K、Rerank top K 和 threshold 注入组合链。建议后续只在组装层注入，不让各 Retriever 自行读取全局 Settings。
- 继续保持单一 EmbeddingModel 注入 ChromaVectorStore 的现有方式；不要在 Naive、Hybrid 或包装 Retriever 中再创建模型客户端。

## 7. 风险结论

- 入库与查询模型不一致：正常 Bootstrap 路径下风险低，因为共享同一实例；直接手工构造不同实例时，已有 identity metadata 可阻止错误查询。
- 向量维度不一致：已记录维度时会及时失败；非空旧 Collection 缺维度 metadata 时仍有上述遗留风险。
- 连接错误 Collection：统一配置和单一 VectorStore 组装已避免 Retriever 使用不同默认名；相对路径受工作目录影响仍需成员四在配置注入层收口。
- 修改 Embedding 模型后必须重建旧索引，系统不会自动迁移或重建。
- 对已记录 identity 的现代 Collection，错误模型/维度能够快速失败；对不完整 legacy metadata 的非空 Collection 尚未覆盖所有失效模式。
