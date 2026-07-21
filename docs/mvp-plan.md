# MVP 与持续发布计划

状态只描述当前仓库可验证能力，不代表生产就绪，也不虚构历史发布时间。

## MVP 0：工程骨架

- **目标**：形成可安装、可协作的 Python 工程和无外部服务闭环。
- **完成功能**：标准 package、公共模型与抽象接口、Mock 闭环、Streamlit 基础页面、基础测试。
- **验收条件**：包可导入；Mock Agent/UI 可运行；公共契约测试通过。
- **建议版本号**：`v0.1.0`。
- **当前状态**：完成。

## MVP 1：Naive RAG

- **目标**：完成单文档问答的真实入库、Top-K 检索、回答和来源链路。
- **完成功能**：TXT/Markdown/PDF/DOCX 解析、清洗、父子切块、Chroma 向量入库、Dense Top-K、
  父块回溯、来源展示；已提供可配置 OpenAI-compatible 生成和无 LLM 的本地 grounded fallback。
- **验收条件**：真实文件可入库；Naive 模式返回规范化证据和 Citation；删除可清理向量。
- **建议版本号**：`v0.2.0`。
- **当前状态**：部分完成——本地闭环完成，但默认 Hash Embedding 和可选 LLM 不等价于生产模型。

## MVP 2：Advanced RAG

- **目标**：增加混合召回、融合和重排，提高多证据问题的可组合性。
- **完成功能**：父子索引、Dense Retrieval、BM25、Hybrid、加权 RRF、Token Overlap Reranker、
  Context Compression、Threshold 和 Multi-Query 接口；Agent Loop 已实现查询重写。
- **验收条件**：Advanced Pipeline 保留来源、分数和文档过滤；单路失败可观察；离线评估可运行。
- **建议版本号**：`v0.3.0`。
- **当前状态**：部分完成——核心组合已完成，Multi-Query 默认仍是 Identity Transformer，尚无
  Cross-Encoder Reranker 和真实评估阈值校准。

## MVP 3：GraphRAG

- **目标**：用轻量知识图支持实体关系证据检索。
- **完成功能**：规则实体关系提取、NetworkX 图结构、文档级删除、一跳/两跳查询、
  `GraphRetriever` 和统一 RetrievedChunk 输出。
- **验收条件**：关系文档可构图；Graph 查询返回受 `document_ids` 限制的证据；图删除有测试。
- **建议版本号**：`v0.4.0`。
- **当前状态**：完成当前课程范围；不等于完整微软 GraphRAG 或生产图平台。

## MVP 4：Agentic RAG

- **目标**：用有界工作流自动选择检索策略并评估回答。
- **完成功能**：analyze、CHAT/NAIVE/ADVANCED/GRAPH Branch、Retriever 路由、generate、evaluate、
  rewrite 和有限 Loop；Streamlit 展示来源、策略和执行轨迹。
- **验收条件**：自动与手动路由测试通过；失败后真实重写；达到 `max_retries` 停止；异常可观察。
- **建议版本号**：`v0.5.0`。
- **当前状态**：完成当前课程范围；仓库当前 HEAD 已存在真实 `v0.5.0` Tag。

## Release 1.0：答辩稳定版

- **目标**：从功能 MVP 收口为可复现、可审计、可演示的稳定版本。
- **完成功能**：计划包含完整集成、需求/架构/接口/工程文档、全量测试、CI、演示 fixtures、
  CHANGELOG 和 Release Notes。
- **验收条件**：develop CI 通过；合入 main 后重复验证；版本元数据一致；组长创建 Tag 和 Release；
  已知限制在 Release Notes 中公开。
- **建议版本号**：`v1.0.0`，只有全部验收完成后方可使用。
- **当前状态**：部分完成——代码、测试和文档已接近收口；main 稳定合并、远程 CI、Tag 和 GitHub
  Release 仍需人工完成。

## 持续发布原则

每个 MVP 都应经历 feature 分支、Pull Request、develop 集成、自动检查和人工冒烟。稳定候选再从
develop 合入 main，由组长在真实对应提交上创建语义化 Tag；禁止仅依据计划提前打 Tag。
