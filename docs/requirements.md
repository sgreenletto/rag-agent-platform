# 项目需求与边界

## 1. 项目背景

RAG Agent Platform 是基于 Streamlit 的模块化智能文档问答系统。当前课程项目支持文档上传与
知识库管理、Naive RAG、Advanced RAG、GraphRAG、Agentic RAG、来源引用和 Agent 执行轨迹。
项目重点是用统一接口比较不同检索策略，并形成可安装、可测试、可演示的本地 MVP。

## 2. 目标用户

- 需要查询企业内部文档的普通用户；
- 需要比较不同 RAG 策略的课程评审者；
- 需要维护演示知识库的管理员。

本阶段不引入多租户、细粒度权限或复杂组织角色。

## 3. 核心用户流程

```text
上传文档
→ 文档解析和切分
→ 建立向量、稀疏和图索引
→ 选择知识源
→ 提交问题
→ 手动或自动选择检索方式
→ 检索
→ 生成
→ 评估和有限重试
→ 返回回答、来源和执行轨迹
```

## 4. 功能需求

| 编号 | 功能 | 输入 | 输出 | 异常或边界 | 验收条件 |
|---|---|---|---|---|---|
| FR-01 | 文档上传 | `.txt`、`.md`、`.pdf`、`.docx` 文件 | 安全命名的上传文件和 `IngestionResult` | 空文件、不支持格式、解析失败时明确报错；不得覆盖同名文件 | 页面可上传当前支持格式，成功后显示父子块数量 |
| FR-02 | 文档列表 | 当前 Repository | `list[DocumentRecord]` | Repository 为空时显示空状态 | 入库成功后列表出现文档且可被选择 |
| FR-03 | 删除文档 | `document_id` | Repository、父子块、向量、图索引同步清理 | 不得只删除 UI 列表项；底层失败需显示错误 | 删除后 `list_documents`、块、向量和图均不再包含该文档 |
| FR-04 | Naive RAG | 查询、可选 `document_ids`、Top-K | Dense + 父块回溯的 `list[RetrievedChunk]` | 空选择范围返回空；结果必须限制在指定文档 | 手动 Naive 模式固定调用 Naive Retriever，结果有统一分数和来源 |
| FR-05 | Advanced RAG | 查询、可选 `document_ids`、Top-K | Dense/BM25/RRF/重排/压缩后的证据 | 单路检索失败按既有策略降级；不得绕过文档过滤 | 手动 Advanced 模式可执行，返回统一 `RetrievedChunk` |
| FR-06 | GraphRAG | 关系查询、可选 `document_ids`、Top-K | 一跳或两跳图证据 | 无可识别实体或图证据时返回空，不伪造关系 | 手动 Graph 模式调用 `GraphRetriever`，可观察 retrieval method |
| FR-07 | Agent 自动路由 | 查询、`mode="agent"` | SIMPLE/COMPLEX/RELATION/CHAT 与对应策略 | LLM 分类失败时记录轨迹并使用规则兜底 | 简单、复杂、关系、问候分别路由 Naive、Advanced、Graph、None |
| FR-08 | 回答生成 | 原始问题和检索证据 | 回答文本和 `Citation` | 空证据不得生成确定性答案；不得引用不存在来源 | 关键证据带 `[1]` 等编号且 Citation 对应 RetrievedChunk |
| FR-09 | 来源引用 | Generator 输出与检索块 | source、page、chunk_id | page 缺失时允许为空；不得伪造页码 | 页面可展开来源，编号与回答上下文一致 |
| FR-10 | 答案评估和有限重试 | 原问题、回答、证据、重试预算 | 评估结果、可选重写查询 | 空检索必须失败；达到 `max_retries` 必须停止 | 首次失败可重写并重试，默认最多重试 2 次且无无限循环 |
| FR-11 | 聊天界面 | 用户消息、模式和知识源选择 | 聊天历史与 AgentResult | 服务初始化或问答失败时展示清晰错误 | `st.chat_input` 可提交，历史在 Session State 中保留 |
| FR-12 | 执行轨迹展示 | `AgentResult.execution_trace` | query type、strategy、retry count、retrieval method 和节点轨迹 | CHAT 可无知识库来源；错误轨迹不得含密钥 | 页面可展开执行信息，并观察自动路由和重试过程 |

## 5. 非功能需求

- **可维护性**：采用 src-layout、单一公共模型和职责明确的 package；根入口保持轻量。
- **模块化**：Retriever、Generator、Evaluator、Repository 和 Agent 通过接口组合。
- **可测试性**：业务编排可注入 Fake/Mock，不依赖真实 API Key；关键 Branch、Loop 和删除行为有测试。
- **配置安全**：密钥仅从环境读取，不提交 `.env`，错误信息和日志不输出密钥。
- **错误可观察性**：配置、解析、检索、生成和评估失败有明确异常或 `AgentResult.error`。
- **基础响应时间**：本地小规模演示应能交互使用；本项目未建立足以承诺精确 SLA 的性能基线。
- **可安装运行**：在 Python 3.12 和 uv 的干净环境中可执行 `uv sync --frozen`、测试和启动命令。

## 6. 项目边界

本次不实现：

- 多用户认证和复杂权限；
- 云端分布式部署；
- Redis 集群；
- 大规模向量数据库集群；
- Embedding 微调；
- 完整微软 GraphRAG；
- 复杂 OCR；
- 跨页表格重建；
- 生产级监控告警；
- 高可用和灾备。

此外，MySQL 配置目前仅为预留，正式元数据实现仍是本地 JSON Repository；默认 Hash Embedding
和规则图抽取面向课程 MVP，不代表生产模型能力。

## 7. 验收标准

- `uv sync --frozen` 成功；
- `import rag_agent_platform` 指向 `src/rag_agent_platform`；
- compileall、Ruff format check、Ruff lint 和 pytest 全部通过；
- Streamlit 可启动并产生 Local URL；
- 页面支持 TXT、Markdown、PDF、DOCX；
- Agent、Naive、Advanced、Graph 四种模式可选择；
- Agent 自动路由、来源和执行轨迹可观察；
- 空上下文拒答，达到最大重试次数后停止；
- 文档删除同步清理 Repository、父子块、向量和图数据；
- 默认测试和 CI 不依赖 LLM API、MySQL 或用户上传文档；
- Git 变更不包含 `.env`、密钥、运行数据库、上传文档、日志或模型权重。
