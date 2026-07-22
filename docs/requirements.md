# 需求与范围

## 1. 项目背景与目标

RAG Agent Platform 是课程演示规模的模块化智能文档问答系统。它解决“把本地制度文档可靠地
入库，并比较 Naive、Advanced、Graph 和 Agentic RAG 工作流”的问题。目标是形成可安装、可测试、
可追踪来源和可演示故障降级的 Python 应用，不把课程 MVP 描述成企业级生产平台。

## 2. 用户角色

- **知识查询用户**：选择已入库文档并获得有引用的回答；
- **知识库维护者**：上传、查看和删除演示文档；
- **课程评审者/开发者**：切换检索策略，查看 execution trace，并复现测试结果。

当前没有登录、租户或角色授权模型，上述角色只是使用场景，不是 RBAC 实现。

## 3. 核心场景

1. 维护者上传 TXT、Markdown、PDF 或 DOCX，系统清洗、父子分块并同步持久化与索引。
2. 用户在限定文档范围内手动选择 RAG 模式，或让 Agent 自动分析并路由。
3. 系统检索、生成、评估；回答写坏时复用证据重新生成，检索不足时才保守重写并重检索。
4. 外部模型临时不可用时有限重试，再用确定性证据合成降级；无证据时明确拒答。
5. 评审者查看策略、计数、引用和 execution trace，验证 Branch、Loop 与终止条件。

## 4. 功能需求

| 编号 | 需求与边界 | 可执行验收 |
|---|---|---|
| FR-01 | 加载 TXT、Markdown、PDF、DOCX；拒绝空文件和不支持格式 | loader 与 pipeline 测试验证格式、元数据和异常 |
| FR-02 | 清洗文本并生成带 parent/child/document 关联的父子块 | cleaner、chunker 测试验证内容与标识 |
| FR-03 | 通过 `DocumentRepository` 持久化文档与块，生产支持 File 和 MySQL 实现 | 两类 Repository 合同与真实 MySQL 测试通过 |
| FR-04 | Child chunk 写入 Chroma，支持过滤、读取、删除和重启恢复 | Chroma 存储测试验证持久化与 document filter |
| FR-05 | 基于 Repository child chunks 建立和刷新 BM25 稀疏索引 | BM25 测试验证排序、过滤和刷新 |
| FR-06 | Naive RAG 执行 Dense 检索并回溯父块 | 手动 Naive 固定调用对应 Retriever，返回统一证据 |
| FR-07 | Advanced RAG 组合 Dense、BM25、RRF、Multi-Query 边界、重排、父块与压缩 | 组合与端到端测试验证分数、去重、过滤 |
| FR-08 | GraphRAG 抽取和持久化轻量实体关系，支持 1–2 跳关系证据 | Graph 构建、过滤、重启和关系检索测试通过 |
| FR-09 | Agent 分析 SIMPLE、COMPLEX、RELATION、CHAT 并路由 Naive、Advanced、Graph、None | 自动路由和手动覆盖测试验证策略 |
| FR-10 | LangGraph 真实执行 Workflow、条件 Branch 和有界 Loop | 节点/条件边测试验证路径与终止 |
| FR-11 | Generator 只基于去重后证据回答，并保持 citation 编号映射 | 生成与 citation 测试验证关键事实和 chunk identity |
| FR-12 | Evaluator 输出 PASS、REGENERATE、REWRITE_RETRIEVE、CLARIFY、REFUSE | 决策测试验证证据充分性与回答缺陷分流 |
| FR-13 | 查询重写必须语义守恒，保留数字和对象，不引入新实体/品牌/子问题 | rewrite 测试验证模型输出并程序化拒绝漂移 |
| FR-14 | REGENERATE 复用查询、策略和证据，独立计数，达到上限后终止 | workflow 测试验证 Retriever 不被重复调用 |
| FR-15 | 真正关键歧义可返回澄清；无相关依据或重试耗尽明确拒答 | CLARIFY/REFUSE 与无答案回归测试通过 |
| FR-16 | 自由生成失败时，从少量相关原文句和 Graph relation 生成有界 grounded fallback | fallback 测试验证长度、相关性、引用与无全文倾倒 |
| FR-17 | LLM 临时网络/429/5xx 有限重试，401 等配置错误不重试 | transport retry 测试验证次数、错误类型与安全摘要 |
| FR-18 | 删除文档同步清理 Repository、Chroma、BM25 和 Graph | coordinated deletion 与各存储测试通过 |
| FR-19 | File/MySQL 文档数据、Chroma 与 Graph 在重建服务后可恢复 | repository、Chroma、Graph 持久化测试通过 |
| FR-20 | Streamlit 提供上传、删除、文档选择、四模式问答、引用和 trace 展示 | AppTest 与应用服务集成测试无异常 |
| FR-21 | `AgentResult.execution_trace` 记录路由、评估、rewrite、regenerate、transport retry 和终止原因 | Agent 工作流测试断言可观察事件 |
| FR-22 | Agent、Naive、Advanced、Graph 四种模式共享 `RetrievedChunk`/`Citation` 契约 | 合同和 AgentService 测试验证公共模型 |

## 5. 非功能需求

| 编号 | 要求 | 验证方式 |
|---|---|---|
| NFR-01 | Python 3.12+、标准 src-layout、只用 `rag_agent_platform` 导入 | package/architecture 测试、compileall |
| NFR-02 | UI、编排、领域接口、基础设施职责清楚，不反向穿透 | AST 架构边界测试 |
| NFR-03 | 高层服务构造函数注入依赖，生产装配集中在 `bootstrap.py` | DI 与 ApplicationServices 测试 |
| NFR-04 | 默认 CI 不调用收费模型、不读取开发者 `.env` | CI 配置合规测试 |
| NFR-05 | 密钥只从环境读取，日志、异常和环境检查不输出具体值 | transport/config/environment 测试 |
| NFR-06 | 所有重试有上限，失败可观察，不产生无限循环或 Streamlit 白屏 | workflow、retry、AppTest |
| NFR-07 | 关键写入/删除和测试相互隔离，运行数据不进入 Git | 临时目录 fixture、Git 检查 |
| NFR-08 | uv lock、compileall、Ruff、pytest 构成可复现质量门禁 | 本地命令与 GitHub Actions |
| NFR-09 | 本地课程数据规模可交互；当前不承诺生产 SLA、吞吐或可用性 | 在限制文档中明确，无虚构性能指标 |

## 6. 系统边界

当前负责：单机/课程演示规模知识库；本地文档入库；File/MySQL 元数据与块持久化；Chroma、BM25
和 NetworkX 索引；四类 RAG；Agent 编排；引用、评估、重试、拒答和确定性降级；Streamlit 管理与
演示。

依赖边界：MySQL 是可选的外部数据库；OpenAI-compatible LLM/Embedding 是可选外部服务，系统只
保证客户端重试和安全降级，不保证服务商可用性或输出质量。

## 7. 非目标

- 高并发、多节点或高可用生产集群；
- 企业级多租户、登录、RBAC、审计日志与合规认证；
- 分布式向量数据库和商业级知识图谱编辑平台；
- 模型训练、微调或 Cross-Encoder 训练；
- 复杂 OCR、跨页表格重建、所有文件格式/语言完全兼容；
- 对外部模型服务的绝对可用性保证；
- 自动化云部署、生产监控、告警、备份和灾备。

## 8. 总体验收标准

- `uv sync --frozen` 后 package 可导入，环境检查给出不泄密的诊断；
- compileall、Ruff format、Ruff lint、全量 pytest 通过；
- MySQL 环境可用时真实 Repository 与 ApplicationServices 集成测试通过；
- 四类模式、五类评估决策、Query Rewrite、transport retry、fallback 和拒答行为有自动测试；
- 删除和重启恢复覆盖 MySQL/Chroma/Graph；
- Streamlit AppTest 无未捕获异常；
- `.env`、真实密钥和运行数据不进入变更。

需求到实现和测试的逐项证据见 [traceability.md](traceability.md)。

## 9. 开发阶段

真实 tag 与能力演进见 [mvp-plan.md](mvp-plan.md)。当前代码是课程最终交付版本 `1.0.0`；
`v1.0.0` tag 将在本次提交合入 main 并通过远程质量门禁后创建，上一正式里程碑为 `v0.7.0`。
