# Agentic RAG Platform

Current stable version: `v1.0.0`

Previous milestone: `v0.7.0`

Project status: **Final course delivery**

模块化智能文档问答平台：把 TXT、Markdown、PDF、DOCX 入库到 File/MySQL、Chroma、BM25 和
NetworkX Graph，并通过 Naive、Advanced、GraphRAG 或 LangGraph Agent 返回有依据、有引用、
可查看 execution trace 的回答。

目标用户是课程评审者、知识查询用户和演示知识库维护者。项目用于单机/课程演示规模的策略比较与
工程实践，不是生产级多租户知识平台。当前代码是用于创建 `v1.0.0` 的最终交付版本；tag 将在
本次提交依次合入 develop、main 并完成远程质量门禁后，由维护者在 main 稳定提交上创建。

## 功能概览

- 文档加载、清洗、父子分块，File/MySQL 文档与 chunk 持久化；
- Chroma Dense、BM25、Hybrid/RRF、Multi-Query 边界、重排、父块回溯和上下文压缩；
- 规则实体关系抽取、NetworkX 持久化、1–2 跳 GraphRAG；
- LangGraph Workflow/Branch/Loop 与 Agent/Naive/Advanced/Graph 手动模式；
- PASS、REGENERATE、REWRITE_RETRIEVE、CLARIFY、REFUSE；
- Query Rewrite 语义守恒、回答评估、无答案拒答、引用映射和 execution trace；
- LLM transport retry、异常回答拦截、确定性 Grounded Fallback；
- 文档删除同步 Repository、Chroma、BM25、Graph，支持重启持久化；
- Streamlit 上传、选择、删除、聊天、引用与执行决策展示。

## v1.0.0 最终状态

本版本已完成课程范围内的文档处理、存储、四类 RAG、Agent 编排、答案评估、失败恢复、引用、
删除同步、Streamlit UI、分层/依赖注入、CI 和工程文档闭环。上一正式工程里程碑是 `v0.7.0`；
从该里程碑到最终版本主要完成了 evaluation decision 分流、Query Rewrite 守恒、transport retry、
Grounded Fallback、人工验收回归和最终工程规范审计。正式发布说明见
[v1.0.0 release notes](docs/releases/v1.0.0.md)。

## 架构概览

```text
Streamlit UI → ApplicationServices
                    ├─ Coordinated Ingestion → Repository + Chroma + BM25 + Graph
                    └─ LangGraph Agent
                         → Retriever → Generator → Evaluator
                         → pass / regenerate / rewrite-retrieve / clarify / refuse
```

`src/rag_agent_platform/bootstrap.py` 是唯一生产 Composition Root。UI 不直接访问数据库；Agent
节点不创建模型、Retriever 或存储；高层服务通过构造函数依赖抽象。详细规则见
[architecture.md](docs/architecture.md) 和 [interfaces.md](docs/interfaces.md)。

## 环境准备

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。PowerShell：

```powershell
uv sync --frozen
Copy-Item .env.example .env
uv run python scripts/check_environment.py
```

环境脚本只显示 credential `configured/missing`，不打印密钥。不要提交 `.env`、上传文档、运行
数据库、Chroma/Graph 数据或日志。

### real、test/hash 与 mock

- `APP_MODE=real`（默认）：真实入库、持久化、Retriever、Graph 和 Agent；模型 provider 可选。
- `EMBEDDING_PROVIDER=hash`：外部服务零依赖的确定性测试/本地模式，不等价于生产语义模型。
- LLM provider 留空：不切换 Mock，仍运行真实检索，并用规则路由/评估和 grounded fallback。
- `APP_MODE=mock`：只在显式测试或 UI 开发时使用，页面会显示警告；真实服务失败不会静默降级。

外部 OpenAI-compatible 示例（值只写入本地 `.env`）：

```dotenv
LLM_PROVIDER=openai-compatible
LLM_MODEL=your-model
LLM_API_KEY=your-key
LLM_BASE_URL=https://your-endpoint.example/v1
```

Embedding 同样支持 `openai-compatible`，对应字段见 `.env.example`。外部模型临时网络/429/5xx 会
有限重试；401/403 等配置错误不重试。

## MySQL 初始化

`.env.example` 以 MySQL 为真实本地示例。先创建专用数据库/账号，再配置 `MYSQL_*`；Repository
首次连接会创建所需表。初始化参考：

```powershell
mysql -u root -p < scripts/init_mysql.sql
uv run pytest tests/test_mysql_repository.py -v -s
```

不需要 MySQL 时设置 `DOCUMENT_REPOSITORY_PROVIDER=file`，元数据写入 `METADATA_PATH`。

## 启动与使用

```powershell
uv run streamlit run app.py
```

默认 Local URL 通常是 `http://localhost:8501`。侧边栏上传并选择文档，然后选择：

- Agent 自动：SIMPLE→Naive、COMPLEX→Advanced、RELATION→Graph、CHAT→None；
- Naive：Dense + parent context；
- Advanced：Dense + BM25 + fusion + reranking + parent/compression；
- GraphRAG：实体关系证据。

可用 `tests/fixtures/enterprise_procurement_policy.txt` 做演示。演示步骤见
[demo-script.md](docs/demo-script.md)。

## 测试和质量门禁

```powershell
uv run python -m compileall src tests scripts app.py
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run pytest tests/test_real_container_integration.py -v -s
uv run pytest -q -k "streamlit or app"
uv run pytest --cov=src/rag_agent_platform --cov-report=term-missing
```

2026-07-22 `v1.0.0` release candidate 最终本地复验为全量 379 passed、语句覆盖率 88%、MySQL
Repository 6 passed、ApplicationServices 5 passed、Streamlit AppTest 1 passed。完整命令与限制见
[acceptance-report.md](docs/acceptance-report.md)。

GitHub Actions 在 push/PR 到 `main`、`develop` 时运行外部模型零依赖的全量质量 job，并在临时
MySQL 8.4 service 中运行 Repository 合同 job。详见 [testing-strategy.md](docs/testing-strategy.md)。

## 常见错误

- `APP_MODE must be either real or mock`：检查 `.env` 中模式拼写。
- LLM/Embedding 缺少 model/key：补齐同一 provider 的字段，环境脚本不会显示具体 key。
- MySQL connection refused：确认服务、host/port、数据库和账号；可临时使用 File provider。
- Chroma/Graph 路径不可写：停止应用，修正目录权限后重启；不要在运行时手工删部分索引。
- `uv sync --frozen` 提示 lock 不一致：开发者先审阅依赖变更并执行 `uv lock`；CI 不自动改 lock。
- 外部模型偶发断连：查看 execution trace 的 transport retry/fallback；持续 401 通常是配置问题。

## 需求、边界和项目演进

- [需求与范围](docs/requirements.md)：FR/NFR、角色、场景、验收和非目标；
- [需求追踪矩阵](docs/traceability.md)：需求—模块—接口—测试；
- [MVP 与真实 tag 演进](docs/mvp-plan.md)；
- [开发步骤](docs/development-plan.md) 与 [工程实践](docs/engineering-practice.md)；
- [部署](docs/deployment.md)、[发布流程](docs/release-process.md)、
  [最终验收](docs/acceptance-report.md)、[v1.0.0 发布说明](docs/releases/v1.0.0.md)。

当前明确不支持：生产级高并发/高可用集群、多租户与 RBAC、分布式向量库、模型训练/微调、商业
图谱编辑平台、复杂 OCR/跨页表格、生产监控告警和自动备份灾备。完整边界以 requirements 为准。

协作约定见 [CONTRIBUTING.md](CONTRIBUTING.md)，正式版本历史见 [CHANGELOG.md](CHANGELOG.md)。
