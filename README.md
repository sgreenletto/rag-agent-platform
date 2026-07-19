# RAG Agent Platform

RAG Agent Platform 是一个四人课程小组共同开发的模块化智能文档问答系统。项目以
Streamlit 为交互入口，最终将逐步实现 Naive RAG、Advanced RAG、Modular RAG、
GraphRAG 与 Agentic RAG，并统一文档来源、检索结果、回答和评估契约。

## 当前阶段

当前版本是“正式工程骨架和 Mock 闭环”，用于稳定公共接口并支持成员并行开发。已实现：

- dataclass 公共数据模型与抽象接口；
- TXT、Markdown 固定长度 Mock 入库；
- 内存 Mock 文档仓库；
- Naive、Advanced、Graph 三个统一契约的 Mock Retriever；
- 规则路由的 Mock AgentService；
- 可上传、选择知识源、聊天并查看来源与执行轨迹的 Streamlit 页面；
- pytest、Ruff、协作文档与接口测试。

这些 Mock 只验证调用链，不代表真实 RAG、真实检索质量或真实 Agentic RAG 已完成。

## 最终目标

后续版本将逐步接入文档解析、父子切块、向量与元数据存储、Dense/BM25/RRF/Reranker、
知识图谱、真实生成与评估、查询改写循环和完整 LangGraph 工作流。

## 技术栈

- Python 3.12+
- uv
- Streamlit
- LangChain Core
- LangGraph
- Pydantic Settings
- pytest / pytest-cov
- Ruff
- Windows PowerShell

## 标准 src-layout

发布名称是 `rag-agent-platform`，Python 导入包名是 `rag_agent_platform`。`src` 仅是源码根
目录，所有正式模块都位于 `src/rag_agent_platform/`，项目代码统一从
`rag_agent_platform` 导入。

```text
rag-agent-platform/
├── app.py
├── src/rag_agent_platform/
│   ├── models/          # 公共数据模型
│   ├── ingestion/       # 入库接口与 Mock
│   ├── storage/         # 存储接口与内存 Mock
│   ├── retrieval/       # Retriever 接口与 Mock
│   ├── graph/           # 图服务接口
│   ├── generation/      # 回答生成接口
│   ├── evaluation/      # 回答评估接口
│   ├── agent/           # Agent 接口、State 与 Mock
│   └── ui/              # Streamlit 页面组装与组件
├── tests/
├── data/
├── docs/
├── scripts/
└── logs/
```

## 环境初始化

确认本机已安装 Python 3.12+ 与 uv，然后在项目根目录执行：

```powershell
uv sync
Copy-Item .env.example .env
```

`.env.example` 只包含空值或安全示例。不要提交真实密钥或密码。

## 启动 Streamlit

唯一推荐启动命令：

```powershell
uv run streamlit run app.py
```

默认访问地址为 `http://localhost:8501`。

## 测试与质量检查

```powershell
uv run python -m compileall src tests app.py
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

## 当前 Mock 能力

上传 TXT 或 Markdown 后，MockIngestionPipeline 会读取文本、固定长度切出父子块并保存到
进程内存；MockAgentService 根据问候、关系词、复杂词和问题长度路由到 NONE、GRAPH、
ADVANCED 或 NAIVE，再返回可展开的 Mock 引用与执行轨迹。PDF、DOCX 上传只显示已预留接口。

## 后续开发计划

详细四人分工与迭代顺序见 `docs/development-plan.md`。总体顺序为真实入库与存储、混合/高级
检索、GraphRAG、LangGraph Agent 与 UI 集成，并持续以公共接口测试保护协作边界。

## 分支说明

- `main`：稳定、可演示版本，不直接开发；
- `develop`：日常集成分支；
- `feature/*`：从 `develop` 创建，通过 Pull Request 合并回 `develop`。

建议功能分支：`feature/ingestion-storage`、`feature/advanced-retrieval`、
`feature/graphrag`、`feature/agent-ui`。
