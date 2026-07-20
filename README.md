# RAG Agent Platform

RAG Agent Platform 是一个采用标准 Python src-layout 的模块化智能文档问答系统。当前版本已把
真实文档入库、Naive/Advanced/Graph 检索、LangGraph Agent、回答生成/评估/重写循环和
Streamlit 页面装配为一个可运行闭环；所有检索模式统一输出 `list[RetrievedChunk]`。

## 已实现能力

- TXT、Markdown、PDF、DOCX 解析、清洗与父子切块；
- JSON 文档元数据仓库、Chroma 子块向量存储及文档级一致性删除；
- 本地 Hash Embedding、Dense Retrieval 与父块回溯；
- BM25、Dense + BM25 Hybrid、加权 RRF、Multi-Query 边界、Token Overlap Reranker、
  句子压缩和阈值组件；
- NetworkX 图存储、规则实体关系抽取、1–2 跳 GraphRAG 和文档过滤；
- 真实 LangGraph `StateGraph`：问题分析、Retriever 分支、生成、评估、查询重写和有界重试；
- Agent 自动、Naive RAG、Advanced RAG、GraphRAG 四种页面模式；
- 文件上传、文档多选、真实删除、聊天历史、引用、检索方式、错误与执行轨迹展示；
- OpenAI-compatible Chat Completions 单一适配器；未配置 LLM 时使用规则分类、真实检索、
  保守抽取式生成和引用校验，不会静默切换到 Mock；
- 单元、契约、LangGraph 集成和真实本地容器测试。

## 架构

```text
Streamlit
→ Service Container
→ LangGraph Agent
→ Retriever Router
   ├─ Naive: Dense → Parent Context
   ├─ Advanced: Dense + BM25 → RRF → Multi-Query → Reranker → Parent → Compression
   └─ Graph: Entity/Relation Extraction → NetworkX GraphRetriever
→ Grounded Generator
→ Conservative Evaluator
→ Rewrite Loop (max_retries=2)
```

根目录 `app.py` 是唯一 Streamlit 入口，长生命周期对象统一由
`rag_agent_platform.bootstrap.build_service_container()` 创建，并通过 `st.cache_resource`
跨 rerun 复用。

## 环境配置

需要 Python 3.12+ 与 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync
Copy-Item .env.example .env
```

默认 `APP_MODE=real`。本地模式无需 API Key，会使用 Hash Embedding 和保守抽取式回答。若需模型
分类、生成、评估和重写，可配置：

```dotenv
LLM_PROVIDER=openai-compatible
LLM_MODEL=your-model
LLM_API_KEY=your-key
LLM_BASE_URL=https://your-endpoint.example/v1
```

不要提交 `.env`、真实密钥、上传文档、Chroma 数据、图数据或日志。`EMBEDDING_PROVIDER` 当前仅
支持空值、`local` 或 `hash`；其他值会在启动时给出明确错误。所有可用配置见 `.env.example`。

## 启动

```powershell
uv run streamlit run app.py
```

默认地址为 `http://localhost:8501`。侧边栏支持 `.txt`、`.md`、`.pdf`、`.docx`，上传成功后会
自动选择该文档。未选择文档表示空检索范围，不会跨范围生成答案。

页面提供四种模式：

- Agent 自动模式：SIMPLE → Naive、COMPLEX → Advanced、RELATION → Graph、CHAT → NONE；
- Naive RAG：强制 Dense + 父块回溯；
- Advanced RAG：强制混合召回、RRF、重排、父块回溯和压缩；
- GraphRAG：强制图检索。

只有显式设置 `APP_MODE=mock` 才会启用 Mock 入库与 Mock Agent，页面会明显警告。

## 测试与质量检查

```powershell
uv run python -c "import rag_agent_platform; print(rag_agent_platform.__file__)"
uv run python -m compileall src tests app.py
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

检索离线评估格式与指标见 `docs/retrieval-evaluation.md`。

## 当前限制

- Hash Embedding 适合本地闭环和测试，不等价于生产语义 Embedding；
- 图实体关系抽取目前是确定性规则实现，不是 LLM/NER 生产抽取器；
- 未配置 LLM 时回答是带真实引用的抽取式证据汇总，综合表达能力有限；
- Advanced 的 Multi-Query 默认使用 Identity Transformer，尚未装配模型查询扩展或
  Cross-Encoder Reranker；
- 文档元数据当前使用 JSON 文件，不是 `.env.example` 中预留的 MySQL；
- PDF 页数会记录在文档元数据，但当前固定长度块切分尚未保留逐页边界，因此 PDF/DOCX 块的
  `page` 可能为空；
- 尚未进行生产级并发、权限、超大文件、远程 Provider、阈值校准与端到端浏览器自动化验证。

分支约定见 `CONTRIBUTING.md`，模块边界见 `docs/interfaces.md`，当前完成情况见
`docs/development-plan.md`。
