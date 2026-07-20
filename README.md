# RAG Agent Platform

RAG Agent Platform 是一个四人课程小组共同开发的模块化智能文档问答系统。项目以
Streamlit 为交互入口，最终将逐步实现 Naive RAG、Advanced RAG、Modular RAG、
GraphRAG 与 Agentic RAG，并统一文档来源、检索结果、回答和评估契约。

## 当前阶段

当前版本以“正式工程骨架和 Mock 闭环”为基线，并已开始接入真实检索组件。已实现：

- dataclass 公共数据模型与抽象接口；
- TXT、Markdown 固定长度 Mock 入库；
- 内存 Mock 文档仓库；
- Naive、Advanced、Graph 三个统一契约的 Mock Retriever；
- 基于 jieba/rank-bm25 的 BM25 Retriever；
- 面向 Chroma 等向量存储适配器的 Dense Retriever；
- 支持加权 RRF、候选去重与可观测降级的 Hybrid Retriever；
- 可插拔 Query Transformer、Multi-Query RRF 合并与查询改写降级；
- 可插拔 Reranker 接口和可运行的 Token Overlap Reranker；
- 保留证据来源的句子级上下文压缩和可组合无答案阈值；
- 支持 JSON/JSONL 问题集、标准排名指标、耗时统计和多模式对比的离线评估；
- 规则路由的 Mock AgentService；
- 可上传、选择知识源、聊天并查看来源与执行轨迹的 Streamlit 页面；
- pytest、Ruff、协作文档与接口测试。

BM25 可对 `ChunkCorpus` 的真实文本快照执行稀疏检索；Dense 目前只完成存储无关的 Retriever
和 Backend 接口，尚未接入真实 Embedding/Chroma。Hybrid 已能融合任意遵守公共契约的 Dense
和 Sparse 实现；Multi-Query 与 Reranker 已提供模型无关接口，但尚未接入真实 LLM Rewrite 或
Cross-Encoder/Rerank API。上下文压缩与无答案策略已有确定性实现，但阈值尚未通过真实评估集
校准。这些 Mock 仍只验证其他调用链，不代表完整生产级 Advanced RAG、真实生成或 Agentic
RAG 已完成。

## 检索与评估模块完成情况

代码位于 `rag_agent_platform.retrieval` 和
`rag_agent_platform.evaluation`。所有 Retriever 对上层保持同一个同步接口：

```python
retrieve(
    query: str,
    document_ids: list[str] | None = None,
    top_k: int = 5,
) -> list[RetrievedChunk]
```

契约约定：空查询和非法 `top_k` 抛出 `ValueError`；`document_ids=None` 表示不限制文档，
`document_ids=[]` 表示没有可检索文档；结果按归一化分数降序排列、按 `chunk_id` 去重且不超过
Top-K。当前完成内容如下：

| 阶段 | 状态 | 已完成内容 |
|---|---|---|
| 1. 公共基础 | 完成 | 公共模型恢复、`BaseRetriever` 契约、`ChunkCorpus`、统一请求校验、分数归一化、过滤、排序和去重 |
| 2. 单路检索 | 完成 | 中英文/技术词 Tokenizer、可刷新 BM25、`DenseSearchBackend`、similarity/distance Dense Retriever |
| 3. 混合检索 | 完成 | 加权 RRF、Dense + Sparse Hybrid、候选扩大、单路异常可观测降级、严格失败模式 |
| 4. 查询增强与重排 | 完成 | `QueryTransformer`、Multi-Query、查询级 RRF、`BaseReranker`、Token Overlap Reranker、重排包装器 |
| 5. 后处理 | 完成 | 句子级上下文压缩、字符预算、组合无答案阈值、Compression/Threshold Retriever 包装器 |
| 6. 离线评估 | 完成 | JSON/JSONL 数据集、Recall@K、Precision@K、Hit Rate@K、MRR、nDCG、无答案准确率、耗时、多模式 Runner 和 CLI |

完整离线组合测试已覆盖：

```text
query
→ MultiQuery
→ Dense + BM25
→ RRF
→ Reranker
→ Compression
→ Threshold
→ list[RetrievedChunk]
```

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

检索评估的数据格式、指标口径和模式对比命令见
[`docs/retrieval-evaluation.md`](docs/retrieval-evaluation.md)。

BM25 使用全量 `ChunkCorpus` 快照构建内存索引；成员一完成新增、更新或删除后，需要由集成层
调用 `BM25Retriever.refresh()`。当前没有自动事件订阅。多查询、混合召回与重排的候选倍数会
逐层相乘，真实服务接入前应由团队装配层设置统一的总候选量和调用预算。

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
