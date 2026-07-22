# 项目架构

当前架构基线：`v1.0.0` 最终课程交付。

项目采用标准 src-layout，正式包是 `rag_agent_platform`。根 `app.py` 只进入 UI；
`bootstrap.build_application_services()` 是生产 Composition Root，Streamlit 通过
`st.cache_resource` 缓存其创建的长生命周期对象。

## 包职责

| 包/模块 | 职责 | 不负责 |
|---|---|---|
| `models` | 跨层 schema 与枚举 | 配置、I/O、业务流程 |
| `ingestion` | loader、清洗、父子分块、入库和多索引协调 | 选择具体数据库 provider |
| `storage` | File/MySQL Repository、Chroma、Corpus 基础设施 | UI 与 Agent 路由 |
| `retrieval` | Retriever 抽象及 Dense/BM25/Hybrid/RRF/父块/压缩组合 | Streamlit 状态与模型配置 |
| `graph` | 关系抽取、NetworkX 持久化、GraphService/Retriever 适配 | 企业级图编辑平台 |
| `generation` | grounded 生成、fallback 抽象和确定性实现 | 检索实现和路由 |
| `evaluation` | 回答决策与离线检索评估 | 修改 Agent state 或重新检索 |
| `agent` | state、节点、路由和 LangGraph 编排 | 创建数据库、Chroma 或模型客户端 |
| `ui` | Streamlit 交互和展示 | 直接访问数据库/索引 |
| `config.py` | 环境配置契约 | 构造服务对象 |
| `bootstrap.py` | 选择实现并装配完整对象图 | 业务规则 |
| `text.py` / `responses.py` | 无方向的共享 tokenizer/稳定响应文本 | 业务编排 |

## 允许依赖方向

```text
app.py → ui → ApplicationServices
                     ↓
          agent / ingestion application services
                     ↓
       Retriever / Generator / Evaluator / Repository / Graph abstractions
                     ↓
       storage, graph, model-provider concrete adapters

models ← all layers (stable contracts)
bootstrap → all abstractions and concrete adapters (composition exception)
```

- UI 只消费 `ApplicationServices`、公共模型和应用服务接口；
- Agent 只接收 `BaseRetriever`、`AnswerGenerator`、`AnswerEvaluator`、`FallbackSynthesizer`、
  `QueryAnalyzer` 与 `QueryRewriter`；
- ingestion 通过 `DocumentRepository`、`ChildChunkVectorStore` 和 `GraphService` 边界协调；
- provider 和路径由 `Settings` 读取，只有 bootstrap/factory 据此选择具体实现；
- `models` 不读取 `.env`；retrieval/storage 不导入 Streamlit；generation/evaluation 不横向依赖
  retrieval 实现。

`tests/test_architecture_boundaries.py` 使用标准库 AST 检查这些 import/实例化规则；合理例外是
`bootstrap.py` 需要知道具体实现，基础设施模块可在自己包内使用 chromadb、pymysql、networkx。

## 运行链路

```text
Streamlit UI
  ↓
ApplicationServices
  ├─ CoordinatedIngestionPipeline
  │    └─ loader → cleaner → parent/child chunker
  │         → File/MySQL Repository + Chroma + BM25 refresh + Graph
  └─ LangGraphAgentService
       analyze_query
       ├─ CHAT/NONE → direct_generate → END
       ├─ SIMPLE/NAIVE → naive_retrieve
       ├─ COMPLEX/ADVANCED → advanced_retrieve
       └─ RELATION/GRAPH → graph_retrieve
                              ↓
                          generate → evaluate
                          ├─ PASS → END
                          ├─ REGENERATE → same evidence → generate
                          ├─ REWRITE_RETRIEVE → rewrite → analyze
                          ├─ CLARIFY → clarification → END
                          └─ REFUSE → insufficient → END
```

回答生成抛出 transport 异常或 regeneration 到达上限时，工作流调用注入的
`FallbackSynthesizer`；具体 `GroundedFallbackSynthesizer` 只选择少量相关原文句或 Graph relation。
证据不足则拒答，不把父文档全文作为答案。

## 计数、状态和可观察性

- `original_query` 始终不变；`current_query` 只在 REWRITE_RETRIEVE 后更新；
- `retry_count` 只统计检索重试，受 `max_retries` 限制；
- `regenerate_count` 只统计同证据重新生成，受 `max_regenerations` 限制；
- transport retry 由 LLM adapter 独立计数，不污染 Agent 两类计数；
- query/strategy history 与节点决策、评分、重试、fallback 和终止原因写入 execution trace。

## 数据与生命周期

Repository provider 可选 `file` 或 `mysql`；`.env.example` 当前以 MySQL 为本地真实运行示例。
Chroma 保存 child 向量，BM25 从 Repository child corpus 构建，NetworkX 图持久化到 Graph 目录。
协调入库服务负责写入后刷新/构图以及文档级联删除。父块可作为生成上下文，但生成有字符预算，
deterministic fallback 只引用实际选中的证据句。

运行数据位于 `.gitignore` 排除的 `data/`；`APP_MODE=mock` 只能显式启用，真实服务失败不会静默
切换为 Mock。
