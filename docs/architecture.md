# 项目架构

项目采用标准 src-layout，正式包为 `rag_agent_platform`。根 `app.py` 只调用 UI，服务构造集中在
`bootstrap.py`，Streamlit 用 `st.cache_resource` 避免重复加载 Chroma、Embedding、图和 Agent。

## 分层与允许依赖方向

```text
app.py
  → ui
      → ApplicationServices / ingestion 与 agent 接口
          → agent
              → BaseRetriever / AnswerGenerator / AnswerEvaluator
          → ingestion
              → DocumentRepository / VectorStore / GraphService
          → retrieval / graph
              → storage 与向量/图抽象
                  → Chroma / NetworkX / 文件 Repository 等基础设施实现

models → 标准库或稳定基础依赖
```

依赖箭头只能由上层指向接口或下一层能力。`models` 不反向依赖业务包；`retrieval` 不读取
Streamlit Session State；`ingestion` 不依赖 UI 或 Agent。`ApplicationServices` 是应用级依赖集合，
`build_application_services()` 是唯一生产装配入口；旧 `ServiceContainer` 与
`build_service_container()` 保留为向后兼容别名。

## 禁止穿透

- UI 不直接创建或操作 Chroma、MySQL、Embedding、BM25 或具体 Retriever；
- Agent 不直接访问数据库、NetworkX、Chroma 或 Streamlit；
- Agent 节点内部不创建 ChatModel、Retriever、数据库或持久连接；
- 上层业务模块不绕过接口依赖具体基础设施；
- 不在多个模块重复装配同一模型、连接或服务图。

上述规则由 `tests/test_architecture_boundaries.py` 的轻量源码扫描保护。具体实现只允许在
`bootstrap.py` 组合，或者留在它所属的基础设施模块内部。

## 运行链路

```text
Streamlit UI
    ↓
ApplicationServices
    ├─ CoordinatedIngestionPipeline
    │   ├─ RealIngestionPipeline → Loader/Cleaner/ParentChildChunker
    │   ├─ FileDocumentRepository
    │   ├─ ChromaVectorStore
    │   ├─ BM25.refresh()
    │   └─ NetworkXGraphService.build/delete_document()
    └─ LangGraphAgentService
        ↓
      analyze_query
        ├─ CHAT/NONE ───────────────→ direct_generate → END
        ├─ SIMPLE/NAIVE ────────────→ naive_retrieve
        ├─ COMPLEX/ADVANCED ────────→ advanced_retrieve
        └─ RELATION/GRAPH ──────────→ graph_retrieve
                                         ↓
                                    generate_answer
                                         ↓
                                    evaluate_answer
                                    ├─ passed → END
                                    ├─ fail + budget → rewrite_query → analyze_query
                                    └─ fail + limit → insufficient_answer → END
```

这同时体现：

- Workflow：`analyze → retrieve → generate → evaluate`；
- Branch：CHAT、NAIVE、ADVANCED、GRAPH 条件边；
- Loop：评估失败且 `retry_count < max_retries` 时重写后回到分析节点。

## 检索装配

- `NaiveRetriever` 是轻量命名适配器，实际委托成员二 `DenseRetriever`，再由
  `ParentContextRetriever` 根据 `parent_id` 回溯成员一父块；
- `AdvancedRetriever` 委托现有 Dense/BM25/Hybrid/RRF/MultiQuery/Reranker/Compression
  组合，不复制检索算法；
- `GraphRetriever` 委托成员三 `NetworkXGraphService`，图证据转换为相同的
  `RetrievedChunk`；
- 所有结果经公共验证层过滤、去重、截断并按 `normalized_score` 降序排列，原始 Dense、BM25、
  RRF 与图分数保留在 metadata。

## 生成、评估与模型

`llm.py` 是唯一 ChatModel 初始化位置，当前支持 OpenAI-compatible chat completions。结构化问题
分类和评估会解析 JSON；模型异常时分类记录轨迹后回落到规则，评估则保守失败。未配置 LLM 时
使用真实上下文的抽取式 `GroundedAnswerGenerator` 和引用校验，不构造虚假答案。

每个上下文编号包含 source、page、retrieval_method 和 content；最终 `Citation` 从同一份
`RetrievedChunk` 顺序构建。生成始终回答 `original_query`，重写只影响检索的 `current_query`。

## 数据与生命周期

文档入库同时写 JSON Repository 和 Chroma；协调适配器随后刷新 BM25 并增量构建图。删除操作
调用真实 pipeline、图服务和 BM25 刷新。运行数据全部位于被 `.gitignore` 排除的 `data/` 子目录。

`APP_MODE=real` 是默认值；`APP_MODE=mock` 只用于显式离线开发，不参与真实模式故障降级。
