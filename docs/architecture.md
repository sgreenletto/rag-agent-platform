# 项目架构

项目采用标准 src-layout，正式包为 `rag_agent_platform`。根 `app.py` 只调用
`rag_agent_platform.ui.app.run_app()`，页面组件与服务初始化留在包内。

## 问答链路

```text
Streamlit UI
    |
    v
AgentService
    |
    +--> Query Analyzer
    +--> Retriever Router
    |       |
    |       +--> Naive Retriever
    |       +--> Advanced Retriever
    |       +--> Graph Retriever
    |
    +--> AnswerGenerator
    +--> AnswerEvaluator
    +--> Rewrite / Retry Loop
    +--> AgentResult -> answer + citations + execution_trace
```

Retriever Router 根据手动模式或问题分类选择策略。所有 Retriever（包括 Graph Retriever）都
必须返回按 `normalized_score` 降序排列的 `list[RetrievedChunk]`，从而让 Generator、Evaluator
和 UI 不依赖具体存储或检索库。

未来的 Rewrite Loop 在评估不通过且未达到 `max_retries` 时，使用 `suggested_query` 更新
`AgentState.current_query` 后重新检索。本阶段只定义 State 和接口；Mock Agent 固定评估通过，
不构建真实 LangGraph 工作流。

## 文档入库链路

```text
Uploaded File
    |
    v
IngestionPipeline
    +--> Loader / Parser
    +--> Cleaner
    +--> Parent Chunk Splitter
    +--> Child Chunk Splitter
    +--> DocumentRepository
    |       +--> Document metadata
    |       +--> Parent chunks
    |       +--> Child chunks
    +--> Vector Store (future)
    +--> GraphService.build (future)
```

当前 `MockIngestionPipeline` 只读取 UTF-8 TXT/Markdown，固定长度切块后写入
`MockDocumentRepository`。PDF/DOCX、Chroma、MySQL、Embedding 与图构建均只保留边界。

## 当前 Mock 闭环

Streamlit 上传文本或使用内置 Mock 文档，用户选择知识源和模式后调用 `MockAgentService`。
自动模式通过简单规则选择 NONE、NAIVE、ADVANCED 或 GRAPH；Mock Retriever 返回规范化证据，
Mock Agent 生成明确标识的占位回答、Citation 和固定执行轨迹。该流程只验证工程集成。
