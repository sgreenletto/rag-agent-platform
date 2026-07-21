# 工程实践说明

## 1. 开发方法

```text
需求分析
→ 设定边界
→ 功能拆分
→ 接口设计
→ 测试定义
→ 最小实现
→ 重构
→ 集成
→ 发布
```

每次迭代先说明输入、输出、失败行为和验收条件，再选择最小可交付范围。算法增强不得绕开公共
契约，也不得把暂未实现的生产能力包装成 MVP 已完成。

## 2. 分层和职责

- `models`：跨模块数据契约，只依赖标准库。
- `ingestion`：文件加载、清洗、切块和入库用例；通过接口接收 Repository、Vector Store、Graph。
- `storage`：Document Repository、Chunk Corpus、Chroma 和 Dense Backend 等基础设施适配器。
- `retrieval`：统一 Retriever 接口及 Dense、BM25、RRF、重排、压缩、父块回溯组合。
- `graph`：实体关系抽取、图存储、GraphService 与标准 Retriever 适配。
- `generation`：回答生成接口和 grounded 实现。
- `evaluation`：回答评估接口、实现以及独立的检索评估工具。
- `agent`：AgentState、节点、路由和 LangGraph 编排，不持有数据库实现细节。
- `ui`：Streamlit 交互、Session State 和展示，仅消费应用服务与公共模型。
- `bootstrap.py`：唯一 Composition Root，集中选择所有具体实现并返回 `ApplicationServices`。

## 3. 允许依赖方向

```text
ui → ApplicationServices / AgentService / IngestionPipeline / models
agent → BaseRetriever / AnswerGenerator / AnswerEvaluator / models
ingestion → DocumentRepository / VectorStore / GraphService / models
retrieval → ChunkCorpus / DenseSearchBackend / models
storage / graph concrete implementations → chromadb / networkx / filesystem
models → Python 标准库
bootstrap → 所有接口和具体实现（唯一装配例外）
```

第三方库只能出现在需要它的基础设施实现或框架适配层。`bootstrap.py` 作为 Composition Root 可以
知道具体类型，上层业务对象只接收接口。

## 4. 禁止穿透规则

- UI 不直接操作 Chroma、MySQL、Embedding 或具体 Retriever；
- UI 不重复构造完整服务图；
- Agent 不直接操作数据库、NetworkX 或 Streamlit；
- Agent 节点内部不创建模型、Retriever 或数据库连接；
- models 不依赖 agent、retrieval、ingestion、storage 或 ui；
- retrieval 不读取 Streamlit Session State；
- 上层服务不选择具体基础设施实现；
- 不在多个位置重复装配模型、连接和索引；
- 不使用 `from src`、`sys.path` 或手工 `PYTHONPATH` 绕过 package 安装。

这些规则由 `tests/test_architecture_boundaries.py` 进行轻量源码扫描保护。

## 5. 依赖注入

项目先定义 `DocumentRepository`、`BaseRetriever`、`AnswerGenerator`、`AnswerEvaluator`、
`GraphService`、`IngestionPipeline` 和 `AgentService` 等抽象边界，再由具体实现满足接口。

- `RealIngestionPipeline` 通过构造函数接收 Repository 和 Vector Store；
- `CoordinatedIngestionPipeline` 接收真实 Pipeline、Chunk Repository、可刷新 Retriever 和图服务；
- `LangGraphAgentService` 接收三个 Retriever、Generator、Evaluator、Analyzer 和 Rewriter；
- `ApplicationServices` 暴露完整对象图；
- Streamlit 只在 `st.cache_resource` 包装的 `build_services()` 中调用 Composition Root；
- 测试用 Fake Retriever、Fake Evaluator、Fake Vector Store 和 Fake Graph 替换真实依赖。

## 6. Git 工作流

```text
feature/*
→ Pull Request 到 develop
→ develop 集成测试与 Streamlit 冒烟
→ Pull Request 到 main
→ main 稳定验证
→ 组长创建 tag
→ GitHub Release
```

提交采用 `feat:`、`fix:`、`refactor:`、`test:`、`docs:`、`chore:` 等 Conventional Commits
前缀。禁止直接在 main 开发或改写共享历史。

## 7. TDD

本项目早期部分模块是实现后补测试，不能声称整个历史完全采用 TDD。当前收尾及后续新增行为应
采用：

1. **Red**：先写描述缺失行为的最小测试并确认它因正确原因失败；
2. **Green**：只实现让测试通过的最小改动；
3. **Refactor**：在全量测试保护下整理命名、依赖和重复代码。

测试提交顺序应反映真实开发过程；本轮未创建提交，也不通过修改 Git 历史伪造测试先行记录。

## 8. 版本和发布

使用语义化版本 `MAJOR.MINOR.PATCH`：

- `v0.1.0`：工程骨架；
- `v0.2.0`：Naive RAG MVP；
- `v0.3.0`：Advanced RAG；
- `v0.4.0`：GraphRAG；
- `v0.5.0`：Agentic RAG；
- `v1.0.0`：最终稳定答辩版。

MAJOR 表示不兼容接口变更，MINOR 表示向后兼容功能，PATCH 表示向后兼容修复。Tag 只能指向
真实完成相应验收的提交；计划版本号不是创建 Tag 的授权。发布时同步项目元数据、CHANGELOG、
Release Notes、CI 结果和已知限制。
