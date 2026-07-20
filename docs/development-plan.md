# 四人开发计划与完成情况

## 当前共同基线

公共 `DocumentRecord`、`RetrievedChunk`、`Citation`、`AgentResult` 和抽象服务接口继续作为全组
唯一契约。真实服务已在 `feature/agent-ui` 完成联合装配，Mock 仅保留给测试和显式 demo 模式。

## 成员一：文档入库与存储（已完成当前阶段）

- TXT、Markdown、PDF、DOCX 解析与清洗；
- 父子切块及 document/chunk/parent/source/page 元数据；
- FileDocumentRepository 和 ChromaVectorStore；
- 文档级 Repository/Vector 删除与测试。

仍待生产化：MySQL Repository、逐页切块映射、并发写入和大文件策略。

## 成员二：Naive 与 Advanced RAG（已完成当前阶段）

- Dense Retriever、BM25、RRF、Hybrid；
- Multi-Query、Reranker、Compression、Threshold；
- 统一归一化、过滤、排序、去重和离线评估；
- 本轮由适配器命名为 Naive/Advanced，并接入父块回溯。

仍待生产化：真实语义 Embedding、LLM Query Transformer、Cross-Encoder Reranker 和阈值校准。

## 成员三：GraphRAG（已完成当前阶段）

- 规则实体关系抽取、NetworkX 图存储；
- 构建、检索、持久化和文档删除；
- GraphRetriever 统一输出与 document_ids 过滤；
- 本轮兼容成员一 `:child:` 块 ID 的文档过滤约定。

仍待生产化：LLM/NER 抽取、图索引重载元数据恢复、复杂实体消歧和大图存储。

## 成员四：Agent、生成与界面（本轮已完成）

- `LangGraphAgentService` 与完整 `AgentState`；
- 结构化优先、规则兜底的问题分析与手动模式固定路由；
- CHAT/NAIVE/ADVANCED/GRAPH 条件分支；
- Grounded Generator、Conservative Evaluator、查询重写和 `max_retries=2` 有界循环；
- 统一 ServiceContainer、真实 Streamlit 入库/删除/选择/问答/引用/轨迹；
- 显式 `APP_MODE=mock`；
- Agent 单元、异常、循环、Fake 集成、真实本地容器和 UI 导入测试；
- README、接口、架构、协作说明和环境示例收尾。

## 后续迭代

1. 选择并评估生产 Embedding/LLM/Reranker Provider；
2. 增加逐页块映射、MySQL/远程对象存储和权限模型；
3. 用真实问题集校准各模式阈值与重试策略；
4. 增加浏览器级 Streamlit E2E、负载与故障注入测试；
5. 在 `develop` 集成验证后再准备稳定演示版本。
