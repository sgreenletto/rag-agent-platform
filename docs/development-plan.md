# 四人开发计划

## 当前共同基线

全组从公共数据模型、抽象接口、Mock 闭环和契约测试开始。成员实现真实模块时必须保持
`RetrievedChunk`、`AgentResult` 等公共输出稳定，避免各分支形成不兼容的数据结构。

## 成员一：文档入库与存储

- PDF、Word 与文本解析；
- 清洗、父子切块；
- Chroma 向量存储；
- MySQL 文档与元数据存储；
- 文档删除的一致性清理与入库测试。

## 成员二：Naive 与 Advanced RAG

- Naive RAG；
- Dense Retriever；
- BM25、RRF、Reranker；
- Advanced RAG 组合检索；
- 检索指标和离线评估。

## 成员三：GraphRAG

- 实体与关系提取；
- 图数据构建与文档级删除；
- GraphRetriever；
- 图证据转换为 `RetrievedChunk`；
- GraphRAG 单元与集成测试。

## 成员四：Agent、生成与界面

- LangGraph AgentState 与工作流；
- 问题分类和 Retriever 路由；
- 回答生成、评估和查询重写循环；
- Streamlit 页面与服务集成；
- 端到端 Mock/真实模块切换测试。

## 建议迭代顺序

1. 各成员基于 `develop` 创建独立 `feature/*` 分支，并保持契约测试通过；
2. 先接入真实文档与单路 Naive 检索，验证端到端数据；
3. 增加 Advanced 与 Graph 检索，但继续输出统一公共模型；
4. 最后将规则 Mock Agent 替换为受重试上限保护的 LangGraph 编排；
5. 每次集成均执行 compileall、Ruff、pytest 和 Streamlit 冒烟测试。
