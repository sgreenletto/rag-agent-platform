# Changelog

本文件采用 [Keep a Changelog](https://keepachangelog.com/) 分组。版本日期来自对应 Git commit；
`1.0.0` 是本次最终交付版本，tag 将在代码合入 main 并通过远程门禁后创建。

## [Unreleased]

当前无未发布功能。

## [1.0.0] - 2026-07-22

### Added

- TXT、Markdown、PDF、DOCX 加载、清洗、父子分块和元数据处理；
- File/MySQL 文档与 chunk 持久化、Chroma Dense 与 BM25 Sparse 检索；
- Naive RAG、Advanced RAG、GraphRAG 与 Agent 自动意图识别/路由；
- 真实 LangGraph Workflow、Branch、Loop；
- PASS、REGENERATE、REWRITE_RETRIEVE、CLARIFY、REFUSE 五类评估决策；
- Query Rewrite 语义守恒、Grounded deterministic fallback、LLM transport retry；
- Citation 映射、execution trace、Streamlit UI 和多存储删除同步；
- 分层、依赖注入、Composition Root、CI、需求追踪、验收、部署与发布文档。

### Changed

- 从上一正式工程里程碑 `v0.7.0` 收口为课程最终交付版本 `v1.0.0`；
- 完善 Python package 依赖方向和 AST 架构门禁；
- 提取中立文本 tokenizer 与稳定响应常量；
- 通过 `FallbackSynthesizer` 完善 Generator、Evaluator、Fallback 的接口注入；
- 按真实 Git 历史统一版本、需求、架构、测试、部署和演示说明。

### Fixed

- 修复 Query Rewrite 引入新品牌、公司、资质和额外子问题的语义漂移；
- 修复生成器把信息技术/维护服务推断为数据库软件销售能力；
- 修复所有评估失败都进入重写重检索的问题；
- 修复整篇父块被直接作为 fallback 输出和异常语义改写返回用户；
- 修复连接重置、超时、429、临时 5xx 无有限重试；
- 修复 Graph 生成失败后不能形成直接关系答案；
- 修复 citation 映射、重复上下文和单一候选归一化置信度问题；
- 修复 MySQL 多连接、持久化和 Repository/Chroma/BM25/Graph 删除同步问题。

### Testing

- 全量自动测试、架构边界和工程合规测试；
- 真实 MySQL Repository 合同、ApplicationServices 集成、Streamlit AppTest；
- Query Rewrite、Agent decision、transport retry、Grounded Fallback 和人工问题回归；
- pytest-cov 语句覆盖率结果记录在最终验收报告。

## [0.7.0] - 2026-07-21

- Git tag `v0.7.0` 指向 `c3cbdcb`；工程规范、CI、分层/依赖注入和真实服务闭环基线合入 develop。

## [0.6.0] - 2026-07-21

- Git tag `v0.6.0` 指向 `9f27f77`；OpenAI-compatible Embedding 与 MySQL Repository 合入。

## [0.5.0] - 2026-07-21

- Git tag `v0.5.0` 指向 `018ee5d`；LangGraph Agent 与 Streamlit 应用集成里程碑。

## [0.4.0] - 2026-07-20

- Git tag `v0.4.0` 指向 `6227dc2`；GraphRAG 模块完成。

## [0.3.0] - 2026-07-21

- Git tag `v0.3.0` 指向 `26e6f1c`；Advanced Retrieval 配置和向量查询校验完成。

## [0.2.0] - 2026-07-20

- Git tag `v0.2.0` 指向 `8fbb930`；文档入库、存储和 Naive RAG MVP 合入。

## [0.1.0] - 2026-07-19

- Git tag `v0.1.0` 指向 `218f01e`；初始化 src-layout、公共接口、基础配置和应用骨架。

## Development milestones

成员步骤 tag（如 `member1-step-*`、`member2-step-*`）记录 Loader、Cleaner、Chunker、Chroma、
Advanced Retrieval、真实 Embedding 和 MySQL 等开发过程；它们不是额外的正式产品版本。
