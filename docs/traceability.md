# 需求追踪矩阵

交付版本：`v1.0.0`。

本表把 [requirements.md](requirements.md) 中的稳定编号映射到当前真实代码与测试。状态“已覆盖”
表示仓库中存在行为断言，不表示生产性能或外部服务已经验收。

| 需求 | 摘要 | 实现模块/关键接口 | 对应测试 | 状态 |
|---|---|---|---|---|
| FR-01 | 文档加载 | `ingestion/loaders.py`、`LoaderRegistry` | `test_ingestion_loaders.py`、`test_ingestion_pipeline.py` | 已覆盖 |
| FR-02 | 清洗与父子分块 | `ingestion/cleaner.py`、`chunker.py` | `test_ingestion_cleaner.py`、`test_ingestion_chunker.py` | 已覆盖 |
| FR-03 | File/MySQL 持久化 | `storage/base.py`、`file_repository.py`、`mysql_repository.py` | `test_document_repository.py`、`test_mysql_repository.py` | 已覆盖 |
| FR-04 | Chroma 索引 | `storage/chroma_store.py`、`chroma_backend.py` | `test_chroma_store.py`、`test_dense_retriever.py` | 已覆盖 |
| FR-05 | BM25 | `retrieval/bm25.py`、`ChunkCorpus` | `test_bm25_retriever.py` | 已覆盖 |
| FR-06 | Naive RAG | `retrieval/pipelines.py`、`parent.py` | `test_retrieval_adapters.py`、`test_agent_service.py` | 已覆盖 |
| FR-07 | Advanced RAG | `hybrid.py`、`fusion.py`、`multi_query.py`、`reranker.py`、`compression.py` | `test_advanced_retrieval_e2e.py` 及各组件测试 | 已覆盖 |
| FR-08 | GraphRAG | `graph/base.py`、`service.py`、`store.py`、`retriever.py` | `test_graph_extractor.py`、`test_graph_store.py`、`test_real_policy_agent_e2e.py` | 已覆盖 |
| FR-09 | Agent 自动路由 | `agent/router.py`、`nodes/analyze.py` | `test_agent_service.py`、`test_query_rewrite.py` | 已覆盖 |
| FR-10 | Workflow/Branch/Loop | `agent/graph.py`、`agent/state.py` | `test_agent_workflow_decisions.py`、`test_agent_service.py` | 已覆盖 |
| FR-11 | 生成与 citation | `generation/base.py`、`service.py`、`models/schemas.py` | `test_generation_service.py`、`test_retrieval_foundation.py` | 已覆盖 |
| FR-12 | 五类评估决策 | `evaluation/base.py`、`service.py` | `test_evaluation_service.py`、`test_agent_workflow_decisions.py` | 已覆盖 |
| FR-13 | Query Rewrite 守恒 | `agent/router.py`、`nodes/rewrite.py` | `test_query_rewrite.py` | 已覆盖 |
| FR-14 | REGENERATE 分流 | `agent/nodes/generate.py`、`evaluate.py` | `test_agent_workflow_decisions.py`、`test_answer_anomaly_flow.py` | 已覆盖 |
| FR-15 | CLARIFY/REFUSE | `agent/nodes/rewrite.py`、`evaluation/service.py` | `test_agent_service.py`、`test_agent_policy_e2e.py` | 已覆盖 |
| FR-16 | Grounded fallback | `generation/fallback.py`、`FallbackSynthesizer` | `test_grounded_fallback.py`、`test_answer_anomaly_flow.py` | 已覆盖 |
| FR-17 | LLM transport retry | `llm.py`、`ChatModel` | `test_llm_transport_retry.py` | 已覆盖 |
| FR-18 | 多存储删除同步 | `ingestion/coordinated.py` | `test_coordinated_ingestion.py`、`test_member1_integration_ready.py` | 已覆盖 |
| FR-19 | 重启持久化 | File/MySQL Repository、`ChromaVectorStore`、`NetworkXGraphService` | `test_document_repository.py`、`test_mysql_repository.py`、`test_chroma_store.py`、`test_graph_store.py` | 已覆盖 |
| FR-20 | Streamlit UI | `ui/app.py`、`components.py`、`session.py` | `test_streamlit_app.py`、`test_real_container_integration.py` | 已覆盖 |
| FR-21 | execution trace | `agent/service.py`、各 node、`llm.py` | `test_agent_workflow_decisions.py`、`test_llm_transport_retry.py` | 已覆盖 |
| FR-22 | 四模式统一契约 | `agent/base.py`、`retrieval/base.py`、`models` | `test_contracts.py`、`test_agent_service.py` | 已覆盖 |
| NFR-01 | Package 规范 | src-layout、`pyproject.toml` | `test_package_imports.py`、`test_architecture_boundaries.py` | 已覆盖 |
| NFR-02 | 分层边界 | 各 `base.py`、`bootstrap.py` | `test_architecture_boundaries.py` | 已覆盖 |
| NFR-03 | 依赖注入 | `ApplicationServices`、各构造函数 | `test_dependency_injection.py` | 已覆盖 |
| NFR-04 | 外部服务隔离 CI | `.github/workflows/ci.yml` | `test_engineering_compliance.py` | 已覆盖 |
| NFR-05 | 配置与密钥安全 | `config.py`、`llm.py`、`check_environment.py` | `test_llm_transport_retry.py`、`test_environment_check.py` | 已覆盖 |
| NFR-06 | 有界失败 | Agent counters、LLM retry budget | `test_agent_workflow_decisions.py`、`test_llm_transport_retry.py`、`test_streamlit_app.py` | 已覆盖 |
| NFR-07 | 测试/运行数据隔离 | pytest `tmp_path`、`.gitignore` | 全量测试与最终 Git 状态检查 | 过程门禁 |
| NFR-08 | 可复现质量门禁 | `uv.lock`、CI workflow | `test_engineering_compliance.py`、本地验证命令 | 已覆盖 |
| NFR-09 | 不虚构生产 SLA | requirements、deployment、README | 文档审阅；尚无性能基准 | 已明确限制 |

若需求、接口或文件名发生变化，应在同一 Pull Request 中更新本矩阵；不能仅把计划条目标记为
“已覆盖”。
