# 测试策略

适用版本：Agentic RAG Platform `1.0.0` 最终课程交付。

## 1. 测试目标

测试优先保护公共契约、检索结果语义、Agent 分流、持久化一致性和失败降级；测试数量不是目标，
每个关键测试应断言业务结果、调用次数、状态或持久化效果，而不是只断言“不抛异常”。

## 2. 测试分层

| 层级 | 典型范围 | 代表文件 |
|---|---|---|
| 单元测试 | cleaner、chunker、tokenizer、指标、阈值、重排 | `test_ingestion_*.py`、`test_retrieval_metrics.py` |
| 接口/合同 | 公共 schema、Repository、Retriever、DI | `test_contracts.py`、`test_document_repository.py`、`test_dependency_injection.py` |
| Retriever 组合 | Dense、BM25、Hybrid、RRF、Advanced、Graph | `test_*_retriever.py`、`test_advanced_retrieval_e2e.py` |
| Agent Workflow | route、Branch、Loop、五类决策、计数和 trace | `test_agent_service.py`、`test_agent_workflow_decisions.py` |
| 回归/降级 | rewrite 漂移、异常改写、transport retry、fallback | `test_query_rewrite.py`、`test_answer_anomaly_flow.py`、`test_llm_transport_retry.py` |
| 持久化集成 | MySQL、Chroma、Graph、删除与重启 | `test_mysql_repository.py`、`test_chroma_store.py`、`test_graph_store.py` |
| 应用集成/UI | 真实 ApplicationServices、Streamlit AppTest | `test_real_container_integration.py`、`test_streamlit_app.py` |

## 3. Fake、Stub 与真实测试边界

- 普通 CI 使用 Fake/Stub ChatModel、Hash Embedding、临时目录或显式 Mock 应用，不访问收费 API，
  也不读取开发者 `.env`。
- MySQL Repository 合同在独立 CI service 和本地真实数据库上运行；测试数据库只使用专用配置。
- `test_real_container_integration.py` 中的“real”指真实 ApplicationServices、Chroma、Graph 和本地
  Hash Embedding，不表示调用外部 LLM。
- 外部 LLM/Embedding smoke 只能在得到数据发送授权和安全配置后人工运行，其结果不能替代自动
  测试，也不作为普通 CI 的前提。

## 4. TDD 与回归驱动事实

Git 历史不能证明项目从第一天起严格测试先行，因此项目不作该声明。早期一些功能提交同时包含
实现与测试，无法从提交顺序证明 Red 在 Green 之前。

当前可确认的是回归测试驱动：Query Rewrite 漂移、REGENERATE/REWRITE_RETRIEVE 分流、LLM
transport retry、父文档全文 fallback、Graph 关系 fallback、删除同步和 MySQL 多连接问题，都有
针对事故行为的测试固定。当前工程审计中，AST 分层测试先失败并定位 generation→retrieval 的
依赖，再把 tokenizer 提升为中立共享能力。未提交工作区中的先后过程不会被伪装成已发布历史。

后续变更采用：先写最小失败测试并确认失败原因（Red）→ 最小实现（Green）→ 全量测试保护下重构
（Refactor）。若只能证明“发现问题后补回归测试”，文档和提交说明应使用
`regression-test-driven fix`，而不是声称严格 TDD。

## 5. CI 测试

`quality` job 固定 Python 3.12，执行 `uv sync --frozen`、compileall、Ruff 和全量 pytest，显式使用
Mock/File/Hash 配置。`mysql-contract` job 使用临时 MySQL 8.4 service，只运行真实 Repository 合同。
任何命令非零退出都会使 job 失败；工作流不含真实密钥。

## 6. 本地真实环境验证

```powershell
uv run python scripts/check_environment.py
uv run pytest tests/test_mysql_repository.py -v -s
uv run pytest tests/test_real_container_integration.py -v -s
uv run pytest -q -k "streamlit or app"
```

真实模型 smoke 应记录 provider 是否成功、transport retry、Agent counters、决策、引用和最终答案；
网络不可用应标记外部环境阻塞，不能改写成“通过”。

## 7. 测试数据与隔离

固定制度文本位于 `tests/fixtures/`；单元和集成测试优先使用 `tmp_path`，不写生产数据目录。MySQL
测试在专用数据库中创建自己的表并清理测试记录。测试不得依赖执行顺序、共享 Session State 或
开发者上传文档。条件 skip 只允许用于明确缺失的外部 MySQL 环境，不使用无理由 xfail。

## 8. 覆盖率与当前限制

项目已安装 `pytest-cov`，覆盖率必须以实际命令输出为准：

```powershell
uv run pytest --cov=src/rag_agent_platform --cov-report=term-missing
```

2026-07-22 `1.0.0` 最终复验为 379 passed、语句覆盖率 88%（3798 statements，451 missed）。
目前没有把观察值设置为强制百分比门槛。分支覆盖、浏览器级 E2E、负载、并发、生产故障注入和
外部模型稳定性仍不是普通 CI 的完整覆盖范围。
