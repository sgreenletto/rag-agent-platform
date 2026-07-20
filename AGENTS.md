# Codex 仓库工作约定

## 项目目标

本项目建设模块化智能文档问答系统，当前已接通真实文档入库、Naive、Advanced、GraphRAG、
Agentic RAG 与 Streamlit。后续变更必须保持公共契约与真实闭环，Mock 只用于测试或显式模式。

## 目录职责

- `app.py`：唯一 Streamlit 入口，只调用 UI 应用；
- `src/rag_agent_platform/models/`：全组公共数据契约；
- `ingestion/`、`storage/`：入库与持久化边界；
- `retrieval/`、`graph/`：统一检索和图服务边界；
- `generation/`、`evaluation/`：回答生成和评估边界；
- `agent/`：AgentService、AgentState 与编排实现；
- `ui/`：Streamlit 组件、Session State 与页面组装；
- `bootstrap.py`：唯一服务装配位置，集中创建存储、Retriever、Graph、模型和 Agent；
- `tests/`：单元测试和契约测试；`docs/`：架构、接口和开发计划。

## 强制约束

- 公共模型和抽象接口不可随意修改；必要变更必须说明影响并同步文档和测试；
- 项目采用标准 src-layout，新模块必须放在 `src/rag_agent_platform/`；
- 所有项目导入从 `rag_agent_platform` 开始，不得使用 `from src...`；
- 不得使用 `sys.path` 或要求设置 `PYTHONPATH` 修复导入；
- 使用 `uv` 管理依赖，不添加当前阶段未使用的重型依赖；
- 不得提交真实密钥、密码、上传文档、日志、数据库或模型权重；
- 正式页面默认使用 `APP_MODE=real`，不得在真实服务失败时静默降级为 Mock；
- Streamlit 长生命周期资源使用 `st.cache_resource`，不得在组件内重复创建 Chroma 或模型；
- 不得自行执行 `git commit` 或 `git push`，也不得执行破坏性 Git 命令；
- 修改前先检查目录、相关文件和 `git status`，保留无关的用户变更；
- 完成后报告修改文件、执行命令、测试结果和仍未实现的真实能力。

## 常用命令

```powershell
uv sync
uv run streamlit run app.py
uv run python -m compileall src tests app.py
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```
