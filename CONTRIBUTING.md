# 协作规范

## 分支策略

- `main` 只保存稳定、可演示版本，不允许直接在 `main` 开发或提交功能；
- `develop` 是日常集成分支；
- 所有 `feature/*` 必须从最新的 `develop` 创建；
- 功能完成后发起 Pull Request，目标分支必须是 `develop`；
- 只有通过完整检查的 `develop` 才能合并到 `main`。

## 提交格式

- `feat:` 新功能
- `fix:` 修复问题
- `refactor:` 重构
- `test:` 添加或修改测试
- `docs:` 文档
- `chore:` 配置、依赖或项目结构

## 合并前检查

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

还需手动启动：

```powershell
uv run streamlit run app.py
```

## 公共接口变更

以下文件属于全组公共契约，不允许成员自行进行破坏性修改：

- `src/rag_agent_platform/models/schemas.py`
- `src/rag_agent_platform/storage/base.py`
- `src/rag_agent_platform/ingestion/base.py`
- `src/rag_agent_platform/retrieval/base.py`
- `src/rag_agent_platform/graph/base.py`
- `src/rag_agent_platform/generation/base.py`
- `src/rag_agent_platform/evaluation/base.py`
- `src/rag_agent_platform/agent/base.py`
- `src/rag_agent_platform/agent/state.py`

确需修改时，先在 Pull Request 中说明原因、字段变化、兼容方案和受影响模块，由全组确认；
同步修改接口文档和契约测试，不允许在单个功能分支中复制或绕开公共模型。

## 包与依赖

- 使用 `uv` 管理环境和依赖；
- 项目采用标准 src-layout，代码从 `rag_agent_platform` 导入；
- 新模块放在 `src/rag_agent_platform/`，不要把 `src` 当成 Python 包；
- 新增依赖必须与当前功能直接相关，并同步更新 `pyproject.toml` 与 `uv.lock`。

## 安全与运行数据

禁止提交 `.env`、API Key、密码、上传的真实文档、Chroma 数据、日志、SQLite 数据库、模型
权重或其他运行产物。提交前检查 `git status`，确认变更范围清晰且全部测试通过。
