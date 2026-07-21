# 协作规范

## 分支策略

- `main` 只保存稳定、可演示版本，不允许直接在 `main` 开发或提交功能；
- `develop` 是日常集成分支；
- 所有 `feature/*` 必须从最新的 `develop` 创建；
- 功能完成后发起 Pull Request，目标分支必须是 `develop`；
- 只有通过完整检查的 `develop` 才能合并到 `main`。
- Pull Request 必须指向上述下一层分支，不允许绕过 `develop` 直接把功能分支合入 `main`；
- Git tag 只能由组长在 `main` 的稳定、已验证提交上创建，其他成员不得自行创建或移动 tag。

## TDD 推荐流程

新增或修复关键行为时采用 Red → Green → Refactor：先写能够复现缺失行为的最小测试并确认失败，
再完成让测试通过的最小实现，最后在测试保护下消除重复和改善结构。历史模块存在实现后补测的
情况，因此不得宣称整个项目历史完全采用 TDD；但新行为必须优先补充行为测试，避免只测 Mock
调用次数而不验证边界和真实适配逻辑。

## 提交格式

- `feat:` 新功能
- `fix:` 修复问题
- `refactor:` 重构
- `test:` 添加或修改测试
- `docs:` 文档
- `chore:` 配置、依赖或项目结构

## 合并前检查

```powershell
uv run python -m compileall src tests app.py
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

本地检查和 GitHub Actions CI 必须全部通过；不得用跳过、忽略退出码或伪造外部服务结果使 CI
静默成功。依赖真实外部服务的测试必须显式标记，并与默认离线测试隔离。

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

当前 `AgentResult` 在原字段之后增加了带默认值的 `retrieved_chunks` 与 `error`，属于构造兼容的
可选扩展。后续扩展同样必须保留旧调用方式。

## 包与依赖

- 使用 `uv` 管理环境和依赖；
- 项目采用标准 src-layout，代码从 `rag_agent_platform` 导入；
- 新模块放在 `src/rag_agent_platform/`，不要把 `src` 当成 Python 包；
- 新增依赖必须与当前功能直接相关，并同步更新 `pyproject.toml` 与 `uv.lock`。
- 服务构造统一放在 `rag_agent_platform.bootstrap`；UI、节点和 Retriever 内不得各自初始化 Provider。
- 上层模块优先依赖抽象接口并使用构造函数注入；禁止 UI 穿透到 Chroma/MySQL，也禁止 Agent
  节点直接操作数据库、图存储或 Streamlit 状态。

## 真实与 Mock 模式

- 默认 `APP_MODE=real`，真实异常必须暴露给 UI；
- 只有明确设置 `APP_MODE=mock` 才能构造 Mock 服务，页面必须标注；
- 单元测试优先依赖注入 Fake/Mock，不允许用硬编码答案冒充真实集成测试；
- 文档新增或删除必须通过 `CoordinatedIngestionPipeline` 同步 Chroma、BM25 和 Graph。

## 安全与运行数据

禁止提交 `.env`、API Key、密码、上传的真实文档、Chroma 数据、日志、SQLite 数据库、模型
权重或其他运行产物。提交前检查 `git status`，确认变更范围清晰且全部测试通过。
