# 发布流程

## 1. 分支和顺序

真实发布顺序为：

```text
需求确认 → 开发 → 定向测试 → 全量测试
→ 更新正式版本元数据、CHANGELOG 与 Release Notes → commit → Pull Request
→ 合并 develop → 稳定验收 → 合并 main
→ 创建 tag → push tag → GitHub Release
```

当前仓库使用 `feature/* → develop → main`。本次 release candidate 的版本元数据已是 `1.0.0`；
远程 CI、真实 MySQL、演示流程和已知限制仍需在稳定验收中确认。本文件描述流程，不授权自动执行
任何 Git 写操作。

## 2. 版本与 Changelog

- 当前最终交付的 `pyproject.toml` 与 `rag_agent_platform.__version__` 均为 `1.0.0`。
- 合入前核对 `CHANGELOG.md`、release notes、验收日期、测试结果和已知限制。
- tag 使用 `vMAJOR.MINOR.PATCH`，应指向已经提交、在 `main` 验收通过的稳定 commit。
- 创建 tag 不会自动修改 `pyproject.toml`，所以元数据变更必须先 commit。

## 3. 验证清单

```powershell
uv sync --frozen
uv run python -m compileall src tests scripts app.py
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run pytest tests/test_mysql_repository.py -v -s
uv run pytest tests/test_real_container_integration.py -v -s
git diff --check
git status --short
```

只有实际运行的项目才可在验收报告中写“通过”；外部环境不可用要标记“未验证/外部环境阻塞”。

## 4. Tag 与历史安全

- 不移动、删除或重建已经发布的 tag；当前真实稳定 tag 以 `git tag --list` 为准。
- 不在未提交工作区或 feature 临时 commit 上创建 release tag。
- 本次 `v1.0.0` 只能在本次变更已经合入 main 的稳定 commit 上创建；现有 `v0.7.0` 保持不变。
- 如团队明确要求在 rebase 后更新个人远程 feature 分支，只能审阅差异后使用
  `git push --force-with-lease`，不能使用裸 `--force`；共享 develop/main 不应靠重写历史同步。
- push tag 后创建 Release Notes，使用 [release-template.md](release-template.md)，链接真实 commit、
  测试证据和已知限制。

## 5. 回滚

已发布 tag 保持不变。需要撤销功能时在分支上创建新的 revert/fix commit，重新走测试和发布流程；
不通过移动旧 tag 伪造发布内容。数据回滚与备份尚未产品化，部署前必须由环境负责人另行制定。
