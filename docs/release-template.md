# Release Notes 模板

> 仅在目标提交完成全部验收后填写并发布；不要预先声称 CI、Tag 或 Release 已完成。

## 版本目标

- 版本：`vX.Y.Z`
- 目标用户与使用场景：
- 本次发布边界：

## 新增功能

- 功能及对应需求编号：
- 用户可观察变化：

## 修复

- 问题：
- 影响范围：
- 修复方式：

## 测试结果

```text
uv sync --frozen:
compileall:
ruff format --check:
ruff check:
pytest:
coverage（如执行）:
Streamlit 冒烟:
GitHub Actions:
```

## 启动方式

```powershell
uv sync --frozen
uv run streamlit run app.py
```

列出新增或变更的环境变量，不得粘贴真实密钥。

## 已知限制

- 尚未实现或未验证的能力：
- 外部服务和本地环境限制：
- 不兼容或风险：

## 迁移说明

- 配置变更：
- 数据迁移：
- 公共接口兼容性：
- 回滚建议：

## 发布确认

- [ ] develop CI 通过
- [ ] main 已完成稳定验证
- [ ] CHANGELOG 已更新
- [ ] package 版本与 Tag 一致
- [ ] Tag 指向本次真实发布提交
- [ ] GitHub Release 内容与本模板一致
