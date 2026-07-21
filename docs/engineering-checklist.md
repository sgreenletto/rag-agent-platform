# 工程验收清单

该清单记录本轮代码与文档状态；远程 CI、main 合并、Tag 和 Release 必须由维护者实际执行后再勾选。

## 需求

- [x] 需求明确
- [x] 边界明确
- [x] 验收标准明确

## 架构

- [x] 分层职责明确
- [x] 无明显跨层穿透
- [x] 使用依赖注入
- [x] 公共接口稳定

## 测试

- [x] Agent 路由测试
- [x] Loop 测试
- [x] document_ids 过滤测试
- [x] 级联删除测试
- [x] 依赖注入测试
- [x] 配置测试

## 质量

- [x] compileall
- [x] Ruff format
- [x] Ruff lint
- [x] pytest
- [x] Streamlit 冒烟测试
- [x] CI 工作流已配置

## 发布

- [ ] 本轮变更完成 develop 集成提交
- [ ] main 稳定
- [ ] 本轮稳定提交已创建 tag
- [ ] GitHub Release
- [x] Release Notes 模板
