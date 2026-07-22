# 开发步骤与最终状态

## 开发方法

```text
需求分析 → 边界确认 → 分层设计 → 接口抽象 → 依赖注入
→ TDD/回归测试 → CI → MVP 演进 → develop/main 发布 → v1.0.0
```

每项需求先定义输入、输出、失败语义和可执行验收；新增行为先以失败测试或回归测试固定，再完成
最小实现和重构。高层依赖抽象，具体 provider 只在 Composition Root 选择。

## 最终已完成能力

- 文档：TXT/Markdown/PDF/DOCX、清洗、父子分块；
- 存储：File/MySQL Repository、Chroma、BM25 corpus、NetworkX Graph；
- 检索：Naive、Advanced、Graph 及统一分数、过滤、去重、引用；
- Agent：自动路由、Workflow、Branch、Loop、五类 evaluation decision、Query Rewrite；
- 稳定性：transport retry、Grounded Fallback、异常回答拦截、无答案拒答；
- 数据一致性：Repository/Chroma/BM25/Graph 删除同步和重启恢复；
- 应用：Composition Root、依赖注入、Streamlit、CI、需求追踪、部署和验收文档。

实现与测试的逐项映射见 [traceability.md](traceability.md)。上述“完成”限于 requirements 中定义的
课程范围，不等于生产级高可用、多租户或商业知识图谱平台。

## v1.0.0 交付步骤

当前工作区是 `1.0.0` release candidate。本次本地门禁通过后，由维护者审阅并提交当前 feature，
依次通过 PR 合入 develop、main；远程 GitHub Actions 在 PR 阶段验证。最后在 main 已提交的稳定
commit 上创建不可移动的 `v1.0.0` tag 和 Release Notes。本轮不执行这些 Git 写操作。

## 交付后的可选增强

真实评估集阈值校准、页级/表格定位、浏览器级 E2E、并发性能、认证授权、生产监控和备份灾备是
后续可选增强，不是本版本未完成的课程需求。
