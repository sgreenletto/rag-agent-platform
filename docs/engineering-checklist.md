# v1.0.0 工程规范验收矩阵

本表把老师的工程要求映射到真实文件、测试与 Git/发布证据。`v1.0.0` 是当前最终 release
candidate；本轮不执行 commit、merge、push 或 tag，因此远程动作保持“PR 阶段执行”。

| 规范要求 | 实现说明 | 文件/测试证据 | 当前状态 | Git/发布证据 |
|---|---|---|---|---|
| 功能划分 | ingestion、storage、retrieval、graph、generation、evaluation、agent、ui 职责独立 | architecture、AST 边界测试 | 通过 | `v0.1.0` 起的模块化历史 |
| 需求分析 | 22 个 FR、9 个 NFR 均有验收方式 | requirements、traceability | 通过 | 本次 v1.0.0 文档 |
| 项目边界 | 课程范围、外部依赖、生产非目标明确 | requirements、deployment | 通过 | v1.0.0 release notes |
| 开发步骤 | 需求→边界→接口→测试→实现→CI→发布 | development-plan | 通过 | 真实 feature/develop/main 流程 |
| 分层设计 | UI/Application/抽象/基础设施依赖方向明确 | architecture、架构测试 | 通过 | v0.7.0 工程基线后增强 |
| Python package | Python 3.12 src-layout、绝对 package import | pyproject、package 测试 | 通过 | `v0.1.0` → `v1.0.0` |
| 控制跨层穿透 | UI 不访问数据库，Agent node 不实例化基础设施 | AST 架构测试 | 通过 | 本次提取中立共享能力 |
| 依赖注入 | 高层构造函数接收接口，bootstrap 集中装配 | interfaces、DI 测试 | 通过 | Composition Root 可审计 |
| 基类/接口+实现 | Repository、Retriever、Generator、Evaluator、Fallback 等有真实实现和 Fake | interfaces、合同测试 | 通过 | 公共接口向后兼容 |
| Git tag/commit/push | v0.1.0～v0.7.0 均为真实 tag；v1.0.0 流程已定义 | mvp-plan、release-process | 本地准备完成 | 本轮禁止 Git 写操作；PR 后执行 |
| 持续优化 | rewrite、评估分流、fallback、retry 等由人工问题驱动改进 | CHANGELOG、回归测试 | 通过 | v0.7.0→v1.0.0 稳定化 |
| 持续发布 | feature→develop→main→tag→Release Notes | CI、release-process | 配置通过 | 远程 CI/发布待 PR 阶段 |
| MVP 演进 | 骨架、入库、检索、Graph、Agent、工程、稳定化完整时间线 | mvp-plan | 通过 | 真实 commit/tag 映射 |
| TDD | 不虚构早期历史；新增修复采用 Red/Green/Refactor 或回归驱动 | testing-strategy、相关测试 | 通过 | 本轮版本测试先失败后修复 |
| CI | Python 3.12、frozen sync、compileall、Ruff、pytest、MySQL job | workflow、合规测试 | 本地配置通过 | 远程 Actions 待 PR 阶段 |
| 环境复现 | `.env.example`、MySQL、环境检查和启动命令完整 | README、deployment、环境测试 | 通过 | v1.0.0 交付文档 |

## 最终本地质量门禁

以下数值在本次版本修改完成后由实际命令更新；不得沿用历史基线：

- [x] `uv lock` 与 `uv sync --frozen`
- [x] compileall
- [x] Ruff format / lint
- [x] 全量 pytest（379 passed）
- [x] pytest-cov（88%）
- [x] 真实 MySQL Repository（6 passed）
- [x] ApplicationServices（5 passed）
- [x] Streamlit AppTest（1 passed）
- [x] `git diff --check`
- [x] `.env` 和运行数据未进入 Git 状态

## 发布动作

- [ ] 审阅并提交 feature 工作区
- [ ] Pull Request 合入 develop，远程 CI 通过
- [ ] develop 稳定验收后合入 main
- [ ] 在 main 稳定 commit 创建 `v1.0.0`
- [ ] push tag 并发布 `docs/releases/v1.0.0.md`

这些动作必须由维护者后续真实执行，不能因本地版本号已是 `1.0.0` 而提前勾选。
