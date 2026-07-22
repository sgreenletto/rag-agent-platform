# MVP 与真实版本演进

本页依据 2026-07-22 实际检查的 Git log、tag 和当前代码整理。`v0.1.0`～`v0.7.0` 均为真实
tag；成员步骤 tag 是开发里程碑。`v1.0.0` 是本次最终交付版本，tag 尚未创建，必须在本次提交
依次合入 develop、main 并通过远程门禁后指向 main 稳定提交。

| 阶段 | 版本或里程碑 | 真实成果 | Git/代码证据 | 状态 |
|---|---|---|---|---|
| 项目初始化 | `v0.1.0` | src-layout、配置、公共接口、基础应用骨架 | tag → `218f01e`，2026-07-19 | 完成 |
| 文档处理 | member1 step 2–5、`v0.2.0` | Loader、Cleaner、父子 Chunker | `755b9c7`、`1e4bffe`、`c0006d6`、`9ed4776`；`v0.2.0` → `8fbb930` | 完成 |
| 存储检索 | member1 step 6–8、member2 step 6、`v0.3.0`、`v0.6.0` | File/MySQL、Chroma、BM25、Dense/融合/重排 | `9fd7e9a`、`06983e8`、`4e9bc7e`、`26e6f1c`、`9f27f77` | 完成 |
| 多 RAG | `v0.2.0`～`v0.4.0` | Naive、Advanced、GraphRAG | `4e9bc7e`；`v0.4.0` → `6227dc2` | 完成 |
| Agent 编排 | `v0.5.0` | Workflow、Branch、Loop、四模式 Streamlit | `f537e64`；tag → `018ee5d` | 完成 |
| 工程里程碑 | `v0.7.0` | 工程规范、CI、分层/DI 和真实服务闭环基线 | tag → `c3cbdcb`，2026-07-21 | 完成 |
| 稳定化 | `v0.7.0 → v1.0.0` | 五类 evaluation decision、rewrite 守恒、fallback、transport retry、citation/trace 和人工问题回归 | 当前真实代码与 2026-07-22 全量测试 | 完成 |
| 最终交付 | `v1.0.0` | 功能、工程、测试、部署、验收和发布文档完整闭环 | 本次 release candidate；tag 待 main 稳定提交 | 最终版本 |

## 正式 tag 时间线

| Tag | 提交 | 日期 | 真实阶段 |
|---|---|---|---|
| `v0.1.0` | `218f01e` | 2026-07-19 | 模块化项目初始化 |
| `v0.2.0` | `8fbb930` | 2026-07-20 | 文档入库、存储和 Naive MVP |
| `v0.3.0` | `26e6f1c` | 2026-07-21 | Advanced Retrieval 配置和校验 |
| `v0.4.0` | `6227dc2` | 2026-07-20 | GraphRAG 完成 |
| `v0.5.0` | `018ee5d` | 2026-07-21 | Agentic RAG 与 Streamlit 集成 |
| `v0.6.0` | `9f27f77` | 2026-07-21 | 真实 Embedding 与 MySQL |
| `v0.7.0` | `c3cbdcb` | 2026-07-21 | 工程实践基线 |
| `v1.0.0` | 本次 main 稳定提交 | 2026-07-22 | 最终课程交付；本轮不创建 tag |

日期按 Git commit 记录，版本顺序与提交日期在协作开发中并不总是严格单调；本页不修改或美化
历史。

## 最终范围与非目标

课程范围功能已经完成，不再把 MySQL、GraphRAG、Agent Workflow、fallback 或工程规范列为未来
计划。生产语义模型校准、浏览器级 E2E、并发/负载、认证/RBAC、监控告警和备份灾备仍属于明确
非目标或可选增强，不影响 `v1.0.0` 课程交付。
