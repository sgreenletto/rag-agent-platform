# Changelog

本文件采用 [Keep a Changelog](https://keepachangelog.com/) 的分组方式，并遵循语义化版本。未确认
发布日期的内容不填写日期。

## [Unreleased]

### Added

- 正式需求与项目边界文档；
- MVP、持续发布和工程实践说明；
- 工程验收清单；
- GitHub Actions CI；
- Release Notes 模板；
- 分层边界、集中装配、协调级联删除、文档过滤和配置安全测试。

### Changed

- `ApplicationServices` 显式暴露 Ingestion、Repository、三类 Retriever、Generator、Evaluator
  和 Agent，并保留原 `ServiceContainer/build_service_container` 兼容名称；
- `RealIngestionPipeline` 改为必须通过构造函数注入 Repository；
- package 版本元数据与仓库现有 `v0.5.0` Tag 对齐。

### Fixed

- 移除 Ingestion 业务类内部选择具体 File Repository 的跨层穿透；
- 补齐 CI、版本、发布与工程过程记录缺口。

### Tested

- Agent Branch、Loop、手动覆盖和文档范围；
- Repository、父子块、向量、图和 BM25 的协调删除；
- Composition Root 和 UI 服务容器类型；
- 源码层间依赖约束与配置错误不泄露密钥。

### Documentation

- README、CONTRIBUTING、AGENTS 和架构文档补充需求、边界、DI、TDD、CI 与发布流程。

当前仓库存在 `v0.5.0` Tag；本轮 Unreleased 内容尚未获得新的版本号、Tag 或 Release。
