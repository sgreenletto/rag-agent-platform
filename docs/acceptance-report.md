# Agentic RAG Platform v1.0.0 最终验收报告

- 验收版本：`1.0.0`
- 目标 Git tag：`v1.0.0`（本轮尚未创建）
- 上一正式里程碑：`v0.7.0`
- 验收日期：2026-07-22
- 验收分支：`feature/engineering-practice`
- 项目状态：课程最终交付 release candidate

## 1. 验收范围

覆盖文档处理、MySQL/File、Chroma、BM25、Graph、三类 RAG、Agent Workflow、五类 evaluation
decision、Query Rewrite、错误恢复、引用/trace、删除同步、持久化、Streamlit、分层/DI、CI 配置和
最终交付文档。生产级多租户、高可用、分布式部署等非目标不在通过范围。

## 2. 功能验收

| 能力 | 验收证据 | 结果 |
|---|---|---|
| TXT/Markdown/PDF/DOCX、清洗、父子分块 | ingestion 单元/管线测试 | 通过 |
| File/MySQL 文档与 chunk | Repository 合同及真实 MySQL | 通过 |
| Chroma / BM25 / Graph | 各存储和 Retriever 测试 | 通过 |
| Naive / Advanced / GraphRAG | adapter、E2E、Agent 手动模式测试 | 通过 |
| Agent 自动路由 | SIMPLE/COMPLEX/RELATION/CHAT 测试 | 通过 |
| Workflow / Branch / Loop | LangGraph 决策与终止测试 | 通过 |
| 五类 evaluation decision | evaluator/workflow tests | 通过 |
| Query Rewrite 守恒 | 数字、实体、问题数和专有名词测试 | 通过 |
| Citation / execution trace | 生成、去重、状态和 retry 断言 | 通过 |
| 无答案拒答 | 火星旅行等无依据问题回归 | 通过 |

## 3. 错误恢复与数据一致性

| 项目 | 真实行为 | 结果 |
|---|---|---|
| 回答缺陷 | 同证据 REGENERATE，不增加 retrieval retry | 通过 |
| 检索不足 | 保守 rewrite → retrieve，受 max_retries 限制 | 通过 |
| LLM 临时错误 | 连接重置、超时、429/5xx 有限 transport retry | 通过 |
| 配置错误 | 401 等不重试，安全错误不泄露 API Key | 通过 |
| 生成持续失败 | 相关原文/Graph relation Grounded Fallback | 通过 |
| 无证据 | REFUSE，不倾倒父文档 | 通过 |
| 删除同步 | Repository、Chroma、BM25、Graph 一致清理 | 通过 |
| 重启持久化 | File/MySQL、Chroma、Graph 恢复测试 | 通过 |

## 4. 应用与工程验收

- Streamlit 使用缓存的 ApplicationServices，初始化失败有页面错误而非白屏；
- `bootstrap.py` 是 Composition Root，UI 和 Agent node 不穿透到具体基础设施；
- Repository、Embedding、Retriever、Graph、Generator、Evaluator、Fallback、Agent 均有稳定边界，
  可注入 Fake/Stub；
- AST 架构测试、需求追踪、版本一致性和环境安全测试通过；
- CI 已配置外部模型零依赖 quality job 和 MySQL 8.4 合同 job。

本地质量门禁已通过；远程 GitHub Actions 将在 PR 阶段执行。本文不把未执行的远程结果写成通过。

## 5. 最终测试结果

版本收口后的实际命令结果：

| 项目 | 结果 |
|---|---|
| `uv lock` / `uv sync --frozen` | 通过；Resolved 130 packages / Checked 129 packages |
| compileall | 通过，`src tests scripts app.py` 全部编译 |
| Ruff format / lint | 140 files already formatted；All checks passed |
| 全量 pytest | 379 passed in 11.68s |
| pytest-cov | 379 passed；3798 statements，451 missed，88% |
| MySQL Repository | 6 passed in 1.57s，包含真实数据库合同 |
| ApplicationServices | 5 passed in 4.48s |
| Streamlit AppTest | 1 passed in 4.34s；无未捕获异常 |
| `git diff --check` | 通过；仅有 Windows LF/CRLF 提示，无 whitespace error |

## 6. 人工演示结果

固定制度文档的数据库采购、紧急采购、付款流程、无答案拒答和 Graph 关系均已转化为自动化人工
问题回归；`demo-script.md` 已准备 3–5 分钟现场流程。本轮版本收口不重新调用外部收费模型，也未
执行答辩现场浏览器演示，因此现场人工演示如实标记为“答辩时执行”，不伪造成已完成。

## 7. 已知限制

- 单机/课程演示规模，不承诺生产 SLA、高并发、高可用或分布式集群；
- 无企业级多租户、认证/RBAC、审计、自动备份和灾备；
- 规则 Graph 抽取、Hash Embedding 不等价于商业图平台和生产语义模型；
- 复杂 OCR、跨页表格、浏览器级 E2E、负载基准和生产监控不在课程范围。

## 8. 最终结论

代码、测试、文档和 Git whitespace/status 门禁均已通过。本代码库满足 `v1.0.0` 课程最终交付的
功能、工程、测试和文档要求。远程 CI、develop/main 合并、tag 和 GitHub Release 是本次提交后的
发布步骤。
