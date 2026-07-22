# 3–5 分钟演示脚本

演示版本：`v1.0.0` 最终课程交付；上一正式里程碑：`v0.7.0`。

## 0:00–0:40 项目与架构

说明这是单机课程规模的模块化 RAG 平台。展示 `src/rag_agent_platform/` 分层和
`bootstrap.py` Composition Root；强调四种模式共享 `RetrievedChunk`、`Citation` 与
`AgentResult`。

## 0:40–1:20 上传与持久化

启动 Streamlit，上传 `tests/fixtures/enterprise_procurement_policy.txt`。展示文档列表与父/子块
数量，说明数据同步进入 Repository、Chroma、BM25 和 Graph；不要展示 `.env` 或密钥。

## 1:20–2:40 三种检索与 Agent

1. Naive：询问“正式员工每年有多少天带薪年假？”，展示十天与引用。
2. Advanced：询问“八万块买数据库那个流程怎么走？”，展示金额分级、信息安全审核、依次审批和
   三家报价；指出不会推断供应商资质。
3. Graph：询问“采购部门与云帆信息技术有限公司存在什么关联关系？”，展示联系/签订合同关系和
   Graph retrieval method。
4. Agent 自动模式下展开 execution trace，指出 analyze、strategy、evaluation decision、
   retry_count 与 regenerate_count。

## 2:40–3:20 拒答与降级

询问“公司是否报销员工的火星旅行费用？”，展示 REFUSE，说明无答案不牵强引用。可通过测试输出
说明 LLM 临时断连会 transport retry，再进入 grounded fallback；演示环境不要故意修改真实密钥。

## 3:20–4:00 删除同步

在 UI 删除刚上传文档，刷新文档列表。说明协调入库服务同步清理 Repository、Chroma、BM25 和
Graph；如时间有限，展示 `test_coordinated_ingestion.py` 的行为断言。

## 4:00–5:00 工程规范总结

展示 requirements 编号、traceability、AST 架构测试、CI 两个 job、测试策略和 v1.0.0 最终验收报告。
明确当前非目标：生产集群、多租户/RBAC、模型训练和高可用。只陈述现场实际通过的命令与真实
稳定 tag，不把开发版本说成已发布版本。
