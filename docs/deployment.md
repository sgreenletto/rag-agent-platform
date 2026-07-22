# 本地部署与运行

本文对应 Agentic RAG Platform `v1.0.0` 最终课程交付版本。

## 1. 当前部署形态

系统是单进程 Streamlit 应用。`app.py` 调用 UI，`bootstrap.py` 在进程内创建 Repository、Chroma、
BM25、NetworkX、Retriever、Generator、Evaluator 和 Agent。MySQL、OpenAI-compatible LLM 与
Embedding 是可选外部端点；Chroma 和 Graph 数据默认持久化在本地目录。

## 2. 准备

需要 Python 3.12+、uv；若选择 MySQL Repository，还需要可连接的 MySQL 8.x 数据库。

```powershell
uv sync --frozen
Copy-Item .env.example .env
uv run python scripts/check_environment.py
```

不要把 `.env` 或密钥提交到 Git。`APP_MODE=real` 使用真实入库/检索；`APP_MODE=mock` 只用于显式
开发演示。`EMBEDDING_PROVIDER=hash` 无需外部端点；配置外部 provider 时必须同时设置模型和密钥。

## 3. MySQL

先创建专用数据库和账号，再按 `.env.example` 配置连接。仓库会在首次初始化时创建所需表；
`scripts/init_mysql.sql` 可用于本地数据库初始化参考。不要使用生产账号运行测试。

```powershell
mysql -u root -p < scripts/init_mysql.sql
uv run pytest tests/test_mysql_repository.py -v -s
```

也可把 `DOCUMENT_REPOSITORY_PROVIDER=file` 用于完全本地元数据文件模式。

## 4. 启动与停止

```powershell
uv run streamlit run app.py
```

浏览器访问终端显示的 Local URL。开发环境停止时在运行终端按 `Ctrl+C`。Streamlit 的
`st.cache_resource` 复用长生命周期服务，修改配置后应重启进程。

## 5. 数据目录与清理

默认数据位于 `data/metadata`、`data/chroma`、`data/graph`、`data/uploads`。优先通过 UI 删除文档，
以同步 Repository、Chroma、BM25 和 Graph。手工删除运行目录会丢失本地索引，操作前应停止应用并
确认路径；项目当前没有自动备份/恢复工具。

## 6. 外部模型

外部 Chat/Embedding 使用 OpenAI-compatible 配置。客户端对临时网络、429 和 5xx 有有限重试，
失败后 Agent 可基于已有证据确定性降级；401/403 等配置错误不会无意义重试。系统不保证第三方
端点 SLA，环境检查只验证配置完整性，不向端点发送业务数据。

## 7. 当前不支持的生产能力

没有容器编排、TLS 终止、反向代理、认证/RBAC、多租户、限流、集中日志、指标告警、分布式锁、
高可用、自动备份、灾备或滚动升级。当前文档是本地/课程演示运行说明，不是生产部署承诺。
