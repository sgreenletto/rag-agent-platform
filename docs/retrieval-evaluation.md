# 检索离线评估

## 数据集格式

评估器支持 UTF-8 `.json` 数组或 `.jsonl`。每条记录示例：

```json
{
  "case_id": "leave-001",
  "question": "员工每年有多少天年假？",
  "relevant_chunk_ids": ["leave-policy"],
  "relevant_document_ids": ["hr-policy"],
  "document_ids": ["hr-policy"],
  "answerable": true,
  "category": "exact"
}
```

- `relevant_chunk_ids`：用于计算 Chunk 级检索指标；可回答问题不能为空；
- `relevant_document_ids`：相关性标注，不会自动传给 Retriever，避免泄露正确答案；
- `document_ids`：模拟用户实际选择的知识范围，只有该字段会作为检索过滤条件；
- `answerable=false`：表示知识库中没有答案，此时 `relevant_chunk_ids` 应为空；
- `category`：建议使用 `exact`、`paraphrase`、`multi-evidence`、`no-answer` 等类别。

`tests/fixtures/retrieval_questions.jsonl` 提供了最小格式示例。正式对比需要将其中 Chunk ID
替换为团队真实入库后的稳定 ID，并扩充覆盖不同问题类别的数据。

## 指标口径

对可回答问题计算：

- Recall@K：Top-K 命中的相关 Chunk 数 / 全部相关 Chunk 数；
- Precision@K：Top-K 命中的相关 Chunk 数 / K；
- Hit Rate@K：Top-K 是否至少命中一个相关 Chunk；
- MRR：第一个相关 Chunk 排名的倒数的平均值；
- nDCG@K：二元相关性下的归一化折损累计增益。

无答案准确率只在 `answerable=false` 样本上计算；Retriever 返回空列表视为判断正确。报告同时
记录每题耗时和平均耗时。重复的检索 Chunk ID 在计算指标前去重。
Top-K 会先按原始排名截断再去重，因此重复结果不会把 K 之外的相关结果抬入评估窗口。
Retriever 异常默认中止整次评估并保留堆栈，不会被记作无答案；构建索引和工厂初始化发生在
计时之外，逐题 latency 只统计 `retrieve()` 调用。

## Retriever 工厂

评估脚本不绑定 Chroma、Embedding 或 LLM。项目集成时提供一个零参数工厂：

```python
def build_retrievers() -> dict[str, BaseRetriever]:
    return {
        "naive": naive_retriever,
        "bm25": bm25_retriever,
        "hybrid": hybrid_retriever,
        "advanced": advanced_retriever,
    }
```

然后执行：

```powershell
uv run python scripts/evaluate_retrieval.py `
  --dataset data/evaluation/questions.jsonl `
  --factory your_module:build_retrievers `
  --top-k 5 `
  --output reports/retrieval-evaluation.json
```

工厂或真实数据尚未接入时不要发布模式优劣结论。阈值、Dense/Sparse 权重、RRF 常数和候选
倍数应依据同一份评估集校准，并保留一份未参与调参的测试集用于最终报告。

当前 JSON 报告包含模式、Top-K、聚合指标、逐题结果与耗时，但尚未自动写入 Git revision、
完整 Retriever 配置、数据集哈希和运行时间戳。形成答辩或可复现实验报告前，应由团队装配层
补充这些运行清单信息。
