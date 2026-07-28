# Wiki Agent Search

`wiki_agent_search` 是实现 `QueryEngineProtocol` 的动态检索模式。它使用 `ToolCallingLanguageModelProtocol` 规划工具调用，并把结构化证据写入 `QueryEvidenceLedger`。

```text
question -> tool planning -> Wiki tools -> evidence ledger -> grounded synthesis
```

默认工具：

| Tool | 作用 |
| --- | --- |
| `read_wiki_index` | 读取页面目录和摘要。 |
| `search_wiki` | 融合 Wiki 向量与全文检索。 |
| `read_wiki_page` | 读取指定 Wiki page。 |
| `read_raw_object` | 读取原始对象作为不可直接引用的辅助上下文。 |

只有在 Wiki 双索引、embedding model 和 tool-calling language model 都可用时，该 mode 才会被默认 registry 暴露。

检索结果保留直接来源与原始来源：

```json
{
  "source": {
    "object_key": "wiki/pages/1-plant-cell.md",
    "object_type": "wiki",
    "origin": {
      "object_key": "raw/biology/plant-cell.pdf",
      "object_type": "pdf"
    }
  }
}
```

`source.object_key` 可交给 `read_wiki_page` 继续阅读，`source.origin.object_key` 用于审计或定位最初进入知识库的原文。citations 会保留同一来源结构。

Agent loop 对 steps、tool calls、单次工具文本和总 evidence 设置预算。最终 `insights[].evidence_ids` 必须引用 ledger 中可引用的 evidence，返回的 results 和 citations 也只来自被选择的证据。

```python
AgenticQueryEngine(
    max_steps=8,
    tool_budget=QueryToolBudget(
        max_tool_calls=8,
        max_tool_result_chars=8_000,
        max_evidence_chars=24_000,
    ),
)
```

工具返回可能被字符预算截断。需要对长页面做穷举判断时，应优先使用 `search_wiki` 定位多个 sections，或者显式提高 `ReadWikiPageTool(max_chars=...)` 与总 evidence budget；模型不能把未读取的页面尾部当成已排除证据。
