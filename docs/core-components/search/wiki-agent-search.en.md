# Wiki Agent Search

`wiki_agent_search` is a dynamic `QueryEngineProtocol` implementation. It uses `ToolCallingLanguageModelProtocol` to plan tools and records structured evidence in `QueryEvidenceLedger`.

```text
question -> tool planning -> Wiki tools -> evidence ledger -> grounded synthesis
```

Default tools:

| Tool | Purpose |
| --- | --- |
| `read_wiki_index` | Read page titles and summaries. |
| `search_wiki` | Fuse Wiki vector and full-text retrieval. |
| `read_wiki_page` | Read one generated Wiki page. |
| `read_raw_object` | Read non-citable supporting source context. |

The default registry exposes this mode only when both Wiki indexes, an embedding model, and a tool-calling language model are available.

Results retain both their direct and original sources:

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

`source.object_key` can be passed to `read_wiki_page`, while `source.origin.object_key` identifies the document originally ingested into the knowledge base for audit or source lookup. Citations retain the same source structure.

The loop bounds model steps, tool calls, individual tool text, and total evidence. Final `insights[].evidence_ids` must reference citable ledger evidence; results and citations are derived only from selected evidence.

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

Tool text can be truncated by these limits. For exhaustive questions over long pages, use `search_wiki` to retrieve multiple sections or explicitly increase `ReadWikiPageTool(max_chars=...)` and the total evidence budget. Unread page tails must not be treated as excluded evidence.
