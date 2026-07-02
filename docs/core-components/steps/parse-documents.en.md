# Parse Documents

`ParseDocuments` is the entry step of a knowledge base build. It reads raw objects from `ObjectStore` and parses them through `DocumentParserRegistry` into unified `ParsedDocument` JSON.

```text
raw objects -> ParsedDocument JSON
```

Later steps no longer care whether the original input was a PDF, HTML page, image, spreadsheet, or text file. They only consume the parsed document protocol.

## Contract

`ParseDocuments` uses:

```text
stores.objects
parsers.documents
```

Default prefixes:

```text
raw/
parsed/
```

Execution flow:

```text
list raw objects
  -> infer file type from object key
  -> choose parser from DocumentParserRegistry
  -> write parsed/{document_id}.json
  -> expose parsed_document_keys
```

The step does not create the object store or parser registry. The recipe provides them.

## Configuration

```python
ParseDocumentsConfig(
    raw_prefix="raw",
    parsed_prefix="parsed",
    skip_unsupported=True,
    object_store=None,
    parser_registry=None,
)
```

| Parameter | Meaning |
| --- | --- |
| `raw_prefix` | Prefix where raw input objects are stored. |
| `parsed_prefix` | Prefix where parsed document JSON is written. |
| `skip_unsupported` | Skip files without a matching parser instead of failing. |
| `object_store` | Named ObjectStore. Defaults to `stores.objects`. |
| `parser_registry` | Named parser registry. Defaults to `parsers.documents`. |

## Requirements

```python
StepRequirements(
    components=frozenset({
        store_ref("objects"),
        parser_ref(),
    })
)
```

## Capabilities

```python
StepCapabilities(
    artifacts=frozenset({
        "parse_documents_result",
        "parsed_document_keys",
    })
)
```

`ParseDocuments` does not unlock a query mode. Query modes are unlocked by later indexing or graph steps.

## Artifacts

`parse_documents_result` is a `ParseDocumentsResult`:

```python
ParseDocumentsResult(
    document_keys=("parsed/doc_abc123.json",),
    skipped_keys=("raw/archive.zip",),
)
```

`parsed_document_keys` is the tuple of written parsed document keys.

## Parsed Output

Each written JSON file represents a `ParsedDocument`:

```python
ParsedDocument(
    document_id="doc_...",
    source=ParsedSource(
        key="raw/paper.pdf",
        name="paper.pdf",
        file_type="pdf",
        content_sha256="...",
    ),
    pages=[
        ParsedPage(page_index=0, text="...")
    ],
)
```

Default object key:

```text
parsed/{document_id}.json
```

`document_id` is stable for the same source content. If a parsed artifact already exists, the step can reuse it during recovery instead of re-running expensive parsers, OCR, MinerU, or VLM description work.

## Unsupported Files

The parser registry decides support:

```text
file suffix -> file_type -> registry.find_parser(file_type)
```

| Setting | Behavior |
| --- | --- |
| `skip_unsupported=True` | Skip the object and record it in `skipped_keys`. |
| `skip_unsupported=False` | Raise an error. |

This lets a recipe enable only the parsers it needs.
