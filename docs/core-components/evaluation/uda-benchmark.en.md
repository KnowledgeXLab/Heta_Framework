# UDA-Benchmark

`UdaBenchmark` integrates one subset of UDA-Benchmark.

UDA-Benchmark targets real document-analysis tasks across finance, tables, papers, and encyclopedia-style QA. It is useful for evaluating recipes that build KBs around concrete documents and answer questions over them.

## Data Layout

Supported subsets:

```text
fin
tat
paper_tab
paper_text
feta
nq
```

Each subset provides:

```text
source documents
    PDF / HTML / text documents

qa csv
    doc_name, q_uid, question, answer_*

extended qa info
    answers, evidence, program, context, and related labels
```

UDA questions are usually bound to `doc_name`. Heta therefore uses `run_units()` to build multiple KBs by document:

```text
ADI_2009.pdf
    -> one KB
    -> questions for ADI_2009

GS_2016.pdf
    -> one KB
    -> questions for GS_2016
```

This avoids mixing unrelated annual reports, tables, fields, and years into one large KB.

## Usage

By default, `UdaBenchmark` downloads QA metadata and source documents:

```python
from heta_framework.evaluation import BenchmarkRunner, UdaBenchmark

benchmark = UdaBenchmark(subset="fin")

result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="uda_fin_vector_v1",
    query_modes=("vector_search",),
)
```

If download fails, preparation fails immediately so network, permission, or data-source problems are visible.

Use local data:

```python
benchmark = UdaBenchmark(
    subset="fin",
    source_root="/data/UDA/dataset/src_doc_files",
    qa_path="/data/UDA/dataset/qa/fin_qa.csv",
    extended_info_path="/data/UDA/dataset/extended_qa_info_bench/bench_fin_qa.json",
)
```

## Source Documents

Without `source_root`, documents are downloaded under:

```text
BenchmarkWorkspace.cache_dir / "uda_{subset}" / "source_documents"
```

Source zips are from the Hugging Face dataset `qinchuanhui/UDA-QA`:

```text
src_doc_files/fin_docs.zip
src_doc_files/tat_docs.zip
src_doc_files/paper_docs.zip
src_doc_files/wiki_feta_docs.zip
src_doc_files/wiki_nq_docs.zip
```

Subset source directories:

| subset | source dirs |
| --- | --- |
| `fin` | `fin_docs` |
| `tat` | `tat_docs` |
| `paper_tab` | `paper_docs` |
| `paper_text` | `paper_docs` |
| `feta` | `wiki_feta_docs` |
| `nq` | `wiki_nq_docs` |

Runner writes documents into the recipe ObjectStore:

```text
raw/benchmarks/uda_{subset}/all/{document_id}/{name}
```

## Case Mapping

Each QA CSV row becomes a `BenchmarkCase`:

```text
case_id = q_uid
query = question
expected.answers = extended answers + answer_* columns
expected.value = extended answers.exe_answer
labels.subset = subset name
```

When extended info exists, evidence uses `source_key_prefix` so query results can be matched to the correct original document without depending on a specific file extension.

## Run Units

`UdaBenchmark.run_units()` groups cases by `doc_name`:

```python
BenchmarkRunUnit(
    unit_id="ADI_2009",
    document_ids=("ADI_2009",),
    case_ids=("ADI/2009/page_49.pdf-1",),
    metadata={"doc_name": "ADI_2009", "subset": "fin"},
)
```

Runner builds:

```text
{knowledge_base_name}-ADI_2009
```

and still returns one aggregated report.

## Default Evaluators

Default:

```text
AnswerContains()
```

When extended evidence is available, add evidence recall:

```python
from heta_framework.evaluation import EvidenceRecallAtK

evaluators=(
    EvidenceRecallAtK(k=5),
    *benchmark.evaluators(),
)
```

## Sources

```text
GitHub: https://github.com/qinchuanhui/UDA-Benchmark
Hugging Face: https://huggingface.co/datasets/qinchuanhui/UDA-QA
Source documents: https://huggingface.co/datasets/qinchuanhui/UDA-QA/tree/main/src_doc_files
```
