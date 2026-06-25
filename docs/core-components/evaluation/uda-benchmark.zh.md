# UDA-Benchmark

`UdaBenchmark` 接入 UDA-Benchmark 的单个 subset。

UDA-Benchmark 面向真实文档分析场景，包含金融、表格、论文和百科类问答。Heta 以 subset 为粒度运行 UDA：

```text
fin
tat
paper_tab
paper_text
feta
nq
```

每个 subset 都提供：

```text
source documents
    原始 PDF / HTML / text 文档

qa csv
    doc_name、q_uid、question、answer_*

extended qa info
    answers、evidence、program、context 等增强标签
```

UDA 的 QA 通常绑定到 `doc_name`。
因此 Heta 不把整个 subset 强行混成一个大 KB，而是由 `run_units()` 按文档生成多 KB 执行单位：

```text
ADI_2009.pdf
    -> 建一个 KB
    -> 跑 ADI_2009 对应的 questions

GS_2016.pdf
    -> 建一个 KB
    -> 跑 GS_2016 对应的 questions
```

这可以避免金融年报、表格字段和年份之间互相串扰，也更容易断点和并发。

## Usage

默认情况下，`UdaBenchmark` 会下载 QA metadata 和当前 subset 需要的 source documents。

```python
from heta_framework.evaluation import BenchmarkRunner, UdaBenchmark

benchmark = UdaBenchmark(
    subset="fin",
)

result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="uda_fin_vector_v1",
    query_modes=("vector_search",),
)
```

如果下载失败，benchmark preparation 会直接失败。这样可以尽早暴露网络、权限或数据源问题。

也可以显式传入本地 source documents 和 metadata：

```python
benchmark = UdaBenchmark(
    subset="fin",
    source_root="/data/UDA/dataset/src_doc_files",
    qa_path="/data/UDA/dataset/qa/fin_qa.csv",
    extended_info_path="/data/UDA/dataset/extended_qa_info_bench/bench_fin_qa.json",
)
```

本地路径优先级更高。只要传入 `source_root`，adapter 就不会下载 source documents。

## Source Documents

不传 `source_root` 时，source documents 会下载到：

```text
BenchmarkWorkspace.cache_dir / "uda_{subset}" / "source_documents"
```

下载来源是官方 Hugging Face dataset `qinchuanhui/UDA-QA` 中的 source zip：

```text
src_doc_files/fin_docs.zip
src_doc_files/tat_docs.zip
src_doc_files/paper_docs.zip
src_doc_files/wiki_feta_docs.zip
src_doc_files/wiki_nq_docs.zip
```

传入本地路径时，`source_root` 应该指向 UDA 的 `dataset/src_doc_files` 目录，或包含对应 subset source dirs 的等价目录。

不同 subset 默认查找的子目录：

| subset | source dirs |
| --- | --- |
| `fin` | `fin_docs` |
| `tat` | `tat_docs` |
| `paper_tab` | `paper_docs` |
| `paper_text` | `paper_docs` |
| `feta` | `wiki_feta_docs` |
| `nq` | `wiki_nq_docs` |

Adapter 会根据 QA CSV 中的 `doc_name` 查找对应原始文档。

下载完成后，Runner 会把这些 `BenchmarkDocument` 写入当前 recipe 的 `ObjectStore`：

```text
raw/benchmarks/uda_{subset}/all/{document_id}/{name}
```

也就是说，下载 cache 只是 benchmark 准备阶段的本地来源；真正参与 KB 构建的是 ObjectStore 里的 raw objects。
多 KB 模式下，每个 run unit 只会把自己需要的 document keys 作为 `source_keys` 传给 recipe。

## Document Mapping

每个唯一 `doc_name` 会变成一个 `BenchmarkDocument`：

```text
document_id
    normalize(doc_name)

name
    source file name

media_type
    根据文件扩展名推断
```

Runner 写入 ObjectStore 后的 raw key：

```text
raw/benchmarks/uda_{subset}/all/{document_id}/{name}
```

例如：

```text
raw/benchmarks/uda_fin/all/ADI_2009/ADI_2009.pdf
```

## Case Mapping

QA CSV 中每一行会变成一个 `BenchmarkCase`：

```text
case_id
    q_uid

query
    question

expected.answers
    extended answers + answer_* columns

expected.value
    extended answers.exe_answer

labels.subset
    subset name
```

如果提供了 extended info，`evidence` 会变成 `BenchmarkEvidence`：

```python
BenchmarkEvidence(
    reference_id="table_1",
    locator={
        "source_key_prefix": "raw/benchmarks/uda_fin/all/ADI_2009/",
    },
    text="...",
    metadata={
        "doc_name": "ADI_2009",
        "evidence_name": "table_1",
    },
)
```

UDA 的 evidence 通常绑定到 `doc_name` 内的表格、文本或上下文片段。`source_key_prefix` 可以避免 adapter 依赖某个具体文件扩展名，同时仍然能判断 query result 是否来自正确原始文档。

## Run Units

`UdaBenchmark.run_units()` 按 `doc_name` 聚合 cases。

例如：

```python
BenchmarkRunUnit(
    unit_id="ADI_2009",
    document_ids=("ADI_2009",),
    case_ids=(
        "ADI/2009/page_49.pdf-1",
        "ADI/2009/page_59.pdf-2",
    ),
    metadata={
        "doc_name": "ADI_2009",
        "subset": "fin",
    },
)
```

Runner 会为这个 unit 构建：

```text
{knowledge_base_name}-ADI_2009
```

最终报告仍然汇总到一次 benchmark run 中。

## Default Evaluators

默认 evaluator：

```text
AnswerContains()
```

如果提供 extended info，可以显式加入 evidence recall：

```python
from heta_framework.evaluation import EvidenceRecallAtK

result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="uda_fin_graph_v1",
    query_modes=("heta_multihop_search",),
    evaluators=(
        EvidenceRecallAtK(k=5),
        *benchmark.evaluators(),
    ),
)
```

## Sources

官方资源：

```text
GitHub: https://github.com/qinchuanhui/UDA-Benchmark
Hugging Face: https://huggingface.co/datasets/qinchuanhui/UDA-QA
Source documents: https://huggingface.co/datasets/qinchuanhui/UDA-QA/tree/main/src_doc_files
```
