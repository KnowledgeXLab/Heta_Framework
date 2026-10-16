# TB Factory source

Repository: https://github.com/wangwenwu2017/terminal-bench-science-task-factory
Commit: `50f6cd6d1ed7e63685227e5b63800d8a22b12657`

`task_factory.py` retains the upstream normalization, validation and materialization functions.
The original model client, batch generation, training export and CLI are omitted;
Data Designer owns generation in this adapter. `prompts.py` retains the original
scientific task format contract. The upstream package metadata contains no license declaration.

Local correction: verifier normalization preserves explicit reward values. A zero
exit status no longer overwrites a rejected submission's reward with 1. Missing
rewards and failing verifier scripts retain reward 0.
Dockerfile heredoc rewriting also preserves the requested Python interpreter
(`python` or `python3`) instead of unconditionally replacing it with `python`.

The adapter now compiles Dockerfiles and shared public inputs before calling the
native validator and materializer. `materialize_candidate(normalize_files=False)`
preserves compiled file bytes and model code, bypassing upstream content-rewriting
heuristics. The default retains upstream normalization for direct legacy callers.
Directory COPY sources are recognized. Raw files under `environment/inputs/` and
`tests/inputs/` are exempt from per-file canary comments so CSV/JSON stays valid;
generated code and Dockerfiles retain the canary requirement. The active candidate
prompt is owned by the adapter; vendored prompts remain provenance references.
Structured data files (CSV/TSV/JSON/JSONL), including hidden reference fixtures,
also remain valid data without comment requirements.
