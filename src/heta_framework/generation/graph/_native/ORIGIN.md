# OmegaDesigner designer workflow

Source: https://github.com/KnowledgeXLab/OmegaDesigner
Commit: 1fadf09c6b81acd3c33554864102c1ca5220d4f4
License: Apache-2.0; LICENSE, NOTICE and original SPDX headers retained.

Includes designer config, seed contract, Data Designer columns/workflow, domain logic and byte-identical assets. Search, ingest, CLI and environment-driven POC entrypoints are not included. Python imports are relocated to heta_framework.generation.graph._native. Heta injects model providers and typed settings directly into the native workflow.

Local adaptations: Settings and all authoring payloads (candidate, spec, implementation and implementation check) expose generation_guidance separately from scientific evidence. The materialization callback optionally namespaces packages by paper identity to prevent same-slug collisions across papers; Heta enables this option. External process environments omit model credentials. Harbor uses the current Python interpreter and Heta's fixed offline verifier Docker environment, preserving Oracle/Nop reward semantics. All other domain logic and asset contents retain the pinned upstream behavior.

The static-to-runtime callback preserves all verdicts and encodes static_summary as JSON text to prevent incompatible nested-list Parquet schemas across row groups. Its on_success version invalidates incompatible boundary checkpoints.
Runtime columns also initialize their primary output before failure returns, keeping rejected rows in the Data Designer dataset instead of raising a missing-column error.
