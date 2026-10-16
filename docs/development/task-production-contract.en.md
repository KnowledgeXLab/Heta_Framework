# Extending Task Production

Task producers integrate with Heta through `TaskProducerProtocol`. See [Task Production Protocols](../core-components/generation/task-protocols.md) for the interface and data types.

## Configuration and Inputs

Define model selection, concurrency, timeouts and production policy in a separate configuration class. Validate configuration at construction and expose it through a read-only `config` property. Use `TaskRequest` for each call's inputs and output location.

Return `TaskRequirements` from `requirements` to declare input names, delivery types and required flags. Use `ObjectInput` for files and strings for text. Producers with multiple input modes return requirements for the configured mode.

## Execution

`produce(request, *, object_store)` performs these steps:

1. Call `requirements.validate(request)` to validate input names, required values and delivery types.
2. Read selected objects and validate file formats and completeness.
3. Prepare a workspace and execute production. The implementation owns model calls, configuration and internal revisions.
4. Run format and runtime checks, retaining acceptance and rejection reasons.
5. Upload deliverable tasks, evidence and reports, then publish the manifest.
6. Return `TaskResult` with manifest and report object keys.

The protocol is structural and does not wrap methods automatically. Implementations own input validation, workspace cleanup and publication.

## Files and Publication

- Use relative POSIX object keys. The supplied Object Store determines the storage backend.
- Use a fresh output prefix for every call and reject overlap with existing outputs or selected inputs.
- Record relative paths, object keys, hashes and file permissions so consumers can reconstruct task packages.
- Keep task files separate from execution evidence. Reports record stage outcomes, actual checks and failure reasons.
- Returned manifest and report keys must be readable. Propagate storage errors that prevent publication.

## Validation and Status

Record format checks, reference-solution runs and negative checks separately. Do not mark unexecuted checks as passed. Exclude rejected candidates from deliverable tasks and retain diagnostics.

Use `succeeded`, `partial` or `failed` for `status`. The manifest determines the delivered count. Normal screening can yield an empty manifest. Each producer documents its specific status rules and validation requirements.

Validation results describe checks that ran; they do not replace scientific review or task-difficulty assessment.
