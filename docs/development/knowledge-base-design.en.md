# Knowledge Base Core Design

This page is a development note for the top-level knowledge base layer in Heta Framework.
It records the boundaries between `KnowledgeRecipe`, `KnowledgeBaseBuilder`, `KnowledgeBase`, run records, and manifests.

## Principle

```text
Recipe describes.
Builder builds.
Step executes.
Record remembers.
KnowledgeBase owns.
Manifest persists.
```

Another boundary:

```text
Recipe validates static logic.
Builder validates runtime reality.
```

## Modules

```text
kb/
  components.py
  validation.py
  state.py
  manifests.py
  cleanup.py
  recipe.py
  builder.py
  knowledge_base.py
```

## Components

`KnowledgeModels`, `KnowledgeStores`, and `KnowledgeParsers` hold runtime components and resolve them by `ComponentRef`.

They do not check connections, and they do not serialize runtime objects.

Component keys:

```text
models.language
models.embedding
models.language.strong
stores.objects
stores.vector
stores.sql
parsers.documents
```

## Validation

`KnowledgeRecipe.validate()` performs static logic validation:

```text
procedures can expand
component refs exist in the recipe
artifact requirements are provided by initial artifacts or previous steps
query requirements are provided by initial queries or previous steps
duplicate artifact outputs produce warnings
```

It does not check:

```text
whether ObjectStore keys exist
whether databases are reachable
whether model APIs are available
whether LLM output is valid
```

Validation is ordered dataflow validation. A recipe does not topologically sort or reorder steps.

## State

`StepRunRecord` records one step:

```text
index
step_name
step_type
status
started_at / finished_at
requirements
capabilities
input_artifacts
output_artifacts
issues
error
```

`RecipeRunRecord` records one full build:

```text
run_id
status
started_at / finished_at
step_records
artifacts
capabilities
issues
```

It is the basis for resume, manifests, and build reports.

## Manifest

`StepManifest`, `KnowledgeRecipeManifest`, and `KnowledgeBaseManifest` are persistent metadata.

Manifests support auditing, display, KB metadata recovery, and checkpoint foundations.
They do not serialize runtime models, stores, or parsers, and they do not try to restore client connections automatically.

## Runtime State

The manifest is final metadata. It is not responsible for hard-interruption recovery during a build.
Hard-interruption recovery is handled by `RecipeRunState`.

When a recipe has `stores.objects`, `KnowledgeBase.create()` uses a reserved prefix in ObjectStore:

```text
_heta/
  knowledge_bases/
    {knowledge_base_name}/
      manifest.json
      latest_run.json
      runs/
        {run_id}/
          state.json
          record.json
```

Responsibilities:

```text
latest_run.json
    Points to the latest run.

state.json
    Mutable build state.
    step started / succeeded / failed are all written here.

record.json
    Immutable record after a run finishes.

manifest.json
    KB metadata, recipe manifest, and final run record.
```

`RecipeRunState` records:

```text
run_id
status
started_at / finished_at
current_step
step_records
artifacts
issues
```

`RecipeRunRecord` remains the final immutable snapshot.
State is for recovery. Record is for reports, manifests, and query capability.

## Cleanup

`cleanup.py` defines the deletion protocol for the KB lifecycle.

Core types:

```text
CleanupTarget
    One persistent resource that can be deleted.

StepCleanupPlan
    Resources created by one step.

KnowledgeBaseDeletePlan
    The full deletion plan aggregated by KnowledgeBase.

KnowledgeBaseDeleteResult
    Deletion result and non-fatal issues.
```

The first version supports four target types:

```text
object_key
    One derived object in ObjectStore.

runtime_prefix
    KnowledgeBase runtime metadata prefix.

sql_table
    Table created by a step in SQLStore.

vector_collection
    Collection created by a step in VectorStore.
```

Boundary:

```text
Step declares cleanup targets.
KnowledgeBase executes cleanup.
ObjectStore raw/ input is outside cleanup scope.
```

This requires every new step to declare what it creates and how it can be cleaned. Deletion logic remains centralized in `KnowledgeBase.delete()` instead of being scattered across steps.

## KnowledgeRecipe

`KnowledgeRecipe` is the static build plan.

Fields:

```text
models
stores
parsers
steps
metadata
```

Methods:

```text
expanded_steps()
get_component(ref)
has_component(ref)
validate(...)
require_valid(...)
manifest()
```

It does not execute, record progress, or access external resources.

## KnowledgeBaseBuilder

`KnowledgeBaseBuilder` builds a knowledge base from a recipe.

Responsibilities:

```text
call recipe.validate()
create StepExecutionContext
run steps in order
diff artifacts before and after each step
record StepRunRecord
collect issues
produce RecipeRunResult
```

It supports:

```text
previous_record
skip_succeeded_steps
```

Resume logic:

```text
previous_record.artifacts are used as initial artifacts
skip succeeded steps when skip_succeeded_steps=True
continue from failed or pending steps
```

## KnowledgeBase

`KnowledgeBase` is the user entry point and build result object.

Fields:

```text
name
description
recipe
run_record
created_at
updated_at
metadata
```

Methods:

```text
create(...)
restore(...)
resume(...)
manifest()
```

`create()` calls `KnowledgeBaseBuilder.build()`.
`restore()` restores KB metadata from a manifest plus a runtime recipe.
`resume()` continues from a previous record and returns a new immutable `KnowledgeBase`.
`delete_plan()` aggregates cleanup plans from steps.
`delete()` deletes derived KB artifacts, persistent indexes, and runtime metadata. It does not delete raw input.

## Create Resume Semantics

`KnowledgeBase.create()` is the single entry point for normal users.

When runtime metadata for the same KB name already exists in ObjectStore:

```text
latest run succeeded
    create() refuses to rebuild, avoiding accidental overwrite.

latest run failed / running
    create() loads state.json, skips succeeded steps, and continues unfinished work.

no latest run
    create() creates a new run_id and state.json.
```

This avoids an extra `resume_existing()` API.
Failed recovery for the same KB name still uses the same `create()` call.

Step-level repeated calls are not handled only by the builder.
Expensive long-running steps should make ObjectStore artifacts naturally idempotent:

```text
ParseDocuments
    reuse parsed/{document_id}.json if it exists.

EmbedChunks
    reuse embeddings/{chunk_id}.json if it exists.

ExtractEntities
    reuse entities/{chunk_id}/*.json if they exist.

ExtractRelations
    reuse relations/{chunk_id}/*.json if they exist.
```

The builder handles run/step-level recovery. Steps handle item/artifact-level reuse.
