# Heta

This directory contains the standalone Python project for the next SDK-oriented
version of Heta. The package distribution name is `heta`.

The first focus is the knowledge base builder:

- `common`: reusable model, embedding, parser, and storage components
- `kb`: knowledge base APIs and pipelines

The current Models component exposes a Heta `LanguageModel` client backed by
LiteLLM for provider adaptation.
