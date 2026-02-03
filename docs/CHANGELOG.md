# Changelog

All notable changes to the `digital.vasic.embeddings` module are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] -- Initial Release

### Added

- Core `provider.EmbeddingProvider` interface with `Embed`, `EmbedBatch`, `Dimensions`, and `Name` methods.
- Shared types: `provider.Config`, `provider.Result`, `provider.TokenUsage`.
- `provider.DefaultConfig()` returning sensible defaults (BatchSize=100, MaxRetries=3, Timeout=30s).
- **OpenAI provider** (`pkg/openai`):
  - Support for `text-embedding-3-small` (1536d), `text-embedding-3-large` (3072d), `text-embedding-ada-002` (1536d).
  - Configurable base URL for Azure OpenAI and proxy compatibility.
- **Cohere provider** (`pkg/cohere`):
  - Support for `embed-english-v3.0` (1024d), `embed-multilingual-v3.0` (1024d), light variants (384d), v2 models.
  - Input type selection: `search_document`, `search_query`, `classification`, `clustering`.
  - Handles both `embeddings` array and `embeddings_by_type.float` response formats.
- **Voyage AI provider** (`pkg/voyage`):
  - Support for `voyage-3` (1024d), `voyage-3-lite` (512d), `voyage-code-3` (1024d), `voyage-finance-2`, `voyage-law-2`, `voyage-large-2` (1536d), `voyage-2`.
  - Input type selection: `document`, `query`.
  - Automatic truncation enabled.
- **Jina AI provider** (`pkg/jina`):
  - Support for `jina-embeddings-v3` (1024d), `jina-embeddings-v2-base-en` (768d), `jina-embeddings-v2-small-en` (512d), multilingual variants, `jina-clip-v1` (768d), `jina-colbert-v2` (128d).
  - Task type selection: `retrieval.document`, `retrieval.query`.
  - Float encoding format.
- **Google Vertex AI provider** (`pkg/google`):
  - Support for `text-embedding-005` (768d), `textembedding-gecko@003`, `text-multilingual-embedding-002`, `text-embedding-004`, gecko multilingual.
  - Automatic endpoint URL construction from project ID, location, and model.
  - Task type set to `RETRIEVAL_DOCUMENT`.
- **AWS Bedrock provider** (`pkg/bedrock`):
  - Dual-model support: Amazon Titan (`amazon.titan-embed-text-v1` 1536d, `amazon.titan-embed-text-v2:0` 1024d, `amazon.titan-embed-image-v1` 1024d) and Cohere on Bedrock (`cohere.embed-english-v3` 1024d, `cohere.embed-multilingual-v3` 1024d).
  - Built-in AWS Signature Version 4 signing (no AWS SDK dependency).
  - Titan models: sequential single-text embedding for batch. Cohere models: native batch support.
- Compile-time interface checks for all providers.
- Full unit test suite using `httptest` mock servers and table-driven tests with `testify`.
- Documentation: README, CLAUDE.md, AGENTS.md, User Guide, Architecture, API Reference, Contributing guide, Mermaid diagrams.
