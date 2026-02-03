# Architecture

This document describes the design decisions, patterns, and structure of the `digital.vasic.embeddings` module.

## Design Goals

1. **Generic and reusable** -- No application-specific dependencies. The module can be imported by any Go project that needs text embeddings.
2. **Minimal dependencies** -- Only `testify` for testing. All HTTP communication uses the standard library `net/http`. AWS signing is implemented from scratch without the AWS SDK.
3. **Uniform interface** -- Every provider implements the same `EmbeddingProvider` interface, enabling provider-agnostic application code.
4. **Compile-time safety** -- All providers include `var _ provider.EmbeddingProvider = (*Client)(nil)` to catch interface violations at compile time.

## Design Patterns

### Strategy Pattern

The `EmbeddingProvider` interface defines a family of embedding algorithms (one per provider). Consuming code programs against the interface, and the concrete strategy is selected at initialization time:

```
EmbeddingProvider (Strategy Interface)
    |
    +-- openai.Client      (ConcreteStrategy)
    +-- cohere.Client       (ConcreteStrategy)
    +-- voyage.Client       (ConcreteStrategy)
    +-- jina.Client         (ConcreteStrategy)
    +-- google.Client       (ConcreteStrategy)
    +-- bedrock.Client      (ConcreteStrategy)
```

The consuming application holds a reference to `provider.EmbeddingProvider` and calls `Embed` / `EmbedBatch` without knowing which backend is in use.

### Factory Pattern

Each provider package exports a `NewClient(config Config) *Client` constructor function. This acts as a factory that:

- Applies default values for unset configuration fields
- Resolves the embedding dimension via `dimensionForModel`
- Creates an `http.Client` with the configured timeout

The consuming application can build a higher-level factory that maps provider names to `NewClient` calls.

### Adapter Pattern

Each provider adapts a different external API into the common `EmbeddingProvider` interface:

| Provider | External API Format | Adaptation |
|----------|-------------------|------------|
| OpenAI | `POST /v1/embeddings` with `input[]` | Direct mapping, index-based reordering |
| Cohere | `POST /v2/embed` with `texts[]` | Handles both `embeddings` and `embeddings_by_type.float` response formats |
| Voyage | `POST /v1/embeddings` with `input[]` | OpenAI-compatible format with index-based reordering |
| Jina | `POST /v1/embeddings` with `input[]` | OpenAI-compatible format with task type and encoding format |
| Google | `POST .../models/{model}:predict` with `instances[]` | Transforms flat text list into `{content, task_type}` instances, extracts `predictions[].embeddings.values` |
| Bedrock | `POST /model/{model}/invoke` | Dual-model adapter: Titan (single-text, `inputText`) and Cohere (batch, `texts[]`). Implements AWS SigV4 signing. |

The Bedrock adapter is the most notable because it adapts two fundamentally different request/response formats (Titan vs. Cohere) behind a single client, dispatching based on the model name prefix.

## Package Structure

```
digital.vasic.embeddings/
    go.mod                 -- Module definition, minimal deps
    pkg/
        provider/          -- Core interface and shared types
            provider.go    -- EmbeddingProvider, Config, Result, TokenUsage
            provider_test.go
        openai/            -- OpenAI adapter
            openai.go
            openai_test.go
        cohere/            -- Cohere adapter
            cohere.go
            cohere_test.go
        voyage/            -- Voyage AI adapter
            voyage.go
            voyage_test.go
        jina/              -- Jina AI adapter
            jina.go
            jina_test.go
        google/            -- Google Vertex AI adapter
            google.go
            google_test.go
        bedrock/           -- AWS Bedrock adapter
            bedrock.go
            bedrock_test.go
```

### Dependency Graph

Provider packages depend only on `pkg/provider`. There are no cross-dependencies between provider packages:

```
pkg/provider  <--  pkg/openai
              <--  pkg/cohere
              <--  pkg/voyage
              <--  pkg/jina
              <--  pkg/google
              <--  pkg/bedrock
```

This flat dependency structure ensures that importing one provider does not pull in code for other providers.

## Internal Structure of Each Provider

Every provider follows the same internal structure:

1. **Constants**: `DefaultBaseURL` (or `DefaultRegion`/`DefaultLocation`), `DefaultModel`
2. **Config struct**: Provider-specific configuration fields with JSON tags
3. **Client struct**: Holds `config`, `httpClient`, and `dimension`
4. **Request/response types**: Private structs mapping to the provider's JSON API
5. **NewClient constructor**: Applies defaults, resolves dimension, creates HTTP client
6. **Interface methods**: `Name()`, `Dimensions()`, `Embed()`, `EmbedBatch()`
7. **dimensionForModel function**: Maps model names to their output dimensions
8. **Compile-time check**: `var _ provider.EmbeddingProvider = (*Client)(nil)`

## HTTP Client Management

Each `Client` creates its own `http.Client` with the configured timeout. This design:

- Avoids sharing clients across providers (isolation)
- Uses Go's default connection pooling via `http.Transport`
- Allows per-provider timeout configuration

All requests use `http.NewRequestWithContext` to support context-based cancellation and timeout.

## Error Handling Strategy

All errors are wrapped with the provider name prefix:

```
<provider>: <operation description>: <underlying error>
```

Examples:
- `openai: failed to marshal request: json: unsupported type`
- `cohere: API error 429 Too Many Requests - {"message":"rate limited"}`
- `bedrock: failed to embed text 2: bedrock: request failed: context canceled`

This convention allows consuming code to identify the source of errors in multi-provider scenarios. All wrapping uses `fmt.Errorf("...: %w", err)` to preserve the error chain for `errors.Is` and `errors.As`.

## Authentication

| Provider | Mechanism | Header |
|----------|-----------|--------|
| OpenAI | Bearer token | `Authorization: Bearer <key>` |
| Cohere | Bearer token | `Authorization: Bearer <key>` |
| Voyage | Bearer token | `Authorization: Bearer <key>` |
| Jina | Bearer token | `Authorization: Bearer <key>` |
| Google | Bearer token | `Authorization: Bearer <token>` |
| Bedrock | AWS Signature V4 | `Authorization: AWS4-HMAC-SHA256 ...` + `X-Amz-Date` |

The Bedrock provider implements AWS Signature Version 4 signing from scratch using `crypto/hmac` and `crypto/sha256`, avoiding a dependency on the AWS SDK.

## Dimension Resolution

Each provider includes a `dimensionForModel` function that maps model identifiers to their output vector dimensions. This is computed once at client creation time and returned by `Dimensions()` without additional API calls. If an unrecognized model is provided, a sensible default is returned (typically the most common dimension for that provider).

## Testing Strategy

- **Unit tests** use `httptest.NewServer` to mock provider HTTP APIs
- **Table-driven tests** with `testify` cover success paths, API errors, malformed JSON, and edge cases
- **Compile-time checks** ensure all providers satisfy the interface
- **Race detection** via `go test -race` is required for all test runs
