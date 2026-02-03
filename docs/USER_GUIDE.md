# User Guide

This guide covers how to use the `digital.vasic.embeddings` module to generate text embeddings across all supported providers.

## Installation

```bash
go get digital.vasic.embeddings
```

Requires Go 1.24.0 or later.

## Core Concepts

All providers implement the `provider.EmbeddingProvider` interface:

```go
type EmbeddingProvider interface {
    Embed(ctx context.Context, text string) ([]float32, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
    Dimensions() int
    Name() string
}
```

This means you can write provider-agnostic code that works with any backend.

## Provider-Agnostic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "digital.vasic.embeddings/pkg/provider"
    "digital.vasic.embeddings/pkg/openai"
)

func generateEmbedding(ctx context.Context, p provider.EmbeddingProvider, text string) {
    embedding, err := p.Embed(ctx, text)
    if err != nil {
        log.Fatalf("embedding failed: %v", err)
    }
    fmt.Printf("Provider: %s, Dimensions: %d, Vector length: %d\n",
        p.Name(), p.Dimensions(), len(embedding))
}

func main() {
    ctx := context.Background()
    client := openai.NewClient(openai.Config{
        APIKey: "your-api-key",
    })
    generateEmbedding(ctx, client, "Hello, world!")
}
```

## OpenAI

Supported models: `text-embedding-3-small` (1536d), `text-embedding-3-large` (3072d), `text-embedding-ada-002` (1536d).

```go
import "digital.vasic.embeddings/pkg/openai"

client := openai.NewClient(openai.Config{
    APIKey: "sk-...",
    Model:  "text-embedding-3-small", // default
})

// Single embedding
ctx := context.Background()
vec, err := client.Embed(ctx, "The quick brown fox")
// vec is []float32 of length 1536

// Batch embedding
vecs, err := client.EmbedBatch(ctx, []string{
    "First document",
    "Second document",
    "Third document",
})
// vecs is [][]float32, one per input text
```

Configuration options:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `APIKey` | `string` | (required) | OpenAI API key |
| `Model` | `string` | `text-embedding-3-small` | Model identifier |
| `BaseURL` | `string` | `https://api.openai.com/v1` | API base URL |
| `Timeout` | `time.Duration` | `30s` | HTTP request timeout |

### Custom Base URL (Azure OpenAI, proxies)

```go
client := openai.NewClient(openai.Config{
    APIKey:  "your-key",
    Model:   "text-embedding-3-small",
    BaseURL: "https://your-proxy.example.com/v1",
})
```

## Cohere

Supported models: `embed-english-v3.0` (1024d), `embed-multilingual-v3.0` (1024d), `embed-english-light-v3.0` (384d), `embed-multilingual-light-v3.0` (384d), `embed-english-v2.0` (4096d), `embed-multilingual-v2.0` (768d).

```go
import "digital.vasic.embeddings/pkg/cohere"

client := cohere.NewClient(cohere.Config{
    APIKey:    "your-cohere-key",
    Model:     "embed-english-v3.0",   // default
    InputType: "search_document",       // default
})

vec, err := client.Embed(ctx, "Search this document")
```

Cohere's `InputType` field controls how embeddings are optimized:

| InputType | Use Case |
|-----------|----------|
| `search_document` | Indexing documents for search (default) |
| `search_query` | Encoding search queries |
| `classification` | Text classification tasks |
| `clustering` | Clustering tasks |

Configuration options:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `APIKey` | `string` | (required) | Cohere API key |
| `Model` | `string` | `embed-english-v3.0` | Model identifier |
| `BaseURL` | `string` | `https://api.cohere.com/v2` | API base URL |
| `InputType` | `string` | `search_document` | Input type for optimization |
| `Timeout` | `time.Duration` | `30s` | HTTP request timeout |

## Voyage AI

Supported models: `voyage-3` (1024d), `voyage-3-lite` (512d), `voyage-code-3` (1024d), `voyage-finance-2` (1024d), `voyage-law-2` (1024d), `voyage-large-2` (1536d), `voyage-large-2-instruct` (1536d), `voyage-2` (1024d).

```go
import "digital.vasic.embeddings/pkg/voyage"

client := voyage.NewClient(voyage.Config{
    APIKey:    "your-voyage-key",
    Model:     "voyage-code-3",    // optimized for code
    InputType: "document",          // default
})

// Embed code snippets
vecs, err := client.EmbedBatch(ctx, []string{
    "func main() { fmt.Println(\"hello\") }",
    "def main(): print('hello')",
})
```

Voyage's `InputType` supports two values:

| InputType | Use Case |
|-----------|----------|
| `document` | Indexing documents (default) |
| `query` | Encoding search queries |

Configuration options:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `APIKey` | `string` | (required) | Voyage AI API key |
| `Model` | `string` | `voyage-3` | Model identifier |
| `BaseURL` | `string` | `https://api.voyageai.com/v1` | API base URL |
| `InputType` | `string` | `document` | Input type |
| `Timeout` | `time.Duration` | `30s` | HTTP request timeout |

## Jina AI

Supported models: `jina-embeddings-v3` (1024d), `jina-embeddings-v2-base-en` (768d), `jina-embeddings-v2-small-en` (512d), `jina-embeddings-v2-base-de` (768d), `jina-embeddings-v2-base-es` (768d), `jina-embeddings-v2-base-zh` (768d), `jina-clip-v1` (768d), `jina-colbert-v2` (128d).

```go
import "digital.vasic.embeddings/pkg/jina"

client := jina.NewClient(jina.Config{
    APIKey: "your-jina-key",
    Model:  "jina-embeddings-v3",       // default
    Task:   "retrieval.document",        // default
})

vec, err := client.Embed(ctx, "Jina AI embeddings are versatile")
```

Jina's `Task` field controls the task-specific optimization:

| Task | Use Case |
|------|----------|
| `retrieval.document` | Indexing documents for retrieval (default) |
| `retrieval.query` | Encoding search queries |

Configuration options:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `APIKey` | `string` | (required) | Jina AI API key |
| `Model` | `string` | `jina-embeddings-v3` | Model identifier |
| `BaseURL` | `string` | `https://api.jina.ai/v1` | API base URL |
| `Task` | `string` | `retrieval.document` | Task type |
| `Timeout` | `time.Duration` | `30s` | HTTP request timeout |

## Google Vertex AI

Supported models: `text-embedding-005` (768d), `textembedding-gecko@003` (768d), `text-multilingual-embedding-002` (768d), `text-embedding-004` (768d), `textembedding-gecko-multilingual@001` (768d).

```go
import "digital.vasic.embeddings/pkg/google"

client := google.NewClient(google.Config{
    ProjectID: "my-gcp-project",
    Location:  "us-central1",            // default
    Model:     "text-embedding-005",      // default
    APIKey:    "your-access-token",
})

vec, err := client.Embed(ctx, "Google Cloud AI embeddings")
```

The Google provider constructs Vertex AI endpoint URLs automatically from `ProjectID`, `Location`, and `Model`. It sends requests to:

```
https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict
```

Configuration options:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ProjectID` | `string` | (required) | Google Cloud project ID |
| `Location` | `string` | `us-central1` | GCP region |
| `Model` | `string` | `text-embedding-005` | Model identifier |
| `APIKey` | `string` | (required) | Access token or API key |
| `BaseURL` | `string` | auto-generated | Override Vertex AI endpoint |
| `Timeout` | `time.Duration` | `30s` | HTTP request timeout |

## AWS Bedrock

Supported models: `amazon.titan-embed-text-v1` (1536d), `amazon.titan-embed-text-v2:0` (1024d), `amazon.titan-embed-image-v1` (1024d), `cohere.embed-english-v3` (1024d), `cohere.embed-multilingual-v3` (1024d).

```go
import "digital.vasic.embeddings/pkg/bedrock"

// Amazon Titan
client := bedrock.NewClient(bedrock.Config{
    Region:    "us-east-1",                     // default
    AccessKey: "AKIA...",
    SecretKey: "your-secret-key",
    Model:     "amazon.titan-embed-text-v2:0",  // default
})

vec, err := client.Embed(ctx, "AWS Bedrock Titan embedding")

// Cohere on Bedrock
cohereClient := bedrock.NewClient(bedrock.Config{
    Region:    "us-east-1",
    AccessKey: "AKIA...",
    SecretKey: "your-secret-key",
    Model:     "cohere.embed-english-v3",
})

vecs, err := cohereClient.EmbedBatch(ctx, []string{"text1", "text2"})
```

The Bedrock provider implements AWS Signature Version 4 signing internally. No external AWS SDK dependency is required.

Note: Amazon Titan models do not natively support batch embedding. `EmbedBatch` on Titan models calls `Embed` sequentially for each input text. Cohere models on Bedrock support native batch embedding.

Configuration options:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Region` | `string` | `us-east-1` | AWS region |
| `AccessKey` | `string` | (required) | AWS access key ID |
| `SecretKey` | `string` | (required) | AWS secret access key |
| `Model` | `string` | `amazon.titan-embed-text-v2:0` | Bedrock model ID |
| `BaseURL` | `string` | auto-generated | Override Bedrock endpoint |
| `Timeout` | `time.Duration` | `30s` | HTTP request timeout |

## Batch Embedding

All providers support `EmbedBatch` for processing multiple texts in a single call. This is more efficient than calling `Embed` in a loop because most providers handle batches server-side.

```go
texts := []string{
    "Document about machine learning",
    "Document about natural language processing",
    "Document about computer vision",
    "Document about reinforcement learning",
}

embeddings, err := client.EmbedBatch(ctx, texts)
if err != nil {
    log.Fatal(err)
}

for i, emb := range embeddings {
    fmt.Printf("Text %d: %d dimensions\n", i, len(emb))
}
```

For large datasets, chunk your inputs according to provider limits and call `EmbedBatch` for each chunk. The `provider.Config.BatchSize` field (default 100) can guide your chunking logic.

## Context and Cancellation

All embedding methods accept a `context.Context` for timeout and cancellation:

```go
// With timeout
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

vec, err := client.Embed(ctx, "time-sensitive embedding")
if err != nil {
    // May be context.DeadlineExceeded
    log.Printf("embedding failed: %v", err)
}
```

## Error Handling

All providers wrap errors with the provider name prefix for easy identification:

```go
vec, err := client.Embed(ctx, "test")
if err != nil {
    // Errors follow the pattern: "<provider>: <description>: <cause>"
    // Examples:
    //   "openai: API error 401 Unauthorized - {\"error\":...}"
    //   "cohere: request failed: context deadline exceeded"
    //   "bedrock: failed to embed text 2: bedrock: API error 400 ..."
    log.Printf("error: %v", err)
}
```

Use `errors.Is` and `errors.As` to inspect wrapped errors:

```go
import "errors"

if errors.Is(err, context.DeadlineExceeded) {
    // Handle timeout
}
```

## Switching Providers at Runtime

Because all providers share the same interface, you can switch providers based on configuration:

```go
func newProvider(name, apiKey string) (provider.EmbeddingProvider, error) {
    switch name {
    case "openai":
        return openai.NewClient(openai.Config{APIKey: apiKey}), nil
    case "cohere":
        return cohere.NewClient(cohere.Config{APIKey: apiKey}), nil
    case "voyage":
        return voyage.NewClient(voyage.Config{APIKey: apiKey}), nil
    case "jina":
        return jina.NewClient(jina.Config{APIKey: apiKey}), nil
    case "google":
        return google.NewClient(google.Config{APIKey: apiKey, ProjectID: "my-project"}), nil
    case "bedrock":
        return bedrock.NewClient(bedrock.Config{AccessKey: apiKey, SecretKey: "..."}), nil
    default:
        return nil, fmt.Errorf("unknown provider: %s", name)
    }
}
```
