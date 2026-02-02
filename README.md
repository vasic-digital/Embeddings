# Embeddings

Generic, reusable Go module for text embedding generation across multiple providers.

## Providers

| Package | Provider | Models |
|---------|----------|--------|
| `pkg/openai` | OpenAI | text-embedding-3-small, text-embedding-3-large, ada-002 |
| `pkg/cohere` | Cohere | embed-english-v3.0, embed-multilingual-v3.0, light variants |
| `pkg/voyage` | Voyage AI | voyage-3, voyage-3-lite, voyage-code-3, voyage-law-2 |
| `pkg/jina` | Jina AI | jina-embeddings-v3, jina-embeddings-v2-*, jina-clip-v1 |
| `pkg/google` | Google Vertex AI | text-embedding-005, text-multilingual-embedding-002 |
| `pkg/bedrock` | AWS Bedrock | Amazon Titan Embed, Cohere on Bedrock |

## Usage

```go
import (
    "digital.vasic.embeddings/pkg/openai"
    "digital.vasic.embeddings/pkg/provider"
)

client := openai.NewClient(openai.Config{
    APIKey: "your-key",
    Model:  "text-embedding-3-small",
})

// Single embedding
embedding, err := client.Embed(ctx, "Hello world")

// Batch embeddings
embeddings, err := client.EmbedBatch(ctx, []string{"text1", "text2"})

// Provider info
fmt.Println(client.Name())       // "openai/text-embedding-3-small"
fmt.Println(client.Dimensions()) // 1536
```

## Interface

All providers implement `provider.EmbeddingProvider`:

```go
type EmbeddingProvider interface {
    Embed(ctx context.Context, text string) ([]float32, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
    Dimensions() int
    Name() string
}
```
