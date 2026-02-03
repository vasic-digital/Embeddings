# API Reference

Complete reference for all exported types, functions, and methods in the `digital.vasic.embeddings` module.

---

## Package `provider`

**Import path**: `digital.vasic.embeddings/pkg/provider`

Defines the core interface and shared types for all embedding providers.

### Interface `EmbeddingProvider`

```go
type EmbeddingProvider interface {
    Embed(ctx context.Context, text string) ([]float32, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
    Dimensions() int
    Name() string
}
```

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `Embed` | `ctx context.Context, text string` | `([]float32, error)` | Generates an embedding vector for a single text |
| `EmbedBatch` | `ctx context.Context, texts []string` | `([][]float32, error)` | Generates embedding vectors for multiple texts |
| `Dimensions` | -- | `int` | Returns the number of dimensions in the embedding vectors |
| `Name` | -- | `string` | Returns the provider name in `provider/model` format |

### Struct `Config`

```go
type Config struct {
    Model      string        `json:"model"`
    BatchSize  int           `json:"batch_size"`
    MaxRetries int           `json:"max_retries"`
    Timeout    time.Duration `json:"timeout"`
}
```

| Field | Type | JSON | Description |
|-------|------|------|-------------|
| `Model` | `string` | `model` | Model identifier |
| `BatchSize` | `int` | `batch_size` | Maximum texts per batch request |
| `MaxRetries` | `int` | `max_retries` | Maximum retry attempts on transient failures |
| `Timeout` | `time.Duration` | `timeout` | HTTP request timeout |

### Function `DefaultConfig`

```go
func DefaultConfig() Config
```

Returns a `Config` with: `BatchSize=100`, `MaxRetries=3`, `Timeout=30s`, `Model=""`.

### Struct `Result`

```go
type Result struct {
    Embeddings [][]float32 `json:"embeddings"`
    Model      string      `json:"model"`
    Usage      TokenUsage  `json:"usage"`
}
```

| Field | Type | JSON | Description |
|-------|------|------|-------------|
| `Embeddings` | `[][]float32` | `embeddings` | Embedding vectors, one per input text |
| `Model` | `string` | `model` | Model identifier that produced the embeddings |
| `Usage` | `TokenUsage` | `usage` | Token usage information |

### Struct `TokenUsage`

```go
type TokenUsage struct {
    PromptTokens int `json:"prompt_tokens"`
    TotalTokens  int `json:"total_tokens"`
}
```

| Field | Type | JSON | Description |
|-------|------|------|-------------|
| `PromptTokens` | `int` | `prompt_tokens` | Number of tokens in the input texts |
| `TotalTokens` | `int` | `total_tokens` | Total tokens consumed |

---

## Package `openai`

**Import path**: `digital.vasic.embeddings/pkg/openai`

Implements `provider.EmbeddingProvider` for OpenAI embedding models.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `DefaultBaseURL` | `"https://api.openai.com/v1"` | Default OpenAI API base URL |
| `DefaultModel` | `"text-embedding-3-small"` | Default embedding model |

### Struct `Config`

```go
type Config struct {
    APIKey  string        `json:"api_key"`
    Model   string        `json:"model"`
    BaseURL string        `json:"base_url"`
    Timeout time.Duration `json:"timeout"`
}
```

| Field | Type | JSON | Default | Description |
|-------|------|------|---------|-------------|
| `APIKey` | `string` | `api_key` | (required) | OpenAI API key |
| `Model` | `string` | `model` | `text-embedding-3-small` | Embedding model |
| `BaseURL` | `string` | `base_url` | `https://api.openai.com/v1` | API base URL |
| `Timeout` | `time.Duration` | `timeout` | `30s` | HTTP request timeout |

### Struct `Client`

Implements `provider.EmbeddingProvider`. Not exported fields.

### Function `NewClient`

```go
func NewClient(config Config) *Client
```

Creates a new OpenAI embedding client. Applies default values for unset fields and resolves the embedding dimension from the model name.

**Dimension mapping**:

| Model | Dimensions |
|-------|-----------|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `text-embedding-ada-002` | 1536 |
| (default) | 1536 |

### Method `(*Client) Embed`

```go
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error)
```

Generates an embedding for a single text. Delegates to `EmbedBatch`.

### Method `(*Client) EmbedBatch`

```go
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
```

Generates embeddings for multiple texts in a single API call to `POST {BaseURL}/embeddings`.

### Method `(*Client) Name`

```go
func (c *Client) Name() string
```

Returns `"openai/{model}"`.

### Method `(*Client) Dimensions`

```go
func (c *Client) Dimensions() int
```

Returns the embedding vector dimensionality for the configured model.

---

## Package `cohere`

**Import path**: `digital.vasic.embeddings/pkg/cohere`

Implements `provider.EmbeddingProvider` for Cohere embedding models.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `DefaultBaseURL` | `"https://api.cohere.com/v2"` | Default Cohere API base URL |
| `DefaultModel` | `"embed-english-v3.0"` | Default embedding model |

### Struct `Config`

```go
type Config struct {
    APIKey    string        `json:"api_key"`
    Model     string        `json:"model"`
    BaseURL   string        `json:"base_url"`
    InputType string        `json:"input_type"`
    Timeout   time.Duration `json:"timeout"`
}
```

| Field | Type | JSON | Default | Description |
|-------|------|------|---------|-------------|
| `APIKey` | `string` | `api_key` | (required) | Cohere API key |
| `Model` | `string` | `model` | `embed-english-v3.0` | Embedding model |
| `BaseURL` | `string` | `base_url` | `https://api.cohere.com/v2` | API base URL |
| `InputType` | `string` | `input_type` | `search_document` | Input type: `search_document`, `search_query`, `classification`, `clustering` |
| `Timeout` | `time.Duration` | `timeout` | `30s` | HTTP request timeout |

### Function `NewClient`

```go
func NewClient(config Config) *Client
```

Creates a new Cohere embedding client.

**Dimension mapping**:

| Model | Dimensions |
|-------|-----------|
| `embed-english-v3.0` | 1024 |
| `embed-multilingual-v3.0` | 1024 |
| `embed-english-light-v3.0` | 384 |
| `embed-multilingual-light-v3.0` | 384 |
| `embed-english-v2.0` | 4096 |
| `embed-multilingual-v2.0` | 768 |
| (default) | 1024 |

### Method `(*Client) Embed`

```go
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error)
```

Generates an embedding for a single text. Delegates to `EmbedBatch`.

### Method `(*Client) EmbedBatch`

```go
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
```

Generates embeddings via `POST {BaseURL}/embed`. Handles both the `embeddings` array response format and the `embeddings_by_type.float` format. Truncation is set to `"END"`.

### Method `(*Client) Name`

```go
func (c *Client) Name() string
```

Returns `"cohere/{model}"`.

### Method `(*Client) Dimensions`

```go
func (c *Client) Dimensions() int
```

Returns the embedding vector dimensionality for the configured model.

---

## Package `voyage`

**Import path**: `digital.vasic.embeddings/pkg/voyage`

Implements `provider.EmbeddingProvider` for Voyage AI embedding models.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `DefaultBaseURL` | `"https://api.voyageai.com/v1"` | Default Voyage AI API base URL |
| `DefaultModel` | `"voyage-3"` | Default embedding model |

### Struct `Config`

```go
type Config struct {
    APIKey    string        `json:"api_key"`
    Model     string        `json:"model"`
    BaseURL   string        `json:"base_url"`
    InputType string        `json:"input_type"`
    Timeout   time.Duration `json:"timeout"`
}
```

| Field | Type | JSON | Default | Description |
|-------|------|------|---------|-------------|
| `APIKey` | `string` | `api_key` | (required) | Voyage AI API key |
| `Model` | `string` | `model` | `voyage-3` | Embedding model |
| `BaseURL` | `string` | `base_url` | `https://api.voyageai.com/v1` | API base URL |
| `InputType` | `string` | `input_type` | `document` | Input type: `document` or `query` |
| `Timeout` | `time.Duration` | `timeout` | `30s` | HTTP request timeout |

### Function `NewClient`

```go
func NewClient(config Config) *Client
```

Creates a new Voyage AI embedding client.

**Dimension mapping**:

| Model | Dimensions |
|-------|-----------|
| `voyage-3` | 1024 |
| `voyage-3-lite` | 512 |
| `voyage-code-3` | 1024 |
| `voyage-finance-2` | 1024 |
| `voyage-law-2` | 1024 |
| `voyage-large-2` | 1536 |
| `voyage-large-2-instruct` | 1536 |
| `voyage-2` | 1024 |
| (default) | 1024 |

### Method `(*Client) Embed`

```go
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error)
```

Generates an embedding for a single text. Delegates to `EmbedBatch`.

### Method `(*Client) EmbedBatch`

```go
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
```

Generates embeddings via `POST {BaseURL}/embeddings`. Truncation is enabled. Results are reordered by the `index` field.

### Method `(*Client) Name`

```go
func (c *Client) Name() string
```

Returns `"voyage/{model}"`.

### Method `(*Client) Dimensions`

```go
func (c *Client) Dimensions() int
```

Returns the embedding vector dimensionality for the configured model.

---

## Package `jina`

**Import path**: `digital.vasic.embeddings/pkg/jina`

Implements `provider.EmbeddingProvider` for Jina AI embedding models.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `DefaultBaseURL` | `"https://api.jina.ai/v1"` | Default Jina AI API base URL |
| `DefaultModel` | `"jina-embeddings-v3"` | Default embedding model |

### Struct `Config`

```go
type Config struct {
    APIKey  string        `json:"api_key"`
    Model   string        `json:"model"`
    BaseURL string        `json:"base_url"`
    Task    string        `json:"task"`
    Timeout time.Duration `json:"timeout"`
}
```

| Field | Type | JSON | Default | Description |
|-------|------|------|---------|-------------|
| `APIKey` | `string` | `api_key` | (required) | Jina AI API key |
| `Model` | `string` | `model` | `jina-embeddings-v3` | Embedding model |
| `BaseURL` | `string` | `base_url` | `https://api.jina.ai/v1` | API base URL |
| `Task` | `string` | `task` | `retrieval.document` | Task type: `retrieval.document` or `retrieval.query` |
| `Timeout` | `time.Duration` | `timeout` | `30s` | HTTP request timeout |

### Function `NewClient`

```go
func NewClient(config Config) *Client
```

Creates a new Jina AI embedding client.

**Dimension mapping**:

| Model | Dimensions |
|-------|-----------|
| `jina-embeddings-v3` | 1024 |
| `jina-embeddings-v2-base-en` | 768 |
| `jina-embeddings-v2-small-en` | 512 |
| `jina-embeddings-v2-base-de` | 768 |
| `jina-embeddings-v2-base-es` | 768 |
| `jina-embeddings-v2-base-zh` | 768 |
| `jina-clip-v1` | 768 |
| `jina-colbert-v2` | 128 |
| (default) | 1024 |

### Method `(*Client) Embed`

```go
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error)
```

Generates an embedding for a single text. Delegates to `EmbedBatch`.

### Method `(*Client) EmbedBatch`

```go
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
```

Generates embeddings via `POST {BaseURL}/embeddings`. Encoding format is set to `"float"`. Results are reordered by the `index` field.

### Method `(*Client) Name`

```go
func (c *Client) Name() string
```

Returns `"jina/{model}"`.

### Method `(*Client) Dimensions`

```go
func (c *Client) Dimensions() int
```

Returns the embedding vector dimensionality for the configured model.

---

## Package `google`

**Import path**: `digital.vasic.embeddings/pkg/google`

Implements `provider.EmbeddingProvider` for Google Vertex AI embedding models.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `DefaultLocation` | `"us-central1"` | Default Google Cloud region |
| `DefaultModel` | `"text-embedding-005"` | Default embedding model |

### Struct `Config`

```go
type Config struct {
    ProjectID string        `json:"project_id"`
    Location  string        `json:"location"`
    Model     string        `json:"model"`
    APIKey    string        `json:"api_key"`
    BaseURL   string        `json:"base_url"`
    Timeout   time.Duration `json:"timeout"`
}
```

| Field | Type | JSON | Default | Description |
|-------|------|------|---------|-------------|
| `ProjectID` | `string` | `project_id` | (required) | Google Cloud project ID |
| `Location` | `string` | `location` | `us-central1` | GCP region |
| `Model` | `string` | `model` | `text-embedding-005` | Embedding model |
| `APIKey` | `string` | `api_key` | (required) | Access token or API key |
| `BaseURL` | `string` | `base_url` | auto-generated from Location | Vertex AI endpoint |
| `Timeout` | `time.Duration` | `timeout` | `30s` | HTTP request timeout |

### Function `NewClient`

```go
func NewClient(config Config) *Client
```

Creates a new Google Vertex AI embedding client.

**Dimension mapping**:

| Model | Dimensions |
|-------|-----------|
| `text-embedding-005` | 768 |
| `textembedding-gecko@003` | 768 |
| `text-multilingual-embedding-002` | 768 |
| `text-embedding-004` | 768 |
| `textembedding-gecko-multilingual@001` | 768 |
| (default) | 768 |

### Method `(*Client) Embed`

```go
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error)
```

Generates an embedding for a single text. Delegates to `EmbedBatch`.

### Method `(*Client) EmbedBatch`

```go
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
```

Generates embeddings via `POST {BaseURL}/projects/{project}/locations/{location}/publishers/google/models/{model}:predict`. Each text is wrapped in an instance with `task_type=RETRIEVAL_DOCUMENT`.

### Method `(*Client) Name`

```go
func (c *Client) Name() string
```

Returns `"google/{model}"`.

### Method `(*Client) Dimensions`

```go
func (c *Client) Dimensions() int
```

Returns the embedding vector dimensionality for the configured model.

---

## Package `bedrock`

**Import path**: `digital.vasic.embeddings/pkg/bedrock`

Implements `provider.EmbeddingProvider` for AWS Bedrock embedding models (Amazon Titan and Cohere on Bedrock).

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `DefaultRegion` | `"us-east-1"` | Default AWS region |
| `DefaultModel` | `"amazon.titan-embed-text-v2:0"` | Default Bedrock model |

### Struct `Config`

```go
type Config struct {
    Region    string        `json:"region"`
    AccessKey string        `json:"access_key"`
    SecretKey string        `json:"secret_key"`
    Model     string        `json:"model"`
    BaseURL   string        `json:"base_url"`
    Timeout   time.Duration `json:"timeout"`
}
```

| Field | Type | JSON | Default | Description |
|-------|------|------|---------|-------------|
| `Region` | `string` | `region` | `us-east-1` | AWS region |
| `AccessKey` | `string` | `access_key` | (required) | AWS access key ID |
| `SecretKey` | `string` | `secret_key` | (required) | AWS secret access key |
| `Model` | `string` | `model` | `amazon.titan-embed-text-v2:0` | Bedrock model ID |
| `BaseURL` | `string` | `base_url` | auto-generated from Region | Bedrock runtime endpoint |
| `Timeout` | `time.Duration` | `timeout` | `30s` | HTTP request timeout |

### Function `NewClient`

```go
func NewClient(config Config) *Client
```

Creates a new AWS Bedrock embedding client.

**Dimension mapping**:

| Model | Dimensions |
|-------|-----------|
| `amazon.titan-embed-text-v1` | 1536 |
| `amazon.titan-embed-text-v2:0` | 1024 |
| `amazon.titan-embed-image-v1` | 1024 |
| `cohere.embed-english-v3` | 1024 |
| `cohere.embed-multilingual-v3` | 1024 |
| (default) | 1536 |

### Method `(*Client) Embed`

```go
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error)
```

Generates an embedding for a single text. Dispatches to `embedTitan` for `amazon.titan*` models or to `embedCohere` for `cohere.*` models. Returns an error for unsupported model prefixes.

### Method `(*Client) EmbedBatch`

```go
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
```

Generates embeddings for multiple texts. For `cohere.*` models, sends a native batch request. For `amazon.titan*` models, calls `Embed` sequentially for each text (Titan does not support batch embedding).

### Method `(*Client) Name`

```go
func (c *Client) Name() string
```

Returns `"bedrock/{model}"`.

### Method `(*Client) Dimensions`

```go
func (c *Client) Dimensions() int
```

Returns the embedding vector dimensionality for the configured model.
