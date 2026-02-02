// Package provider defines the core interfaces and types for embedding providers.
// All embedding backends implement EmbeddingProvider to offer a unified API for
// generating text embeddings across different services.
package provider

import (
	"context"
	"time"
)

// EmbeddingProvider defines the interface for all embedding backends.
type EmbeddingProvider interface {
	// Embed generates an embedding vector for the given text.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch generates embedding vectors for multiple texts.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the number of dimensions in the embedding vectors
	// produced by this provider.
	Dimensions() int

	// Name returns the provider name (e.g., "openai/text-embedding-3-small").
	Name() string
}

// Config holds common configuration for embedding providers.
type Config struct {
	// Model is the model identifier used by the provider.
	Model string `json:"model"`

	// BatchSize is the maximum number of texts per batch request.
	BatchSize int `json:"batch_size"`

	// MaxRetries is the maximum number of retry attempts on transient failures.
	MaxRetries int `json:"max_retries"`

	// Timeout is the HTTP request timeout for embedding API calls.
	Timeout time.Duration `json:"timeout"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		BatchSize:  100,
		MaxRetries: 3,
		Timeout:    30 * time.Second,
	}
}

// Result holds the output of an embedding request.
type Result struct {
	// Embeddings contains the embedding vectors, one per input text.
	Embeddings [][]float32 `json:"embeddings"`

	// Model is the model identifier that produced the embeddings.
	Model string `json:"model"`

	// Usage contains token usage information for the request.
	Usage TokenUsage `json:"usage"`
}

// TokenUsage contains token consumption details for an embedding request.
type TokenUsage struct {
	// PromptTokens is the number of tokens in the input texts.
	PromptTokens int `json:"prompt_tokens"`

	// TotalTokens is the total number of tokens consumed.
	TotalTokens int `json:"total_tokens"`
}
