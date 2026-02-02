// Package jina implements the EmbeddingProvider interface for Jina AI's
// embedding models (jina-embeddings-v3, jina-embeddings-v2-base-en, etc.).
package jina

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"digital.vasic.embeddings/pkg/provider"
)

// Default values for Jina AI embedding configuration.
const (
	DefaultBaseURL = "https://api.jina.ai/v1"
	DefaultModel   = "jina-embeddings-v3"
)

// Config holds Jina AI-specific configuration.
type Config struct {
	// APIKey is the Jina AI API key for authentication.
	APIKey string `json:"api_key"`

	// Model is the embedding model to use.
	Model string `json:"model"`

	// BaseURL is the base URL for the Jina AI API.
	BaseURL string `json:"base_url"`

	// Task specifies the task type (e.g., "retrieval.document", "retrieval.query").
	Task string `json:"task"`

	// Timeout is the HTTP request timeout.
	Timeout time.Duration `json:"timeout"`
}

// Client implements provider.EmbeddingProvider for Jina AI.
type Client struct {
	config     Config
	httpClient *http.Client
	dimension  int
}

// embedRequest represents a Jina embed API request.
type embedRequest struct {
	Input          []string `json:"input"`
	Model          string   `json:"model"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
	Task           string   `json:"task,omitempty"`
}

// embedResponse represents a Jina embed API response.
type embedResponse struct {
	Model  string `json:"model"`
	Object string `json:"object"`
	Usage  struct {
		TotalTokens  int `json:"total_tokens"`
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
	Data []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// NewClient creates a new Jina AI embedding client.
func NewClient(config Config) *Client {
	if config.BaseURL == "" {
		config.BaseURL = DefaultBaseURL
	}
	if config.Model == "" {
		config.Model = DefaultModel
	}
	if config.Task == "" {
		config.Task = "retrieval.document"
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	dimension := dimensionForModel(config.Model)

	return &Client{
		config:    config,
		dimension: dimension,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

// Name returns the provider name.
func (c *Client) Name() string {
	return fmt.Sprintf("jina/%s", c.config.Model)
}

// Dimensions returns the embedding vector dimensionality.
func (c *Client) Dimensions() int {
	return c.dimension
}

// Embed generates an embedding for a single text.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := c.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("jina: no embedding returned")
	}
	return embeddings[0], nil
}

// EmbedBatch generates embeddings for multiple texts.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := embedRequest{
		Input:          texts,
		Model:          c.config.Model,
		EncodingFormat: "float",
		Task:           c.config.Task,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("jina: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("%s/embeddings", c.config.BaseURL),
		bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("jina: failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.config.APIKey))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("jina: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("jina: API error %s - %s", resp.Status, string(respBody))
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("jina: failed to parse response: %w", err)
	}

	embeddings := make([][]float32, len(result.Data))
	for _, item := range result.Data {
		embeddings[item.Index] = item.Embedding
	}

	return embeddings, nil
}

// dimensionForModel returns the embedding dimension for the given model.
func dimensionForModel(model string) int {
	switch model {
	case "jina-embeddings-v3":
		return 1024
	case "jina-embeddings-v2-base-en":
		return 768
	case "jina-embeddings-v2-small-en":
		return 512
	case "jina-embeddings-v2-base-de", "jina-embeddings-v2-base-es",
		"jina-embeddings-v2-base-zh":
		return 768
	case "jina-clip-v1":
		return 768
	case "jina-colbert-v2":
		return 128
	default:
		return 1024
	}
}

// Compile-time interface check.
var _ provider.EmbeddingProvider = (*Client)(nil)
