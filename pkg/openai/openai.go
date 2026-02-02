// Package openai implements the EmbeddingProvider interface for OpenAI's
// embedding models (text-embedding-3-small, text-embedding-3-large, ada-002).
package openai

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

// Default values for OpenAI embedding configuration.
const (
	DefaultBaseURL = "https://api.openai.com/v1"
	DefaultModel   = "text-embedding-3-small"
)

// Config holds OpenAI-specific configuration.
type Config struct {
	// APIKey is the OpenAI API key for authentication.
	APIKey string `json:"api_key"`

	// Model is the embedding model to use.
	// Supported: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002.
	Model string `json:"model"`

	// BaseURL is the base URL for the OpenAI API. Defaults to https://api.openai.com/v1.
	BaseURL string `json:"base_url"`

	// Timeout is the HTTP request timeout. Defaults to 30s.
	Timeout time.Duration `json:"timeout"`
}

// Client implements provider.EmbeddingProvider for OpenAI.
type Client struct {
	config     Config
	httpClient *http.Client
	dimension  int
}

// embedRequest is the request body for the OpenAI embeddings endpoint.
type embedRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

// embedResponse is the response body from the OpenAI embeddings endpoint.
type embedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// NewClient creates a new OpenAI embedding client.
func NewClient(config Config) *Client {
	if config.BaseURL == "" {
		config.BaseURL = DefaultBaseURL
	}
	if config.Model == "" {
		config.Model = DefaultModel
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
	return fmt.Sprintf("openai/%s", c.config.Model)
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
		return nil, fmt.Errorf("openai: no embedding returned")
	}
	return embeddings[0], nil
}

// EmbedBatch generates embeddings for multiple texts.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := embedRequest{
		Input: texts,
		Model: c.config.Model,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("openai: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("%s/embeddings", c.config.BaseURL),
		bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.config.APIKey))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai: API error %s - %s", resp.Status, string(respBody))
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("openai: failed to parse response: %w", err)
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
	case "text-embedding-3-small":
		return 1536
	case "text-embedding-3-large":
		return 3072
	case "text-embedding-ada-002":
		return 1536
	default:
		return 1536
	}
}

// Compile-time interface check.
var _ provider.EmbeddingProvider = (*Client)(nil)
