// Package voyage implements the EmbeddingProvider interface for Voyage AI's
// embedding models (voyage-3, voyage-3-lite, voyage-code-3, etc.).
package voyage

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

// Default values for Voyage AI embedding configuration.
const (
	DefaultBaseURL = "https://api.voyageai.com/v1"
	DefaultModel   = "voyage-3"
)

// Config holds Voyage AI-specific configuration.
type Config struct {
	// APIKey is the Voyage AI API key for authentication.
	APIKey string `json:"api_key"`

	// Model is the embedding model to use.
	Model string `json:"model"`

	// BaseURL is the base URL for the Voyage AI API.
	BaseURL string `json:"base_url"`

	// InputType specifies the type of input ("document" or "query").
	InputType string `json:"input_type"`

	// Timeout is the HTTP request timeout.
	Timeout time.Duration `json:"timeout"`
}

// Client implements provider.EmbeddingProvider for Voyage AI.
type Client struct {
	config     Config
	httpClient *http.Client
	dimension  int
}

// embedRequest represents a Voyage embed API request.
type embedRequest struct {
	Input      []string `json:"input"`
	Model      string   `json:"model"`
	InputType  string   `json:"input_type,omitempty"`
	Truncation bool     `json:"truncation,omitempty"`
}

// embedResponse represents a Voyage embed API response.
type embedResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// NewClient creates a new Voyage AI embedding client.
func NewClient(config Config) *Client {
	if config.BaseURL == "" {
		config.BaseURL = DefaultBaseURL
	}
	if config.Model == "" {
		config.Model = DefaultModel
	}
	if config.InputType == "" {
		config.InputType = "document"
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
	return fmt.Sprintf("voyage/%s", c.config.Model)
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
		return nil, fmt.Errorf("voyage: no embedding returned")
	}
	return embeddings[0], nil
}

// EmbedBatch generates embeddings for multiple texts.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := embedRequest{
		Input:      texts,
		Model:      c.config.Model,
		InputType:  c.config.InputType,
		Truncation: true,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("voyage: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("%s/embeddings", c.config.BaseURL),
		bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("voyage: failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.config.APIKey))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("voyage: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("voyage: API error %s - %s", resp.Status, string(respBody))
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("voyage: failed to parse response: %w", err)
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
	case "voyage-3":
		return 1024
	case "voyage-3-lite":
		return 512
	case "voyage-code-3":
		return 1024
	case "voyage-finance-2":
		return 1024
	case "voyage-law-2":
		return 1024
	case "voyage-large-2", "voyage-large-2-instruct":
		return 1536
	case "voyage-2":
		return 1024
	default:
		return 1024
	}
}

// Compile-time interface check.
var _ provider.EmbeddingProvider = (*Client)(nil)
