// Package cohere implements the EmbeddingProvider interface for Cohere's
// embedding models (embed-english-v3.0, embed-multilingual-v3.0, etc.).
package cohere

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

// Default values for Cohere embedding configuration.
const (
	DefaultBaseURL = "https://api.cohere.com/v2"
	DefaultModel   = "embed-english-v3.0"
)

// Config holds Cohere-specific configuration.
type Config struct {
	// APIKey is the Cohere API key for authentication.
	APIKey string `json:"api_key"`

	// Model is the embedding model to use.
	Model string `json:"model"`

	// BaseURL is the base URL for the Cohere API.
	BaseURL string `json:"base_url"`

	// InputType specifies the type of input for embedding.
	// Common values: "search_document", "search_query", "classification", "clustering".
	InputType string `json:"input_type"`

	// Timeout is the HTTP request timeout.
	Timeout time.Duration `json:"timeout"`
}

// Client implements provider.EmbeddingProvider for Cohere.
type Client struct {
	config     Config
	httpClient *http.Client
	dimension  int
}

// embedRequest represents a Cohere embed API request.
type embedRequest struct {
	Texts     []string `json:"texts"`
	Model     string   `json:"model"`
	InputType string   `json:"input_type"`
	Truncate  string   `json:"truncate,omitempty"`
}

// embedResponse represents a Cohere embed API response.
type embedResponse struct {
	ID            string         `json:"id"`
	Embeddings    [][]float32    `json:"embeddings"`
	EmbeddingsObj *embeddingsObj `json:"embeddings_by_type,omitempty"`
	Texts         []string       `json:"texts"`
}

// embeddingsObj represents typed embeddings.
type embeddingsObj struct {
	Float [][]float32 `json:"float,omitempty"`
}

// NewClient creates a new Cohere embedding client.
func NewClient(config Config) *Client {
	if config.BaseURL == "" {
		config.BaseURL = DefaultBaseURL
	}
	if config.Model == "" {
		config.Model = DefaultModel
	}
	if config.InputType == "" {
		config.InputType = "search_document"
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
	return fmt.Sprintf("cohere/%s", c.config.Model)
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
		return nil, fmt.Errorf("cohere: no embedding returned")
	}
	return embeddings[0], nil
}

// EmbedBatch generates embeddings for multiple texts.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := embedRequest{
		Texts:     texts,
		Model:     c.config.Model,
		InputType: c.config.InputType,
		Truncate:  "END",
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("cohere: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("%s/embed", c.config.BaseURL),
		bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("cohere: failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.config.APIKey))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cohere: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("cohere: API error %s - %s", resp.Status, string(respBody))
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("cohere: failed to parse response: %w", err)
	}

	if result.Embeddings != nil {
		return result.Embeddings, nil
	}
	if result.EmbeddingsObj != nil && result.EmbeddingsObj.Float != nil {
		return result.EmbeddingsObj.Float, nil
	}

	return nil, fmt.Errorf("cohere: no embeddings in response")
}

// dimensionForModel returns the embedding dimension for the given model.
func dimensionForModel(model string) int {
	switch model {
	case "embed-english-v3.0", "embed-multilingual-v3.0":
		return 1024
	case "embed-english-light-v3.0", "embed-multilingual-light-v3.0":
		return 384
	case "embed-english-v2.0":
		return 4096
	case "embed-multilingual-v2.0":
		return 768
	default:
		return 1024
	}
}

// Compile-time interface check.
var _ provider.EmbeddingProvider = (*Client)(nil)
