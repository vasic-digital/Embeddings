// Package google implements the EmbeddingProvider interface for Google Vertex AI's
// embedding models (text-embedding-005, text-multilingual-embedding-002, etc.).
package google

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

// Default values for Google Vertex AI embedding configuration.
const (
	DefaultLocation = "us-central1"
	DefaultModel    = "text-embedding-005"
)

// Config holds Google Vertex AI-specific configuration.
type Config struct {
	// ProjectID is the Google Cloud project ID.
	ProjectID string `json:"project_id"`

	// Location is the Google Cloud region (e.g., "us-central1").
	Location string `json:"location"`

	// Model is the embedding model to use.
	Model string `json:"model"`

	// APIKey is the API key or access token for authentication.
	APIKey string `json:"api_key"`

	// BaseURL overrides the default Vertex AI endpoint.
	BaseURL string `json:"base_url"`

	// Timeout is the HTTP request timeout.
	Timeout time.Duration `json:"timeout"`
}

// jsonMarshaler abstracts JSON marshaling for dependency injection in tests.
type jsonMarshaler interface {
	Marshal(v interface{}) ([]byte, error)
}

// defaultMarshaler is the production JSON marshaler.
type defaultMarshaler struct{}

func (defaultMarshaler) Marshal(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}

// Client implements provider.EmbeddingProvider for Google Vertex AI.
type Client struct {
	config     Config
	httpClient *http.Client
	dimension  int
	marshaler  jsonMarshaler
}

// embedRequest represents a Google embedding API request.
type embedRequest struct {
	Instances []embedInstance `json:"instances"`
}

// embedInstance represents a single embedding input.
type embedInstance struct {
	Content  string `json:"content"`
	TaskType string `json:"task_type,omitempty"`
}

// embedResponse represents a Google embedding API response.
type embedResponse struct {
	Predictions []struct {
		Embeddings struct {
			Values     []float32 `json:"values"`
			Statistics struct {
				TokenCount int `json:"token_count"`
			} `json:"statistics"`
		} `json:"embeddings"`
	} `json:"predictions"`
}

// NewClient creates a new Google Vertex AI embedding client.
func NewClient(config Config) *Client {
	if config.Location == "" {
		config.Location = DefaultLocation
	}
	if config.Model == "" {
		config.Model = DefaultModel
	}
	if config.BaseURL == "" {
		config.BaseURL = fmt.Sprintf(
			"https://%s-aiplatform.googleapis.com/v1", config.Location)
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
		marshaler: defaultMarshaler{},
	}
}

// Name returns the provider name.
func (c *Client) Name() string {
	return fmt.Sprintf("google/%s", c.config.Model)
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
		return nil, fmt.Errorf("google: no embedding returned")
	}
	return embeddings[0], nil
}

// EmbedBatch generates embeddings for multiple texts.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	instances := make([]embedInstance, len(texts))
	for i, text := range texts {
		instances[i] = embedInstance{
			Content:  text,
			TaskType: "RETRIEVAL_DOCUMENT",
		}
	}

	reqBody := embedRequest{Instances: instances}

	body, err := c.marshaler.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("google: failed to marshal request: %w", err)
	}

	url := fmt.Sprintf(
		"%s/projects/%s/locations/%s/publishers/google/models/%s:predict",
		c.config.BaseURL, c.config.ProjectID, c.config.Location, c.config.Model)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("google: failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.config.APIKey))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("google: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("google: API error %s - %s", resp.Status, string(respBody))
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("google: failed to parse response: %w", err)
	}

	embeddings := make([][]float32, len(result.Predictions))
	for i, pred := range result.Predictions {
		embeddings[i] = pred.Embeddings.Values
	}

	return embeddings, nil
}

// dimensionForModel returns the embedding dimension for the given model.
func dimensionForModel(model string) int {
	switch model {
	case "text-embedding-005", "textembedding-gecko@003":
		return 768
	case "text-multilingual-embedding-002":
		return 768
	case "text-embedding-004":
		return 768
	case "textembedding-gecko-multilingual@001":
		return 768
	default:
		return 768
	}
}

// Compile-time interface check.
var _ provider.EmbeddingProvider = (*Client)(nil)
