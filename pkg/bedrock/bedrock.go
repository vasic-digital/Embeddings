// Package bedrock implements the EmbeddingProvider interface for AWS Bedrock's
// embedding models (Amazon Titan, Cohere on Bedrock).
package bedrock

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"digital.vasic.embeddings/pkg/provider"
)

// Default values for AWS Bedrock embedding configuration.
const (
	DefaultRegion = "us-east-1"
	DefaultModel  = "amazon.titan-embed-text-v2:0"
)

// Config holds AWS Bedrock-specific configuration.
type Config struct {
	// Region is the AWS region (e.g., "us-east-1").
	Region string `json:"region"`

	// AccessKey is the AWS access key ID.
	AccessKey string `json:"access_key"`

	// SecretKey is the AWS secret access key.
	SecretKey string `json:"secret_key"`

	// Model is the Bedrock model ID.
	Model string `json:"model"`

	// BaseURL overrides the default Bedrock endpoint.
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

// Client implements provider.EmbeddingProvider for AWS Bedrock.
type Client struct {
	config     Config
	httpClient *http.Client
	dimension  int
	marshaler  jsonMarshaler
}

// titanRequest represents an AWS Titan embedding request.
type titanRequest struct {
	InputText string `json:"inputText"`
}

// titanResponse represents an AWS Titan embedding response.
type titanResponse struct {
	Embedding      []float32 `json:"embedding"`
	InputTextToken int       `json:"inputTextTokenCount"`
}

// cohereRequest represents a Bedrock Cohere embedding request.
type cohereRequest struct {
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type"`
}

// cohereResponse represents a Bedrock Cohere embedding response.
type cohereResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// NewClient creates a new AWS Bedrock embedding client.
func NewClient(config Config) *Client {
	if config.Region == "" {
		config.Region = DefaultRegion
	}
	if config.Model == "" {
		config.Model = DefaultModel
	}
	if config.BaseURL == "" {
		config.BaseURL = fmt.Sprintf(
			"https://bedrock-runtime.%s.amazonaws.com", config.Region)
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
	return fmt.Sprintf("bedrock/%s", c.config.Model)
}

// Dimensions returns the embedding vector dimensionality.
func (c *Client) Dimensions() int {
	return c.dimension
}

// Embed generates an embedding for a single text.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	if strings.HasPrefix(c.config.Model, "amazon.titan") {
		return c.embedTitan(ctx, text)
	}
	if strings.HasPrefix(c.config.Model, "cohere.") {
		embeddings, err := c.embedCohere(ctx, []string{text})
		if err != nil {
			return nil, err
		}
		if len(embeddings) == 0 {
			return nil, fmt.Errorf("bedrock: no embedding returned")
		}
		return embeddings[0], nil
	}
	return nil, fmt.Errorf("bedrock: unsupported model: %s", c.config.Model)
}

// EmbedBatch generates embeddings for multiple texts.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if strings.HasPrefix(c.config.Model, "cohere.") {
		return c.embedCohere(ctx, texts)
	}

	// Titan models do not support batch; call individually.
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := c.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("bedrock: failed to embed text %d: %w", i, err)
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

// embedTitan generates an embedding using a Titan model.
func (c *Client) embedTitan(ctx context.Context, text string) ([]float32, error) {
	reqBody := titanRequest{InputText: text}

	body, err := c.marshaler.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("bedrock: failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/model/%s/invoke", c.config.BaseURL, c.config.Model)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bedrock: failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	c.signRequest(req, body)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bedrock: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock: API error %s - %s", resp.Status, string(respBody))
	}

	var result titanResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("bedrock: failed to parse response: %w", err)
	}

	return result.Embedding, nil
}

// embedCohere generates embeddings using a Cohere model on Bedrock.
func (c *Client) embedCohere(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := cohereRequest{
		Texts:     texts,
		InputType: "search_document",
	}

	body, err := c.marshaler.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("bedrock: failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/model/%s/invoke", c.config.BaseURL, c.config.Model)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bedrock: failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	c.signRequest(req, body)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bedrock: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock: API error %s - %s", resp.Status, string(respBody))
	}

	var result cohereResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("bedrock: failed to parse response: %w", err)
	}

	return result.Embeddings, nil
}

// signRequest signs an HTTP request with AWS Signature Version 4.
func (c *Client) signRequest(req *http.Request, body []byte) {
	t := time.Now().UTC()
	amzDate := t.Format("20060102T150405Z")
	dateStamp := t.Format("20060102")

	hashedPayload := sha256Hash(body)

	canonicalHeaders := fmt.Sprintf(
		"content-type:%s\nhost:%s\nx-amz-date:%s\n",
		req.Header.Get("Content-Type"), req.URL.Host, amzDate)
	signedHeaders := "content-type;host;x-amz-date"

	canonicalRequest := fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s",
		req.Method, req.URL.Path, req.URL.RawQuery,
		canonicalHeaders, signedHeaders, hashedPayload)

	credentialScope := fmt.Sprintf(
		"%s/%s/bedrock/aws4_request", dateStamp, c.config.Region)
	stringToSign := fmt.Sprintf("AWS4-HMAC-SHA256\n%s\n%s\n%s",
		amzDate, credentialScope, sha256Hash([]byte(canonicalRequest)))

	kDate := hmacSHA256([]byte("AWS4"+c.config.SecretKey), dateStamp)
	kRegion := hmacSHA256(kDate, c.config.Region)
	kService := hmacSHA256(kRegion, "bedrock")
	kSigning := hmacSHA256(kService, "aws4_request")
	signature := hex.EncodeToString(hmacSHA256(kSigning, stringToSign))

	authHeader := fmt.Sprintf(
		"AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		c.config.AccessKey, credentialScope, signedHeaders, signature)

	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("Authorization", authHeader)
}

// sha256Hash computes the hex-encoded SHA-256 hash of data.
func sha256Hash(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// hmacSHA256 computes HMAC-SHA256.
func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}

// dimensionForModel returns the embedding dimension for the given model.
func dimensionForModel(model string) int {
	switch model {
	case "amazon.titan-embed-text-v1":
		return 1536
	case "amazon.titan-embed-text-v2:0":
		return 1024
	case "amazon.titan-embed-image-v1":
		return 1024
	case "cohere.embed-english-v3":
		return 1024
	case "cohere.embed-multilingual-v3":
		return 1024
	default:
		return 1536
	}
}

// Compile-time interface check.
var _ provider.EmbeddingProvider = (*Client)(nil)
