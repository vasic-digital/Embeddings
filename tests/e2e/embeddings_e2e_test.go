package e2e

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"digital.vasic.embeddings/pkg/openai"
	"digital.vasic.embeddings/pkg/provider"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIClient_FullEmbeddingLifecycle_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	mockEmbeddings := map[string][]float32{
		"hello": {0.1, 0.2, 0.3, 0.4},
		"world": {0.5, 0.6, 0.7, 0.8},
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Input []string `json:"input"`
				Model string   `json:"model"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)

			data := make([]map[string]interface{}, len(req.Input))
			for i, text := range req.Input {
				emb, ok := mockEmbeddings[text]
				if !ok {
					emb = []float32{0.0, 0.0, 0.0, 0.0}
				}
				data[i] = map[string]interface{}{
					"embedding": emb,
					"index":     i,
				}
			}

			response := map[string]interface{}{
				"data":  data,
				"model": req.Model,
				"usage": map[string]int{
					"prompt_tokens": len(req.Input) * 5,
					"total_tokens":  len(req.Input) * 5,
				},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(response)
		},
	))
	defer server.Close()

	client := openai.NewClient(openai.Config{
		APIKey:  "test-api-key",
		Model:   "text-embedding-3-small",
		BaseURL: server.URL,
		Timeout: 10 * time.Second,
	})

	// Step 1: Single embed
	ctx := t.Context()
	embedding, err := client.Embed(ctx, "hello")
	require.NoError(t, err)
	assert.Equal(t, []float32{0.1, 0.2, 0.3, 0.4}, embedding)

	// Step 2: Batch embed
	embeddings, err := client.EmbedBatch(ctx, []string{"hello", "world"})
	require.NoError(t, err)
	assert.Equal(t, 2, len(embeddings))
	assert.Equal(t, []float32{0.1, 0.2, 0.3, 0.4}, embeddings[0])
	assert.Equal(t, []float32{0.5, 0.6, 0.7, 0.8}, embeddings[1])

	// Step 3: Verify provider metadata
	assert.Equal(t, "openai/text-embedding-3-small", client.Name())
	assert.Equal(t, 1536, client.Dimensions())
}

func TestOpenAIClient_APIErrorHandling_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error": {"message": "invalid api key"}}`))
		},
	))
	defer server.Close()

	client := openai.NewClient(openai.Config{
		APIKey:  "invalid-key",
		BaseURL: server.URL,
	})

	ctx := t.Context()
	_, err := client.Embed(ctx, "test")
	assert.Error(t, err, "should return error for unauthorized request")
	assert.Contains(t, err.Error(), "401")
}

func TestOpenAIClient_EmptyResponse_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			response := map[string]interface{}{
				"data":  []interface{}{},
				"model": "text-embedding-3-small",
				"usage": map[string]int{
					"prompt_tokens": 0,
					"total_tokens":  0,
				},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(response)
		},
	))
	defer server.Close()

	client := openai.NewClient(openai.Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := t.Context()
	_, err := client.Embed(ctx, "test")
	assert.Error(t, err, "empty response should return error")
}

func TestProviderConfigSerialization_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	cfg := provider.Config{
		Model:      "text-embedding-3-large",
		BatchSize:  50,
		MaxRetries: 5,
		Timeout:    60 * time.Second,
	}

	data, err := json.Marshal(cfg)
	require.NoError(t, err)

	var decoded provider.Config
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)
	assert.Equal(t, cfg.Model, decoded.Model)
	assert.Equal(t, cfg.BatchSize, decoded.BatchSize)
	assert.Equal(t, cfg.MaxRetries, decoded.MaxRetries)
}

func TestOpenAIClient_ServerError_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error": {"message": "internal server error"}}`))
		},
	))
	defer server.Close()

	client := openai.NewClient(openai.Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := t.Context()
	_, err := client.EmbedBatch(ctx, []string{"test"})
	assert.Error(t, err, "server error should be propagated")
}

func TestTokenUsage_Structure_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	usage := provider.TokenUsage{
		PromptTokens: 150,
		TotalTokens:  150,
	}

	data, err := json.Marshal(usage)
	require.NoError(t, err)

	var decoded provider.TokenUsage
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)
	assert.Equal(t, 150, decoded.PromptTokens)
	assert.Equal(t, 150, decoded.TotalTokens)
}
