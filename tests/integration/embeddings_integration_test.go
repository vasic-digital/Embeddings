package integration

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"digital.vasic.embeddings/pkg/openai"
	"digital.vasic.embeddings/pkg/provider"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIClient_InterfaceCompliance_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")  // SKIP-OK: #short-mode
	}

	// Verify that OpenAI Client implements EmbeddingProvider at compile time
	var _ provider.EmbeddingProvider = (*openai.Client)(nil)

	client := openai.NewClient(openai.Config{
		APIKey: "test-key",
		Model:  "text-embedding-3-small",
	})
	assert.NotNil(t, client)
	assert.Equal(t, "openai/text-embedding-3-small", client.Name())
	assert.Equal(t, 1536, client.Dimensions())
}

func TestOpenAIClient_ModelDimensions_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")  // SKIP-OK: #short-mode
	}

	tests := []struct {
		model      string
		dimensions int
	}{
		{"text-embedding-3-small", 1536},
		{"text-embedding-3-large", 3072},
		{"text-embedding-ada-002", 1536},
		{"unknown-model", 1536}, // default fallback
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			client := openai.NewClient(openai.Config{
				APIKey: "test-key",
				Model:  tc.model,
			})
			assert.Equal(t, tc.dimensions, client.Dimensions())
		})
	}
}

func TestOpenAIClient_DefaultConfig_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")  // SKIP-OK: #short-mode
	}

	// Empty model and base URL should use defaults
	client := openai.NewClient(openai.Config{
		APIKey: "test-key",
	})
	assert.Equal(t, "openai/text-embedding-3-small", client.Name())
	assert.Equal(t, 1536, client.Dimensions())
}

func TestOpenAIClient_EmbedBatch_WithMockServer_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")  // SKIP-OK: #short-mode
	}

	// Create a mock HTTP server that returns embedding responses
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, http.MethodPost, r.Method)
			assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
			assert.Contains(t, r.Header.Get("Authorization"), "Bearer ")

			response := map[string]interface{}{
				"data": []map[string]interface{}{
					{
						"embedding": []float32{0.1, 0.2, 0.3},
						"index":     0,
					},
					{
						"embedding": []float32{0.4, 0.5, 0.6},
						"index":     1,
					},
				},
				"model": "text-embedding-3-small",
				"usage": map[string]int{
					"prompt_tokens": 10,
					"total_tokens":  10,
				},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(response)
		},
	))
	defer server.Close()

	client := openai.NewClient(openai.Config{
		APIKey:  "test-key",
		Model:   "text-embedding-3-small",
		BaseURL: server.URL,
	})

	ctx := t.Context()
	embeddings, err := client.EmbedBatch(ctx, []string{"hello", "world"})
	require.NoError(t, err)
	assert.Equal(t, 2, len(embeddings))
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, embeddings[0])
	assert.Equal(t, []float32{0.4, 0.5, 0.6}, embeddings[1])
}

func TestOpenAIClient_Embed_WithMockServer_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")  // SKIP-OK: #short-mode
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			response := map[string]interface{}{
				"data": []map[string]interface{}{
					{
						"embedding": []float32{0.7, 0.8, 0.9},
						"index":     0,
					},
				},
				"model": "text-embedding-3-small",
				"usage": map[string]int{
					"prompt_tokens": 5,
					"total_tokens":  5,
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
	embedding, err := client.Embed(ctx, "test input")
	require.NoError(t, err)
	assert.Equal(t, []float32{0.7, 0.8, 0.9}, embedding)
}

func TestProviderConfig_Defaults_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")  // SKIP-OK: #short-mode
	}

	cfg := provider.DefaultConfig()
	assert.Equal(t, 100, cfg.BatchSize)
	assert.Equal(t, 3, cfg.MaxRetries)
	assert.Greater(t, cfg.Timeout.Seconds(), float64(0))
}

func TestProviderResult_Structure_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")  // SKIP-OK: #short-mode
	}

	result := provider.Result{
		Embeddings: [][]float32{{0.1, 0.2}, {0.3, 0.4}},
		Model:      "test-model",
		Usage: provider.TokenUsage{
			PromptTokens: 20,
			TotalTokens:  20,
		},
	}

	assert.Equal(t, 2, len(result.Embeddings))
	assert.Equal(t, "test-model", result.Model)
	assert.Equal(t, 20, result.Usage.PromptTokens)

	// Verify JSON serialization
	data, err := json.Marshal(result)
	require.NoError(t, err)

	var decoded provider.Result
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)
	assert.Equal(t, result.Model, decoded.Model)
	assert.Equal(t, result.Usage.TotalTokens, decoded.Usage.TotalTokens)
}
