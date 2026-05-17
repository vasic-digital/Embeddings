package integration

import (
	"encoding/json"
	"os"
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

// TestOpenAIClient_EmbedBatch_LiveAPI_Integration: per CONST-050(A),
// integration tests MUST exercise the real backing service.
// Previously this test stood up a `httptest.NewServer` with
// hardcoded float-vector responses — CONST-050(A) violation
// (mock servers in integration tests certify the JSON serialisation
// path but cannot detect wrong endpoint, auth-header format,
// model-name mismatches, token-counting errors, batch-size limits —
// all of which would appear to end users as broken features).
//
// Fix per CLAUDE.md "Acceptance demo" guidance ("Without
// OPENAI_API_KEY the live tests skip — that's OK per DoD"):
// SKIP-OK when no real API key set; live HTTP POST to OpenAI when
// OPENAI_API_KEY is provided. Unit-level JSON-serialisation
// coverage stays in pkg/openai/openai_test.go where mock servers
// are CONST-050(A)-permitted.
func TestOpenAIClient_EmbedBatch_LiveAPI_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode") // SKIP-OK: #short-mode
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("SKIP-OK: #embeddings-live-key-required — OPENAI_API_KEY env var not set; integration tests MUST hit the real provider per CONST-050(A), and mock-server fakes are restricted to unit tests (see pkg/openai/openai_test.go)")
	}
	client := openai.NewClient(openai.Config{
		APIKey: apiKey,
		Model:  "text-embedding-3-small",
	})

	ctx := t.Context()
	embeddings, err := client.EmbedBatch(ctx, []string{"hello", "world"})
	require.NoError(t, err)
	require.Equal(t, 2, len(embeddings))
	// Real embeddings have client.Dimensions() (1536 for
	// text-embedding-3-small) float32 values per item.
	assert.Equal(t, client.Dimensions(), len(embeddings[0]))
	assert.Equal(t, client.Dimensions(), len(embeddings[1]))
}

// TestOpenAIClient_Embed_LiveAPI_Integration: SKIP-OK pattern per
// CONST-050(A); see above.
func TestOpenAIClient_Embed_LiveAPI_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode") // SKIP-OK: #short-mode
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("SKIP-OK: #embeddings-live-key-required — OPENAI_API_KEY env var not set; integration tests MUST hit the real provider per CONST-050(A)")
	}
	client := openai.NewClient(openai.Config{
		APIKey: apiKey,
	})

	ctx := t.Context()
	embedding, err := client.Embed(ctx, "test input")
	require.NoError(t, err)
	assert.Equal(t, client.Dimensions(), len(embedding))
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
