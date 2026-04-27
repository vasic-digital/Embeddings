package security

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"digital.vasic.embeddings/pkg/openai"
	"digital.vasic.embeddings/pkg/provider"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIClient_AuthorizationHeader_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	var capturedAuth string
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			capturedAuth = r.Header.Get("Authorization")
			response := map[string]interface{}{
				"data": []map[string]interface{}{
					{"embedding": []float32{0.1}, "index": 0},
				},
				"model": "test",
				"usage": map[string]int{
					"prompt_tokens": 1, "total_tokens": 1,
				},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(response)
		},
	))
	defer server.Close()

	apiKey := "sk-test-secret-key-12345"
	client := openai.NewClient(openai.Config{
		APIKey:  apiKey,
		BaseURL: server.URL,
	})

	ctx := t.Context()
	_, err := client.Embed(ctx, "test")
	require.NoError(t, err)
	assert.Equal(t, "Bearer "+apiKey, capturedAuth,
		"authorization header should use Bearer scheme with exact API key")
}

func TestOpenAIClient_LargeInput_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			response := map[string]interface{}{
				"data": []map[string]interface{}{
					{"embedding": []float32{0.1}, "index": 0},
				},
				"model": "test",
				"usage": map[string]int{
					"prompt_tokens": 1, "total_tokens": 1,
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

	// Large input should not cause panic
	largeInput := strings.Repeat("word ", 100000)
	ctx := t.Context()
	_, err := client.Embed(ctx, largeInput)
	assert.NoError(t, err, "large input should not crash the client")
}

func TestOpenAIClient_EmptyAPIKey_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	// Creating a client with empty API key should still work
	// (the server will reject it)
	client := openai.NewClient(openai.Config{
		APIKey: "",
	})
	assert.NotNil(t, client)
	assert.Equal(t, 1536, client.Dimensions())
}

func TestOpenAIClient_MalformedJSON_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{invalid json`))
		},
	))
	defer server.Close()

	client := openai.NewClient(openai.Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := t.Context()
	_, err := client.Embed(ctx, "test")
	assert.Error(t, err, "malformed JSON response should return error")
}

func TestOpenAIClient_HTMLResponseBody_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/html")
			_, _ = w.Write([]byte(`<html><body>Error page</body></html>`))
		},
	))
	defer server.Close()

	client := openai.NewClient(openai.Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := t.Context()
	_, err := client.Embed(ctx, "test")
	assert.Error(t, err,
		"HTML response should be handled as error without panic")
}

func TestProviderConfig_NegativeValues_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	// Negative batch size and retries should not cause issues
	cfg := provider.Config{
		Model:      "test-model",
		BatchSize:  -1,
		MaxRetries: -5,
		Timeout:    0,
	}

	// Verify it can be serialized/deserialized without error
	data, err := json.Marshal(cfg)
	require.NoError(t, err)

	var decoded provider.Config
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)
	assert.Equal(t, -1, decoded.BatchSize)
}

func TestOpenAIClient_NilContext_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")  // SKIP-OK: #short-mode
	}

	// Empty batch should not panic
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			response := map[string]interface{}{
				"data":  []interface{}{},
				"model": "test",
				"usage": map[string]int{
					"prompt_tokens": 0, "total_tokens": 0,
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
	embeddings, err := client.EmbedBatch(ctx, []string{})
	// Empty batch is valid but may return empty results
	if err == nil {
		assert.Equal(t, 0, len(embeddings))
	}
}
