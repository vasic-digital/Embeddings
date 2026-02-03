package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	tests := []struct {
		name              string
		config            Config
		expectedModel     string
		expectedBaseURL   string
		expectedDimension int
	}{
		{
			name:              "default_values",
			config:            Config{APIKey: "test-key"},
			expectedModel:     DefaultModel,
			expectedBaseURL:   DefaultBaseURL,
			expectedDimension: 1536,
		},
		{
			name: "custom_model_3_large",
			config: Config{
				APIKey: "test-key",
				Model:  "text-embedding-3-large",
			},
			expectedModel:     "text-embedding-3-large",
			expectedBaseURL:   DefaultBaseURL,
			expectedDimension: 3072,
		},
		{
			name: "ada_002",
			config: Config{
				APIKey: "test-key",
				Model:  "text-embedding-ada-002",
			},
			expectedModel:     "text-embedding-ada-002",
			expectedBaseURL:   DefaultBaseURL,
			expectedDimension: 1536,
		},
		{
			name: "custom_base_url",
			config: Config{
				APIKey:  "test-key",
				BaseURL: "https://custom.api.com/v1",
			},
			expectedModel:     DefaultModel,
			expectedBaseURL:   "https://custom.api.com/v1",
			expectedDimension: 1536,
		},
		{
			name: "unknown_model_defaults",
			config: Config{
				APIKey: "test-key",
				Model:  "unknown-model",
			},
			expectedModel:     "unknown-model",
			expectedDimension: 1536,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(tt.config)
			assert.NotNil(t, client)
			assert.Equal(t, tt.expectedDimension, client.Dimensions())
			if tt.expectedBaseURL != "" {
				assert.Contains(t, client.Name(), tt.expectedModel)
			}
		})
	}
}

func TestClient_Name(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		expected string
	}{
		{"small", "text-embedding-3-small", "openai/text-embedding-3-small"},
		{"large", "text-embedding-3-large", "openai/text-embedding-3-large"},
		{"ada", "text-embedding-ada-002", "openai/text-embedding-ada-002"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(Config{APIKey: "key", Model: tt.model})
			assert.Equal(t, tt.expected, client.Name())
		})
	}
}

func TestClient_Dimensions(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		expected int
	}{
		{"text-embedding-3-small", "text-embedding-3-small", 1536},
		{"text-embedding-3-large", "text-embedding-3-large", 3072},
		{"text-embedding-ada-002", "text-embedding-ada-002", 1536},
		{"unknown_model", "unknown", 1536},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(Config{APIKey: "key", Model: tt.model})
			assert.Equal(t, tt.expected, client.Dimensions())
		})
	}
}

func TestClient_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/embeddings", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))

		var req embedRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "text-embedding-3-small", req.Model)
		assert.Equal(t, []string{"test text"}, req.Input)

		response := embedResponse{
			Data: []struct {
				Embedding []float32 `json:"embedding"`
				Index     int       `json:"index"`
			}{
				{Embedding: make([]float32, 1536), Index: 0},
			},
			Model: "text-embedding-3-small",
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	})

	embedding, err := client.Embed(context.Background(), "test text")

	assert.NoError(t, err)
	assert.Len(t, embedding, 1536)
}

func TestClient_EmbedBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embedResponse{
			Data: []struct {
				Embedding []float32 `json:"embedding"`
				Index     int       `json:"index"`
			}{
				{Embedding: make([]float32, 1536), Index: 0},
				{Embedding: make([]float32, 1536), Index: 1},
				{Embedding: make([]float32, 1536), Index: 2},
			},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	})

	embeddings, err := client.EmbedBatch(context.Background(), []string{"a", "b", "c"})

	assert.NoError(t, err)
	assert.Len(t, embeddings, 3)
	for _, emb := range embeddings {
		assert.Len(t, emb, 1536)
	}
}

func TestClient_Embed_NoEmbeddingReturned(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embedResponse{
			Data: []struct {
				Embedding []float32 `json:"embedding"`
				Index     int       `json:"index"`
			}{},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "test text")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no embedding returned")
}

func TestClient_Embed_APIError(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       string
		errContain string
	}{
		{
			name:       "unauthorized",
			statusCode: http.StatusUnauthorized,
			body:       `{"error": "invalid_api_key"}`,
			errContain: "401",
		},
		{
			name:       "rate_limited",
			statusCode: http.StatusTooManyRequests,
			body:       `{"error": "rate limit"}`,
			errContain: "429",
		},
		{
			name:       "server_error",
			statusCode: http.StatusInternalServerError,
			body:       `{"error": "internal error"}`,
			errContain: "500",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			client := NewClient(Config{
				APIKey:  "test-key",
				BaseURL: server.URL,
				Timeout: 5 * time.Second,
			})

			_, err := client.Embed(context.Background(), "test")
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.errContain)
		})
	}
}

func TestClient_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Timeout: 100 * time.Millisecond,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Embed(ctx, "test text")
	assert.Error(t, err)
}

func TestClient_EmbedBatch_IndexOrdering(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return in reverse order to test index-based placement
		response := embedResponse{
			Data: []struct {
				Embedding []float32 `json:"embedding"`
				Index     int       `json:"index"`
			}{
				{Embedding: []float32{2.0}, Index: 1},
				{Embedding: []float32{1.0}, Index: 0},
			},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	})

	embeddings, err := client.EmbedBatch(context.Background(), []string{"a", "b"})
	assert.NoError(t, err)
	assert.Len(t, embeddings, 2)
	assert.Equal(t, []float32{1.0}, embeddings[0])
	assert.Equal(t, []float32{2.0}, embeddings[1])
}

func TestClient_EmbedBatch_InvalidURL(t *testing.T) {
	// Control characters in URL cause http.NewRequestWithContext to fail
	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: "http://example.com\x00invalid",
		Timeout: 5 * time.Second,
	})

	_, err := client.EmbedBatch(context.Background(), []string{"test"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create request")
}

func TestClient_EmbedBatch_RequestFailure(t *testing.T) {
	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: "http://invalid-host-that-does-not-exist.local:99999",
		Timeout: 100 * time.Millisecond,
	})

	_, err := client.EmbedBatch(context.Background(), []string{"test"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "request failed")
}

func TestClient_EmbedBatch_JSONDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{invalid json`))
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	})

	_, err := client.EmbedBatch(context.Background(), []string{"test"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse response")
}

// =========================================================================
// Additional Tests for 100% Coverage
// =========================================================================

func TestClient_Embed_ReturnError(t *testing.T) {
	// Test that Embed properly propagates errors from EmbedBatch
	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: "http://invalid-host-that-does-not-exist.local:99999",
		Timeout: 100 * time.Millisecond,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "request failed")
}

// mockMarshaler is a test marshaler that can be configured to return errors.
type mockMarshaler struct {
	err error
}

func (m mockMarshaler) Marshal(v interface{}) ([]byte, error) {
	if m.err != nil {
		return nil, m.err
	}
	return json.Marshal(v)
}

func TestClient_EmbedBatch_MarshalError(t *testing.T) {
	client := NewClient(Config{
		APIKey:  "test-key",
		Timeout: 5 * time.Second,
	})
	client.marshaler = mockMarshaler{err: fmt.Errorf("mock marshal error")}

	_, err := client.EmbedBatch(context.Background(), []string{"test"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to marshal request")
}
