package cohere

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
		expectedDimension int
	}{
		{
			name:              "default_values",
			config:            Config{APIKey: "test-key"},
			expectedModel:     DefaultModel,
			expectedDimension: 1024,
		},
		{
			name:              "multilingual_v3",
			config:            Config{APIKey: "k", Model: "embed-multilingual-v3.0"},
			expectedModel:     "embed-multilingual-v3.0",
			expectedDimension: 1024,
		},
		{
			name:              "light_v3",
			config:            Config{APIKey: "k", Model: "embed-english-light-v3.0"},
			expectedModel:     "embed-english-light-v3.0",
			expectedDimension: 384,
		},
		{
			name:              "multilingual_light_v3",
			config:            Config{APIKey: "k", Model: "embed-multilingual-light-v3.0"},
			expectedDimension: 384,
		},
		{
			name:              "english_v2",
			config:            Config{APIKey: "k", Model: "embed-english-v2.0"},
			expectedDimension: 4096,
		},
		{
			name:              "multilingual_v2",
			config:            Config{APIKey: "k", Model: "embed-multilingual-v2.0"},
			expectedDimension: 768,
		},
		{
			name:              "unknown_model",
			config:            Config{APIKey: "k", Model: "unknown"},
			expectedDimension: 1024,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(tt.config)
			assert.NotNil(t, client)
			assert.Equal(t, tt.expectedDimension, client.Dimensions())
		})
	}
}

func TestClient_Name(t *testing.T) {
	client := NewClient(Config{APIKey: "k", Model: "embed-english-v3.0"})
	assert.Equal(t, "cohere/embed-english-v3.0", client.Name())
}

func TestClient_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/embed", r.URL.Path)
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer ")

		var req embedRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "embed-english-v3.0", req.Model)
		assert.Equal(t, "search_document", req.InputType)
		assert.Equal(t, "END", req.Truncate)

		response := embedResponse{
			ID:         "test-id",
			Embeddings: [][]float32{make([]float32, 1024)},
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
	assert.Len(t, embedding, 1024)
}

func TestClient_EmbedBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embedResponse{
			Embeddings: [][]float32{
				make([]float32, 1024),
				make([]float32, 1024),
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
}

func TestClient_Embed_EmbeddingsObjFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"id": "test-id",
			"embeddings_by_type": map[string]interface{}{
				"float": [][]float32{make([]float32, 1024)},
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

	embedding, err := client.Embed(context.Background(), "test")
	assert.NoError(t, err)
	assert.Len(t, embedding, 1024)
}

func TestClient_Embed_NoEmbeddingReturned(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{"id": "test-id"}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no embedding")
}

func TestClient_Embed_APIError(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		errContain string
	}{
		{"unauthorized", http.StatusUnauthorized, "401"},
		{"rate_limited", http.StatusTooManyRequests, "429"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				_, _ = w.Write([]byte(`{"error": "test"}`))
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

	_, err := client.Embed(ctx, "test")
	assert.Error(t, err)
}

func TestClient_CustomInputType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		assert.Equal(t, "search_query", req.InputType)

		response := embedResponse{
			Embeddings: [][]float32{make([]float32, 1024)},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:    "test-key",
		BaseURL:   server.URL,
		InputType: "search_query",
		Timeout:   5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "query text")
	assert.NoError(t, err)
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

// =========================================================================
// Additional Tests for 100% Coverage
// =========================================================================

func TestClient_Embed_EmptyEmbeddingsReturned(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embedResponse{
			ID:         "test-id",
			Embeddings: [][]float32{},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no embedding")
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
