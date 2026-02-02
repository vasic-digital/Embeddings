package voyage

import (
	"context"
	"encoding/json"
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
		expectedDimension int
	}{
		{"default", Config{APIKey: "k"}, 1024},
		{"voyage-3-lite", Config{APIKey: "k", Model: "voyage-3-lite"}, 512},
		{"voyage-code-3", Config{APIKey: "k", Model: "voyage-code-3"}, 1024},
		{"voyage-finance-2", Config{APIKey: "k", Model: "voyage-finance-2"}, 1024},
		{"voyage-law-2", Config{APIKey: "k", Model: "voyage-law-2"}, 1024},
		{"voyage-large-2", Config{APIKey: "k", Model: "voyage-large-2"}, 1536},
		{"voyage-large-2-instruct", Config{APIKey: "k", Model: "voyage-large-2-instruct"}, 1536},
		{"voyage-2", Config{APIKey: "k", Model: "voyage-2"}, 1024},
		{"unknown", Config{APIKey: "k", Model: "unknown"}, 1024},
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
	client := NewClient(Config{APIKey: "k", Model: "voyage-3"})
	assert.Equal(t, "voyage/voyage-3", client.Name())
}

func TestClient_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/embeddings", r.URL.Path)
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer ")

		var req embedRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "voyage-3", req.Model)
		assert.Equal(t, "document", req.InputType)
		assert.True(t, req.Truncation)

		response := embedResponse{
			Object: "list",
			Data: []struct {
				Object    string    `json:"object"`
				Embedding []float32 `json:"embedding"`
				Index     int       `json:"index"`
			}{
				{Object: "embedding", Embedding: make([]float32, 1024), Index: 0},
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

	embedding, err := client.Embed(context.Background(), "test text")
	assert.NoError(t, err)
	assert.Len(t, embedding, 1024)
}

func TestClient_EmbedBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embedResponse{
			Data: []struct {
				Object    string    `json:"object"`
				Embedding []float32 `json:"embedding"`
				Index     int       `json:"index"`
			}{
				{Embedding: make([]float32, 1024), Index: 0},
				{Embedding: make([]float32, 1024), Index: 1},
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

func TestClient_Embed_NoEmbeddingReturned(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embedResponse{
			Data: []struct {
				Object    string    `json:"object"`
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
		{"rate_limited", http.StatusTooManyRequests, "429"},
		{"unauthorized", http.StatusUnauthorized, "401"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				_, _ = w.Write([]byte(`{"error": "test"}`))
			}))
			defer server.Close()

			client := NewClient(Config{
				APIKey:  "key",
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
		APIKey:  "key",
		BaseURL: server.URL,
		Timeout: 100 * time.Millisecond,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Embed(ctx, "test")
	assert.Error(t, err)
}

func TestClient_EmbedBatch_IndexOrdering(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embedResponse{
			Data: []struct {
				Object    string    `json:"object"`
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
		APIKey:  "key",
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	})

	embeddings, err := client.EmbedBatch(context.Background(), []string{"a", "b"})
	assert.NoError(t, err)
	assert.Equal(t, []float32{1.0}, embeddings[0])
	assert.Equal(t, []float32{2.0}, embeddings[1])
}
