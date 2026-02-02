package jina

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
		{"v2-base-en", Config{APIKey: "k", Model: "jina-embeddings-v2-base-en"}, 768},
		{"v2-small-en", Config{APIKey: "k", Model: "jina-embeddings-v2-small-en"}, 512},
		{"v2-base-de", Config{APIKey: "k", Model: "jina-embeddings-v2-base-de"}, 768},
		{"v2-base-es", Config{APIKey: "k", Model: "jina-embeddings-v2-base-es"}, 768},
		{"v2-base-zh", Config{APIKey: "k", Model: "jina-embeddings-v2-base-zh"}, 768},
		{"clip-v1", Config{APIKey: "k", Model: "jina-clip-v1"}, 768},
		{"colbert-v2", Config{APIKey: "k", Model: "jina-colbert-v2"}, 128},
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
	client := NewClient(Config{APIKey: "k", Model: "jina-embeddings-v3"})
	assert.Equal(t, "jina/jina-embeddings-v3", client.Name())
}

func TestClient_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/embeddings", r.URL.Path)
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer ")

		var req embedRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "jina-embeddings-v3", req.Model)
		assert.Equal(t, "retrieval.document", req.Task)
		assert.Equal(t, "float", req.EncodingFormat)

		response := embedResponse{
			Model: "jina-embeddings-v3",
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Object: "embedding", Index: 0, Embedding: make([]float32, 1024)},
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
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Index: 0, Embedding: make([]float32, 1024)},
				{Index: 1, Embedding: make([]float32, 1024)},
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
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
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
		{"forbidden", http.StatusForbidden, "403"},
		{"server_error", http.StatusInternalServerError, "500"},
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

func TestClient_CustomTask(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		assert.Equal(t, "retrieval.query", req.Task)

		response := embedResponse{
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Index: 0, Embedding: make([]float32, 1024)},
			},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		APIKey:  "key",
		BaseURL: server.URL,
		Task:    "retrieval.query",
		Timeout: 5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "query")
	assert.NoError(t, err)
}
