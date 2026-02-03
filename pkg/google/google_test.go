package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNewClient(t *testing.T) {
	tests := []struct {
		name              string
		config            Config
		expectedDimension int
	}{
		{
			name:              "default",
			config:            Config{ProjectID: "proj", APIKey: "k"},
			expectedDimension: 768,
		},
		{
			name:              "gecko",
			config:            Config{ProjectID: "p", APIKey: "k", Model: "textembedding-gecko@003"},
			expectedDimension: 768,
		},
		{
			name:              "multilingual",
			config:            Config{ProjectID: "p", APIKey: "k", Model: "text-multilingual-embedding-002"},
			expectedDimension: 768,
		},
		{
			name:              "embedding_004",
			config:            Config{ProjectID: "p", APIKey: "k", Model: "text-embedding-004"},
			expectedDimension: 768,
		},
		{
			name:              "gecko_multilingual",
			config:            Config{ProjectID: "p", APIKey: "k", Model: "textembedding-gecko-multilingual@001"},
			expectedDimension: 768,
		},
		{
			name:              "unknown",
			config:            Config{ProjectID: "p", APIKey: "k", Model: "unknown"},
			expectedDimension: 768,
		},
		{
			name:              "default_location",
			config:            Config{ProjectID: "p", APIKey: "k"},
			expectedDimension: 768,
		},
		{
			name:              "custom_location",
			config:            Config{ProjectID: "p", APIKey: "k", Location: "europe-west1"},
			expectedDimension: 768,
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
	client := NewClient(Config{ProjectID: "p", APIKey: "k"})
	assert.Equal(t, "google/text-embedding-005", client.Name())
}

func TestClient_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Contains(t, r.URL.Path, "predict")
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer ")

		response := embedResponse{
			Predictions: []struct {
				Embeddings struct {
					Values     []float32 `json:"values"`
					Statistics struct {
						TokenCount int `json:"token_count"`
					} `json:"statistics"`
				} `json:"embeddings"`
			}{
				{Embeddings: struct {
					Values     []float32 `json:"values"`
					Statistics struct {
						TokenCount int `json:"token_count"`
					} `json:"statistics"`
				}{Values: make([]float32, 768)}},
			},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		ProjectID: "test-project",
		Location:  "us-central1",
		APIKey:    "test-key",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	embedding, err := client.Embed(context.Background(), "test text")
	assert.NoError(t, err)
	assert.Len(t, embedding, 768)
}

func TestClient_EmbedBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embedResponse{
			Predictions: []struct {
				Embeddings struct {
					Values     []float32 `json:"values"`
					Statistics struct {
						TokenCount int `json:"token_count"`
					} `json:"statistics"`
				} `json:"embeddings"`
			}{
				{Embeddings: struct {
					Values     []float32 `json:"values"`
					Statistics struct {
						TokenCount int `json:"token_count"`
					} `json:"statistics"`
				}{Values: make([]float32, 768)}},
				{Embeddings: struct {
					Values     []float32 `json:"values"`
					Statistics struct {
						TokenCount int `json:"token_count"`
					} `json:"statistics"`
				}{Values: make([]float32, 768)}},
			},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		ProjectID: "test-project",
		Location:  "us-central1",
		APIKey:    "test-key",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	embeddings, err := client.EmbedBatch(context.Background(), []string{"a", "b"})
	assert.NoError(t, err)
	assert.Len(t, embeddings, 2)
}

func TestClient_Embed_NoEmbeddingReturned(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := embedResponse{
			Predictions: []struct {
				Embeddings struct {
					Values     []float32 `json:"values"`
					Statistics struct {
						TokenCount int `json:"token_count"`
					} `json:"statistics"`
				} `json:"embeddings"`
			}{},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		ProjectID: "p",
		APIKey:    "k",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
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
		{"forbidden", http.StatusForbidden, "403"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				_, _ = w.Write([]byte(`{"error": "test"}`))
			}))
			defer server.Close()

			client := NewClient(Config{
				ProjectID: "p",
				APIKey:    "k",
				BaseURL:   server.URL,
				Timeout:   5 * time.Second,
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
		ProjectID: "p",
		APIKey:    "k",
		BaseURL:   server.URL,
		Timeout:   100 * time.Millisecond,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Embed(ctx, "test")
	assert.Error(t, err)
}

func TestClient_RequestBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		assert.Len(t, req.Instances, 1)
		assert.Equal(t, "hello world", req.Instances[0].Content)
		assert.Equal(t, "RETRIEVAL_DOCUMENT", req.Instances[0].TaskType)

		response := embedResponse{
			Predictions: []struct {
				Embeddings struct {
					Values     []float32 `json:"values"`
					Statistics struct {
						TokenCount int `json:"token_count"`
					} `json:"statistics"`
				} `json:"embeddings"`
			}{
				{Embeddings: struct {
					Values     []float32 `json:"values"`
					Statistics struct {
						TokenCount int `json:"token_count"`
					} `json:"statistics"`
				}{Values: make([]float32, 768)}},
			},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		ProjectID: "p",
		APIKey:    "k",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "hello world")
	assert.NoError(t, err)
}

func TestClient_EmbedBatch_InvalidURL(t *testing.T) {
	// Control characters in URL cause http.NewRequestWithContext to fail
	client := NewClient(Config{
		ProjectID: "p",
		APIKey:    "k",
		BaseURL:   "http://example.com\x00invalid",
		Timeout:   5 * time.Second,
	})

	_, err := client.EmbedBatch(context.Background(), []string{"test"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create request")
}

func TestClient_EmbedBatch_RequestFailure(t *testing.T) {
	client := NewClient(Config{
		ProjectID: "p",
		APIKey:    "k",
		BaseURL:   "http://invalid-host-that-does-not-exist.local:99999",
		Timeout:   100 * time.Millisecond,
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
		ProjectID: "p",
		APIKey:    "k",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
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
		ProjectID: "p",
		APIKey:    "k",
		BaseURL:   "http://invalid-host-that-does-not-exist.local:99999",
		Timeout:   100 * time.Millisecond,
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
		ProjectID: "p",
		APIKey:    "k",
		Timeout:   5 * time.Second,
	})
	client.marshaler = mockMarshaler{err: fmt.Errorf("mock marshal error")}

	_, err := client.EmbedBatch(context.Background(), []string{"test"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to marshal request")
}
