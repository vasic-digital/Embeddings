package bedrock

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
		{"default", Config{AccessKey: "a", SecretKey: "s"}, 1024},
		{"titan_v1", Config{AccessKey: "a", SecretKey: "s", Model: "amazon.titan-embed-text-v1"}, 1536},
		{"titan_v2", Config{AccessKey: "a", SecretKey: "s", Model: "amazon.titan-embed-text-v2:0"}, 1024},
		{"titan_image", Config{AccessKey: "a", SecretKey: "s", Model: "amazon.titan-embed-image-v1"}, 1024},
		{"cohere_english", Config{AccessKey: "a", SecretKey: "s", Model: "cohere.embed-english-v3"}, 1024},
		{"cohere_multilingual", Config{AccessKey: "a", SecretKey: "s", Model: "cohere.embed-multilingual-v3"}, 1024},
		{"unknown", Config{AccessKey: "a", SecretKey: "s", Model: "unknown"}, 1536},
		{"default_region", Config{AccessKey: "a", SecretKey: "s"}, 1024},
		{"custom_region", Config{AccessKey: "a", SecretKey: "s", Region: "eu-west-1"}, 1024},
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
	client := NewClient(Config{AccessKey: "a", SecretKey: "s"})
	assert.Equal(t, "bedrock/amazon.titan-embed-text-v2:0", client.Name())
}

func TestClient_Embed_Titan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Contains(t, r.URL.Path, "invoke")

		response := titanResponse{
			Embedding:      make([]float32, 1024),
			InputTextToken: 5,
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		Region:    "us-east-1",
		AccessKey: "test-key",
		SecretKey: "test-secret",
		Model:     "amazon.titan-embed-text-v2:0",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	embedding, err := client.Embed(context.Background(), "test text")
	assert.NoError(t, err)
	assert.Len(t, embedding, 1024)
}

func TestClient_Embed_Cohere(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := cohereResponse{
			Embeddings: [][]float32{make([]float32, 1024)},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		Region:    "us-east-1",
		AccessKey: "test-key",
		SecretKey: "test-secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	embedding, err := client.Embed(context.Background(), "test text")
	assert.NoError(t, err)
	assert.Len(t, embedding, 1024)
}

func TestClient_EmbedBatch_Titan(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		response := titanResponse{
			Embedding:      make([]float32, 1024),
			InputTextToken: 5,
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		Region:    "us-east-1",
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "amazon.titan-embed-text-v2:0",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	embeddings, err := client.EmbedBatch(context.Background(), []string{"a", "b"})
	assert.NoError(t, err)
	assert.Len(t, embeddings, 2)
	assert.Equal(t, 2, callCount)
}

func TestClient_EmbedBatch_Cohere(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := cohereResponse{
			Embeddings: [][]float32{
				make([]float32, 1024),
				make([]float32, 1024),
			},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		Region:    "us-east-1",
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	embeddings, err := client.EmbedBatch(context.Background(), []string{"a", "b"})
	assert.NoError(t, err)
	assert.Len(t, embeddings, 2)
}

func TestClient_Embed_UnsupportedModel(t *testing.T) {
	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "unsupported-model",
		Timeout:   5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported model")
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
				_, _ = w.Write([]byte(`{"message": "error"}`))
			}))
			defer server.Close()

			client := NewClient(Config{
				AccessKey: "key",
				SecretKey: "secret",
				Model:     "amazon.titan-embed-text-v2:0",
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
		AccessKey: "key",
		SecretKey: "secret",
		BaseURL:   server.URL,
		Timeout:   100 * time.Millisecond,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Embed(ctx, "test")
	assert.Error(t, err)
}

func TestSHA256Hash(t *testing.T) {
	hash := sha256Hash([]byte("test"))
	assert.Len(t, hash, 64)
	assert.NotEmpty(t, hash)
}

func TestHmacSHA256(t *testing.T) {
	result := hmacSHA256([]byte("key"), "data")
	assert.NotEmpty(t, result)
	assert.Len(t, result, 32)
}

func TestClient_SignRequest(t *testing.T) {
	client := NewClient(Config{
		Region:    "us-east-1",
		AccessKey: "AKID",
		SecretKey: "secret",
		Timeout:   5 * time.Second,
	})

	req, _ := http.NewRequest(http.MethodPost, "https://example.com/test", nil)
	req.Header.Set("Content-Type", "application/json")

	client.signRequest(req, []byte("test body"))

	assert.NotEmpty(t, req.Header.Get("Authorization"))
	assert.NotEmpty(t, req.Header.Get("X-Amz-Date"))
	assert.Contains(t, req.Header.Get("Authorization"), "AWS4-HMAC-SHA256")
	assert.Contains(t, req.Header.Get("Authorization"), "AKID")
}

func TestClient_Embed_Cohere_NoEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := cohereResponse{
			Embeddings: [][]float32{},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no embedding")
}

func TestClient_Embed_Titan_JSONDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{invalid json`))
	}))
	defer server.Close()

	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "amazon.titan-embed-text-v2:0",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse response")
}

func TestClient_Embed_Cohere_JSONDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{invalid json`))
	}))
	defer server.Close()

	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse response")
}

func TestClient_Embed_Cohere_APIError(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		errContain string
	}{
		{"unauthorized", http.StatusUnauthorized, "401"},
		{"bad_request", http.StatusBadRequest, "400"},
		{"service_unavailable", http.StatusServiceUnavailable, "503"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				_, _ = w.Write([]byte(`{"message": "error"}`))
			}))
			defer server.Close()

			client := NewClient(Config{
				AccessKey: "key",
				SecretKey: "secret",
				Model:     "cohere.embed-english-v3",
				BaseURL:   server.URL,
				Timeout:   5 * time.Second,
			})

			_, err := client.Embed(context.Background(), "test")
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.errContain)
		})
	}
}

func TestClient_EmbedBatch_Titan_Failure(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		// Fail on second request
		if callCount == 2 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"message": "server error"}`))
			return
		}
		response := titanResponse{
			Embedding:      make([]float32, 1024),
			InputTextToken: 5,
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "amazon.titan-embed-text-v2:0",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	_, err := client.EmbedBatch(context.Background(), []string{"first", "second", "third"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to embed text 1")
}

func TestClient_Embed_Titan_RequestFailure(t *testing.T) {
	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "amazon.titan-embed-text-v2:0",
		BaseURL:   "http://invalid-host-that-does-not-exist.local:99999",
		Timeout:   100 * time.Millisecond,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "request failed")
}

func TestClient_Embed_Cohere_RequestFailure(t *testing.T) {
	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   "http://invalid-host-that-does-not-exist.local:99999",
		Timeout:   100 * time.Millisecond,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "request failed")
}

func TestClient_EmbedBatch_Cohere_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"message": "rate limited"}`))
	}))
	defer server.Close()

	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	_, err := client.EmbedBatch(context.Background(), []string{"a", "b", "c"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "429")
}

func TestClient_Embed_Titan_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "amazon.titan-embed-text-v2:0",
		BaseURL:   server.URL,
		Timeout:   10 * time.Second,
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := client.Embed(ctx, "test")
	assert.Error(t, err)
}

func TestClient_Embed_Cohere_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   server.URL,
		Timeout:   10 * time.Second,
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := client.Embed(ctx, "test")
	assert.Error(t, err)
}

func TestClient_EmbedBatch_EmptyTexts(t *testing.T) {
	tests := []struct {
		name  string
		model string
	}{
		{"titan_empty", "amazon.titan-embed-text-v2:0"},
		{"cohere_empty", "cohere.embed-english-v3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tt.model == "cohere.embed-english-v3" {
					response := cohereResponse{Embeddings: [][]float32{}}
					_ = json.NewEncoder(w).Encode(response)
				} else {
					response := titanResponse{Embedding: make([]float32, 1024)}
					_ = json.NewEncoder(w).Encode(response)
				}
			}))
			defer server.Close()

			client := NewClient(Config{
				AccessKey: "key",
				SecretKey: "secret",
				Model:     tt.model,
				BaseURL:   server.URL,
				Timeout:   5 * time.Second,
			})

			embeddings, err := client.EmbedBatch(context.Background(), []string{})
			assert.NoError(t, err)
			assert.Len(t, embeddings, 0)
		})
	}
}

func TestDimensionForModel(t *testing.T) {
	tests := []struct {
		model    string
		expected int
	}{
		{"amazon.titan-embed-text-v1", 1536},
		{"amazon.titan-embed-text-v2:0", 1024},
		{"amazon.titan-embed-image-v1", 1024},
		{"cohere.embed-english-v3", 1024},
		{"cohere.embed-multilingual-v3", 1024},
		{"some-unknown-model", 1536},
		{"", 1536},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			dim := dimensionForModel(tt.model)
			assert.Equal(t, tt.expected, dim)
		})
	}
}

func TestClient_Embed_Titan_InvalidURL(t *testing.T) {
	// Control characters in URL cause http.NewRequestWithContext to fail
	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "amazon.titan-embed-text-v2:0",
		BaseURL:   "http://example.com\x00invalid",
		Timeout:   5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create request")
}

func TestClient_Embed_Cohere_InvalidURL(t *testing.T) {
	// Control characters in URL cause http.NewRequestWithContext to fail
	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   "http://example.com\x00invalid",
		Timeout:   5 * time.Second,
	})

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create request")
}

// =========================================================================
// Additional Tests for 100% Coverage
// =========================================================================

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

func TestClient_Embed_Titan_MarshalError(t *testing.T) {
	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "amazon.titan-embed-text-v2:0",
		Timeout:   5 * time.Second,
	})
	client.marshaler = mockMarshaler{err: fmt.Errorf("mock marshal error")}

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to marshal request")
}

func TestClient_Embed_Cohere_MarshalError(t *testing.T) {
	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		Timeout:   5 * time.Second,
	})
	client.marshaler = mockMarshaler{err: fmt.Errorf("mock marshal error")}

	_, err := client.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to marshal request")
}

func TestClient_EmbedBatch_Cohere_RequestFailure(t *testing.T) {
	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   "http://invalid-host-that-does-not-exist.local:99999",
		Timeout:   100 * time.Millisecond,
	})

	_, err := client.EmbedBatch(context.Background(), []string{"a", "b"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "request failed")
}

func TestClient_EmbedBatch_Cohere_JSONDecodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{invalid json`))
	}))
	defer server.Close()

	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   server.URL,
		Timeout:   5 * time.Second,
	})

	_, err := client.EmbedBatch(context.Background(), []string{"a", "b"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse response")
}

func TestClient_EmbedBatch_Cohere_InvalidURL(t *testing.T) {
	client := NewClient(Config{
		AccessKey: "key",
		SecretKey: "secret",
		Model:     "cohere.embed-english-v3",
		BaseURL:   "http://example.com\x00invalid",
		Timeout:   5 * time.Second,
	})

	_, err := client.EmbedBatch(context.Background(), []string{"a", "b"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create request")
}
