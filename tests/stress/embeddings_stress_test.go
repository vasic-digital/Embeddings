package stress

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"digital.vasic.embeddings/pkg/openai"
	"digital.vasic.embeddings/pkg/provider"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIClient_ConcurrentEmbedRequests_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			response := map[string]interface{}{
				"data": []map[string]interface{}{
					{"embedding": []float32{0.1, 0.2, 0.3}, "index": 0},
				},
				"model": "text-embedding-3-small",
				"usage": map[string]int{
					"prompt_tokens": 5, "total_tokens": 5,
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

	const goroutines = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			ctx := t.Context()
			text := fmt.Sprintf("Embedding text number %d", idx)
			embedding, err := client.Embed(ctx, text)
			assert.NoError(t, err,
				"embed should succeed in goroutine %d", idx)
			assert.Equal(t, 3, len(embedding),
				"embedding should have 3 dimensions in goroutine %d", idx)
		}(i)
	}
	wg.Wait()
}

func TestOpenAIClient_ConcurrentBatchEmbedRequests_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Input []string `json:"input"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)

			data := make([]map[string]interface{}, len(req.Input))
			for i := range req.Input {
				data[i] = map[string]interface{}{
					"embedding": []float32{0.1, 0.2, 0.3},
					"index":     i,
				}
			}
			response := map[string]interface{}{
				"data":  data,
				"model": "text-embedding-3-small",
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
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	const goroutines = 50
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			ctx := t.Context()
			texts := []string{
				fmt.Sprintf("text A %d", idx),
				fmt.Sprintf("text B %d", idx),
				fmt.Sprintf("text C %d", idx),
			}
			embeddings, err := client.EmbedBatch(ctx, texts)
			assert.NoError(t, err,
				"batch embed should succeed in goroutine %d", idx)
			assert.Equal(t, 3, len(embeddings),
				"should return 3 embeddings in goroutine %d", idx)
		}(i)
	}
	wg.Wait()
}

func TestOpenAIClient_ConcurrentCreation_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	const goroutines = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)

	clients := make([]*openai.Client, goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			clients[idx] = openai.NewClient(openai.Config{
				APIKey: fmt.Sprintf("key-%d", idx),
				Model:  "text-embedding-3-small",
			})
		}(i)
	}
	wg.Wait()

	for i, c := range clients {
		require.NotNil(t, c, "client %d should not be nil", i)
		assert.Equal(t, 1536, c.Dimensions())
	}
}

func TestProviderConfig_ConcurrentSerialization_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	cfg := provider.DefaultConfig()
	cfg.Model = "test-model"

	const goroutines = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			data, err := json.Marshal(cfg)
			assert.NoError(t, err, "marshal should succeed in goroutine %d", idx)

			var decoded provider.Config
			err = json.Unmarshal(data, &decoded)
			assert.NoError(t, err, "unmarshal should succeed in goroutine %d", idx)
			assert.Equal(t, "test-model", decoded.Model)
		}(i)
	}
	wg.Wait()
}

func TestProviderResult_ConcurrentAccess_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	const goroutines = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)

	results := make([]provider.Result, goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			emb := make([][]float32, 5)
			for j := range emb {
				emb[j] = []float32{float32(idx), float32(j)}
			}
			results[idx] = provider.Result{
				Embeddings: emb,
				Model:      fmt.Sprintf("model-%d", idx),
				Usage: provider.TokenUsage{
					PromptTokens: idx * 10,
					TotalTokens:  idx * 10,
				},
			}
		}(i)
	}
	wg.Wait()

	for i, r := range results {
		assert.Equal(t, 5, len(r.Embeddings), "result %d should have 5 embeddings", i)
		assert.Equal(t, fmt.Sprintf("model-%d", i), r.Model)
	}
}
