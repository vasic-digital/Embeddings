package benchmark

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"digital.vasic.embeddings/pkg/openai"
	"digital.vasic.embeddings/pkg/provider"
)

func BenchmarkOpenAIClient_NewClient(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark test in short mode")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = openai.NewClient(openai.Config{
			APIKey: "test-key",
			Model:  "text-embedding-3-small",
		})
	}
}

func BenchmarkOpenAIClient_Name(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark test in short mode")
	}

	client := openai.NewClient(openai.Config{
		APIKey: "test-key",
		Model:  "text-embedding-3-small",
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = client.Name()
	}
}

func BenchmarkOpenAIClient_Dimensions(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark test in short mode")
	}

	client := openai.NewClient(openai.Config{
		APIKey: "test-key",
		Model:  "text-embedding-3-large",
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = client.Dimensions()
	}
}

func BenchmarkOpenAIClient_Embed(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark test in short mode")
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

	ctx := b.Context()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.Embed(ctx, "benchmark text")
	}
}

func BenchmarkOpenAIClient_EmbedBatch(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark test in short mode")
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

	texts := []string{"text one", "text two", "text three", "text four", "text five"}
	ctx := b.Context()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.EmbedBatch(ctx, texts)
	}
}

func BenchmarkProviderConfig_Marshal(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark test in short mode")
	}

	cfg := provider.DefaultConfig()
	cfg.Model = "text-embedding-3-small"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = json.Marshal(cfg)
	}
}

func BenchmarkProviderResult_Marshal(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark test in short mode")
	}

	result := provider.Result{
		Embeddings: make([][]float32, 10),
		Model:      "text-embedding-3-small",
		Usage:      provider.TokenUsage{PromptTokens: 100, TotalTokens: 100},
	}
	for i := range result.Embeddings {
		result.Embeddings[i] = make([]float32, 1536)
		for j := range result.Embeddings[i] {
			result.Embeddings[i][j] = float32(j) * 0.001
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, _ := json.Marshal(result)
		var decoded provider.Result
		_ = json.Unmarshal(data, &decoded)
	}
}

func BenchmarkMultipleClients_Creation(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark test in short mode")
	}

	models := []string{
		"text-embedding-3-small",
		"text-embedding-3-large",
		"text-embedding-ada-002",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, model := range models {
			_ = openai.NewClient(openai.Config{
				APIKey: fmt.Sprintf("key-%d", i),
				Model:  model,
			})
		}
	}
}
