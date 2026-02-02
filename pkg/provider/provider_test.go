package provider

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestDefaultConfig(t *testing.T) {
	tests := []struct {
		name     string
		validate func(t *testing.T, cfg Config)
	}{
		{
			name: "batch_size_default",
			validate: func(t *testing.T, cfg Config) {
				assert.Equal(t, 100, cfg.BatchSize)
			},
		},
		{
			name: "max_retries_default",
			validate: func(t *testing.T, cfg Config) {
				assert.Equal(t, 3, cfg.MaxRetries)
			},
		},
		{
			name: "timeout_default",
			validate: func(t *testing.T, cfg Config) {
				assert.Equal(t, 30*time.Second, cfg.Timeout)
			},
		},
		{
			name: "model_empty_by_default",
			validate: func(t *testing.T, cfg Config) {
				assert.Empty(t, cfg.Model)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tt.validate(t, cfg)
		})
	}
}

func TestConfig_Fields(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		check  func(t *testing.T, cfg Config)
	}{
		{
			name: "custom_model",
			config: Config{
				Model:      "custom-model",
				BatchSize:  50,
				MaxRetries: 5,
				Timeout:    10 * time.Second,
			},
			check: func(t *testing.T, cfg Config) {
				assert.Equal(t, "custom-model", cfg.Model)
				assert.Equal(t, 50, cfg.BatchSize)
				assert.Equal(t, 5, cfg.MaxRetries)
				assert.Equal(t, 10*time.Second, cfg.Timeout)
			},
		},
		{
			name:   "zero_value",
			config: Config{},
			check: func(t *testing.T, cfg Config) {
				assert.Empty(t, cfg.Model)
				assert.Zero(t, cfg.BatchSize)
				assert.Zero(t, cfg.MaxRetries)
				assert.Zero(t, cfg.Timeout)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.check(t, tt.config)
		})
	}
}

func TestResult_Fields(t *testing.T) {
	tests := []struct {
		name   string
		result Result
		check  func(t *testing.T, r Result)
	}{
		{
			name: "with_embeddings",
			result: Result{
				Embeddings: [][]float32{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}},
				Model:      "test-model",
				Usage: TokenUsage{
					PromptTokens: 10,
					TotalTokens:  10,
				},
			},
			check: func(t *testing.T, r Result) {
				assert.Len(t, r.Embeddings, 2)
				assert.Equal(t, "test-model", r.Model)
				assert.Equal(t, 10, r.Usage.PromptTokens)
				assert.Equal(t, 10, r.Usage.TotalTokens)
			},
		},
		{
			name:   "empty_result",
			result: Result{},
			check: func(t *testing.T, r Result) {
				assert.Nil(t, r.Embeddings)
				assert.Empty(t, r.Model)
				assert.Zero(t, r.Usage.PromptTokens)
				assert.Zero(t, r.Usage.TotalTokens)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.check(t, tt.result)
		})
	}
}

func TestTokenUsage_Fields(t *testing.T) {
	tests := []struct {
		name  string
		usage TokenUsage
		check func(t *testing.T, u TokenUsage)
	}{
		{
			name: "non_zero",
			usage: TokenUsage{
				PromptTokens: 100,
				TotalTokens:  150,
			},
			check: func(t *testing.T, u TokenUsage) {
				assert.Equal(t, 100, u.PromptTokens)
				assert.Equal(t, 150, u.TotalTokens)
			},
		},
		{
			name:  "zero_value",
			usage: TokenUsage{},
			check: func(t *testing.T, u TokenUsage) {
				assert.Zero(t, u.PromptTokens)
				assert.Zero(t, u.TotalTokens)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.check(t, tt.usage)
		})
	}
}

// TestEmbeddingProviderInterface verifies that the interface is correctly defined
// by creating a compile-time check with a mock implementation.
func TestEmbeddingProviderInterface(t *testing.T) {
	// This test verifies the interface contract at compile time.
	var _ EmbeddingProvider = (*mockProvider)(nil)
	assert.True(t, true, "EmbeddingProvider interface is well-defined")
}

// mockProvider is a compile-time check for the EmbeddingProvider interface.
type mockProvider struct{}

func (m *mockProvider) Embed(_ context.Context, _ string) ([]float32, error) {
	return nil, nil
}

func (m *mockProvider) EmbedBatch(_ context.Context, _ []string) ([][]float32, error) {
	return nil, nil
}

func (m *mockProvider) Dimensions() int { return 0 }
func (m *mockProvider) Name() string    { return "" }
