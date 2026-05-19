# Embeddings

Generic, reusable Go module for text embedding generation across multiple providers. Single `EmbeddingProvider` interface, seven backend packages, real HTTP transport, no production-side mocks. Module path: `digital.vasic.embeddings` (Go 1.24.0). Standalone — no consuming-project context leaks; safe to incorporate at any owning project's root per CONST-051(C).

## Providers

| Package | Provider | Models | Dim |
|---------|----------|--------|-----|
| `pkg/openai` | OpenAI | text-embedding-3-small, text-embedding-3-large, ada-002 | 1536 / 3072 / 1536 |
| `pkg/cohere` | Cohere | embed-english-v3.0, embed-multilingual-v3.0, light variants | 1024 |
| `pkg/voyage` | Voyage AI | voyage-3, voyage-3-lite, voyage-code-3, voyage-law-2 | 1024 |
| `pkg/jina` | Jina AI | jina-embeddings-v3, jina-embeddings-v2-*, jina-clip-v1 | 1024 |
| `pkg/google` | Google Vertex AI | text-embedding-005, text-multilingual-embedding-002 | 768 |
| `pkg/bedrock` | AWS Bedrock | Amazon Titan Embed, Cohere on Bedrock (SigV4-signed) | 1024 / 1536 |
| `pkg/provider` | — | Common interface, Config, Result, TokenUsage | — |

## Usage

```go
import (
    "context"
    "digital.vasic.embeddings/pkg/openai"
    "digital.vasic.embeddings/pkg/provider"
)

client := openai.NewClient(openai.Config{
    APIKey: "your-key",
    Model:  "text-embedding-3-small",
})

// Single embedding
embedding, err := client.Embed(context.Background(), "Hello world")

// Batch embeddings (order-preserving)
embeddings, err := client.EmbedBatch(context.Background(),
    []string{"text1", "text2"})

// Provider metadata
fmt.Println(client.Name())       // "openai/text-embedding-3-small"
fmt.Println(client.Dimensions()) // 1536

// Interface contract (compile-time + runtime)
var _ provider.EmbeddingProvider = client
```

Every provider Client implements the same four-method interface — drop-in interchangeable as long as the dimensional shape requirement of your downstream code is satisfied.

## Interface

```go
type EmbeddingProvider interface {
    Embed(ctx context.Context, text string) ([]float32, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
    Dimensions() int
    Name() string
}
```

## Build & Test

```bash
make build                  # go build ./...
make test                   # GOMAXPROCS=2 go test -count=1 -race -p 1 ./...
make test-short             # unit tests only
make test-bench             # benchmarks
make test-coverage          # coverage report (coverage.html)
make fmt                    # gofmt + goimports
make vet                    # go vet
make lint                   # golangci-lint
```

`make test` runs the full unit suite across all seven packages with `-race` and ordered package execution (avoids cross-package httptest port collisions).

## Round-249 Anti-bluff guarantees

This module ships with round-249 deep-doc + Challenge enrichment under CONST-035 / Article XI §11.9 / CONST-050(B). The guarantees below are mechanically verified — not promises.

1. **Bilingual round-trip across every provider.** `challenges/runner/main.go` drives the real `EmbedBatch` path of every Client (openai, cohere, voyage, jina, google, bedrock-titan) through a real net/http loopback `httptest.Server`. The runner asserts that the original UTF-8 bytes of each input (English, Serbian-Cyrillic, Japanese, Arabic, Simplified Chinese) survive `json.Marshal` and arrive at the server byte-for-byte. Failure to round-trip exits non-zero.
2. **Dimensional contract.** For every provider × every locale the runner asserts `len(out[i]) == Client.Dimensions()`. Any dim drift is a hard FAIL — no metadata-only PASS.
3. **Index-ordering preservation.** `EmbedBatch` is required to place the embedding for `texts[i]` at `out[i]` regardless of server-side ordering — exercised in unit tests (TestClient_EmbedBatch_IndexOrdering for OpenAI + Voyage) and at runtime by the runner's positional `assertCapturedTexts` over the captured-input slice.
4. **Interface contract verified at runtime.** Each package ships a compile-time blank assignment (`var _ provider.EmbeddingProvider = (*Client)(nil)`); the runner restates this at runtime through a `[]provider.EmbeddingProvider` slice over six freshly-constructed clients, asserting `Name() != ""` and `Dimensions() > 0`.
5. **Paired-mutation deep-doc gate.** `challenges/embeddings_describe_challenge.sh --anti-bluff-mutate` plants a deliberate ledger-vs-source rename mismatch (`EmbedBatch → EmbedBogus_MUTATED` in a tmp ledger copy) and asserts the gate FAILS with exit 99. This proves the gate actually catches ledger/source drift instead of rubber-stamping it. Without the flag the gate runs normal validation and MUST exit 0.
6. **Live integration tests SKIP-OK when no API key is provided.** Per CONST-050(A), integration tests MUST hit the real backing service — they cannot use mock servers. When `OPENAI_API_KEY` is absent the live tests skip with `SKIP-OK: #embeddings-live-key-required`; when present they POST against `https://api.openai.com/v1/embeddings` and assert the real dimensional shape.

### Run the round-249 gate

```bash
# Real cross-provider bilingual exerciser (real HTTP loopback, real Client
# transport, real JSON marshal/unmarshal):
go run ./challenges/runner/ -fixtures tests/fixtures/i18n/payloads.json
# Expected: 36 PASS, 0 FAIL across 6 providers × 5 locales + 6 contract checks.

# Paired-mutation describe gate:
bash challenges/embeddings_describe_challenge.sh                  # exit 0
bash challenges/embeddings_describe_challenge.sh --anti-bluff-mutate  # exit 99

# Existing scripted Challenges (per CONST-050(B) test-type matrix):
bash challenges/scripts/embeddings_compile_challenge.sh
bash challenges/scripts/embeddings_unit_challenge.sh
bash challenges/scripts/embeddings_functionality_challenge.sh
bash challenges/scripts/chaos_failure_injection_challenge.sh
bash challenges/scripts/ddos_health_flood_challenge.sh
bash challenges/scripts/scaling_horizontal_challenge.sh
bash challenges/scripts/stress_sustained_load_challenge.sh
bash challenges/scripts/ui_terminal_interaction_challenge.sh
bash challenges/scripts/ux_end_to_end_flow_challenge.sh
```

## Test-type coverage matrix (CONST-050(B))

| Test type | Location |
|-----------|----------|
| Unit | `pkg/*/`*_test.go` |
| Integration | `tests/integration/` (SKIP-OK without `OPENAI_API_KEY`) |
| End-to-end | `tests/e2e/` |
| Security | `tests/security/` |
| Stress | `tests/stress/` |
| Benchmark | `tests/benchmark/` |
| Challenges | `challenges/scripts/` + `challenges/embeddings_describe_challenge.sh` |
| Bilingual fixtures | `tests/fixtures/i18n/payloads.json` (round-249) |

See `docs/test-coverage.md` for the per-symbol → per-test ledger.

## Governance

This submodule inherits the constitution from its parent consuming project (HelixCode, etc.). See `CONSTITUTION.md`, `CLAUDE.md`, `AGENTS.md` at the module root for the full anti-bluff anchor set: Article XI §11.9, CONST-033, CONST-035, CONST-036, CONST-042 (no secret leak), CONST-047 (recursive cascade), CONST-048 (full-automation coverage), CONST-050 (no-fakes-beyond-unit-tests + 100% test-type coverage), CONST-051 (decoupled / project-not-aware), CONST-053 (.gitignore hygiene), CONST-055 (post-pull validation), CONST-060 (fetch-before-edit), CONST-061 (pre-force-push merge-first).

## Adding a new provider

1. Create `pkg/<name>/<name>.go` implementing `provider.EmbeddingProvider`.
2. Add the compile-time check: `var _ provider.EmbeddingProvider = (*Client)(nil)`.
3. Create `pkg/<name>/<name>_test.go` with table-driven tests; mock HTTP via `httptest` is permitted in unit tests only (CONST-050(A)).
4. Extend `docs/test-coverage.md` with the new package's symbol → test rows.
5. Extend `challenges/runner/main.go` with a new provider case (loopback `httptest.Server` + `EmbedBatch` round-trip + bilingual `assertCapturedTexts`).
6. Run `make test`, the runner, and `bash challenges/embeddings_describe_challenge.sh` — all three must exit 0.
