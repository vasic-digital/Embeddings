# Embeddings Test Coverage Ledger (round-249)

Round-249 deep-doc enrichment under CONST-035 / Article XI §11.9 / CONST-050(B).

This document is the authoritative mapping of every exported symbol to the test sources that exercise it. Drift between this file and `go test -cover` output is a CONST-035 bluff at the documentation-truth layer — fix the document OR add the missing test, never silently leave the gap.

## Verbatim 2026-05-19 operator mandate (CONST-049 §11.4.17)

> "all existing tests and Challenges do work in anti-bluff manner - they MUST confirm that all tested codebase really works as expected! We had been in position that all tests do execute with success and all Challenges as well, but in reality the most of the features does not work and can't be used! This MUST NOT be the case and execution of tests and Challenges MUST guarantee the quality, the completition and full usability by end users of the product!"

## Test-type matrix (CONST-050(B))

| Test type | Location | Status |
|-----------|----------|--------|
| Unit | `pkg/*/`*_test.go` | PRESENT — every package |
| Integration | `tests/integration/` | PRESENT (SKIP-OK without `OPENAI_API_KEY`) |
| End-to-end | `tests/e2e/` | PRESENT |
| Security | `tests/security/` | PRESENT |
| Stress | `tests/stress/` | PRESENT |
| Benchmark | `tests/benchmark/` | PRESENT |
| Challenges | `challenges/scripts/` + `challenges/embeddings_describe_challenge.sh` | PRESENT — incl. round-249 paired-mutation describe gate |
| Bilingual fixtures | `tests/fixtures/i18n/` | PRESENT (round-249) |

## `pkg/provider` — core interface

| Symbol | Kind | Test source(s) |
|--------|------|---------------|
| `EmbeddingProvider` | interface | `pkg/provider/provider_test.go` (TestEmbeddingProviderInterface); runtime restated by `challenges/runner/main.go` (contract section) |
| `Config` | struct | `pkg/provider/provider_test.go` (TestConfig_Fields, TestDefaultConfig) |
| `DefaultConfig` | constructor | `pkg/provider/provider_test.go` (TestDefaultConfig) |
| `Result` | struct | `pkg/provider/provider_test.go` (TestResult_Fields); JSON round-trip in `tests/integration/embeddings_integration_test.go` (TestProviderResult_Structure_Integration) |
| `TokenUsage` | struct | `pkg/provider/provider_test.go` (TestTokenUsage_Fields) |

## `pkg/openai`

| Symbol | Kind | Test source(s) |
|--------|------|---------------|
| `DefaultBaseURL` / `DefaultModel` | const | `pkg/openai/openai_test.go` (TestNewClient default-fill paths) |
| `Config` | struct | `pkg/openai/openai_test.go` (TestNewClient) |
| `Client` | struct | `pkg/openai/openai_test.go` (every test); `challenges/runner/main.go` (openai loopback case) |
| `NewClient` | constructor | `pkg/openai/openai_test.go` (TestNewClient); `tests/integration/embeddings_integration_test.go` (TestOpenAIClient_*) |
| `Client.Name` | method | `pkg/openai/openai_test.go` (TestClient_Name); `challenges/runner/main.go` (contract section) |
| `Client.Dimensions` | method | `pkg/openai/openai_test.go` (TestClient_Dimensions); runner dim-shape assertion |
| `Client.Embed` | method | `pkg/openai/openai_test.go` (TestClient_Embed, TestClient_Embed_NoEmbeddingReturned, TestClient_Embed_APIError, TestClient_Embed_ReturnError); live SKIP-OK in `tests/integration/` (TestOpenAIClient_Embed_LiveAPI_Integration) |
| `Client.EmbedBatch` | method | `pkg/openai/openai_test.go` (TestClient_EmbedBatch, TestClient_EmbedBatch_IndexOrdering, TestClient_EmbedBatch_InvalidURL, TestClient_EmbedBatch_RequestFailure, TestClient_EmbedBatch_JSONDecodeError, TestClient_EmbedBatch_MarshalError, TestClient_ContextCancellation); `challenges/runner/main.go` (5-locale loopback round-trip) |

## `pkg/cohere`

| Symbol | Kind | Test source(s) |
|--------|------|---------------|
| `DefaultBaseURL` / `DefaultModel` | const | `pkg/cohere/cohere_test.go` (TestNewClient default-fill) |
| `Config` | struct | `pkg/cohere/cohere_test.go` (TestNewClient) |
| `Client` | struct | `pkg/cohere/cohere_test.go`; `challenges/runner/main.go` (cohere case) |
| `NewClient` | constructor | `pkg/cohere/cohere_test.go` (TestNewClient) |
| `Client.Name` | method | `pkg/cohere/cohere_test.go` (TestClient_Name) |
| `Client.Dimensions` | method | runner dim-shape assertion |
| `Client.Embed` | method | `pkg/cohere/cohere_test.go` (TestClient_Embed, TestClient_Embed_EmbeddingsObjFormat, TestClient_Embed_NoEmbeddingReturned, TestClient_Embed_APIError, TestClient_Embed_EmptyEmbeddingsReturned, TestClient_Embed_ReturnError, TestClient_ContextCancellation, TestClient_CustomInputType) |
| `Client.EmbedBatch` | method | `pkg/cohere/cohere_test.go` (TestClient_EmbedBatch, TestClient_EmbedBatch_InvalidURL, TestClient_EmbedBatch_RequestFailure, TestClient_EmbedBatch_JSONDecodeError, TestClient_EmbedBatch_MarshalError); `challenges/runner/main.go` (5-locale loopback round-trip) |

## `pkg/voyage`

| Symbol | Kind | Test source(s) |
|--------|------|---------------|
| `DefaultBaseURL` / `DefaultModel` | const | `pkg/voyage/voyage_test.go` (TestNewClient default-fill) |
| `Config` | struct | `pkg/voyage/voyage_test.go` (TestNewClient) |
| `Client` | struct | `pkg/voyage/voyage_test.go`; `challenges/runner/main.go` (voyage case) |
| `NewClient` | constructor | `pkg/voyage/voyage_test.go` (TestNewClient) |
| `Client.Name` | method | `pkg/voyage/voyage_test.go` (TestClient_Name) |
| `Client.Dimensions` | method | runner dim-shape assertion |
| `Client.Embed` | method | `pkg/voyage/voyage_test.go` (TestClient_Embed, TestClient_Embed_NoEmbeddingReturned, TestClient_Embed_APIError, TestClient_Embed_ReturnError, TestClient_ContextCancellation, TestClient_CustomInputType) |
| `Client.EmbedBatch` | method | `pkg/voyage/voyage_test.go` (TestClient_EmbedBatch, TestClient_EmbedBatch_IndexOrdering, TestClient_EmbedBatch_InvalidURL, TestClient_EmbedBatch_RequestFailure, TestClient_EmbedBatch_JSONDecodeError, TestClient_EmbedBatch_MarshalError); `challenges/runner/main.go` (5-locale loopback round-trip) |

## `pkg/jina`

| Symbol | Kind | Test source(s) |
|--------|------|---------------|
| `DefaultBaseURL` / `DefaultModel` | const | `pkg/jina/jina_test.go` (TestNewClient default-fill) |
| `Config` | struct | `pkg/jina/jina_test.go` (TestNewClient) |
| `Client` | struct | `pkg/jina/jina_test.go`; `challenges/runner/main.go` (jina case) |
| `NewClient` | constructor | `pkg/jina/jina_test.go` (TestNewClient) |
| `Client.Name` | method | `pkg/jina/jina_test.go` (TestClient_Name) |
| `Client.Dimensions` | method | runner dim-shape assertion |
| `Client.Embed` | method | `pkg/jina/jina_test.go` (TestClient_Embed, TestClient_Embed_NoEmbeddingReturned, TestClient_Embed_APIError, TestClient_Embed_ReturnError, TestClient_ContextCancellation, TestClient_CustomTask) |
| `Client.EmbedBatch` | method | `pkg/jina/jina_test.go` (TestClient_EmbedBatch, TestClient_EmbedBatch_InvalidURL, TestClient_EmbedBatch_RequestFailure, TestClient_EmbedBatch_JSONDecodeError, TestClient_EmbedBatch_MarshalError); `challenges/runner/main.go` (5-locale loopback round-trip) |

## `pkg/google`

| Symbol | Kind | Test source(s) |
|--------|------|---------------|
| `DefaultLocation` / `DefaultModel` / `DefaultBaseURL` | const | `pkg/google/google_test.go` (TestNewClient default-fill) |
| `Config` | struct | `pkg/google/google_test.go` (TestNewClient) |
| `Client` | struct | `pkg/google/google_test.go`; `challenges/runner/main.go` (google case) |
| `NewClient` | constructor | `pkg/google/google_test.go` (TestNewClient) |
| `Client.Name` | method | `pkg/google/google_test.go` (TestClient_Name) |
| `Client.Dimensions` | method | runner dim-shape assertion |
| `Client.Embed` | method | `pkg/google/google_test.go` (TestClient_Embed, TestClient_Embed_NoEmbeddingReturned, TestClient_Embed_APIError, TestClient_Embed_ReturnError, TestClient_ContextCancellation, TestClient_RequestBody) |
| `Client.EmbedBatch` | method | `pkg/google/google_test.go` (TestClient_EmbedBatch, TestClient_EmbedBatch_InvalidURL, TestClient_EmbedBatch_RequestFailure, TestClient_EmbedBatch_JSONDecodeError, TestClient_EmbedBatch_MarshalError); `challenges/runner/main.go` (5-locale loopback round-trip) |

## `pkg/bedrock`

| Symbol | Kind | Test source(s) |
|--------|------|---------------|
| `DefaultRegion` / `DefaultModel` | const | `pkg/bedrock/bedrock_test.go` (TestNewClient default-fill, TestDimensionForModel) |
| `Config` | struct | `pkg/bedrock/bedrock_test.go` (TestNewClient) |
| `Client` | struct | `pkg/bedrock/bedrock_test.go`; `challenges/runner/main.go` (bedrock-titan case) |
| `NewClient` | constructor | `pkg/bedrock/bedrock_test.go` (TestNewClient) |
| `Client.Name` | method | `pkg/bedrock/bedrock_test.go` (TestClient_Name) |
| `Client.Dimensions` | method | runner dim-shape assertion |
| `Client.Embed` (Titan + Cohere routes) | method | `pkg/bedrock/bedrock_test.go` (TestClient_Embed_Titan, TestClient_Embed_Cohere, TestClient_Embed_UnsupportedModel, TestClient_Embed_APIError, TestClient_ContextCancellation, plus _Titan_/_Cohere_ failure paths) |
| `Client.EmbedBatch` (Titan loop + Cohere batch) | method | `pkg/bedrock/bedrock_test.go` (TestClient_EmbedBatch_Titan, TestClient_EmbedBatch_Cohere, TestClient_EmbedBatch_Cohere_APIError, TestClient_EmbedBatch_Titan_Failure, TestClient_EmbedBatch_Cohere_RequestFailure, TestClient_EmbedBatch_EmptyTexts); `challenges/runner/main.go` (Titan loop, 5-locale loopback round-trip) |
| AWS SigV4 helpers (`signRequest`, `sha256Hash`, `hmacSHA256`) | internal | `pkg/bedrock/bedrock_test.go` (TestSHA256Hash, TestHmacSHA256, TestClient_SignRequest) |

## Edge cases (round-249)

- Empty input batch (Titan loop short-circuit) — `pkg/bedrock/bedrock_test.go` (TestClient_EmbedBatch_EmptyTexts).
- Unknown model dimension fallback — `pkg/openai/openai_test.go` (TestClient_Dimensions default branch), `pkg/bedrock/bedrock_test.go` (TestDimensionForModel).
- Index ordering preservation across out-of-order server response — `pkg/openai/openai_test.go` (TestClient_EmbedBatch_IndexOrdering), `pkg/voyage/voyage_test.go` (TestClient_EmbedBatch_IndexOrdering).
- JSON marshal failure path — `pkg/{openai,cohere,jina,google,bedrock}/*_test.go` (TestClient_EmbedBatch_MarshalError variants).
- Context cancellation mid-flight — `pkg/{openai,cohere,voyage,jina,google,bedrock}/*_test.go` (TestClient_ContextCancellation variants).
- Bilingual UTF-8 byte preservation through Marshal → POST body → server decode — `challenges/runner/main.go` (assertCapturedTexts on every provider).
- Live API SKIP-OK when no key set — `tests/integration/embeddings_integration_test.go` (TestOpenAIClient_EmbedBatch_LiveAPI_Integration, TestOpenAIClient_Embed_LiveAPI_Integration); per CONST-050(A) mocks restricted to unit tests, integration MUST hit real provider.

## Paired-mutation Challenge

`challenges/embeddings_describe_challenge.sh` accepts `--anti-bluff-mutate` to plant a deliberate ledger-vs-source mismatch (renames every `EmbedBatch` occurrence in a tmp copy of the ledger to `EmbedBogus_MUTATED`) and asserts the gate FAILS with exit 99. Without the flag the gate runs normal validation and MUST exit 0. Composition: CONST-035 (anti-bluff) × CONST-050(B) (paired mutation) × CONST-047 (cascade).

## Anti-bluff acceptance criteria

1. `GOMAXPROCS=2 go test -count=1 -race -p 1 ./pkg/...` exits 0 — all 7 packages PASS (verified round-249).
2. `bash challenges/embeddings_describe_challenge.sh` exits 0 (gate PASS on clean tree).
3. `bash challenges/embeddings_describe_challenge.sh --anti-bluff-mutate` exits 99 (gate correctly fails on planted mutation).
4. Every symbol in this ledger appears in the listed test source verbatim — no metadata-only / configuration-only ledger entries.
5. `challenges/runner/main.go` exits 0 with 36 PASS lines (6 providers × 5 locales + 6 interface-contract checks) when invoked with the bilingual fixture set.
