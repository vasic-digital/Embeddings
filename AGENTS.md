# AGENTS.md -- Multi-Agent Coordination Guide

This document provides guidance for AI agents working with the Embeddings module (`digital.vasic.embeddings`). It covers coordination patterns, boundaries, and conventions that agents must follow when making changes.

## Module Identity

- **Module path**: `digital.vasic.embeddings`
- **Language**: Go 1.24.0
- **Purpose**: Standalone, generic library for text embedding generation across 6 providers
- **No application-specific dependencies**: This module must remain fully generic and reusable

## Agent Roles

### Code Agent

Responsible for modifying source code in `pkg/`. Must:

- Follow the `EmbeddingProvider` interface contract defined in `pkg/provider/provider.go`
- Include compile-time interface checks (`var _ provider.EmbeddingProvider = (*Client)(nil)`)
- Use `fmt.Errorf("<provider>: ...: %w", err)` for all error wrapping
- Ensure every provider has a `Config` struct, `Client` struct, `NewClient` constructor, and private `dimensionForModel` function
- Implement `Embed` by delegating to `EmbedBatch` with a single-element slice (except Bedrock Titan which invokes directly)
- Run `go test ./... -count=1 -race` after any change

### Test Agent

Responsible for test files (`*_test.go`). Must:

- Use table-driven tests with `testify`
- Use `httptest.NewServer` for mock HTTP servers in unit tests
- Name tests as `Test<Struct>_<Method>_<Scenario>`
- Cover success, error, edge cases (empty input, invalid JSON, HTTP errors)
- Never introduce external network calls in unit tests

### Documentation Agent

Responsible for `*.md` files and `docs/`. Must:

- Keep all documentation consistent with actual source code
- Update API_REFERENCE.md when exported types or functions change
- Update USER_GUIDE.md when new providers are added
- Update ARCHITECTURE.md when design patterns change

### Integration Agent

Responsible for coordinating this module with consuming projects (e.g., HelixAgent). Must:

- Never introduce application-specific types into this module
- Ensure the `go.mod` dependency list stays minimal (only `testify` for testing)
- Validate that changes do not break downstream `go.sum` integrity

## Package Boundaries

```
pkg/provider/    -- Core interface and shared types. NO provider-specific logic.
pkg/openai/      -- OpenAI implementation only. Imports only pkg/provider.
pkg/cohere/      -- Cohere implementation only. Imports only pkg/provider.
pkg/voyage/      -- Voyage AI implementation only. Imports only pkg/provider.
pkg/jina/        -- Jina AI implementation only. Imports only pkg/provider.
pkg/google/      -- Google Vertex AI implementation only. Imports only pkg/provider.
pkg/bedrock/     -- AWS Bedrock implementation only. Imports only pkg/provider.
```

No provider package may import another provider package. All cross-provider coordination happens in the consuming application, not in this module.

## Adding a New Provider

When an agent adds a new provider, the following files must be created or updated:

1. `pkg/<name>/<name>.go` -- Implementation with `Config`, `Client`, `NewClient`, `Embed`, `EmbedBatch`, `Dimensions`, `Name`, `dimensionForModel`
2. `pkg/<name>/<name>_test.go` -- Table-driven tests with mock HTTP server
3. `README.md` -- Add row to the providers table
4. `CLAUDE.md` -- Add to the provider packages list
5. `docs/USER_GUIDE.md` -- Add usage example
6. `docs/API_REFERENCE.md` -- Document all exported types and functions
7. `docs/ARCHITECTURE.md` -- Update if new patterns are introduced
8. `docs/CHANGELOG.md` -- Add entry under Unreleased or new version

## Coordination Protocols

### Before Modifying `pkg/provider/provider.go`

Any change to the core interface is a breaking change. Agents must:

1. Verify no downstream consumers will break
2. Update ALL provider implementations simultaneously
3. Update ALL test files
4. Update API_REFERENCE.md

### Before Adding Dependencies

The module intentionally has a minimal dependency footprint (only `testify` for tests). Adding a new dependency requires:

1. Justification that standard library alternatives are insufficient
2. Verification the dependency is well-maintained and has a compatible license
3. Update to `go.mod` and `go.sum`

### Conflict Resolution

If multiple agents modify the same file:

1. The agent modifying `pkg/provider/` has priority (interface changes cascade)
2. Test agent changes are rebased on top of code agent changes
3. Documentation agent runs last to capture final state

## Quality Gates

All agents must ensure the following pass before considering work complete:

```bash
go test ./... -count=1 -race    # All tests pass with race detection
go vet ./...                    # No vet warnings
gofmt -l .                      # No formatting issues
```

## File Ownership

| Path | Primary Agent | Secondary |
|------|--------------|-----------|
| `pkg/provider/provider.go` | Code Agent | -- |
| `pkg/*/` (implementations) | Code Agent | Test Agent |
| `*_test.go` | Test Agent | Code Agent |
| `docs/` | Documentation Agent | -- |
| `CLAUDE.md` | Documentation Agent | Code Agent |
| `AGENTS.md` | Documentation Agent | -- |
| `go.mod`, `go.sum` | Integration Agent | Code Agent |
