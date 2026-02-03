# Contributing

Thank you for your interest in contributing to the Embeddings module. This guide covers the development workflow, coding standards, and review process.

## Prerequisites

- Go 1.24.0 or later
- Git with SSH access configured
- `gofmt` and `goimports` (included with Go)
- `testify` (installed automatically via `go mod tidy`)

## Getting Started

1. Clone the repository via SSH:
   ```bash
   git clone <repository-url>
   cd Embeddings
   ```

2. Verify everything builds and passes tests:
   ```bash
   go build ./...
   go test ./... -count=1 -race
   go vet ./...
   ```

## Development Workflow

### Branch Naming

Use conventional prefixes:

- `feat/<description>` -- New features (e.g., `feat/add-mistral-provider`)
- `fix/<description>` -- Bug fixes (e.g., `fix/cohere-response-parsing`)
- `refactor/<description>` -- Code restructuring
- `test/<description>` -- Test improvements
- `docs/<description>` -- Documentation changes

### Commit Messages

Follow Conventional Commits:

```
<type>(<scope>): <description>
```

Examples:
- `feat(voyage): add voyage-3-large model support`
- `fix(bedrock): handle empty Titan response`
- `test(openai): add timeout edge case`
- `docs(api): update Cohere dimension table`

### Code Style

- Run `gofmt` on all Go files before committing
- Group imports: stdlib, then third-party, then internal (blank line separated)
- Line length should not exceed 100 characters where practical
- Use `camelCase` for unexported identifiers, `PascalCase` for exported
- Error messages must start with the package name: `fmt.Errorf("openai: ...")`
- Always wrap errors with `%w` for chain inspection
- Use `defer` for resource cleanup (e.g., `resp.Body.Close()`)

### Adding a New Provider

1. Create `pkg/<name>/<name>.go`:
   - Define `Config` struct with JSON tags
   - Define `Client` struct with `config`, `httpClient`, `dimension` fields
   - Define private request/response structs for the API
   - Implement `NewClient(config Config) *Client` with default values
   - Implement all four `EmbeddingProvider` methods
   - Add `dimensionForModel` function
   - Add compile-time check: `var _ provider.EmbeddingProvider = (*Client)(nil)`

2. Create `pkg/<name>/<name>_test.go`:
   - Use `httptest.NewServer` for mock HTTP
   - Table-driven tests covering: successful embed, successful batch, API error, malformed response
   - Test `Name()` and `Dimensions()` for each supported model

3. Update documentation:
   - `README.md` -- Add row to providers table
   - `CLAUDE.md` -- Add to provider packages list
   - `docs/USER_GUIDE.md` -- Add usage section with code examples
   - `docs/API_REFERENCE.md` -- Document all exported types and functions
   - `docs/CHANGELOG.md` -- Add entry

4. Run the full test suite:
   ```bash
   go test ./... -count=1 -race
   go vet ./...
   ```

## Testing Requirements

### Unit Tests

- Every exported function and method must have tests
- Use table-driven tests with descriptive test names
- Use `httptest.NewServer` to mock HTTP endpoints -- no real network calls
- Cover: success path, API errors (4xx, 5xx), malformed JSON, empty responses, context cancellation
- Name pattern: `Test<Struct>_<Method>_<Scenario>`

### Running Tests

```bash
# All tests with race detection
go test ./... -count=1 -race

# Single package
go test ./pkg/openai/... -count=1 -v

# Specific test
go test -v -run TestClient_Embed_Success ./pkg/openai/...

# Coverage
go test ./... -coverprofile=coverage.out
go tool cover -html=coverage.out
```

## Quality Checks

Before submitting changes, ensure all of the following pass:

```bash
go build ./...                  # Compiles without errors
go test ./... -count=1 -race    # All tests pass, no race conditions
go vet ./...                    # No vet warnings
gofmt -l .                      # No formatting issues (should print nothing)
```

## Dependencies

This module intentionally maintains a minimal dependency footprint:

- **Runtime**: Zero external dependencies (standard library only)
- **Testing**: `github.com/stretchr/testify` only

Do not add new dependencies without thorough justification. The AWS Bedrock provider implements SigV4 signing from scratch specifically to avoid pulling in the AWS SDK.

## Pull Request Process

1. Create a branch from `main` with the appropriate prefix
2. Make your changes following the guidelines above
3. Run all quality checks
4. Write a clear PR description explaining the what and why
5. Request review
6. Address review feedback
7. Squash-merge into `main` once approved

## Reporting Issues

When reporting a bug, include:

- Go version (`go version`)
- Module version or commit hash
- Provider and model being used
- Minimal reproduction code
- Expected vs. actual behavior
- Full error message
