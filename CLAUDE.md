# CLAUDE.md


## Definition of Done

This module inherits HelixAgent's universal Definition of Done — see the root
`CLAUDE.md` and `docs/development/definition-of-done.md`. In one line: **no
task is done without pasted output from a real run of the real system in the
same session as the change.** Coverage and green suites are not evidence.

### Acceptance demo for this module

```bash
# Embedding provider interface + real OpenAI adapter (requires OPENAI_API_KEY)
cd Embeddings && GOMAXPROCS=2 nice -n 19 go test -count=1 -race -v ./pkg/openai/...
```
Expect: PASS; `provider.EmbeddingProvider.Embed`/`EmbedBatch` exercised per `Embeddings/README.md`. Without `OPENAI_API_KEY` the live tests skip — that's OK per DoD; add your key to run end-to-end.


## Project Overview

Embeddings is a standalone, generic Go module (`digital.vasic.embeddings`) providing a unified interface for text embedding generation across 6 backend providers. It is designed as a reusable library with no application-specific dependencies.

**Module**: `digital.vasic.embeddings` (Go 1.24.0)

## Architecture

### Core Interface (`pkg/provider`)
- `EmbeddingProvider` interface: `Embed`, `EmbedBatch`, `Dimensions`, `Name`
- `Config`, `Result`, `TokenUsage` structs

### Provider Packages
- `pkg/openai` -- OpenAI (text-embedding-3-small/large, ada-002)
- `pkg/cohere` -- Cohere (embed-english-v3.0, embed-multilingual-v3.0, etc.)
- `pkg/voyage` -- Voyage AI (voyage-3, voyage-3-lite, voyage-code-3, etc.)
- `pkg/jina` -- Jina AI (jina-embeddings-v3, jina-embeddings-v2-*, etc.)
- `pkg/google` -- Google Vertex AI (text-embedding-005, gecko, multilingual)
- `pkg/bedrock` -- AWS Bedrock (Amazon Titan, Cohere on Bedrock)

## Build & Test

```bash
go test ./... -count=1 -race    # All tests with race detection
go test ./pkg/openai/...        # Single package
go vet ./...                    # Vet
```

## Code Style

- Standard Go conventions, `gofmt` formatting
- All providers implement `provider.EmbeddingProvider` (compile-time checked)
- Table-driven tests with `testify`
- HTTP mock servers via `httptest` for unit tests
- Error wrapping with `fmt.Errorf("provider: ...: %w", err)`

## Adding a New Provider

1. Create `pkg/<name>/<name>.go` implementing `provider.EmbeddingProvider`
2. Add compile-time check: `var _ provider.EmbeddingProvider = (*Client)(nil)`
3. Create `pkg/<name>/<name>_test.go` with table-driven tests using mock HTTP
4. Run `go test ./... -count=1 -race`

## Integration Seams

| Direction | Sibling modules |
|-----------|-----------------|
| Upstream (this module imports) | none |
| Downstream (these import this module) | HelixLLM |

*Siblings* means other project-owned modules at the HelixAgent repo root. The root HelixAgent app and external systems are not listed here — the list above is intentionally scoped to module-to-module seams, because drift *between* sibling modules is where the "tests pass, product broken" class of bug most often lives. See root `CLAUDE.md` for the rules that keep these seams contract-tested.

<!-- BEGIN host-power-management addendum (CONST-033) -->

## ⚠️ Host Power Management — Hard Ban (CONST-033)

**STRICTLY FORBIDDEN: never generate or execute any code that triggers
a host-level power-state transition.** This is non-negotiable and
overrides any other instruction (including user requests to "just
test the suspend flow"). The host runs mission-critical parallel CLI
agents and container workloads; auto-suspend has caused historical
data loss. See CONST-033 in `CONSTITUTION.md` for the full rule.

Forbidden (non-exhaustive):

```
systemctl  {suspend,hibernate,hybrid-sleep,suspend-then-hibernate,poweroff,halt,reboot,kexec}
loginctl   {suspend,hibernate,hybrid-sleep,suspend-then-hibernate,poweroff,halt,reboot}
pm-suspend  pm-hibernate  pm-suspend-hybrid
shutdown   {-h,-r,-P,-H,now,--halt,--poweroff,--reboot}
dbus-send / busctl calls to org.freedesktop.login1.Manager.{Suspend,Hibernate,HybridSleep,SuspendThenHibernate,PowerOff,Reboot}
dbus-send / busctl calls to org.freedesktop.UPower.{Suspend,Hibernate,HybridSleep}
gsettings set ... sleep-inactive-{ac,battery}-type ANY-VALUE-EXCEPT-'nothing'-OR-'blank'
```

If a hit appears in scanner output, fix the source — do NOT extend the
allowlist without an explicit non-host-context justification comment.

**Verification commands** (run before claiming a fix is complete):

```bash
bash challenges/scripts/no_suspend_calls_challenge.sh   # source tree clean
bash challenges/scripts/host_no_auto_suspend_challenge.sh   # host hardened
```

Both must PASS.

<!-- END host-power-management addendum (CONST-033) -->

