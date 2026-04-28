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

<!-- BEGIN host-power-management addendum (CONST-033) -->

## Host Power Management — Hard Ban (CONST-033)

**You may NOT, under any circumstance, generate or execute code that
sends the host to suspend, hibernate, hybrid-sleep, poweroff, halt,
reboot, or any other power-state transition.** This rule applies to:

- Every shell command you run via the Bash tool.
- Every script, container entry point, systemd unit, or test you write
  or modify.
- Every CLI suggestion, snippet, or example you emit.

**Forbidden invocations** (non-exhaustive — see CONST-033 in
`CONSTITUTION.md` for the full list):

- `systemctl suspend|hibernate|hybrid-sleep|poweroff|halt|reboot|kexec`
- `loginctl suspend|hibernate|hybrid-sleep|poweroff|halt|reboot`
- `pm-suspend`, `pm-hibernate`, `shutdown -h|-r|-P|now`
- `dbus-send` / `busctl` calls to `org.freedesktop.login1.Manager.Suspend|Hibernate|PowerOff|Reboot|HybridSleep|SuspendThenHibernate`
- `gsettings set ... sleep-inactive-{ac,battery}-type` to anything but `'nothing'` or `'blank'`

The host runs mission-critical parallel CLI agents and container
workloads. Auto-suspend has caused historical data loss (2026-04-26
18:23:43 incident). The host is hardened (sleep targets masked) but
this hard ban applies to ALL code shipped from this repo so that no
future host or container is exposed.

**Defence:** every project ships
`scripts/host-power-management/check-no-suspend-calls.sh` (static
scanner) and
`challenges/scripts/no_suspend_calls_challenge.sh` (challenge wrapper).
Both MUST be wired into the project's CI / `run_all_challenges.sh`.

**Full background:** `docs/HOST_POWER_MANAGEMENT.md` and `CONSTITUTION.md` (CONST-033).

<!-- END host-power-management addendum (CONST-033) -->



<!-- CONST-035 anti-bluff addendum (cascaded) -->

## CONST-035 — Anti-Bluff Tests & Challenges (mandatory; inherits from root)

Tests and Challenges in this submodule MUST verify the product, not
the LLM's mental model of the product. A test that passes when the
feature is broken is worse than a missing test — it gives false
confidence and lets defects ship to users. Functional probes at the
protocol layer are mandatory:

- TCP-open is the FLOOR, not the ceiling. Postgres → execute
  `SELECT 1`. Redis → `PING` returns `PONG`. ChromaDB → `GET
  /api/v1/heartbeat` returns 200. MCP server → TCP connect + valid
  JSON-RPC handshake. HTTP gateway → real request, real response,
  non-empty body.
- Container `Up` is NOT application healthy. A `docker/podman ps`
  `Up` status only means PID 1 is running; the application may be
  crash-looping internally.
- No mocks/fakes outside unit tests (already CONST-030; CONST-035
  raises the cost of a mock-driven false pass to the same severity
  as a regression).
- Re-verify after every change. Don't assume a previously-passing
  test still verifies the same scope after a refactor.
- Verification of CONST-035 itself: deliberately break the feature
  (e.g. `kill <service>`, swap a password). The test MUST fail. If
  it still passes, the test is non-conformant and MUST be tightened.

## CONST-033 clarification — distinguishing host events from sluggishness

Heavy container builds (BuildKit pulling many GB of layers, parallel
podman/docker compose-up across many services) can make the host
**appear** unresponsive — high load average, slow SSH, watchers
timing out. **This is NOT a CONST-033 violation.** Suspend / hibernate
/ logout are categorically different events. Distinguish via:

- `uptime` — recent boot? if so, the host actually rebooted.
- `loginctl list-sessions` — session(s) still active? if yes, no logout.
- `journalctl ... | grep -i 'will suspend\|hibernate'` — zero broadcasts
  since the CONST-033 fix means no suspend ever happened.
- `dmesg | grep -i 'killed process\|out of memory'` — OOM kills are
  also NOT host-power events; they're memory-pressure-induced and
  require their own separate fix (lower per-container memory limits,
  reduce parallelism).

A sluggish host under build pressure recovers when the build finishes;
a suspended host requires explicit unsuspend (and CONST-033 should
make that impossible by hardening `IdleAction=ignore` +
`HandleSuspendKey=ignore` + masked `sleep.target`,
`suspend.target`, `hibernate.target`, `hybrid-sleep.target`).

If you observe what looks like a suspend during heavy builds, the
correct first action is **not** "edit CONST-033" but `bash
challenges/scripts/host_no_auto_suspend_challenge.sh` to confirm the
hardening is intact. If hardening is intact AND no suspend
broadcast appears in journal, the perceived event was build-pressure
sluggishness, not a power transition.

<!-- BEGIN no-session-termination addendum (CONST-036) -->

## User-Session Termination — Hard Ban (CONST-036)

**You may NOT, under any circumstance, generate or execute code that
ends the currently-logged-in user's desktop session, kills their
`user@<UID>.service` user manager, or indirectly forces them to
manually log out / power off.** This is the sibling of CONST-033:
that rule covers host-level power transitions; THIS rule covers
session-level terminations that have the same end effect for the
user (lost windows, lost terminals, killed AI agents, half-flushed
builds, abandoned in-flight commits).

**Why this rule exists.** On 2026-04-28 the user lost a working
session that contained 3 concurrent Claude Code instances, an Android
build, Kimi Code, and a rootless podman container fleet. The
`user.slice` consumed 60.6 GiB peak / 5.2 GiB swap, the GUI became
unresponsive, the user was forced to log out and then power off via
the GNOME shell. The host could not auto-suspend (CONST-033 was in
place and verified) and the kernel OOM killer never fired — but the
user had to manually end the session anyway, because nothing
prevented overlapping heavy workloads from saturating the slice.
CONST-036 closes that loophole at both the source-code layer and the
operational layer. See
`docs/issues/fixed/SESSION_LOSS_2026-04-28.md` in the HelixAgent
project.

**Forbidden direct invocations** (non-exhaustive):

- `loginctl terminate-user|terminate-session|kill-user|kill-session`
- `systemctl stop user@<UID>` / `systemctl kill user@<UID>`
- `gnome-session-quit`
- `pkill -KILL -u $USER` / `killall -u $USER`
- `dbus-send` / `busctl` calls to `org.gnome.SessionManager.Logout|Shutdown|Reboot`
- `echo X > /sys/power/state`
- `/usr/bin/poweroff`, `/usr/bin/reboot`, `/usr/bin/halt`

**Indirect-pressure clauses:**

1. Do not spawn parallel heavy workloads casually; check `free -h`
   first; keep `user.slice` under 70% of physical RAM.
2. Long-lived background subagents go in `system.slice`. Rootless
   podman containers die with the user manager.
3. Document AI-agent concurrency caps in CLAUDE.md.
4. Never script "log out and back in" recovery flows.

**Defence:** every project ships
`scripts/host-power-management/check-no-session-termination-calls.sh`
(static scanner) and
`challenges/scripts/no_session_termination_calls_challenge.sh`
(challenge wrapper). Both MUST be wired into the project's CI /
`run_all_challenges.sh`.

<!-- END no-session-termination addendum (CONST-036) -->
