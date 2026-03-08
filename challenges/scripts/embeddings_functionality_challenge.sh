#!/usr/bin/env bash
# embeddings_functionality_challenge.sh - Validates Embeddings module core functionality
# Checks 6 embedding providers, provider interface, batch processing, config types
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODULE_NAME="Embeddings"

PASS=0
FAIL=0
TOTAL=0

pass() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); echo "  FAIL: $1"; }

echo "=== ${MODULE_NAME} Functionality Challenge ==="
echo ""

# --- Section 1: Required packages (7) ---
echo "Section 1: Required packages (7)"

for pkg in openai cohere voyage jina google bedrock provider; do
    echo "Test: Package pkg/${pkg} exists"
    if [ -d "${MODULE_DIR}/pkg/${pkg}" ]; then
        pass "Package pkg/${pkg} exists"
    else
        fail "Package pkg/${pkg} missing"
    fi
done

# --- Section 2: Provider interface ---
echo ""
echo "Section 2: Provider interface"

echo "Test: EmbeddingProvider interface exists"
if grep -q "type EmbeddingProvider interface" "${MODULE_DIR}/pkg/provider/"*.go 2>/dev/null; then
    pass "EmbeddingProvider interface exists"
else
    fail "EmbeddingProvider interface missing"
fi

echo "Test: Result struct exists"
if grep -q "type Result struct" "${MODULE_DIR}/pkg/provider/"*.go 2>/dev/null; then
    pass "Result struct exists"
else
    fail "Result struct missing"
fi

echo "Test: TokenUsage struct exists"
if grep -q "type TokenUsage struct" "${MODULE_DIR}/pkg/provider/"*.go 2>/dev/null; then
    pass "TokenUsage struct exists"
else
    fail "TokenUsage struct missing"
fi

echo "Test: Provider Config struct exists"
if grep -q "type Config struct" "${MODULE_DIR}/pkg/provider/"*.go 2>/dev/null; then
    pass "Provider Config struct exists"
else
    fail "Provider Config struct missing"
fi

# --- Section 3: Each provider has Client and Config ---
echo ""
echo "Section 3: Provider implementations"

for provider in openai cohere voyage jina google bedrock; do
    echo "Test: ${provider} has Client struct"
    if grep -q "type Client struct" "${MODULE_DIR}/pkg/${provider}/"*.go 2>/dev/null; then
        pass "${provider} Client struct exists"
    else
        fail "${provider} Client struct missing"
    fi

    echo "Test: ${provider} has Config struct"
    if grep -q "type Config struct" "${MODULE_DIR}/pkg/${provider}/"*.go 2>/dev/null; then
        pass "${provider} Config struct exists"
    else
        fail "${provider} Config struct missing"
    fi
done

# --- Section 4: Batch embedding support ---
echo ""
echo "Section 4: Batch embedding support"

echo "Test: Embed or EmbedBatch method exists somewhere"
if grep -rqE "func.*\b(Embed|EmbedBatch|EmbedTexts)\b" "${MODULE_DIR}/pkg/" 2>/dev/null; then
    pass "Embed method exists in provider implementations"
else
    fail "No Embed method found in provider implementations"
fi

echo "Test: EmbeddingProvider interface has Embed method"
if grep -rqE "(Embed|EmbedTexts|EmbedBatch)" "${MODULE_DIR}/pkg/provider/"*.go 2>/dev/null; then
    pass "EmbeddingProvider interface includes embed method"
else
    fail "EmbeddingProvider interface missing embed method"
fi

# --- Section 5: Provider count ---
echo ""
echo "Section 5: Provider count"

provider_count=0
for provider in openai cohere voyage jina google bedrock; do
    if [ -d "${MODULE_DIR}/pkg/${provider}" ]; then
        provider_count=$((provider_count + 1))
    fi
done
echo "Test: At least 6 embedding providers present (found: ${provider_count})"
if [ "$provider_count" -ge 6 ]; then
    pass "At least 6 embedding providers present"
else
    fail "Expected at least 6 providers, found ${provider_count}"
fi

# --- Section 6: Source structure completeness ---
echo ""
echo "Section 6: Source structure"

echo "Test: Each package has non-test Go source files"
all_have_source=true
for pkg in openai cohere voyage jina google bedrock provider; do
    non_test=$(find "${MODULE_DIR}/pkg/${pkg}" -name "*.go" ! -name "*_test.go" -type f 2>/dev/null | wc -l)
    if [ "$non_test" -eq 0 ]; then
        fail "Package pkg/${pkg} has no non-test Go files"
        all_have_source=false
    fi
done
if [ "$all_have_source" = true ]; then
    pass "All packages have non-test Go source files"
fi

echo ""
echo "=== Results: ${PASS}/${TOTAL} passed, ${FAIL} failed ==="
[ "${FAIL}" -eq 0 ] && exit 0 || exit 1
