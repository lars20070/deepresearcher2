# Optimizations for `utils.py` and `test_utils.py`

## Overview

Six optimizations identified across the two files, ranging from type-safety bugs to DRY violations and unnecessary network calls in tests.

---

## Changes

### 1. Switch `fetch_full_page_content` from `urllib` to `requests` (`utils.py:78-157`)

The function manually handles gzip/deflate/brotli decompression (~15 lines) using `urllib`. The `requests` library (already imported) handles this automatically.

**Replace** the entire function body with `requests.get()`, catching `requests.exceptions.*` instead of `urllib.error.*`.

**Remove imports** no longer needed: `gzip`, `urllib.error`, `urllib.request`, `zlib`, `brotli`.

### 2. Fix type safety: `fetch_full_page_content` signature `HttpUrl` -> `str` (`utils.py:78`)

Four callers pass raw `str` but the signature expects `HttpUrl`. The function internally does `str(url)` anyway.

- Change signature to `url: str`
- Remove `HttpUrl(url)` validation wrapper in `duckduckgo_search` (lines 204-208)
- Remove `from pydantic import HttpUrl, ValidationError` import (no longer used in utils.py)

### 3. Extract `_resolve_content` helper to eliminate DRY violation (`utils.py`)

All 5 search functions (excl. perplexity) repeat the fetch-content/fallback/truncate pattern. Extract into:

```python
def _resolve_content(
    url: str | None,
    existing_content: str | None,
    fallback: str | None,
    max_content_length: int | None,
) -> str | None:
```

This helper also incorporates the `should_fetch` optimization (Issue 4 below).

### 4. Add `should_fetch` optimization to `brave_search`, `serper_search`, `searxng_search`

Currently these always call `fetch_full_page_content()` even when `max_content_length` is small and a summary is already available. The `_resolve_content` helper includes the smart `should_fetch` logic already used by `duckduckgo_search` and `tavily_search`.

For brave/serper/searxng, pass `existing_content=summary` so the fetch is skipped when the summary already meets the length constraint.

### 5. Extract shared test assertion helper (`test_utils.py`)

`test_brave_search`, `test_serper_search`, `test_searxng_search` have identical assertion blocks. Extract into `_assert_search_results()`. Keep separate test functions to preserve VCR cassette naming.

### 6. Mock `test_fetch_full_page_content` (`test_utils.py:25-43`)

Currently hits Wikipedia live (slow, flaky). Mock `requests.get` to return canned HTML. Remove unused `from pydantic import HttpUrl, ValidationError` import from test file.

---

## Files to modify

| File | Changes |
|------|---------|
| `src/deepresearcher2/utils.py` | Rewrite `fetch_full_page_content` to use `requests`; change signature to `str`; add `_resolve_content` helper; simplify all 5 search functions; clean imports |
| `tests/test_utils.py` | Mock `test_fetch_full_page_content`; extract `_assert_search_results` helper; simplify VCR tests; clean imports |

## Implementation order

1. `utils.py`: Rewrite `fetch_full_page_content` (requests + str signature) + clean imports
2. `utils.py`: Add `_resolve_content` helper
3. `utils.py`: Refactor all 5 search functions to use `_resolve_content`
4. `test_utils.py`: Mock `test_fetch_full_page_content`
5. `test_utils.py`: Extract `_assert_search_results` and simplify VCR tests
6. `test_utils.py`: Clean imports

## Verification

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyright .
uv run pytest tests/test_utils.py -v
uv run pytest -n auto
```
