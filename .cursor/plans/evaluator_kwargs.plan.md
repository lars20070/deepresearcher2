---
name: ""
overview: ""
todos: []
isProject: false
---

# Enable Passing Arbitrary Parameters to Evaluators

## Problem Summary

Currently, `EvaluationStrategy` is defined as:

```python
EvaluationStrategy = Callable[[Item], Coroutine[Any, Any, None]]
```

This doesn't allow passing any additional configuration. Users who want to customize `bradley_terry_evaluation` (e.g., change `criterion`, `max_standard_deviation`, or `temperature`) have no way to do so through the `@pytest.mark.assay()` marker.

## Proposed Solution

Use a **Protocol with `**kwargs`** pattern, where all marker kwargs (except reserved ones like `generator` and `evaluator`) are automatically passed to the evaluator function.

## Implementation Steps

### 1. Update `EvaluationStrategy` Type Definition

Change from a simple `Callable` type alias to a `Protocol` that documents the expected signature:

```python
from typing import Protocol

class EvaluationStrategy(Protocol):
    """Protocol for evaluation strategy functions.
    
    Evaluators receive the pytest Item and any additional kwargs from the marker.
    Custom evaluators should accept **kwargs for forward compatibility.
    """
    async def __call__(self, item: Item, **kwargs: Any) -> None: ...
```

### 2. Define Reserved Marker Keywords

Add a constant to clearly separate marker-specific kwargs from evaluator kwargs:

```python
# Reserved kwargs that are consumed by the plugin, not passed to evaluator
_RESERVED_MARKER_KWARGS = frozenset({"generator", "evaluator"})
```

### 3. Update `pytest_runtest_makereport` Hook

Extract and pass evaluator kwargs:

```python
# Extract all non-reserved kwargs to pass to evaluator
evaluator_kwargs = {k: v for k, v in marker.kwargs.items() if k not in _RESERVED_MARKER_KWARGS}

# Run the async evaluation strategy synchronously
try:
    asyncio.run(evaluator(item, **evaluator_kwargs))
except Exception:
    logger.exception("Error during evaluation in pytest_runtest_makereport.")
```

### 4. Update `bradley_terry_evaluation` Signature

Add explicit parameters with defaults, plus `**kwargs` for forward compatibility:

```python
async def bradley_terry_evaluation(
    item: Item,
    *,
    criterion: str = "Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
    max_standard_deviation: float = 2.0,
    temperature: float = 0.0,
    timeout: int = 300,
    **kwargs: Any,  # Accept and ignore unknown kwargs for forward compatibility
) -> None:
```

### 5. Update Marker Documentation

Update the marker registration in `pytest_configure`:

```python
config.addinivalue_line(
    "markers",
    "assay(generator=None, evaluator=bradley_terry_evaluation, **kwargs): "
    "Mark the test for AI agent evaluation (assay). "
    "Args: "
    "generator - optional callable returning a Dataset for test cases; "
    "evaluator - optional async callable(Item, **kwargs) -> None for custom evaluation strategy "
    "(defaults to Bradley-Terry tournament); "
    "**kwargs - additional keyword arguments passed to the evaluator.",
)
```

## Usage Examples

After implementation, users can customize evaluations like this:

```python
# Customize criterion and temperature
@pytest.mark.assay(
    generator=generate_evaluation_cases,
    criterion="Which response is more helpful and accurate?",
    temperature=0.1,
)
async def test_helpfulness(assay: AssayContext) -> None:
    ...

# Customize max_standard_deviation for stricter convergence
@pytest.mark.assay(
    evaluator=bradley_terry_evaluation,
    max_standard_deviation=1.5,
    timeout=600,
)
async def test_with_strict_convergence(assay: AssayContext) -> None:
    ...

# Custom evaluator with custom parameters
@pytest.mark.assay(
    evaluator=my_custom_evaluator,
    my_custom_param="value",
    another_param=42,
)
async def test_with_custom_eval(assay: AssayContext) -> None:
    ...
```

## Benefits

1. **Backwards compatible** - Existing evaluators that don't accept kwargs will still work (Python allows extra kwargs to be ignored via `**kwargs`)
2. **Type-safe** - Protocol clearly defines the expected interface
3. **Clean API** - No need for wrapper objects or configuration classes
4. **Flexible** - Any evaluator can define its own parameters
5. **Discoverable** - IDE autocomplete can show available parameters for `bradley_terry_evaluation`

## Files to Modify

- [src/deepresearcher2/plugin.py](src/deepresearcher2/plugin.py) - Update type, hook, and default evaluator

## Testing Considerations

- Add a test that passes custom kwargs through the marker
- Verify backwards compatibility with existing tests
- Test that unknown kwargs are handled gracefully