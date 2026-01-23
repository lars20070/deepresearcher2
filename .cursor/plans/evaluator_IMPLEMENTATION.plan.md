---
name: Evaluator Callable Class Pattern
overview: Implement the callable class pattern (structlog-style) for pytest evaluator plugins, where configuration lives in each evaluator's `__init__` and the Protocol defines only the execution interface.
todos:
  - id: protocol-evalresult
    content: Define `EvalResult` dataclass and `Evaluator` Protocol in plugin.py
    status: completed
  - id: evaluator-class
    content: Create `BradleyTerryEvaluator` class with `__init__` configuration and `__call__` execution
    status: completed
  - id: update-hook
    content: Update `pytest_runtest_makereport` to use `BradleyTerryEvaluator()` as default and handle `EvalResult`
    status: completed
  - id: update-marker-docs
    content: Update marker documentation in `pytest_configure` to reflect new pattern
    status: completed
  - id: deprecate-function
    content: Add deprecation warning to `bradley_terry_evaluation()` function for backwards compatibility
    status: completed
  - id: update-tests
    content: Update tests in `test_plugin.py` for new class-based evaluator pattern
    status: completed
isProject: false
---

# Callable Class Pattern for Evaluator Configuration

## Design Overview

Adopt the **callable class pattern** where evaluators are instantiated with configuration parameters and the marker receives a **configured instance**:

```python
# User writes this:
@pytest.mark.assay(evaluator=BradleyTerryEvaluator(criterion="...", temperature=0.1))
async def test_example(): ...
```

The Protocol defines only the `__call__` signature (what the plugin invokes), while each evaluator class documents its own configuration via `__init__` with full type hints.

## Current State

- [plugin.py](src/deepresearcher2/plugin.py): `EvaluationStrategy = Callable[[Item], Coroutine[Any, Any, None]]` (line 37)
- `bradley_terry_evaluation` is a standalone async function (lines 311-371)
- Marker invocation: `asyncio.run(evaluator(item))` (line 306)

## Implementation

### Step 1: Define the Evaluator Protocol and EvalResult dataclass

Add to [plugin.py](src/deepresearcher2/plugin.py):

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class EvalResult:
    """Result from an evaluator execution."""
    score: float | None = None
    passed: bool = True
    details: dict[str, Any] | None = None

class Evaluator(Protocol):
    """Protocol for evaluation strategy callables.
    
    The Protocol defines ONLY what the plugin needs to call.
    Evaluator implementations configure themselves via __init__.
    """
    def __call__(self, item: Item) -> Coroutine[Any, Any, EvalResult]: ...
```

The Protocol uses `Item` as the sole parameter since the plugin passes the test item. Each evaluator extracts what it needs from `item.funcargs["assay"]` and `item.stash[AGENT_RESPONSES_KEY]`.

### Step 2: Create the BradleyTerryEvaluator class

Convert `bradley_terry_evaluation()` to a callable class. This is the primary change:

```python
class BradleyTerryEvaluator:
    """Evaluates test outputs using Bradley-Terry tournament scoring.
    
    Configuration is set at instantiation; __call__ runs the evaluation.
    """
    
    def __init__(
        self,
        *,
        criterion: str = "Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
        max_standard_deviation: float = 2.0,
        temperature: float = 0.0,
        timeout: int = 300,
    ):
        """Configure the evaluator.
        
        Args:
            criterion: The evaluation criterion for pairwise comparison.
            max_standard_deviation: Convergence threshold for adaptive strategy.
            temperature: Model sampling temperature (0.0 = deterministic).
            timeout: Model request timeout in seconds.
        """
        self.criterion = criterion
        self.max_standard_deviation = max_standard_deviation
        self.temperature = temperature
        self.timeout = timeout
    
    async def __call__(self, item: Item) -> EvalResult:
        """Run Bradley-Terry tournament on baseline and novel responses."""
        # ... implementation using self.criterion, self.temperature, etc.
```

### Step 3: Update `pytest_runtest_makereport` hook

The hook already supports callables; minimal changes needed. Update type annotation and logging:

```python
# Line ~299 in plugin.py
evaluator: Evaluator = marker.kwargs.get("evaluator", BradleyTerryEvaluator())
if not callable(evaluator):
    logger.error(f"Invalid evaluator type: {type(evaluator)}. Expected callable.")
    return

# Run the async evaluation
try:
    result = asyncio.run(evaluator(item))
    logger.info(f"Evaluation result: score={result.score}, passed={result.passed}")
except Exception:
    logger.exception("Error during evaluation in pytest_runtest_makereport.")
```

Note: The default is now `BradleyTerryEvaluator()` (an instance), not a function reference.

### Step 4: Update marker documentation

Update `pytest_configure` (lines 56-72):

```python
config.addinivalue_line(
    "markers",
    "assay(generator=None, evaluator=BradleyTerryEvaluator()): "
    "Mark the test for AI agent evaluation (assay). "
    "Args: "
    "generator - optional callable returning a Dataset for test cases; "
    "evaluator - optional Evaluator instance for custom evaluation strategy "
    "(defaults to BradleyTerryEvaluator with default settings). "
    "Configure evaluators by instantiating with parameters: "
    "evaluator=BradleyTerryEvaluator(criterion='...', temperature=0.1)",
)
```

### Step 5: Deprecate the old function-based API (optional)

Keep `bradley_terry_evaluation` as a thin wrapper for backwards compatibility:

```python
async def bradley_terry_evaluation(item: Item) -> None:
    """DEPRECATED: Use BradleyTerryEvaluator() instead.
    
    This function is provided for backwards compatibility.
    """
    import warnings
    warnings.warn(
        "bradley_terry_evaluation() is deprecated. "
        "Use BradleyTerryEvaluator() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    evaluator = BradleyTerryEvaluator()
    await evaluator(item)
```

## Files to Modify

- [src/deepresearcher2/plugin.py](src/deepresearcher2/plugin.py): All implementation changes
- [tests/test_plugin.py](tests/test_plugin.py): Update tests for new pattern

## Testing Strategy

1. **Unit tests for BradleyTerryEvaluator**:

   - Test `__init__` stores configuration correctly
   - Test `__call__` uses configured values
   - Test default values match current behavior

2. **Protocol conformance tests**:

   - Verify `BradleyTerryEvaluator` satisfies `Evaluator` protocol
   - Test custom evaluator classes work with the marker

3. **Integration tests**:

   - Test marker with configured evaluator instance
   - Test backwards compatibility with function-based evaluators

4. **Update existing tests** in `test_plugin.py`:

   - `test_pytest_runtest_makereport_runs_evaluation`: Use `BradleyTerryEvaluator()`
   - `test_bradley_terry_evaluation_*`: Test class-based evaluator

## Usage Examples After Implementation

```python
# Default behavior (same as before)
@pytest.mark.assay()
async def test_default(): ...

# Custom criterion
@pytest.mark.assay(evaluator=BradleyTerryEvaluator(
    criterion="Which response is more helpful?",
    temperature=0.1,
))
async def test_custom_criterion(): ...

# Future custom evaluator
class MyCustomEvaluator:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    async def __call__(self, item: Item) -> EvalResult:
        # Custom logic
        return EvalResult(score=0.9, passed=True)

@pytest.mark.assay(evaluator=MyCustomEvaluator(threshold=0.7))
async def test_custom_evaluator(): ...
```

## Benefits

1. **Type safety**: Protocol enforces `__call__` signature; `__init__` has full type hints
2. **IDE discoverability**: Autocomplete shows all configuration options when typing `BradleyTerryEvaluator(`
3. **Backwards compatibility**: Add new `__init__` parameters with defaults without breaking existing code
4. **Validation**: Can add Pydantic/attrs validation in `__init__` if needed
5. **Clean API**: No special kwargs handling in the plugin; evaluators are self-contained