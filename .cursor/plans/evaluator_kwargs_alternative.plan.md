---
name: ""
overview: ""
todos: []
isProject: false
---

# Designing Python callable protocols for pytest plugins

**Protocol with `**kwargs` is not the best approach.** For pytest plugin evaluator callables, the structlog-style pattern—using a class with `__init__` for configuration and `__call__` for execution—provides superior type safety, IDE discoverability, and ergonomics. If `**kwargs` semantics are truly required, TypedDict with `Unpack` (Python 3.12+) is the only type-safe option, though it comes with significant trade-offs.

## The core problem with Protocol and **kwargs

Defining a Protocol with catch-all `**kwargs` creates fundamental typing problems. When a Protocol's `__call__` defines `*args, **kwargs`, Pyright and mypy will reject functions that don't have those exact parameter signatures:

```python
class EvaluatorProtocol(Protocol):
    def __call__(self, result: Any, **kwargs: Any) -> float: ...

def simple_evaluator(result: Any) -> float:  # No kwargs
    return float(result)

# ERROR: Parameter "**kwargs" has no corresponding parameter
evaluator: EvaluatorProtocol = simple_evaluator  # Fails type check!
```

This defeats the purpose of accepting arbitrary kwargs—implementers are forced to add unused `**kwargs` to their signatures, creating boilerplate and potential bugs. The **explicit parameters approach** in Protocols is vastly superior for type safety and IDE support.

## Four viable patterns for pytest evaluator configuration

### Pattern 1: Callable class with Protocol (recommended)

This pattern, used successfully by structlog, separates configuration (constructor) from execution (`__call__`):

```python
from typing import Protocol, Any

class Evaluator(Protocol):
    """Fixed signature Protocol - explicit about what __call__ receives."""
    def __call__(self, result: Any, context: dict[str, Any]) -> float: ...

# Implementation: configuration via __init__, execution via __call__
class ThresholdEvaluator:
    def __init__(self, threshold: float = 0.5, strict: bool = False):
        self.threshold = threshold
        self.strict = strict
    
    def __call__(self, result: Any, context: dict[str, Any]) -> float:
        score = float(result.get("score", 0))
        if self.strict and score < self.threshold:
            return 0.0
        return score

# Usage in pytest marker
@pytest.mark.evaluate(evaluator=ThresholdEvaluator(threshold=0.7, strict=True))
def test_model_accuracy(): ...
```

The marker receives a **configured instance**, not a callable plus kwargs. Each evaluator class documents its own configuration options with full type hints, defaults, and docstrings. IDEs provide complete autocomplete when instantiating `ThresholdEvaluator(...)`.

### Pattern 2: TypedDict with Unpack (Python 3.12+)

If kwargs semantics are genuinely required, PEP 692's `Unpack[TypedDict]` is the **only way** to get typed `**kwargs`:

```python
from typing import TypedDict, Unpack, NotRequired, Protocol

class EvaluatorKwargs(TypedDict):
    threshold: NotRequired[float]
    normalize: NotRequired[bool]
    weights: NotRequired[dict[str, float]]

class TypedEvaluator(Protocol):
    def __call__(
        self, 
        result: Any, 
        **kwargs: Unpack[EvaluatorKwargs]
    ) -> float: ...

def weighted_evaluator(
    result: Any, 
    **kwargs: Unpack[EvaluatorKwargs]
) -> float:
    threshold = kwargs.get("threshold", 0.5)
    weights = kwargs.get("weights", {})
    # Type checker knows these types
    return sum(result[k] * v for k, v in weights.items())
```

This provides IDE autocomplete for kwargs keys and type checking for values. However, it requires Python 3.12+ (or `typing_extensions`), has inconsistent type checker support for edge cases, and forces all implementations to use the same kwargs signature.

### Pattern 3: Dataclass configuration objects

Frozen dataclasses provide excellent IDE support and immutability:

```python
from dataclasses import dataclass
from typing import Protocol, Any

@dataclass(frozen=True, kw_only=True)
class EvaluatorConfig:
    threshold: float = 0.5
    normalize: bool = True
    metric: str = "accuracy"

class ConfiguredEvaluator(Protocol):
    def __call__(
        self, 
        result: Any, 
        config: EvaluatorConfig
    ) -> float: ...

# Marker usage
@pytest.mark.evaluate(
    evaluator=my_evaluator,
    config=EvaluatorConfig(threshold=0.8, metric="f1")
)
def test_classifier(): ...
```

This separates configuration from the callable entirely. The `kw_only=True` parameter (Python 3.10+) forces named arguments, improving readability. Frozen dataclasses prevent accidental mutation.

### Pattern 4: functools.partial (limited typing)

For simple cases, `functools.partial` pre-binds configuration:

```python
from functools import partial

def threshold_evaluator(
    result: Any, 
    threshold: float = 0.5, 
    strict: bool = False
) -> float:
    score = result.get("score", 0)
    return 0.0 if strict and score < threshold else score

# Create configured versions
strict_eval = partial(threshold_evaluator, threshold=0.9, strict=True)
lenient_eval = partial(threshold_evaluator, threshold=0.3)
```

**Critical limitation**: mypy and pyright have poor support for `partial` typing (issue open since 2016). The resulting `partial[float]` type loses parameter information, breaking IDE autocomplete and static analysis.

## How major pytest plugins handle this

The established plugins reveal a clear pattern: **explicit configuration over arbitrary kwargs**.

**hypothesis** uses a custom immutable settings class with runtime validation. The `@settings(max_examples=200)` decorator accepts explicit, documented parameters—not arbitrary kwargs. Each setting has type hints, defaults, validation, and documentation.

**pytest-benchmark** takes the simpler approach of marker kwargs without validation:
```python
@pytest.mark.benchmark(group="math", min_rounds=5, warmup=True)
```
This works but provides no type safety or IDE support for the kwargs.

**pytest-asyncio** uses explicitly named marker kwargs (`loop_scope="module"`) with deprecation warnings when changing APIs. This demonstrates the backwards compatibility pattern: add new parameter names while deprecating old ones with clear warnings.

**pytest's Mark class** itself is a frozen dataclass storing `args: tuple[Any, ...]` and `kwargs: Mapping[str, Any]`—it doesn't try to type the contents, leaving validation to plugins.

## Type checker support comparison

| Approach | mypy | pyright | IDE autocomplete |
|----------|------|---------|------------------|
| Protocol + explicit params | ✅ Full | ✅ Full | ⭐⭐⭐⭐⭐ |
| TypedDict + Unpack | ✅ Supported | ✅ Excellent | ⭐⭐⭐⭐ |
| Callable class pattern | ✅ Full | ✅ Full | ⭐⭐⭐⭐⭐ |
| functools.partial | ⚠️ Loses types | ⚠️ Loses types | ⭐⭐ |
| Protocol + `**kwargs: Any` | ❌ Breaks | ❌ Breaks | ⭐ |

ParamSpec (PEP 612) is designed for **decorator signature preservation**, not configuration passing. It's excellent for wrapping functions while preserving their signatures but doesn't solve the evaluator configuration problem.

## Recommended architecture for pytest evaluator plugins

For a pytest plugin accepting evaluator callables through markers, the cleanest design combines the callable class pattern with a fixed Protocol:

```python
from typing import Protocol, Any, TypeVar
from dataclasses import dataclass

# The Protocol defines ONLY what the plugin needs to call
class Evaluator(Protocol):
    def __call__(self, result: Any, test_name: str) -> EvalResult: ...

@dataclass
class EvalResult:
    score: float
    passed: bool
    details: dict[str, Any] | None = None

# Evaluator implementations configure themselves
class LLMJudgeEvaluator:
    """Evaluates using an LLM judge with configurable criteria."""
    
    def __init__(
        self, 
        model: str = "gpt-4",
        criteria: list[str] | None = None,
        temperature: float = 0.0
    ):
        self.model = model
        self.criteria = criteria or ["accuracy", "relevance"]
        self.temperature = temperature
    
    def __call__(self, result: Any, test_name: str) -> EvalResult:
        # Implementation using self.model, self.criteria, etc.
        ...

# Marker receives configured instance
@pytest.mark.llm_eval(evaluator=LLMJudgeEvaluator(model="claude-3", criteria=["safety"]))
def test_response_safety(): ...
```

This design achieves several goals simultaneously. **Type safety** is complete—the Protocol enforces the `__call__` signature, and each evaluator's `__init__` has full type hints. **IDE discoverability** is excellent because users get autocomplete when typing `LLMJudgeEvaluator(` showing all configuration options. **Backwards compatibility** is straightforward: add new `__init__` parameters with defaults without breaking existing code. **Validation** can be added in `__init__` using Pydantic, attrs, or manual checks.

## Backwards compatibility strategies

When evolving the configuration API:

1. **Adding parameters**: Always provide defaults—new parameters with defaults never break existing code
2. **Renaming parameters**: Support both old and new names, emit `DeprecationWarning` for old names (pytest-asyncio pattern)
3. **Removing parameters**: Deprecation cycle of 2+ minor versions with clear migration docs
4. **TypedDict evolution**: Use `NotRequired` for new keys; removal is always breaking

For the callable class pattern, version your evaluator base classes if you need to change the Protocol signature:

```python
class EvaluatorV1(Protocol):  # Original
    def __call__(self, result: Any) -> float: ...

class EvaluatorV2(Protocol):  # Extended
    def __call__(self, result: Any, context: TestContext) -> EvalResult: ...
```

## Conclusion

**Avoid Protocol with `**kwargs: Any`**—it breaks type checking and provides no IDE support. For pytest evaluator plugins, the callable class pattern (structlog-style) offers the best balance of flexibility, type safety, and developer experience. Configuration lives in each evaluator's `__init__` with full type hints; the Protocol defines only the execution interface. If you need true kwargs semantics, TypedDict with `Unpack` is viable but requires Python 3.12+ and careful consideration of the trade-offs. The most successful pytest plugins (hypothesis, pytest-asyncio) use explicit, documented configuration parameters rather than arbitrary kwargs.