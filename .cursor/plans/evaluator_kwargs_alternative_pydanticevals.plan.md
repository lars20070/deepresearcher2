# Pydantic Evals architecture limits tournament-style evaluators to post-processing

Pydantic Evals' evaluator architecture is fundamentally **case-level only**—evaluators receive a single `EvaluatorContext` containing one case's data, with **no access to the Dataset or other cases**. For Bradley-Terry or comparative evaluators that need multiple cases, the framework provides no built-in extension points, requiring a two-phase architecture: standard pydantic-evals evaluation followed by external tournament processing.

## Core architecture centers on isolated case evaluation

The library follows a code-first philosophy built around four core classes. **Dataset** is a generic `BaseModel` containing a collection of `Case` objects and dataset-level evaluators, with type parameters `Dataset[InputsT, OutputT, MetadataT]`. **Case** represents a single test scenario with `inputs`, `expected_output`, `metadata`, and optional case-specific evaluators. **Evaluator** is the abstract base class all evaluators must inherit from, decorated with `@dataclass`. **EvaluatorContext** is the sole input to all evaluators, containing everything needed to evaluate a single case.

The `Evaluator` class signature reveals the fundamental design constraint:

```python
@dataclass(repr=False)
class Evaluator(Generic[InputsT, OutputT, MetadataT], metaclass=_StrictABCMeta):
    @abstractmethod
    def evaluate(
        self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]
    ) -> EvaluatorOutput | Awaitable[EvaluatorOutput]:
        ...
```

The `EvaluatorContext` dataclass provides **only single-case data**: `name`, `inputs`, `output`, `expected_output`, `metadata`, `duration`, `attributes`, `metrics`, and `span_tree` for OpenTelemetry traces. Critically, there is no reference to the `Dataset`, no access to other cases' inputs or outputs, and no aggregated statistics.

## Evaluator return types offer flexibility but remain case-scoped

Evaluators can return several types bundled under `EvaluatorOutput`:

- **Boolean assertions** (`bool`): Pass/fail checks appearing as ✔ or ✗ in reports
- **Numeric scores** (`float`): Quality metrics typically in 0.0-1.0 range
- **String labels** (`str`): Categorical classifications
- **EvaluationReason**: Value with explanatory `reason` string attached
- **Dictionary mapping**: Multiple results from one evaluator via `dict[str, EvaluationScalar | EvaluationReason]`
- **Empty dict** (`{}`): Conditional skip—evaluator doesn't apply to this case

The multiple-results pattern enables comprehensive single-case checks but doesn't enable cross-case access:

```python
@dataclass
class ComprehensiveCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float | str]:
        return {
            'valid_format': self._check_format(ctx.output),
            'quality_score': self._score_quality(ctx.output),
            'category': self._classify(ctx.output),
        }
```

## Dataset.evaluate() executes cases concurrently with no cross-case hooks

When `dataset.evaluate(task)` runs, it processes all cases through `task_group_gather` with optional `max_concurrency` limiting via `anyio.Semaphore`. Each case executes independently: the task runs with case inputs, an `EvaluatorContext` is constructed with results, all applicable evaluators (dataset-level + case-specific) execute against that context, and results aggregate into a `ReportCase`. After all cases complete, `EvaluationReport` is assembled.

The architecture provides **no extension points between evaluator execution and report assembly**. There is no `on_all_cases_complete()` hook, no aggregate evaluator concept, and no way for an evaluator to request other cases' data. This design enables efficient concurrent execution but architecturally precludes comparative evaluation within the framework.

GitHub Issue #1413 ("Calculating custom aggregate metrics for evals") confirms this limitation was raised and closed as "not planned." Users requesting precision/recall or similar cross-case metrics were advised to post-process `EvaluationReport.cases` externally.

## EvaluatorContext provides rich single-case data including span trees

The context exposes comprehensive per-case information evaluators can access:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str \| None` | Case identifier |
| `inputs` | `InputsT` | Task inputs for this case |
| `output` | `OutputT` | Actual output produced |
| `expected_output` | `OutputT \| None` | Expected output if provided |
| `metadata` | `MetadataT \| None` | Case metadata (difficulty, category, etc.) |
| `duration` | `float` | Task execution time in seconds |
| `attributes` | `dict[str, Any]` | Custom attributes set via `set_eval_attribute()` |
| `metrics` | `dict[str, int \| float]` | Custom metrics set via `increment_eval_metric()` |
| `span_tree` | `SpanTree` | OpenTelemetry spans for behavioral evaluation |

The `span_tree` property enables sophisticated behavioral evaluation—checking tool calls, execution flow, and timing—but only within a single case's execution trace.

## Other frameworks provide native comparative evaluation patterns

**LangSmith** offers `evaluate_comparative()` which natively handles pairwise preference scoring across experiments with built-in position-bias mitigation via randomization. **Inspect AI** (UK AISI) separates `Scorer` (per-sample) from `Metric` (aggregate), where `Metric` receives `list[SampleScore]` for cross-sample computations. **Vertex AI** provides pairwise model evaluation computing `candidate_model_win_rate` and `baseline_model_win_rate` automatically.

The Inspect AI architecture offers the cleanest model for pydantic-evals extension: separating single-case scoring from aggregate metrics would allow tournament evaluators to receive all scores after case-level evaluation completes.

## Recommended architecture for pytest plugin with Bradley-Terry support

For a pytest plugin integrating pydantic-evals with tournament-style evaluators, implement a **two-phase architecture**:

**Phase 1: Standard pydantic-evals evaluation** collects per-case results using the existing `Evaluator` protocol:

```python
from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

@dataclass
class CaseCollectorEvaluator(Evaluator):
    """Standard evaluator that stores outputs for later tournament comparison."""
    collected: list  # Thread-safe collection for concurrent execution
    
    def evaluate(self, ctx: EvaluatorContext) -> float:
        self.collected.append((ctx.name, ctx.output, ctx.inputs))
        return 1.0  # Placeholder score
```

**Phase 2: Tournament evaluation** operates on the `EvaluationReport` post-completion:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class TournamentEvaluator(Protocol):
    """Protocol for evaluators requiring access to all cases."""
    
    def __call__(
        self, 
        cases: list[tuple[str, Any, Any]]  # (name, output, inputs)
    ) -> dict[str, float]:
        """Return scores/rankings for all cases."""
        ...

class BradleyTerryEvaluator:
    """Tournament evaluator using Bradley-Terry model."""
    
    def __call__(self, cases: list[tuple[str, Any, Any]]) -> dict[str, float]:
        # Pairwise LLM comparisons
        # Bradley-Terry coefficient estimation
        # Return normalized scores per case
        ...
```

**Integration wrapper** ties both phases together:

```python
class ComparativeEvaluationRunner:
    def __init__(
        self, 
        dataset: Dataset,
        case_evaluators: list[Evaluator],
        tournament_evaluators: list[TournamentEvaluator]
    ):
        self.dataset = dataset
        self.case_evaluators = case_evaluators
        self.tournament_evaluators = tournament_evaluators
    
    async def run(self, task) -> ComparativeReport:
        # Phase 1: Standard evaluation
        report = await self.dataset.evaluate(task)
        
        # Phase 2: Tournament evaluation
        cases = [(c.name, c.output, c.inputs) for c in report.cases]
        tournament_results = {
            type(e).__name__: e(cases) 
            for e in self.tournament_evaluators
        }
        
        return ComparativeReport(
            standard_report=report,
            tournament_results=tournament_results
        )
```

## Key implementation considerations for type safety

For the callable class pattern with Protocol, ensure evaluators are `@dataclass` decorated to match pydantic-evals conventions. Use generic type parameters `Evaluator[InputsT, OutputT, MetadataT]` for IDE discoverability. The `@runtime_checkable` decorator on `Protocol` enables `isinstance()` checks for evaluator type differentiation.

When running with concurrency, tournament evaluators that collect state during Phase 1 must use thread-safe collections (e.g., `threading.Lock` protecting list appends) or run with `max_concurrency=1`. The cleaner approach is extracting case data from `EvaluationReport.cases` post-completion, avoiding shared state entirely.

## Conclusion

Pydantic Evals provides an excellent foundation for single-case evaluation with type-safe generics, flexible return types, and OpenTelemetry integration. However, its architecture fundamentally precludes comparative/tournament evaluation within the evaluator protocol. The recommended approach separates concerns: use pydantic-evals `Evaluator` for per-case scoring, then implement `TournamentEvaluator` as a separate Protocol operating on collected results. This maintains clean integration with pydantic-evals classes while enabling Bradley-Terry, ELO, and other multi-case ranking algorithms.