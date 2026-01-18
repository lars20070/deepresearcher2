# Pytest-recording VCR recordings and cassettes
_Exported on 10/01/2026 at 19:55:02 CET from Cursor (2.3.29)_

---

**User**

Please explain how the `pytest-recording` package with VCR recordings, replays and `cassettes` subfolders works.

---

**Cursor**

Let me also look at some example test files and cassettes to show you the full picture:

Now I have a complete picture. Let me explain how `pytest-recording` with VCR cassettes works:

## How `pytest-recording` Works

**pytest-recording** is a pytest plugin that wraps [VCR.py](https://vcrpy.readthedocs.io/) to record and replay HTTP interactions. This makes tests deterministic and fast by avoiding real network calls after the initial recording.

### Core Concept

1. **First run (record mode)**: The test makes real HTTP requests, and all request/response pairs are saved to a **cassette** (a YAML file)
2. **Subsequent runs (playback mode)**: The test replays responses from the cassette file instead of making real network calls

### Directory Structure

Cassettes are organized by test module in subfolders under `tests/cassettes/`:

```
tests/cassettes/
  - test_example/          # Cassettes for tests/test_example.py
    - test_weather_agent.yaml
    - test_agent_delegation.yaml
  - test_graph/            # Cassettes for tests/test_graph.py
    - test_websearch_with_reflection.yaml
    - test_summarizesearchresults.yaml
  - test_utils/            # Cassettes for tests/test_utils.py
    - test_brave_search.yaml
    - test_serper_search.yaml
```

The naming convention is: `tests/cassettes/<module_name>/<test_function_name>.yaml`

### Usage in Tests

Simply add the `@pytest.mark.vcr()` decorator:

```17:21:tests/test_graph.py
@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_websearch_without_reflection(topic: str) -> None:
    """
    Test the websearch node.
```

### Configuration

The project configures VCR via the `vcr_config` fixture in `conftest.py`:

```139:162:tests/conftest.py
@pytest.fixture
def vcr_config() -> dict[str, object]:
    """
    Configure VCR recordings for tests with @pytest.mark.vcr() decorator.

    When on bare metal, our host is `localhost`. When in a dev container, our host is `host.docker.internal`.
    `uri_spoofing` ensures that VCR cassettes are read or recorded as if the host was `localhost`.
    See ./tests/cassettes/*/*.yaml.

    Returns:
        dict[str, object]: VCR configuration settings.
    """

    def uri_spoofing(request: Request) -> Request:
        if request.uri and "host.docker.internal" in request.uri:
            # Replace host.docker.internal with localhost.
            request.uri = request.uri.replace("host.docker.internal", "localhost")
        return request

    return {
        "ignore_localhost": False,  # We want to record local SearXNG and Ollama requests.
        "filter_headers": ["authorization", "x-api-key"],
        "decode_compressed_response": True,
        "before_record_request": uri_spoofing,
    }
```

Key configuration options:
- **`filter_headers`**: Removes sensitive headers (API keys) from recordings
- **`ignore_localhost: False`**: Records local service requests (Ollama, SearXNG)
- **`before_record_request`**: Normalizes `host.docker.internal` → `localhost` for dev container compatibility
- **`decode_compressed_response`**: Makes cassettes human-readable

### Recording Modes

You can control recording via command line:

```bash
# Playback only (default, fails if no cassette exists)
uv run pytest tests/test_graph.py::test_websearch -v

# Record new cassettes (makes real HTTP calls)
uv run pytest tests/test_graph.py::test_websearch -v --record-mode=once

# Re-record all cassettes (overwrites existing)
uv run pytest tests/test_graph.py::test_websearch -v --record-mode=all
```

### Benefits

1. **Fast CI**: No network calls during test runs
2. **Deterministic**: Same response every time
3. **No API costs**: Recorded responses don't require API keys
4. **Debugging**: Cassette YAML files are human-readable for inspection

---

**User**

Great. Below you find an example of a `cassettes` subfolder.
```
tests
├── cassettes
│   └── test_example
│       └── test_agent_delegation.yaml
└── test_example.py
```
In a similar way, I want to store the baseline data and auxiliary data for AI agent evaluations in `assays` subfolders.
```
tests
├── assays
│   └── test_curiosity
│       └── test_search_queries.yaml
└── test_curiosity.py
```
I want to stay close to the `pytest` framework and use hooks.

@test_evals.py (245-270)  In a first step, I need to store a `Dataset` of `Cases` as a `yaml` file in a `assays` subfolder. What is the best way to set this up? In the case of `pytest-recording`, the `@pytest.mark.vcr()` decorator filles the `cassettes` subfolder automatically. The user interface for the developer is very clean. How can I fill `cassettes` with the cases for the evaluations?

Please think carefully about the problem. Come up with three options for the software framework design. Discuss the pros and cons of each design and present the option that you would recommend.

---

**Cursor**

This is an excellent design question. Let me analyze the problem and propose three different approaches.

## Problem Analysis

You want a framework that:
1. Stores `Dataset[Case]` objects as YAML files in `tests/assays/<module>/<test>.yaml`
2. Provides a clean decorator-based API like `@pytest.mark.vcr()`
3. Supports both "record" (generate & save) and "playback" (load) modes
4. Integrates with pytest hooks

Let me read how pydantic-evals handles Dataset serialization to understand the API:

Now I have a complete picture. Let me present three design options:

---

## Option 1: Fixture-Based Approach with `conftest.py` Hook

**Concept**: Create a `assay_dataset` fixture that automatically loads/saves datasets based on test name, similar to how `vcr_config` works.

```python
# tests/conftest.py
import os
from pathlib import Path
from typing import Any, Generator
import pytest
from pydantic_evals import Dataset, Case

@pytest.fixture
def assay_path(request: pytest.FixtureRequest) -> Path:
    """Compute the assay file path from test module and function name."""
    module_name = Path(request.fspath).stem  # e.g., "test_curiosity"
    test_name = request.node.name.split("[")[0]  # Handle parametrized tests
    return Path(request.fspath).parent / "assays" / module_name / f"{test_name}.yaml"

@pytest.fixture
def assay_dataset(
    request: pytest.FixtureRequest, 
    assay_path: Path
) -> Generator[Dataset, None, None]:
    """Load dataset from assay file, or provide empty dataset for recording."""
    record_mode = request.config.getoption("--assay-record", default=False)
    
    if assay_path.exists() and not record_mode:
        dataset = Dataset.from_file(assay_path)
    else:
        dataset = Dataset(cases=[])
    
    yield dataset
    
    # Auto-save if recording
    if record_mode and dataset.cases:
        assay_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_file(assay_path)
```

**Usage in tests:**
```python
@pytest.mark.asyncio
async def test_search_queries(assay_dataset: Dataset) -> None:
    if not assay_dataset.cases:
        # Generate cases (record mode)
        cases = [Case(name=f"case_{i}", inputs={"topic": t}) for i, t in enumerate(TOPICS)]
        assay_dataset.cases.extend(cases)
    
    for case in assay_dataset.cases:
        # Run evaluation...
```

| Pros | Cons |
|------|------|
| Simple implementation | Requires explicit fixture injection |
| Follows pytest conventions | Two code paths (record vs playback) in test |
| No custom markers needed | Less "magical" than VCR |
| Easy to understand | Dataset mutation feels awkward |

---

## Option 2: Marker + Hook-Based Approach (VCR-style)

**Concept**: Create a `@pytest.mark.assay()` marker that works like `@pytest.mark.vcr()`, using pytest hooks to wrap test execution.

```python
# tests/conftest.py
import pytest
from pathlib import Path
from pydantic_evals import Dataset

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "assay: marks test to use assay dataset files")

@pytest.fixture
def assay_config() -> dict[str, Any]:
    """Override in tests to customize assay behavior."""
    return {
        "record_mode": "none",  # "none", "once", "all"
    }

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    """Wrap test execution to inject assay dataset."""
    marker = item.get_closest_marker("assay")
    if marker is None:
        yield
        return
    
    # Compute path
    module_name = Path(item.fspath).stem
    test_name = item.name.split("[")[0]
    assay_path = Path(item.fspath).parent / "assays" / module_name / f"{test_name}.yaml"
    
    # Get record mode from marker or CLI
    record_mode = marker.kwargs.get("record_mode", "none")
    if item.config.getoption("--assay-record", default=False):
        record_mode = "once"
    
    # Load or create dataset
    if assay_path.exists() and record_mode == "none":
        dataset = Dataset.from_file(assay_path)
    else:
        dataset = Dataset(cases=[])
    
    # Inject into test via a special attribute
    item.funcargs["assay"] = AssayContext(dataset=dataset, path=assay_path, record_mode=record_mode)
    
    yield
    
    # Save after test if recording
    context = item.funcargs.get("assay")
    if context and context.record_mode != "none" and context.dataset.cases:
        context.path.parent.mkdir(parents=True, exist_ok=True)
        context.dataset.to_file(context.path)

@dataclass
class AssayContext:
    dataset: Dataset
    path: Path
    record_mode: str
```

**Usage in tests:**
```python
@pytest.mark.assay(record_mode="once")
@pytest.mark.asyncio
async def test_search_queries(assay: AssayContext) -> None:
    if assay.record_mode != "none":
        # Generate and populate cases
        for i, topic in enumerate(TOPICS):
            assay.dataset.cases.append(Case(name=f"case_{i}", inputs={"topic": topic}))
    
    for case in assay.dataset.cases:
        # Run evaluation...
```

| Pros | Cons |
|------|------|
| Clean decorator API like VCR | More complex implementation |
| Familiar pattern for users of pytest-recording | Still requires conditional logic in tests |
| Supports `record_mode` parameter | Hook-based magic can be harder to debug |
| CLI integration (`--assay-record`) | `AssayContext` adds a new abstraction |

---

## Option 3: Generator Function + Decorator (Separation of Concerns)

**Concept**: Separate the "case generation" from "case evaluation" using a dedicated generator function and a decorator that caches the result.

```python
# tests/assay_framework.py
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar
from pydantic_evals import Dataset, Case

T = TypeVar("T")

def assay(
    generator: Callable[[], list[Case]] | None = None,
    *,
    record_mode: str = "none",
):
    """
    Decorator that provides a dataset to the test.
    
    - If assay file exists: loads from file
    - If assay file missing or record_mode="once": calls generator, saves to file
    """
    def decorator(test_func: Callable) -> Callable:
        @wraps(test_func)
        async def wrapper(*args, **kwargs):
            # Derive path from test function
            test_file = Path(test_func.__code__.co_filename)
            module_name = test_file.stem
            assay_path = test_file.parent / "assays" / module_name / f"{test_func.__name__}.yaml"
            
            # Determine if we need to generate
            should_record = record_mode == "all" or (record_mode == "once" and not assay_path.exists())
            
            if assay_path.exists() and not should_record:
                dataset = Dataset.from_file(assay_path)
            elif generator:
                cases = generator()
                dataset = Dataset(cases=cases)
                assay_path.parent.mkdir(parents=True, exist_ok=True)
                dataset.to_file(assay_path)
            else:
                raise FileNotFoundError(f"No assay file at {assay_path} and no generator provided")
            
            # Inject dataset as first argument
            return await test_func(dataset, *args, **kwargs)
        return wrapper
    return decorator
```

**Usage in tests:**
```python
def generate_curiosity_cases() -> list[Case]:
    """Generator function for test cases - only called when recording."""
    topics = ["pangolin trafficking", "molecular gastronomy", ...]
    return [Case(name=f"case_{i}", inputs={"topic": t}) for i, t in enumerate(topics)]

@assay(generator=generate_curiosity_cases, record_mode="once")
@pytest.mark.asyncio
async def test_search_queries(dataset: Dataset) -> None:
    # Clean test code - no conditional logic!
    for case in dataset.cases:
        result = await run_agent(case.inputs["topic"])
        assert result is not None
```

| Pros | Cons |
|------|------|
| **Clean separation**: generation vs evaluation | Custom decorator, not pure pytest |
| **No conditional logic in tests** | Less discoverable than markers |
| Generator function is reusable | Doesn't use pytest hooks directly |
| Test code focuses only on evaluation | Async wrapper adds complexity |
| Easy to understand data flow | Need to handle sync/async variants |

---

## Recommendation: **Option 3 (Generator + Decorator)**

I recommend **Option 3** for these reasons:

### 1. **Clean Separation of Concerns**
The VCR pattern works because HTTP recording is fully automatic—no user code decides what to record. But for evaluation datasets, you *must* write generation logic somewhere. Option 3 makes this explicit:
- **Generator function**: Defines *what* cases to create
- **Decorator**: Handles *when* to generate and *where* to store
- **Test function**: Focuses only on *evaluation*

### 2. **No Conditional Logic in Tests**
Options 1 and 2 require `if record_mode:` blocks in your test code. Option 3 keeps tests clean—they always receive a populated dataset.

### 3. **Reusable Generators**
Generator functions can be shared across tests:
```python
# tests/evals/generators.py
def curiosity_topics() -> list[Case]:
    return [Case(name=f"case_{i}", inputs={"topic": t}) for i, t in enumerate(TOPICS)]

# tests/evals/test_curiosity.py
@assay(generator=curiosity_topics, record_mode="once")
async def test_search_queries(dataset): ...

@assay(generator=curiosity_topics, record_mode="once") 
async def test_query_diversity(dataset): ...
```

### 4. **Graceful Upgrade Path**
Start simple, then add pytest integration later:
```python
# Phase 1: Simple decorator (Option 3)
# Phase 2: Add CLI flag via conftest.py
# Phase 3: Add pytest marker for IDE discovery
```

### Implementation Suggestion

Combine Option 3's clean design with Option 2's pytest integration:

```python
# tests/conftest.py
def pytest_addoption(parser):
    parser.addoption("--assay-record", action="store_true", help="Record new assay datasets")

# tests/assay.py - The decorator respects the CLI flag
def assay(generator=None, record_mode="auto"):
    # If record_mode="auto", check pytest CLI flag
    ...
```

This gives you:
- Clean test code (Option 3)
- CLI integration (Option 2)  
- Familiar pytest patterns

Would you like me to implement a complete working version of Option 3 with pytest CLI integration?

