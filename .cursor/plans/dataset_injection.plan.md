You can combine pytest_runtest_call with a generator by using the hook to resolve the generator, load/create the dataset, and inject it into the test's arguments. Here's how:

## Implementation
```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pytest
from pytest import Config, Item, Parser, Session

from .evals.evals import EvalPlayer
from .logger import logger

if TYPE_CHECKING:
    from pydantic_evals import Case, Dataset

ASSAY_MODES = ("evaluate", "new_baseline")
PLAYERS_KEY = pytest.StashKey[list[EvalPlayer]]()
DATASET_KEY = pytest.StashKey["Dataset"]()


@dataclass
class AssayMarkerConfig:
    """Configuration extracted from @pytest.mark.assay() marker."""
    generator: Callable[[], list[Case]] | None = None
    record_mode: str = "auto"  # "auto", "always", "never"


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("assay", "Assay evaluation options")
    group.addoption(
        "--assay-mode",
        action="store",
        default="evaluate",
        choices=ASSAY_MODES,
        help='Assay mode: "evaluate" (use existing baseline) or "new_baseline" (regenerate).',
    )


def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers",
        "assay(generator=None, record_mode='auto'): Mark test for AI agent evaluation.",
    )


def _get_assay_path(item: Item) -> Path:
    """Compute assay file path from test location."""
    test_file = Path(item.fspath)
    module_name = test_file.stem  # e.g., "test_curiosity"
    test_name = item.name.split("[")[0]  # Handle parametrized tests
    return test_file.parent / "assays" / module_name / f"{test_name}.yaml"


def _should_regenerate(item: Item, marker_config: AssayMarkerConfig, assay_path: Path) -> bool:
    """Determine if we should call the generator to create new cases."""
    cli_mode = item.config.getoption("--assay-mode")

    if marker_config.record_mode == "always":
        return True
    if marker_config.record_mode == "never":
        return False

    # record_mode == "auto": check CLI and file existence
    if cli_mode == "new_baseline":
        return True
    return not assay_path.exists()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Item):
    """Inject dataset into assay-marked tests before execution."""
    from pydantic_evals import Case, Dataset

    marker = item.get_closest_marker("assay")
    if marker is None:
        yield
        return

    # Extract marker configuration
    marker_config = AssayMarkerConfig(
        generator=marker.kwargs.get("generator"),
        record_mode=marker.kwargs.get("record_mode", "auto"),
    )

    assay_path = _get_assay_path(item)
    should_regenerate = _should_regenerate(item, marker_config, assay_path)

    # Load or generate dataset
    if should_regenerate:
        if marker_config.generator is None:
            raise ValueError(
                f"No assay file at {assay_path} and no generator provided. "
                "Either create the file or pass generator= to @pytest.mark.assay()."
            )
        logger.info(f"Generating new baseline: {assay_path}")
        cases = marker_config.generator()
        dataset: Dataset[Any, Any, Any] = Dataset(cases=cases)

        # Save to file
        assay_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_file(assay_path)
    else:
        logger.info(f"Loading existing baseline: {assay_path}")
        dataset = Dataset.from_file(assay_path)

    # Inject into test function arguments
    item.funcargs["assay_path"] = assay_path
    item.funcargs["assay_dataset"] = dataset

    # Store in stash for hooks to access
    item.stash[DATASET_KEY] = dataset

    # Initialize players collection
    item.stash[PLAYERS_KEY] = []

    yield  # Test runs here
```
## Generator Definition
```python
from pydantic_evals import Case


def generate_curiosity_cases() -> list[Case]:
    """Generator for curiosity evaluation test cases."""
    topics = [
        "pangolin trafficking networks",
        "molecular gastronomy techniques",
        "quantum entanglement applications",
        "bioluminescent organisms",
        "ancient navigation methods",
    ]
    return [
        Case(
            name=f"topic_{i}",
            inputs={"topic": topic},
        )
        for i, topic in enumerate(topics)
    ]
```
## Test Usage
```python
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic_evals import Dataset

from deepresearcher2.plugin import PLAYERS_KEY
from deepresearcher2.evals.evals import EvalPlayer
from tests.generators import generate_curiosity_cases

if TYPE_CHECKING:
    pass


@pytest.mark.assay(generator=generate_curiosity_cases, record_mode="auto")
@pytest.mark.asyncio
async def test_search_queries(
    request: pytest.FixtureRequest,
    assay_path: Path,          # Injected by hook
    assay_dataset: Dataset,    # Injected by hook
) -> None:
    """
    Clean test code—no conditional logic for loading/generating.
    The hook handles everything based on --assay-mode CLI flag.
    """
    players: list[EvalPlayer] = []

    for idx, case in enumerate(assay_dataset.cases):
        # Run your agent
        result = await run_agent_query(case.inputs["topic"])

        # Create players for evaluation
        player_baseline = EvalPlayer(idx=idx, item=case.inputs.get("query", ""))
        player_novel = EvalPlayer(idx=idx + len(assay_dataset.cases), item=result)
        players.extend([player_baseline, player_novel])

    # Pass to plugin for tournament
    request.node.stash[PLAYERS_KEY] = players
```
## CLI Usage
```bash
# Use existing baseline (default)
uv run pytest tests/test_curiosity.py -v

# Regenerate baseline using generator function
uv run pytest tests/test_curiosity.py -v --assay-mode=new_baseline
```
## Data Flow
```
@pytest.mark.assay(generator=fn)
           │
           ▼
pytest_runtest_call (hookwrapper=True)
           │
           ├─► Check --assay-mode CLI flag
           │
           ├─► If "new_baseline" or file missing:
           │       └─► Call generator() → Save to YAML
           │
           ├─► Else: Load from YAML
           │
           ├─► Inject into item.funcargs["assay_dataset"]
           │
           └─► yield → Test executes with dataset
```