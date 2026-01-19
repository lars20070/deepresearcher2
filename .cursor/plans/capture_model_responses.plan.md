Capturing data "by magic" (without explicitly calling a recording function in every test) is the hallmark of a great plugin. While Python doesn't allow you to easily trigger a hook *whenever* a specific type like `list[int]` is created without massive performance overhead, you can achieve a "magic" feel using **Instrumentation** or **Context Management**.

For AI agents, the cleanest way to do this is to intercept the agent's response-generating method.

---

## 1. The "Magic" Approach: Monkeypatching / Instrumentation

If your AI agents use a consistent method (e.g., `.run()` or `.query()`), your plugin can automatically "wrap" that method to capture the output and stash it in the pytest `item`.

### The Plugin Logic (`plugin.py`)

```python
import pytest

# Define a Stash key to store the responses
AGENT_RESPONSES = pytest.StashKey[list]()

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    # This runs BEFORE the test function execution
    # 1. Initialize the stash for this test
    item.stash[AGENT_RESPONSES] = []

    # 2. You could theoretically monkeypatch the Agent class here
    # For example, if all agents inherit from a BaseAgent:
    # original_method = BaseAgent.generate
    # def wrapped_generate(self, *args, **kwargs):
    #     res = original_method(self, *args, **kwargs)
    #     item.stash[AGENT_RESPONSES].append(res)
    #     return res
    
    yield # The test runs here

```

### The Evaluation Hook

You can define a custom hook that other people (or your own plugin) can use to evaluate these responses once the test finishes.

```python
# Define the hook specification
def pytest_evaluate_agent_responses(item, responses):
    """ Hook to evaluate agent responses collected during a test. """

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        responses = item.stash.get(AGENT_RESPONSES, [])
        if responses:
            # Trigger your custom evaluation hook
            item.config.hook.pytest_evaluate_agent_responses(
                item=item, 
                responses=responses
            )

```

---

## 2. The "Semi-Magic" Approach: The Context Manager Fixture

Detecting a `list[int]` globally is dangerous because pytest itself and other libraries use lists of integers constantly. A cleaner "semi-magic" way is to provide a fixture that acts as a collector.

### The Plugin Logic

```python
import pytest

class AgentCollector:
    def __init__(self):
        self.responses = []

    def __call__(self, response):
        # The test just 'calls' this object to log data
        self.responses.append(response)

@pytest.fixture
def capture_agent(request):
    collector = AgentCollector()
    yield collector
    # After test, move data to stash for the hooks
    request.node.stash[AGENT_RESPONSES] = collector.responses

```

### The Test (Clean API)

```python
def test_ai_logic(capture_agent):
    agent = MyAgent()
    response = agent.ask("What is 2+2?")
    
    # Just passing it to the fixture is enough; 
    # the hooks handle the rest behind the scenes.
    capture_agent(response) 

```

---

## 3. The "True Magic" (Advanced): Object Proxies

If you truly want to detect any `list[int]` without the user doing anything, you would need to use a **Trace Function** (`sys.settrace`). However, this is **not recommended** for pytest plugins because it slows down the test suite by 10x–100x.

**A better alternative for "Magic" detection:**
If your agents are objects, you can use a fixture that searches the local variables of the test function after it finishes.

```python
import inspect

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield
    # AFTER the test has run, inspect the local variables of the test function
    # Note: This requires the test to still have its frame available, 
    # which usually requires catching it in a wrapper.
    
    # Better: Use the 'funcargs' or 'request' to find Agent objects

```

---

## Which should you choose?

I recommend a combination: **A Custom Fixture + `item.stash**`.

1. **The Fixture** provides a clear contract: "If you want this evaluated, put it here."
2. **The Stash** allows your `pytest_runtest_makereport` hook to grab that data and perform the evaluation (e.g., calculating LLM accuracy or cost) and attach it to the final test report.

### Next Step

Would you like me to show you how to **define the Hook Specification** so that users of your plugin can write their own custom evaluation logic in their `conftest.py`?