---
name: Refactor Plugin to Use capture_run_messages()
overview: Replace monkey-patching Agent.run() with PydanticAI's official capture_run_messages() context manager for capturing baseline responses in the assay plugin.
todos: []
isProject: false
---

# Plugin Refactor: Capturing Agent Responses

## Problem

The current implementation in `plugin.py` captures PydanticAI agent responses by monkey-patching `Agent.run()`. This works but is not the most elegant approach.

## Current Approach (Monkey-patching `Agent.run`)

The current implementation captures `AgentRunResult` objects by monkey-patching `Agent.run()`. This works but has drawbacks:

- Complex implementation with `ContextVar` for async safety
- Only captures final results, not intermediate messages (tool calls, retries, etc.)
- Not using the official testing API

## Better Approach: `capture_run_messages()`

PydanticAI provides a dedicated context manager for testing that captures the complete message exchange:

```python
from pydantic_ai import capture_run_messages, ModelMessagesTypeAdapter

with capture_run_messages() as messages:
    result = await agent.run("...")
    
# messages now contains all ModelRequest and ModelResponse objects
# including tool calls, retries, system prompts, etc.

# Built-in JSON serialization:
from pydantic_core import to_json
json_bytes = to_json(messages)

# Or to Python objects:
from pydantic_core import to_jsonable_python
json_dict = to_jsonable_python(messages)
```

## Key Benefits

| Aspect | Monkey-patching `Agent.run` | `capture_run_messages()` |
|--------|---------------------------|-------------------------|
| Captures | Only final `AgentRunResult` | Full message history (requests, responses, tool calls, retries) |
| API stability | Fragile (internal method) | Official testing API |
| Serialization | Manual | Built-in `ModelMessagesTypeAdapter` |
| Detail level | Output only | Complete conversation trace |
| Async safety | Manual `ContextVar` | Handled internally |

## Recommended Refactor

Here's how the `pytest_runtest_call` hook could be refactored:

```python
from pydantic_ai import capture_run_messages, ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python

# New stash key for captured messages
CAPTURED_MESSAGES_KEY = pytest.StashKey[list[Any]]()

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Item) -> Generator[None, None, None]:
    if not _is_assay(item):
        yield
        return

    logger.info("Capturing agent messages via capture_run_messages()")
    
    with capture_run_messages() as messages:
        yield  # Run the test
    
    # Store captured messages (full conversation history)
    item.stash[CAPTURED_MESSAGES_KEY] = list(messages)
    
    # These are already serializable via:
    # json_data = to_jsonable_python(messages)
```

This gives you:

1. **Full conversation traces** - Not just outputs, but every request/response including tool calls and retries
2. **Official API** - Less likely to break with PydanticAI updates  
3. **Built-in serialization** - Use `ModelMessagesTypeAdapter` or `to_json()` directly
4. **Simpler code** - No manual `ContextVar` management needed

## References

- PydanticAI Testing Documentation: https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md
- PydanticAI Message History: https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md