#!/usr/bin/env python3

import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


def basic_chat() -> None:
    """
    Basic chat interface with the agent.
    https://youtu.be/2FsN4f4z2CY
    """

    logfire.info("Starting basic chat.")

    model = "llama3.3"
    # model = "qwen2.5:72b"
    # model = "qwq:32b"
    ollama_model = OpenAIModel(
        model_name=model,
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1",
        ),
    )

    agent = Agent(
        ollama_model,
        system_prompt="You are a helpful assistant.",
    )

    result = None
    while True:
        user_message = input(">>> ")
        if user_message.lower() in {"exit", "quit", "bye", "stop"}:
            break

        result = agent.run_sync(
            user_message,
            message_history=result.all_messages() if result else None,
        )
        print(result.data)
