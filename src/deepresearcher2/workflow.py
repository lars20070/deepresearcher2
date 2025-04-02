#!/usr/bin/env python3

import logfire
import rizaio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from deepresearcher2 import logger


def riza_example() -> None:
    """
    Example by Riza team. https://riza.io
    https://youtu.be/2FsN4f4z2CY

    Returns:
        None
    """

    logfire.info("Starting Rizza example.")

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

    @agent.tool_plain
    def execute_code(code: str) -> str:
        """
        Execute Python code

        Use print() to write the output of your code to stdout.
        Use only the Python standard library and build in modules.
        For example, do not use pandas, but you can use csv.
        Use httpx to make http requests.

        Args:
            code (str): The code to execute.

        Returns:
            str: The output of the code execution.
        """
        load_dotenv()

        print(f"Executing code:\n{code}")
        riza = rizaio.Riza()
        result = riza.command.exec(
            language="PYTHON",
            code=code,
        )

        if result.exit_code != 0:
            raise ModelRetry(result.stderr)
        if result.stdout == "":
            raise ModelRetry("Code executed successfully, but no output was returned. Ensure your code includes print statements for output.")

        print(f"Execution output:\n{result.stdout}")
        return result.stdout

    user_message = "What is the capital of France?"
    result = agent.run_sync(user_message)

    while user_message not in {"exit", "quit"}:
        print(f"Result:\n{result.data}")
        user_message = input("> ")
        result = agent.run_sync(
            user_message,
            message_history=result.all_messages(),
        )


def main() -> None:
    """
    Main function for the script.

    Returns:
        None
    """

    logger.info("Starting main function.")
    logfire.info("Starting main function.")

    riza_example()


if __name__ == "__main__":
    main()
