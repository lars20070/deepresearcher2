"""
deepresearcher2 public interface
"""

from .logger import logger  # noqa: I001
from .examples import basic_chat, chat_with_python, mcp_server

__all__ = [
    "logger",
    "basic_chat",
    "chat_with_python",
    "mcp_server",
]
