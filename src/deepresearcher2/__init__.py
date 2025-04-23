"""
deepresearcher2 public interface
"""

from .config import config
from .examples import basic_chat, chat_with_python, mcp_server
from .logger import logger

__all__ = [
    "config",
    "logger",
    "basic_chat",
    "chat_with_python",
    "mcp_server",
]
