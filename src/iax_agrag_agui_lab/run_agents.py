"""FastAPI app wiring the main SSE, history, and state endpoints together.

This example demonstrates how to:
- Register the main AGUI SSE endpoint.
- Register history endpoints (list threads, delete thread, get message snapshot).
- Register state endpoints (patch state, get state snapshot).
- Extract `user_id` and `session_id` from requests in a clear, minimal way.

Run locally:
    uvicorn app:app --reload

Replace `hello_agent` with your real ADK agent implementation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from agui.adk_agui_agent_server import AdkAguiAgentServer
from agents.hello_agent import hello_agent
from dotenv import load_dotenv

try:
    from uvicorn.logging import DefaultFormatter as _BaseFormatter
except ImportError:  # pragma: no cover - uvicorn may not be present during tests
    _BaseFormatter = logging.Formatter


def _format_json_lines(value: Any, indent: int = 0) -> List[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: List[str] = []
        for key, child in value.items():
            child_lines = _format_json_lines(child, indent + 2)
            if len(child_lines) == 1:
                lines.append(f"{prefix}{key}: {child_lines[0].lstrip()}")
            else:
                lines.append(f"{prefix}{key}:")
                lines.extend(child_lines)
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            item_lines = _format_json_lines(item, indent + 2)
            if len(item_lines) == 1:
                lines.append(f"{prefix}- {item_lines[0].lstrip()}")
            else:
                lines.append(f"{prefix}-")
                lines.extend(item_lines)
        return lines or [f"{prefix}-"]
    if isinstance(value, str):
        pieces = value.splitlines()
        if len(pieces) == 1:
            return [f"{prefix}{pieces[0]}"]
        return [f"{prefix}{pieces[0]}"] + [f"{prefix}{piece}" for piece in pieces[1:]]
    return [f"{prefix}{value}"]


class JsonAwareFormatter(_BaseFormatter):
    def format(
        self, record: logging.LogRecord
    ) -> str:  # pragma: no cover - formatting logic
        original_msg, original_args = record.msg, record.args
        try:
            pretty = self._maybe_pretty_json(record)
            if pretty:
                record.msg = pretty
                record.args = ()
            return super().format(record)
        finally:
            record.msg, record.args = original_msg, original_args

    def _maybe_pretty_json(self, record: logging.LogRecord) -> str | None:
        message = record.getMessage()
        if not isinstance(message, str):
            return None
        stripped = message.strip()
        if not stripped or (stripped[0] not in "{[" or stripped[-1] not in "}]"):
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        lines = _format_json_lines(parsed)
        return "\n".join(lines)


_LOGGING_INITIALISED = False


def _configure_console_logging() -> None:
    global _LOGGING_INITIALISED
    if _LOGGING_INITIALISED:
        return
    formatter = (
        JsonAwareFormatter("%(levelprefix)s %(message)s", use_colors=False)
        if _BaseFormatter is not logging.Formatter
        else JsonAwareFormatter("%(levelname)s %(message)s")
    )
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
    for logger_name in ("generic", "adk", "adk_agui_middleware"):
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            for handler in logger.handlers:
                handler.setFormatter(formatter)
        logger.propagate = False
    _LOGGING_INITIALISED = True


# Cargar variables de entorno
load_dotenv()
_configure_console_logging()

from agents.coordinator_agent import coordinator
from agents.pizza_agent import pizzeria_bot


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await AdkAguiAgentServer(hello_agent, agui_main_path="/hello-adk-agui").register_app(
        app, initialState={}
    )
    await AdkAguiAgentServer(coordinator, agui_main_path="/coordinator").register_app(
        app, initialState={}
    )
    await AdkAguiAgentServer(pizzeria_bot, agui_main_path="/pizza").register_app(
        app, initialState={"pizza_created": False, "delivery_info": "null"}
    )
    yield
    # Shutdown (si necesitas limpiar algo)


app = FastAPI(title="AGUI Context + History + State", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:1111",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run(
        "src.iax_agrag_agui_lab.run_agents:app", host="0.0.0.0", port=1111, reload=True
    )
