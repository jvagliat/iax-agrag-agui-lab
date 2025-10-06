"""FastAPI app wiring the main SSE, history, and state endpoints together.

This example demonstrates how to:
- Register the main AGUI SSE endpoint.
- Register history endpoints (list threads, delete thread, get message snapshot).
- Register state endpoints (patch state, get state snapshot).
- Extract `user_id` and `session_id` from requests in a clear, minimal way.

Run locally:
    uvicorn app:app --reload

Replace `DemoAgent` with your real ADK agent implementation.
"""

from __future__ import annotations

from typing import Any

from ag_ui.core import RunAgentInput
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from adk_agui_middleware import (
    SSEService,
    register_agui_endpoint,
    register_agui_history_endpoint,
    register_state_endpoint,
)
from adk_agui_middleware.data_model.config import (
    HistoryConfig,
    PathConfig,
    RunnerConfig,
    StateConfig,
)
from adk_agui_middleware.data_model.context import ConfigContext
from adk_agui_middleware.service.history_service import HistoryService
from adk_agui_middleware.service.state_service import StateService


# Optional import for local dev clarity; examples still load without ADK installed.
try:  # pragma: no cover - optional at example time
    from google.adk.agents import BaseAgent  # type: ignore
    from google.adk.sessions import Session  # type: ignore
except Exception:  # pragma: no cover - dev convenience
    BaseAgent = object  # type: ignore[misc,assignment]

    class Session:  # type: ignore[py311-valid-type]
        id: str
        last_update_time: float
        state: dict[str, Any]

async def extract_user_id_main(_: RunAgentInput, request: Request) -> str:
    """User id for the main SSE endpoint (from `X-User-Id`, defaults to `guest`).

    The AGUI middleware may await this value. Provide an async callable when the
    middleware expects an awaitable. A simple async function keeps compatibility
    with both sync and async middleware paths.
    """
    # Extract user ID from HTTP header for authentication/authorization
    return request.headers.get("X-User-Id", "guest")


async def extract_user_id_history(request: Request) -> str:
    """User id for history/state endpoints (from `X-User-Id`, defaults to `guest`).

    Made async for the same reason as `extract_user_id_main` — the middleware
    may call and await this function.
    """
    # Same user ID extraction logic for history and state endpoints
    return request.headers.get("X-User-Id", "guest")


async def extract_session_id_history(request: Request) -> str:
    """Session id for history/state endpoints, read from the path parameter.

    Made async to match the signature expected by middleware that may await
    configuration callables.
    """
    # Path parameter name follows the default config: /.../{thread_id}
    # This maps the URL path parameter to session/thread identification
    return str(request.path_params.get("thread_id"))


async def format_thread_list(session_list: list[Session]) -> list[dict[str, Any]]:
    """Return threads in a simple client-friendly structure.

    Includes an optional title if present in session state.
    """
    formatted: list[dict[str, Any]] = []
    for s in session_list:
        # Create base thread information with ID and timestamp
        row: dict[str, Any] = {
            "threadId": s.id,
            "lastUpdateTime": str(int(getattr(s, "last_update_time", 0))),
        }
        # Add optional thread title from session state if available
        title = getattr(s, "state", {}).get("threadTitle")
        if title:
            row["threadTitle"] = str(title)
        formatted.append(row)
    return formatted


# Instantiate your agent and minimal middleware wiring
from src.iax_agrag_agui_lab.agents.hello_agent import hello_agent 
agent: Any = hello_agent  # Replace with your BaseAgent instance

# Configuration for in-memory services (suitable for development/testing)

# Main configuration context for the SSE service
config_context = ConfigContext(
    app_name="demo-app",  # Application identifier
    user_id=extract_user_id_main,  # User ID extraction for main endpoint
)

# SSE service handles the main chat/agent interaction endpoint
sse_service = SSEService(
    agent=agent,  # The agent that processes user requests
    config_context=config_context,  # Context extraction configuration
)

# History service manages conversation threads and message history
history_service = HistoryService(
    HistoryConfig(
        app_name="demo-app",  # Must match SSE service app name
        user_id=extract_user_id_history,  # User ID extraction for history endpoints
        session_id=extract_session_id_history,  # Session/thread ID extraction
        get_thread_list=format_thread_list,  # Custom thread list formatting
    )
)

# State service manages session state persistence and retrieval
state_service = StateService(
    StateConfig(
        app_name="demo-app",  # Must match SSE service app name
        user_id=extract_user_id_history,  # User ID extraction for state endpoints
        session_id=extract_session_id_history,  # Session/thread ID extraction
    )
)


# Create FastAPI application with comprehensive AGUI integration
app = FastAPI(title="AGUI Context + History + State")

# CORS: allow local development frontends (http://localhost:3000 and 127.0.0.1)
# Adjust origins as needed for production.
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

# Main SSE endpoint (POST) for running your agent
# This endpoint handles user interactions and streams responses
register_agui_endpoint(
    app=app,
    sse_service=sse_service,
    path_config=PathConfig(
        agui_main_path="/hello-adk-agui"),  # Available at POST /agui
)

# History endpoints (GET list, DELETE thread, GET message snapshot)
# These endpoints manage conversation history and thread management
register_agui_history_endpoint(
    app=app,
    history_service=history_service,
    # Provides: GET /threads, DELETE /threads/{thread_id}, GET /threads/{thread_id}/messages
)

# State endpoints (PATCH state, GET state snapshot)
# These endpoints handle session state persistence and retrieval
register_state_endpoint(
    app=app,
    state_service=state_service,
    # Provides: PATCH /threads/{thread_id}/state, GET /threads/{thread_id}/state
)


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run("notebooks.run_hello_agent:app", host="0.0.0.0", port=1111, reload=True)