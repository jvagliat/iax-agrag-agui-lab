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

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from iax_agrag_agui_lab.agui.adk_agui_agent_server import AdkAguiAgentServer
from agents.hello_agent import hello_agent 
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()
app = FastAPI(title="AGUI Context + History + State")

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

AdkAguiAgentServer(hello_agent, agui_main_path="/hello-adk-agui").register_app(app)


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run("src.iax_agrag_agui_lab.run_hello_agent:app", host="0.0.0.0", port=1111, reload=True)