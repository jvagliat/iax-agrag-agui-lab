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


import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from agui.adk_agui_agent_server import AdkAguiAgentServer
from agents.hello_agent import hello_agent
from dotenv import load_dotenv

from debug import configure_console_logging

load_dotenv()
configure_console_logging()

from agents.coordinator_agent import coordinator
from agents.pizza_agent import pizzeria_bot
from agents.agrag.agentic_rag import agentic_rag_bot

from agents.agrag.agentic_rag_multi_query import agentic_rag_multi_query_bot


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await AdkAguiAgentServer(
        hello_agent, agui_main_path="/hello-adk-agui"
    ).register_app(app, initialState={})
    await AdkAguiAgentServer(coordinator, agui_main_path="/coordinator").register_app(
        app, initialState={}
    )
    await AdkAguiAgentServer(pizzeria_bot, agui_main_path="/pizza").register_app(
        app, initialState={"pizza_created": False, "delivery_info": "null"}
    )
    await AdkAguiAgentServer(
        agentic_rag_bot, agui_main_path="/agentic-rag"
    ).register_app(
        app,
        initialState={
            "triage_result": "",
            "retrieved_chunks": "",
            "final_response": "",
        },
    )

    await AdkAguiAgentServer(
        agentic_rag_multi_query_bot, agui_main_path="/mq-agentic-rag"
    ).register_app(
        app,
        initialState={
            "triage_result": "",
            "retrieved_chunks": "",
            "final_response": "",
        },
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
        "src.iax_agrag_agui_lab.run_agents:app", 
        host="0.0.0.0",
        port=8000,
        reload=True
    )
