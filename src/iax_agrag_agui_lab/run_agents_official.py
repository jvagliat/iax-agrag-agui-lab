"""
FastAPI app with OFFICIAL adk-agui-middleware integration.
Exposes agentic_rag_multi_query_bot using official AGUI protocol.

Run:
    poetry run uvicorn src.iax_agrag_agui_lab.run_agents_official:app --port 2222 --reload
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

# Configure logging BEFORE any other imports
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Component-specific loggers for ADK middleware
logging.getLogger('adk_agent').setLevel(logging.INFO)
logging.getLogger('event_translator').setLevel(logging.INFO)
logging.getLogger('session_manager').setLevel(logging.INFO)
logging.getLogger('endpoint').setLevel(logging.INFO)

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from adk_agui_middleware.data_model.context import ConfigContext
from ag_ui.core import RunAgentInput
from ag_ui_adk import ADKAgent

from agents.agrag.agentic_rag_multi_query import agentic_rag_multi_query_bot
from agents.agrag.workana_rag_agent import workana_rag_bot

# Dynamic Identification
# Recommended for multi-tenant applications:
def extract_app(input: RunAgentInput) -> str:
    """Extract app name from request context."""
    for ctx in input.context:
        if ctx.description == "app":
            return ctx.value
    return "default_app"

def extract_user(input: RunAgentInput) -> str:
    """Extract user ID from request context."""
    for ctx in input.context:
        if ctx.description == "user":
            return ctx.value
    return f"anonymous_{input.thread_id}"
# Create FastAPI application
app = FastAPI(title="AGUI Official - AGRAG Multi-Query")

# Exception handler to log errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    error_msg = f"Error: {str(exc)}\n{traceback.format_exc()}"
    print(f"\n{'='*80}\nERROR 500:\n{error_msg}\n{'='*80}\n")
    return {"error": str(exc), "detail": traceback.format_exc()}

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:2222",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
from agents.coordinator_agent import coordinator

agent = ADKAgent(
    adk_agent=coordinator,              # Required: The ADK agent to embed
    app_name_extractor=extract_app,
    user_id_extractor=extract_user,

    session_timeout_seconds=1200,    # Optional: Session timeout (default: 20 minutes)
    cleanup_interval_seconds=300,    # Optional: Cleanup interval (default: 5 minutes)
    # max_sessions_per_user=10,        # Optional: Max sessions per user (default: 10)
    use_in_memory_services=True,     # Optional: Use in-memory services (default: True)
    execution_timeout_seconds=600,   # Optional: Execution timeout (default: 10 minutes)
    tool_timeout_seconds=300,        # Optional: Tool timeout (default: 5 minutes)
    max_concurrent_executions=5      # Optional: Max concurrent executions (default: 5)
)
from ag_ui_adk import add_adk_fastapi_endpoint

# Add endpoint with custom path
add_adk_fastapi_endpoint(
    app,
    agent,
    path="/coordinator"  # Custom endpoint path
)

# Multiple agents on different endpoints
#add_adk_fastapi_endpoint(app, general_agent, path="agrag-official")
# Register Workana RAG agent on a separate endpoint
# workana_agent = ADKAgent(
#     adk_agent=workana_rag_bot,
#     app_name_extractor=extract_app,
#     user_id_extractor=extract_user,

#     session_timeout_seconds=1200,
#     cleanup_interval_seconds=300,
#     use_in_memory_services=True,
#     execution_timeout_seconds=600,
#     tool_timeout_seconds=300,
#     max_concurrent_executions=5,
# )

# add_adk_fastapi_endpoint(
#     app,
#     workana_agent,
#     path="/workana_rag"
# )

# Configure LangSmith tracing
from langsmith.integrations.otel import configure
import os
configure(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    project_name=os.getenv("LANGSMITH_PROJECT")
)  

# Optional: can also use LANGSMITH_PROJECT and LANGSMITH_API_KEY environment variables


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.iax_agrag_agui_lab.run_agents_official:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
