# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental lab for integrating Google Agent Development Kit (ADK) agents with the AGUI (Agent UI Protocol) middleware. The project demonstrates multi-agent orchestration patterns using ADK's hierarchical agent system, Neo4j graph database integration, and FastAPI-based server endpoints.

## Key Technologies

- **Google ADK (Agent Development Kit)**: Framework for building LLM-based agents with hierarchical orchestration
- **adk-agui-middleware**: Bridge between ADK agents and AGUI protocol for browser-based chat interfaces
- **Neo4j**: Graph database integration for agent data persistence/retrieval
- **FastAPI**: Web framework for exposing agent endpoints via SSE (Server-Sent Events)
- **Poetry**: Dependency management (Python 3.13 required)

## Running the Application

### Start the multi-agent server:
```bash
poetry run python -m src.iax_agrag_agui_lab.run_agents
```

Server runs on `http://0.0.0.0:1111` with multiple agent endpoints:
- `/hello-adk-agui` - Simple hello agent
- `/coordinator` - Coordinator agent example
- `/pizza` - Multi-agent pizza ordering system (Cajero → Chef + Delivery)

### Environment Setup

Create a `.env` file with:
```
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

## Architecture Patterns

### Multi-Agent Orchestration

The codebase demonstrates ADK's agent hierarchy:

1. **Parent-Child Relationships**: Parent agents coordinate sub_agents (see [pizza_agent.py](src/iax_agrag_agui_lab/agents/pizza_agent.py))
   - `Cajero` (parent) → `KitchenPipeline` (child) → `Chef` + `Delivery` (sequential sub-agents)

2. **Sequential Agents**: `SequentialAgent` executes sub_agents in order, each reading/writing to shared state
   - Uses `output_key` to store results in session state
   - Subsequent agents reference prior results via `{variable_name}` in instructions

3. **Transfer Pattern**: Parent agents use `transfer_to_agent` to delegate work to sub_agents

### AGUI Integration via AdkAguiAgentServer

The [AdkAguiAgentServer](src/iax_agrag_agui_lab/agui/adk_agui_agent_server.py) wrapper:
- Registers SSE endpoints for streaming agent responses
- Manages session state, conversation history, and thread management
- Provides REST endpoints: `/threads`, `/threads/{thread_id}/messages`, `/threads/{thread_id}/state`
- Auto-creates in-memory sessions per agent with configurable `initialState`

### Neo4j Integration

The [neo4j_for_adk.py](src/iax_agrag_agui_lab/data/neo4j_for_adk.py) module:
- Provides ADK-compatible query responses (`tool_success` / `tool_error` format)
- Converts Neo4j types (Node, Relationship, Path) to Python dicts via `to_python()`
- Singleton `graphdb` instance with auto-cleanup on exit

## Design Documents

[high_level_orchestration_design.md](docs/high_level_orchestration_design.md) contains scaffolding blueprints for porting agent workflows to ADK, Agno, and CrewAI frameworks with shared abstractions for:
- Conversation memory
- Initial query classification/routing
- Planning and multi-query RAG retrieval
- Response generation with document citations

## Development Notes

### Session State Management

ADK agents use session state (dict) that persists across conversation turns:
- Initialize via `session_service.create_session(state={...})`
- Agents write to state using `output_key` parameter
- Access state variables in `instruction` prompts via `{key_name}` syntax

### Custom Agent Instructions

When defining `LlmAgent` instructions:
- Be explicit about input sources (user message vs state variables)
- Define expected output format precisely (especially for coordination agents)
- For sequential pipelines, clearly document state dependencies between agents

### CORS Configuration

The FastAPI app allows origins: `localhost:3000`, `127.0.0.1:3000`, `localhost:1111` for local AGUI client development.
