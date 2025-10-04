
# Alternative Orchestration Framework Designs

This document captures a first-iteration blueprint for porting the current LangGraph-based research agent into higher-level orchestration frameworks: Google Agent Development Kit (ADK), Agno, and CrewAI. The goal is to provide a copy-paste friendly scaffold that stays flexible, keeps the implementation simple, and preserves the agent capabilities already present in this repository.

The common functional requirements we carry over are:

- **Conversation memory** with long-term persistence hooks.
- **Initial discernment** that classifies the user query as legal, general, or requiring clarification before any heavy work occurs.
- **Planning pass** that decides whether multiple semantic (RAG) lookups are necessary.
- **Response writer** that composes the final answer citing the retrieved documents.
- **Retriever abstraction** that plugs into the existing vector database tools without rewriting indexing logic.

To keep the three blueprints aligned, every design reuses the same conceptual modules:

```python
class RetrievalQuery(NamedTuple):
    text: str
    filters: dict[str, Any] | None = None
    top_k: int = 5

class RetrievedDocument(TypedDict):
    id: str
    score: float
    content: str
    metadata: dict[str, Any]

class VectorRetriever(Protocol):
    async def retrieve(self, query: RetrievalQuery) -> Sequence[RetrievedDocument]:
        """Invoke the underlying RAG tool. Implement this with your existing stack."""

class DocumentBuffer(Protocol):
    def extend(self, docs: Sequence[RetrievedDocument]) -> None: ...
    def snapshot(self) -> Sequence[RetrievedDocument]: ...
```

Replace the `VectorRetriever` implementation with whichever MCP server or SDK you already have; the rest of the scaffolding only expects this interface.

---

## 1. Google Agent Development Kit (ADK)

### Module layout

```
/agents
  router_agent.py       # wraps the first-pass classification + clarification logic
  planner_agent.py      # determines how many RAG searches are needed
  researcher_group.py   # orchestrates parallel MCP tool sessions
  writer_agent.py       # drafts the final answer with document citations
/shared
  memory.py             # conversation + document buffers backed by Firestore/Redis/etc.
  retrievers.py         # adapters returning `VectorRetriever`
workflow.py             # Async entry point wiring the agents together
```

### Key components

1. **Conversation memory**
   - Use `ConversableAgent` with a custom `history_storage` that reads/writes to your preferred backend.
   - Mirror the current LangGraph message schema to ease portability.

2. **Initial discernment**
   - Implement `RouterAgent(ConversableAgent)` that loads few-shot examples through `BaseExampleProvider`.
   - Output a structured payload:
     ```python
     class RouterOutput(TypedDict):
         route: Literal["legal", "needs_info", "general"]
         rationale: str
         missing_fields: list[str]
     ```
   - If `route == "needs_info"`, reply immediately and short-circuit the workflow.

3. **Planning vs. single lookup**
   - `PlannerAgent` receives the `RouterOutput` plus the user message history, returning:
     ```python
     class PlanOutput(TypedDict):
         multi_query: bool
         queries: list[RetrievalQuery]
         notes: str
     ```
   - Attach Vertex AI Search or other tools to expand the plan when `multi_query` is `True`.

4. **Parallel research**
   - Build `ResearcherGroup` with `GroupChat` so each planned query spawns an agent session that calls `VectorRetriever.retrieve` via `MCPToolset`.
   - Aggregate documents into a shared `DocumentBuffer`.

5. **Writer**
   - `WriterAgent` consumes the conversation history and `DocumentBuffer.snapshot()` to emit the final response template.
   - Include placeholders for legal disclaimers when `route == "legal"`.

6. **Orchestration**
   - `workflow.py` coordinates the async steps:
     ```python
     async def handle_turn(user_input: str, *, memory: ConversationMemory) -> str:
         router_output = await router_agent.step(user_input, memory=memory)
         if router_output["route"] == "needs_info":
             return router_output["clarification_prompt"]
         plan = await planner_agent.step(router_output, memory=memory)
         documents = await researcher_group.run(plan["queries"], retriever)
         return await writer_agent.step(
             memory=memory,
             documents=documents,
             plan_notes=plan["notes"],
             route=router_output["route"],
         )
     ```

### Extensibility hooks

- Swap memory backends by replacing the `history_storage` dependency injection.
- Introduce compliance reviewers by inserting another `ConversableAgent` between research and writer.
- Adopt live audio or other ADK tools by plugging additional toolchains into the `ResearcherGroup`.

---

## 2. Agno

### Module layout

```
/workflows
  legal_research.yaml   # declarative wiring of tasks + conditions
/agents
  classifier.py         # wraps the router logic via @tool
  planner.py            # decides multi-query vs. single query
  retriever_team.py     # orchestrates parallel retriever executions
  writer.py             # final response agent
/shared
  state.py              # Pydantic models for workflow state
  retrievers.py         # adapters exposing `VectorRetriever`
```

### State schema

```python
class ConversationState(BaseModel):
    user_id: str
    history: list[dict[str, str]]
    route: Literal["legal", "needs_info", "general"] | None = None
    plan: list[RetrievalQuery] = []
    documents: list[RetrievedDocument] = []
    planner_notes: str | None = None
```

Persist `ConversationState` via Agno's built-in Redis or SQL providers to retain memories between turns.

### Workflow outline (YAML)

```yaml
name: legal_research
state: ConversationState
steps:
  - id: classify
    tool: classifier.router
    on_result:
      - when: "result.route == 'needs_info'"
        to: respond_with_clarification
      - default: plan
  - id: plan
    tool: planner.make_plan
    to: retrieve
  - id: retrieve
    parallel:
      for_each: "result.queries"
      as: query
      tool: retriever_team.run_query
    aggregate: retriever_team.merge_results
    to: write
  - id: write
    tool: writer.compose
    end: true
  - id: respond_with_clarification
    tool: writer.ask_for_details
    end: true
```

### Tool signatures

```python
@tool
def router(state: ConversationState, user_input: str) -> ConversationState: ...

@tool
def make_plan(state: ConversationState) -> ConversationState: ...

@tool
def run_query(state: ConversationState, query: RetrievalQuery, retriever: VectorRetriever) -> list[RetrievedDocument]: ...

@tool
def merge_results(state: ConversationState, results: list[list[RetrievedDocument]]) -> ConversationState: ...

@tool
def compose(state: ConversationState) -> str: ...
```

The retriever tools simply invoke your existing vector search interface and push documents into `state.documents`. The writer reads the same state to generate the final answer.

### Extensibility hooks

- Add monitoring by subscribing to workflow events; Agno streams them by default.
- Register extra teams (e.g., compliance) and branch from the YAML using additional `when` clauses.
- Swap memory behavior by changing the configured state provider without touching tool code.

---

## 3. CrewAI

### Module layout

```
/crews
  flow.py               # defines the CrewAI Flow with ordered stages
  agents.py             # router, planner, researcher, writer definitions
  tasks.py              # task templates binding agents + tools
/shared
  memory.py             # wraps Crew memory layers (short/long/entity)
  retrievers.py         # adapters returning `VectorRetriever`
```

### Memory strategy

- Use `CrewMemory` with short-term and entity memories enabled. Persist summaries per conversation ID for continuity.
- Mirror the existing message roles to minimize prompt rework.

### Agents

```python
router_agent = Agent(
    role="Router",
    goal="Clasificar si la consulta es legal, general o requiere datos",
    tools=[legal_classifier_tool],
    memory=True,
)

planner_agent = Agent(
    role="Planner",
    goal="Decidir si se necesitan búsquedas múltiples y generar queries",
    tools=[query_expander_tool],
    memory=True,
)

researcher_agent = Agent(
    role="Researcher",
    goal="Ejecutar queries en paralelo usando herramientas RAG",
    tools=[vector_retriever_tool],
    memory=False,
)

writer_agent = Agent(
    role="Writer",
    goal="Redactar la respuesta final citando documentos",
    memory=True,
)
```

Expose `vector_retriever_tool` as a thin wrapper over your `VectorRetriever` implementation so you can swap the backend later.

### Flow definition

```python
class LegalResearchFlow(Flow):
    def create_flow(self):
        router_task = Task(
            description="Clasifica la consulta y determina si falta información",
            agent=router_agent,
        )

        planner_task = Task(
            description="Analiza la consulta y arma la lista de búsquedas",
            agent=planner_agent,
            context=[router_task],
        )

        research_task = Task(
            description="Ejecuta cada query planificada y guarda documentos",
            agent=researcher_agent,
            context=[planner_task],
            tools=[
                Tool(
                    name="vector_retriever",
                    func=make_retriever_tool(vector_retriever_instance),
                )
            ],
            async_execution=True,
        )

        writer_task = Task(
            description="Redacta la respuesta final usando documentos y memoria",
            agent=writer_agent,
            context=[router_task, planner_task, research_task],
        )

        return router_task >> planner_task >> research_task >> writer_task
```

### Execution helper

```python
def run_flow(user_input: str, *, memory: CrewMemory, retriever: VectorRetriever) -> str:
    flow = LegalResearchFlow(memory=memory, retriever=retriever)
    result = flow.kickoff(inputs={"user_input": user_input})
    if result.context["router"].route == "needs_info":
        return result.context["router"].clarification_prompt
    return result.output
```

### Extensibility hooks

- Introduce compliance reviews by inserting an extra task before the writer and chaining it via `>>`.
- Enable asynchronous tool execution for heavy MCP calls (already set on `research_task`).
- Attach telemetry by listening to flow events (`flow.add_observer`).

---

## Implementation checklist

1. Select one framework and copy the corresponding module layout + code skeleton into your project.
2. Plug your existing vector search logic into the `VectorRetriever` adapter.
3. Flesh out prompts/examples for router, planner, and writer based on domain requirements.
4. Add persistence backing (Redis, Firestore, SQL) for conversation memory if not already configured.
5. Run a dry test per framework to ensure the router short-circuits when clarification is needed and that the planner can emit single or multi-query plans.

This blueprint should let you move quickly while leaving ample space for future extensions like compliance checks, analytics, or additional toolchains.