
# Conceptual Example: Defining Hierarchy
from typing import List
from google.adk.agents import LlmAgent, BaseAgent

from agents.agrag.agentic_rag_multi_query import agentic_rag_multi_query_bot
from agents.agrag.workana_rag_agent import workana_rag_bot
from agents.web_search_agent import web_search_agent
from agents.coder_agent import coder_agent


class AgentData:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def to_dict(self) -> dict:
        # Safely access fields that may not exist on BaseAgent implementations.
        name = getattr(self.agent, "name", "")
        description = getattr(self.agent, "description", "")

        # Prefer singular `instruction` (used by LlmAgent),
        # else fall back to `instructions`; default to empty string.
        if hasattr(self.agent, "instruction"):
            instructions = getattr(self.agent, "instruction")
        elif hasattr(self.agent, "instructions"):
            instructions = getattr(self.agent, "instructions")
        else:
            instructions = ""

        # Ensure JSON-serializable sub_agents (list of names) if available.
        if hasattr(self.agent, "sub_agents") and isinstance(self.agent.sub_agents, list):
            sub_agents = [getattr(a, "name", "") for a in self.agent.sub_agents]
        else:
            sub_agents = []

        return {
            "name": name,
            "description": description,
            "instructions": instructions if isinstance(instructions, str) else str(instructions),
            "sub_agents": sub_agents,
        }


def get_platform_agents() -> List[dict]:
    """Return available platform agents with basic info as JSON-serializable dicts."""
    agents = [
        AgentData(agentic_rag_multi_query_bot),
        AgentData(workana_rag_bot),
        AgentData(web_search_agent),
        AgentData(coder_agent),
    ]
    # FunctionTool expects JSON-serializable outputs; return plain dicts.
    return [a.to_dict() for a in agents]


from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
llm = LiteLlm(model="openai/gpt-4.1-mini", stream_options={"include_usage": True})

platform_specialist = LlmAgent(
    name="PlatformSpecialist",
    model=llm,
    description="Experto en consultas sobre los agentes disponibles.",
    instruction="""
    Eres un experto en los agentes disponibles en la plataforma.
    Puedes responder preguntas sobre capacidades y funciones de cada agente.
    Usa la herramienta `list_agents` para obtener nombre, descripción e instrucciones.
    """,
    tools=[FunctionTool(
        func=get_platform_agents,
    )],
    output_key="PlatformSpecialist.plataform",
)
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 71°F"



# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model=llm,
    description="Coordinador de agentes: enruta consultas al agente adecuado.",
    instruction="""
    # Rol
    Eres un asistente cordial a cargo de un laboratorio de IA con múltiples agentes a los que puedes delegar tareas.

    # Reglas de ruteo (ejemplos):
    - Consultas de Workana (políticas, help desk, pagos, disputas) -> Workana RAG.
    - Consultas sobre iattraxia/IAX (arquitectura, funcionalidades, agentes) -> Agentic RAG Multi-Query.
    - Búsquedas generales en la web (noticias, conocimiento abierto) -> WebSearchAgent.
    - Si tienes dudas, consulta al PlatformSpecialist usando la herramienta `transfer_to_agent`.

    # Instrucciones
    - Explica brevemente al usuario tu lógica de delegación antes de transferir.
    - Responde siempre en el idioma del usuario.
    - Usa formato Markdown.
    """,
    sub_agents=[
        agentic_rag_multi_query_bot,
        workana_rag_bot,
        web_search_agent,
        platform_specialist,
        coder_agent,
    ], 
    tools=[FunctionTool(
        func=get_weather,
    )],
)

# Framework automatically sets:
# assert greeter.parent_agent == coordinator
# assert task_doer.parent_agent == coordinator
