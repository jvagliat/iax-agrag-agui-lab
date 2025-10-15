
# Conceptual Example: Defining Hierarchy
from typing import List
from google.adk.agents import LlmAgent, BaseAgent

from agents.agrag.agentic_rag_multi_query import agentic_rag_multi_query_bot
from agents.agrag.workana_rag_agent import workana_rag_bot
from agents.web_search_agent import web_search_agent


class AgentData:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def to_dict(self):
        return {
            "name": self.agent.name,
            "description": self.agent.description,
            "instructions": self.agent.instructions,
        }


def get_platform_agents() -> List[AgentData]:
    """Return available platform agents with basic info."""
    return [
        AgentData(agentic_rag_multi_query_bot),
        AgentData(workana_rag_bot),
        AgentData(web_search_agent),
    ]


from google.adk.tools import FunctionTool

platform_specialist = LlmAgent(
    name="PlatformSpecialist",
    model="gemini-2.0-flash",
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


# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash",
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
    ]
)

# Framework automatically sets:
# assert greeter.parent_agent == coordinator
# assert task_doer.parent_agent == coordinator

