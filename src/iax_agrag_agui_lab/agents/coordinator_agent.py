
# Conceptual Example: Defining Hierarchy
from typing import List
from google.adk.agents import LlmAgent, BaseAgent


from agents.agrag.agentic_rag_multi_query import agentic_rag_multi_query_bot
from agents.agrag.workana_rag_agent import workana_rag_bot
from agents.web_search_agent import web_search_agent
from agents.tools.tavily_search_tool import create_adk_tavily_search_tool

class AgentData():
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def to_dict(self):
        return {
            "name": self.agent.name,
            "description": self.agent.description,
            "instructions": self.agent.instructions,
        }

def get_platform_agents() -> List[AgentData]:
    """Function to return available platform agents.
    For each agent, return its name, description, and instructions.

    Returns: list of available agents
    """
    return [
        AgentData(agentic_rag_multi_query_bot),
        AgentData(workana_rag_bot),
        AgentData(web_search_agent),
    ]

from google.adk.tools import FunctionTool

platform_specialist = LlmAgent(
    name="PlatformSpecialist",
    model="gemini-2.0-flash",
    description="Expert in platform-specific queries.",
    instruction="""
    Eres un experto en los agentes disponibles en la plataforma. 

    Puedes responder preguntas sobre las capacidades y funciones de cada agente.

    Para acceder a la información de cada agente, puedes usar la herramienta `list_agents` que te proporciona los detalles de cada agente disponible.
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
    description="""
    # Role: Coordinador de Agentes
    Eres un asistente muy cordial a cargo de un labotorio de
    inteligencia artificial en el que se implementan otros multiples agentes
    a los puedes delegar trabajos.

    Tu única labor es identificar a quién delegar cada tarea.

    # Instrucciones:
    - Una vez que recibes el mensaje explica al usuario cual es tu lógica de la delegación.
    - Si no estás seguro de a quién delegar preguntale al agente {PlatformSpecialist}.

    # Importante
    - Siempre responde en el mismo idioma de la pregunta.
    - Responde en formato markdown.
    """,
    sub_agents=[ 
        agentic_rag_multi_query_bot,
        workana_rag_bot, 
        web_search_agent, 
        platform_specialist
    ]

)

# Framework automatically sets:
# assert greeter.parent_agent == coordinator
# assert task_doer.parent_agent == coordinator