from agents.tools.tavily_search_tool import create_adk_tavily_search_tool
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
llm = LiteLlm(model="openai/gpt-4.1-mini", stream_options={"include_usage": True})

web_search_agent = LlmAgent(
    name="WebSearchAgent",
    model=llm,
    description="Agente para responder preguntas usando búsqueda web (Tavily).",
    instruction="""
    Responderás preguntas consultando la web siempre usando la herramienta `tavily_search`.

    Reglas:
    - No inventes. Si no hay evidencia suficiente, dilo explícitamente.
    - Cita 2–5 fuentes al final, con formato Markdown: "Fuentes:" seguido de lista con título y URL (cuando estén disponibles).
    - Resume en 1–3 párrafos. Prioriza claridad y exactitud.
    - Responde en el mismo idioma del usuario.

    Formato final sugerido:
    [respuesta breve y precisa]

    Fuentes:
    - [Título 1](URL)
    - [Título 2](URL)
    """,
    tools=[create_adk_tavily_search_tool()] 
)
