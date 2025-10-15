from agents.tools.tavily_search_tool import create_adk_tavily_search_tool
from google.adk.agents import LlmAgent

web_search_agent = LlmAgent(
    name="WebSearchAgent",
    model="gemini-2.0-flash",
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
