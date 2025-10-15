"""
Agentic RAG System - Versión Multi-Query
Pipeline: Triage + Query Generation + Multi-Retrieval + Synthesis
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from agents.agrag.query_iax_docs_tool import query_iax_documentation_rag
from google.adk.models.lite_llm import LiteLlm

# ==================== HERRAMIENTAS ====================
vector_search_tool = FunctionTool(
    func=query_iax_documentation_rag,
)

# ==================== AGENTES ====================

# 1) TRIAGE AGENT - Clasifica consultas (OpenAI)
llm = LiteLlm(model="openai/gpt-4.1-mini", stream_options={"include_usage": True})

triage_agent = LlmAgent(
    name="TriageAgent",
    model=llm,
    description="Clasifica consultas del usuario: GENERAL o ESPECÍFICA",
    instruction="""
    Eres un asistente que clasifica consultas sobre iattraxia y la plataforma IAX.

    - GENERAL: saludos, cortesía, conversación casual, agradecimientos.
    - ESPECÍFICA: preguntas sobre iattraxia, IAX, desarrollo de software con IA, agentes, automatizaciones.

    Si es GENERAL: responde amigable y brevemente, preséntate como asistente de iattraxia/IAX.
    Si es ESPECÍFICA: indica "Déjame buscar esa información en la documentación...".

    Importante: No inventes. Si no sabes, dilo. Informa cuando ejecutes tareas o delegues.
    """,
    output_key="TriageAgent.response",
    sub_agents=[],
)


# 2) QUERY GENERATOR - Genera múltiples consultas diversas
query_generator_agent = LlmAgent(
    name="QueryGeneratorAgent",
    model=llm,
    description="Genera EXACTAMENTE 3 consultas de búsqueda diversas",
    instruction="""
    Eres experto en formular consultas de búsqueda.

    Genera EXACTAMENTE 3 consultas diferentes que cubran ángulos complementarios.
    Devuelve SOLO un arreglo JSON con 3 strings. Sin texto adicional.

    Ejemplo:
    [
      "funcionalidades de la plataforma IAX",
      "arquitectura y componentes del sistema IAX",
      "casos de uso de agentes autónomos en IAX"
    ]
    """,
    output_key="QueryGeneratorAgent.generated_queries",
    sub_agents=[],
)


# 3) MULTI-RETRIEVAL AGENT - Ejecuta búsquedas con las 3 consultas
multi_retrieval_agent = LlmAgent(
    name="MultiRetrievalAgent",
    model=llm,
    description="Ejecuta búsquedas vectoriales con las consultas generadas",
    instruction="""
    Las consultas generadas están en: {QueryGeneratorAgent.generated_queries}

    Tareas:
    1) Usa la herramienta `vector_search` con CADA una de las 3 consultas.
    2) Reúne todos los resultados obtenidos.

    Formato de salida (JSON estricto):
    {
      "retrieved_chunks": [
         {"content": "...", "source": "...", "score": 0.0},
         ...
      ],
      "by_query": {
        "<consulta_1>": [ {"content": "...", "source": "..."}, ... ],
        "<consulta_2>": [ ... ],
        "<consulta_3>": [ ... ]
      }
    }

    Notas:
    - Ejecuta las 3 búsquedas aunque algunas den pocos resultados.
    - Extrae "source" de la metadata de los documentos cuando esté disponible.
    """,
    output_key="MultiRetrievalAgent.retrieved_chunks",
    tools=[vector_search_tool],
    sub_agents=[],
)


# 4) SYNTHESIZER AGENT - Genera respuesta final con citas
synthesizer_agent = LlmAgent(
    name="SynthesizerAgent",
    model=llm,
    description="Genera respuesta final integrando múltiples búsquedas",
    instruction="""
    Los chunks recuperados de múltiples búsquedas están en: {MultiRetrievalAgent.retrieved_chunks}

    Tareas:
    1) Lee TODOS los chunks recuperados.
    2) Identifica complementariedades y elimina redundancias.
    3) Redacta respuesta clara y bien estructurada.
    4) Cita fuentes (título/ID y URL si existe) al final.
    5) Si hay contradicciones, menciónalas.
    6) Adapta longitud (1–5 párrafos máx.). Si no hay suficiente info, admítelo.

    Formato de salida (Markdown):
    - Respuesta estructurada (introducción + puntos principales + info complementaria)
    - Sección final "Fuentes:" con una lista de referencias
    """,
    output_key="MultiRetrievalAgent.final_response",
)


# ==================== CONFIGURACIÓN DE SUB-AGENTES ====================
research_pipeline = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[query_generator_agent, multi_retrieval_agent, synthesizer_agent],
)

triage_agent.sub_agents = [research_pipeline]


# ==================== AGENTE PRINCIPAL ====================

agentic_rag_multi_query_bot = triage_agent
