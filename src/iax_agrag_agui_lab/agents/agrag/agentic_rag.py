"""
Agentic RAG System - Versión simple para pruebas
Pipeline: Triage + Retrieval + Synthesis
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from iax_agrag_agui_lab.agents.agrag.query_iax_docs_tool import query_iax_documentation_rag

# ==================== HERRAMIENTAS ====================
vector_search_tool = FunctionTool(
    func=query_iax_documentation_rag,
)

# ==================== AGENTES ====================

# 1) TRIAGE AGENT - Clasifica consultas
triage_agent = LlmAgent(
    name="TriageAgent",
    model="gemini-2.0-flash",
    description="Clasifica consultas del usuario: GENERAL o ESPECÍFICA",
    instruction="""
    Eres un asistente que clasifica consultas de usuarios.

    Determina el tipo de consulta:
    - GENERAL: saludos, cortesía, conversación casual.
    - ESPECÍFICA: sobre iattraxia, plataforma IAX, desarrollo de software con IA, agentes o automatizaciones.

    Si es GENERAL: responde de forma amable y breve.
    Si es ESPECÍFICA: indica "Déjame buscar esa información para ti..." y luego USA transfer_to_agent para llamar a 'RetrievalAgent'.

    Responde en el mismo idioma del usuario.
    """,
    output_key="triage_result",
    sub_agents=[]  # Se configurará después
)


# 2) RETRIEVAL AGENT - Busca y evalúa información
retrieval_agent = LlmAgent(
    name="RetrievalAgent",
    model="gemini-2.0-flash",
    description="Busca información relevante en la base de conocimiento",
    instruction="""
    Eres un experto en recuperación de información.

    Tareas:
    1) Identifica keywords de la consulta del usuario.
    2) USA la herramienta `vector_search` con dichas keywords.
    3) Evalúa la relevancia de los resultados.
    4) Si hay buenos resultados, transfiere a 'SynthesizerAgent' con transfer_to_agent.
    5) Si no encuentras nada útil, informa con cortesía.

    Reglas:
    - Siempre usa `vector_search` antes de decidir.
    - No inventes información.
    - Resume claramente qué encontraste (máx. 3 líneas) antes de transferir.
    """,
    output_key="retrieved_chunks",
    tools=[vector_search_tool],
    sub_agents=[]  # Se configurará después
)


# 3) SYNTHESIZER AGENT - Genera respuesta con citas
synthesizer_agent = LlmAgent(
    name="SynthesizerAgent",
    model="gemini-2.0-flash",
    description="Redacta respuesta final citando fuentes",
    instruction="""
    Eres un redactor técnico experto.

    Los chunks recuperados están en: {retrieved_chunks?}

    Tareas:
    1) Lee cuidadosamente los chunks recuperados.
    2) Redacta una respuesta clara y estructurada que responda la pregunta.
    3) CITA las fuentes al final.
    4) Si la información es parcial, indícalo. No inventes.

    Formato de salida (Markdown):
    - Párrafo(s) de respuesta (1–3 párrafos).
    - Sección final "Fuentes:" con lista de referencias. Incluye título/ID y URL si está disponible en metadata.
    """,
    output_key="final_response"
)


# ==================== CONFIGURACIÓN DE SUB-AGENTES ====================

# Pipeline de Retrieval + Synthesis
retrieval_pipeline = SequentialAgent(
    name="RetrievalPipeline",
    sub_agents=[retrieval_agent, synthesizer_agent]
)

# Triage tiene acceso al pipeline completo
triage_agent.sub_agents = [retrieval_pipeline]


# ==================== AGENTE PRINCIPAL ====================

agentic_rag_bot = triage_agent

