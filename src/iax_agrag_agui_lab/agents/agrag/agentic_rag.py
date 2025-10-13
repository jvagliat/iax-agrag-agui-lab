"""
🔍 Agentic RAG System - Versión Simple para Pruebas
Sistema de RAG con Triage → Retrieval → Synthesis
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool
from typing import Any

from iax_agrag_agui_lab.agents.agrag.query_iax_docs_tool import query_iax_documentation_rag

# ==================== HERRAMIENTAS ====================
vector_search_tool = FunctionTool(
    func=query_iax_documentation_rag,
)

# ==================== AGENTES ====================

# 1️⃣ TRIAGE AGENT - Clasifica consultas
triage_agent = LlmAgent(
    name="TriageAgent",
    model="gemini-2.0-flash",
    description="Clasifica consultas del usuario: GENERAL o SPECIFIC",
    instruction="""
    Eres un asistente que clasifica consultas de usuarios.

    Analiza la pregunta del usuario y determina:

    - **GENERAL**: Saludos, preguntas simples, conversación casual
      Ejemplos: "hola", "¿cómo estás?", "gracias", "¿qué puedes hacer?"

    - **SPECIFIC**: Preguntas especificas sobre la empresa, iattraxia o sus servicios, 
    o aspectos relacionados con la plataforma IAX, desarrollo de software, IA, Agentes o automatizaciones. 

    **Si es GENERAL**: Responde directamente de forma amigable y breve.

    **Si es SPECIFIC**: Di algo como "Déjame buscar esa información para ti..."
    y luego USA la herramienta transfer_to_agent para llamar a 'RetrievalAgent'.

    Sé conversacional y amigable en tu respuesta.
    """,
    output_key="triage_result",
    sub_agents=[]  # Se configurará después
)


# 3️⃣ RETRIEVAL AGENT - Busca y evalúa información
retrieval_agent = LlmAgent(
    name="RetrievalAgent",
    model="gemini-2.0-flash",
    description="Busca información relevante en la base de conocimiento",
    instruction="""
    Eres un experto en recuperación de información.

    El usuario preguntó: (lee el mensaje del usuario)

    **Tu trabajo**:
    1. Identifica las palabras clave principales de la pregunta
    2. USA la herramienta `vector_search` con esas keywords
    3. Evalúa si los resultados son relevantes
    4. Si los chunks son buenos, transfiere a 'SynthesizerAgent' con transfer_to_agent
    5. Si no encuentras nada útil, díselo al usuario educadamente

    **Importante**: Siempre usa la herramienta vector_search antes de decidir.
    """,
    output_key="retrieved_chunks",
    tools=[vector_search_tool],

    sub_agents=[]  # Se configurará después
)


# 4️⃣ SYNTHESIZER AGENT - Genera respuesta con citas
synthesizer_agent = LlmAgent(
    name="SynthesizerAgent",
    model="gemini-2.0-flash",
    description="Redacta respuesta final citando fuentes",
    instruction="""
    Eres un redactor técnico experto.

    El usuario preguntó: (lee el mensaje original del usuario)
    Los chunks recuperados están en: {retrieved_chunks?}

    **Tu trabajo**:
    1. Lee cuidadosamente los chunks recuperados
    2. Genera una respuesta clara y estructurada que responda la pregunta
    3. CITA las fuentes al final: "📚 Fuentes: [fuente1], [fuente2]"
    4. Si la información es parcial, indícalo
    5. Usa un tono profesional pero amigable

    **Formato de respuesta**:
    - Markdown bien especificado y delimitado
    - Párrafo principal respondiendo la pregunta
    - Información adicional relevante (si aplica)
    - Línea en blanco
    - 📚 Fuentes: [lista de fuentes citadas]

    NO inventes información que no esté en los chunks.

    """,
    output_key="final_response"
)


# ==================== CONFIGURACIÓN DE SUB-AGENTES ====================

# Pipeline de Retrieval → Synthesis
retrieval_pipeline = SequentialAgent(
    name="RetrievalPipeline",
    sub_agents=[retrieval_agent, synthesizer_agent]
)

# Triage tiene acceso al pipeline completo
triage_agent.sub_agents = [retrieval_pipeline]


# ==================== AGENTE PRINCIPAL ====================

agentic_rag_bot = triage_agent
