"""
🔍 Agentic RAG System - Multi-Query Version
Sistema de RAG con Triage → Query Generation → Multi-Retrieval → Synthesis
Usando OpenAI y estrategia multi-query inspirada en sistema legal
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool
from typing import Any

from agents.agrag.query_docs_tool import query_iax_documentation_rag
from google.adk.models.lite_llm import LiteLlm

# ==================== HERRAMIENTAS ====================
vector_search_tool = FunctionTool(
    func=query_iax_documentation_rag,
)

# ==================== AGENTES ====================

# 1️⃣ TRIAGE AGENT - Clasifica consultas (usando OpenAI)
triage_agent = LlmAgent(
    name="TriageAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Clasifica consultas del usuario: GENERAL o SPECIFIC",
    instruction="""
    Eres un asistente que clasifica consultas de usuarios sobre iattraxia y la plataforma IAX.

    Analiza la pregunta del usuario y determina:

    - **GENERAL**: Saludos, preguntas simples, conversación casual, agradecimientos
      Ejemplos: "hola", "¿cómo estás?", "gracias", "¿qué puedes hacer?", "buenos días"

    - **SPECIFIC**: Preguntas específicas sobre:
      * iattraxia (la empresa, servicios, equipo)
      * Plataforma IAX (funcionalidades, arquitectura, uso)
      * Desarrollo de software con IA
      * Agentes autónomos o automatizaciones
      * Tecnologías relacionadas

    **Si es GENERAL**: Responde directamente de forma amigable y breve. Preséntate como
    asistente especializado en iattraxia/IAX y ofrece ayuda.

    **Si es SPECIFIC**: Di algo como "Déjame buscar esa información en la documentación..."
    y luego USA transfer_to_agent para llamar a 'QueryGeneratorAgent'.

    Sé conversacional y amigable.
    """,
    output_key="triage_result",
    sub_agents=[]  # Se configurará después
)


# 2️⃣ QUERY GENERATOR - Genera múltiples consultas diversas
query_generator_agent = LlmAgent(
    name="QueryGeneratorAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Genera 3 consultas de búsqueda diversas para maximizar cobertura",
    instruction="""
    Eres un experto en formulación de consultas de búsqueda.

    El usuario preguntó: (lee el mensaje del usuario)

    **Tu trabajo**:
    Genera EXACTAMENTE 3 consultas de búsqueda diferentes para buscar información relevante.
    Las consultas deben ser:
    - **Diversas**: Enfocar diferentes aspectos de la pregunta
    - **Específicas**: Usar términos técnicos relevantes (IAX, iattraxia, agentes, automatización, etc.)
    - **Complementarias**: Cubrir diferentes ángulos de la misma necesidad

    **Formato de salida**:
    Devuelve SOLO las 3 consultas, una por línea, sin numeración ni explicaciones adicionales.

    Ejemplo de salida esperada:
    funcionalidades de la plataforma IAX
    arquitectura y componentes del sistema IAX
    casos de uso de agentes autónomos en IAX

    Después de generar las consultas, USA transfer_to_agent para llamar a 'MultiRetrievalAgent'.
    """,
    output_key="generated_queries",
    sub_agents=[]  # Se configurará después
)


# 3️⃣ MULTI-RETRIEVAL AGENT - Ejecuta búsquedas con las 3 consultas
multi_retrieval_agent = LlmAgent(
    name="MultiRetrievalAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Ejecuta búsquedas vectoriales con las consultas generadas",
    instruction="""
    Eres un experto en recuperación de información.

    Las consultas generadas están en: {generated_queries}

    **Tu trabajo**:
    1. Lee las 3 consultas generadas
    2. USA la herramienta `vector_search` con CADA una de las 3 consultas
    3. Recopila todos los resultados obtenidos
    4. Si obtienes resultados relevantes, transfiere a 'SynthesizerAgent' con transfer_to_agent
    5. Si no encuentras nada útil en ninguna consulta, díselo al usuario educadamente

    **Importante**:
    - Ejecuta las 3 búsquedas aunque algunas den resultados
    - Los resultados combinados darán mejor cobertura al Synthesizer
    """,
    output_key="retrieved_chunks",
    tools=[vector_search_tool],
    sub_agents=[]  # Se configurará después
)


# 4️⃣ SYNTHESIZER AGENT - Genera respuesta final con citas
synthesizer_agent = LlmAgent(
    name="SynthesizerAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Genera respuesta final integrando información de múltiples búsquedas",
    instruction="""
    Eres un redactor técnico experto especializado en iattraxia y la plataforma IAX.

    El usuario preguntó: (lee el mensaje original del usuario)
    Los chunks recuperados de múltiples búsquedas están en: {retrieved_chunks}

    **Tu trabajo**:
    1. Lee cuidadosamente TODOS los chunks recuperados de las diferentes consultas
    2. Identifica información complementaria y elimina redundancias
    3. Genera una respuesta clara, coherente y bien estructurada
    4. CITA las fuentes usando formato: 📚 **Fuentes**: [fuente1], [fuente2]
    5. Si la información es parcial o hay lagunas, indícalo explícitamente
    6. Usa tono profesional pero accesible

    **Formato de respuesta**:
    - Usa **markdown** con viñetas, negritas para términos clave
    - Estructura lógica: introducción → puntos principales → información complementaria
    - Citas distribuidas en el texto cuando sea relevante
    - Sección final "📚 **Fuentes**:" con lista de documentos consultados

    **Reglas estrictas**:
    - NO inventes información que no esté en los chunks
    - Si múltiples chunks contradicen, menciona ambas perspectivas
    - Adapta longitud según complejidad (1 párrafo a 5 párrafos máximo)
    - Si no hay suficiente información, admítelo y sugiere reformular la pregunta
    """,
    output_key="final_response"
)


# ==================== CONFIGURACIÓN DE SUB-AGENTES ====================

# Pipeline: Query Generation → Multi-Retrieval → Synthesis
research_pipeline = SequentialAgent(
    name="ResearchPipeline",
    # instruction="Cada vez que avances en la secuencia ve USANDO transfer_to_agent al siguiente agente.",
    sub_agents=[query_generator_agent, multi_retrieval_agent, synthesizer_agent]
)

# Triage tiene acceso al pipeline completo
triage_agent.sub_agents = [research_pipeline]


# ==================== AGENTE PRINCIPAL ====================

agentic_rag_multi_query_bot = triage_agent
