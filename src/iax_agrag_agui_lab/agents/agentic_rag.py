"""
🔍 Agentic RAG System - Versión Simple para Pruebas
Sistema de RAG con Triage → Retrieval → Synthesis
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool
from typing import Any


# ==================== MOCK TOOLS ====================

def mock_vector_search(query: str, top_k: int = 3) -> str:
    """Mock de búsqueda vectorial - retorna chunks simulados"""
    # Base de datos mock de documentación técnica
    mock_db = {
        "python": [
            {"content": "Python es un lenguaje de programación interpretado de alto nivel. Creado por Guido van Rossum en 1991.", "source": "python_intro.md:1-2"},
            {"content": "Python usa tipado dinámico y gestión automática de memoria. Soporta múltiples paradigmas: orientado a objetos, imperativo, funcional.", "source": "python_features.md:5-7"},
            {"content": "Para instalar paquetes en Python se usa pip: 'pip install nombre-paquete'. El gestor de entornos virtuales más común es venv.", "source": "python_tools.md:12-15"}
        ],
        "javascript": [
            {"content": "JavaScript es un lenguaje de programación que se ejecuta principalmente en navegadores web. Creado por Brendan Eich en 1995.", "source": "js_intro.md:1-2"},
            {"content": "JavaScript usa eventos y callbacks para manejar asincronía. Las promesas y async/await simplifican el código asíncrono.", "source": "js_async.md:8-10"},
            {"content": "Node.js permite ejecutar JavaScript en el servidor. NPM es el gestor de paquetes estándar.", "source": "js_runtime.md:3-5"}
        ],
        "adk": [
            {"content": "Google ADK (Agent Development Kit) es un framework para construir agentes LLM jerárquicos. Los agentes pueden coordinarse usando transfer_to_agent.", "source": "adk_overview.md:1-3"},
            {"content": "Los agentes ADK comparten estado mediante session state (dict). Se accede con {variable_name} en las instrucciones.", "source": "adk_state.md:10-12"},
            {"content": "Para crear un agente básico: LlmAgent(name='Agent', model='gemini-2.0-flash', instruction='...', output_key='result').", "source": "adk_quickstart.md:15-17"}
        ],
        "default": [
            {"content": "Documentación general del sistema. Esta es una base de conocimiento de ejemplo para demostrar el sistema de RAG agéntico.", "source": "README.md:1"},
            {"content": "El sistema incluye documentación sobre múltiples lenguajes de programación y frameworks.", "source": "index.md:5"},
        ]
    }

    # Selección simple basada en keywords
    query_lower = query.lower()
    selected_docs = mock_db["default"]

    if "python" in query_lower or "pip" in query_lower or "venv" in query_lower:
        selected_docs = mock_db["python"]
    elif "javascript" in query_lower or "js" in query_lower or "node" in query_lower:
        selected_docs = mock_db["javascript"]
    elif "adk" in query_lower or "agente" in query_lower or "agent" in query_lower:
        selected_docs = mock_db["adk"]

    # Limitar a top_k resultados
    results = selected_docs[:top_k]

    # Formatear resultados
    formatted = "\n\n".join([
        f"[{i+1}] {doc['content']}\nFuente: {doc['source']}"
        for i, doc in enumerate(results)
    ])

    return formatted


vector_search_tool = FunctionTool(
    func=mock_vector_search,
    # name="vector_search",
    # description="Busca información en la base de conocimiento vectorial. Retorna chunks relevantes con sus fuentes."
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

    - **SPECIFIC**: Preguntas técnicas que requieren buscar en documentación
      Ejemplos: "¿cómo instalar Python?", "explica async en JS", "qué es ADK"

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
