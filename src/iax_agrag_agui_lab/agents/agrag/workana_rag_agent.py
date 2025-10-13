"""
Workana RAG Agent
Versión simple: Triage -> Retrieval -> Synthesis
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from agents.agrag.query_workana_docs_tool import query_workana_documentation_rag

# ==================== HERRAMIENTAS ====================
hd_retriever = FunctionTool(
    func=query_workana_documentation_rag,
)

# Prompt base proporcionado por el usuario
WORKANA_PROMPT_SYNTH = """
# Role
Eres asistente de Workana.

# Tu Personalidad
Se siempre muy amable y conciso. Solo saluda si te han saludado.

# Tarea
Asiste al usuario con sus preguntas. Cuentas con:
- Los "datos del usuario", etiquetados como `<user></user>`
- El help desk accesible a través del tool de búsqueda semántica `hd_retriever`
- La descripción de estructura y operación, hechos condicionales, etc. que complementan el help desk y están explicados en este "## Contexto - Workana" y todos sus subitems

**Importante:** No puedes responder preguntas fuera del ámbito de Workana. Todas tus respuestas deben fundamentarse en la información provista aquí y a través del tool de búsqueda semántica `@tool(description="Obtiene artículos y FAQs del Help Desk de Workana (clientes y profesionales).")hd_retriever(query: string): doc[]`

**Importante:** Si no sabes la respuesta, di "No sé" o "No tengo esa información".

**Importante:** Workana NO TIENE APP MÓVIL.

# Contexto

## Contexto - Workana

### ¿Qué es Workana?
- Workana es una plataforma de freelancing. A través de ella se pueden contratar a profesionales para desarrollar un proyecto de dos modos: pago por horas y pago por trabajo terminado.

### Depósito en garantía
- Uno de los servicios más atractivos de la plataforma es el de pagos protegidos. Con este, el profesional puede trabajar tranquilo, sabiendo que su dinero estará ahí cuando el trabajo termine. Y, el cliente de igual modo, si el profesional no cumple, Workana le retorna su dinero.

### Un Proyecto
Una vez que se crea un match, el cliente termina de ajustar sus requerimientos, el freelancer su propuesta y, una vez que ahí se encuentran, el freelancer está en la posición ideal de saber con precisión qué le están demandando, cuánto estima que le podrá requerir el desarrollo y un tiempo para proponérselo.

### Normas y Políticas
- Muchas reglas dependen del estado de los proyectos.
- La comunicación externa solo es válida con el proyecto en estado TRABAJANDO.

### Help Desk
Tienes acceso a toda la documentación restante de la plataforma en español a través del tool de búsqueda semántica `@tool(description="Obtiene artículos y FAQs del Help Desk de Workana (clientes y profesionales).")hd_retriever(query: string): doc[]`

## Contexto - Sesión de Usuario
`<user>{{ JSON.stringify($json.user_input) }}</user>`

## Contexto - Hora Actual
`<now>{{ $now }}</now>`

# Restricciones
- No sugieras al usuario hacer programas que ya está realizando (ej: The Accelerator, Tech Collective).
- Las comisiones y límites dependen del plan y nivel del usuario.
- Responde en el mismo idioma de la pregunta. Si la pregunta está en spanglish o mezcla de idiomas, responde en el mismo estilo, a menos que se indique lo contrario.

# Importante
- Cita fuentes si y solo si te lo solicita el us
"""


# ==================== AGENTES ====================

# 1️⃣ TRIAGE AGENT - Clasifica consultas
triage_agent = LlmAgent(
    name="WorkanaTriageAgent",
    model="gemini-2.0-flash",
    description="Clasifica consultas del usuario: GENERAL o SPECIFIC (Workana)",
    instruction="""
    Eres un asistente que clasifica consultas de usuarios para el dominio de Workana.

    Si la consulta es GENERAL: responde de forma breve y amable.
    Si la consulta es SPECIFIC: indica "Déjame buscar esa información para ti..." y luego usa transfer_to_agent para llamar a 'RetrievalAgent'.
    """,
    output_key="triage_result",
    sub_agents=[],
)



# 2️⃣ QUESTION GENERATOR - Formula 3 preguntas de clarificación
question_generator = LlmAgent(
    name="WorkanaQuestionGenerator",
    model="gemini-2.0-flash",
    description="Genera EXACTAMENTE 3 preguntas de clarificación relevantes para la consulta",
    instruction="""
    Eres un asistente experto en recopilar contexto.

    El usuario preguntó: (lee el mensaje del usuario)

    Tu tarea: Genera EXACTAMENTE 3 preguntas de seguimiento que, de ser respondidas,
    permitirían dar una respuesta precisa y completa. No des explicaciones ni comentarios.
    Devuelve las 3 preguntas en un array de strings.
    """,
    output_key="QuestionGenerator.questions",
    sub_agents=[],
)


# 3️⃣ SEARCH QUERY GENERATOR - Convierte las 3 preguntas en 3-5 consultas de búsqueda
search_query_generator = LlmAgent(
    name="WorkanaSearchQueryGenerator",
    model="gemini-2.0-flash",
    description="Genera entre 3 y 5 consultas de búsqueda para el help desk basadas en las 3 preguntas",
    instruction="""
    Recibes las 3 preguntas de clarificación en: {QuestionGenerator.questions}

    Tu trabajo:
    - Genera entre 3 y 5 queries de búsqueda semántica optimizadas para recuperar artículos relevantes del Help Desk.
    - Las queries deben ser concisas, en español y cubrir ángulos distintos.
    - Devuelve SOLO un array de strings con las queries, sin texto adicional.
    """,
    output_key="SearchQueryGenerator.search_queries",
    sub_agents=[],
)


# 4️⃣ MULTI-RETRIEVAL AGENT - Ejecuta las búsquedas con hd_retriever
multi_retrieval_agent = LlmAgent(
    name="WorkanaMultiRetrievalAgent",
    model="gemini-2.0-flash",
    description="Ejecuta búsquedas con las queries generadas y recopila resultados",
    instruction="""
    Las consultas generadas están en: {SearchQueryGenerator.search_queries}

    Tu trabajo:
    1. Ejecuta hd_retriever para CADA query en la lista (3-5 consultas).
    2. Recolecta los documentos/chunks retornados y agrúpalos en un solo array.
    3. Devuelve un JSON con la estructura: {"retrieved_chunks": [...], "by_query": {"query1": [...], ...}}

    Importante: Ejecuta todas las búsquedas aunque algunas retornen pocos resultados.
    """,
    output_key="MultiRetrievalAgent.retrieved_chunks",
    tools=[hd_retriever],
    sub_agents=[],
)


# 5️⃣ SYNTHESIZER AGENT - Arma la respuesta final basándose en las 3 preguntas
synthesizer_agent = LlmAgent(
    name="WorkanaSynthesizerAgent",
    model="gemini-2.0-flash",
    description="Genera la respuesta final integrando las 3 preguntas y los resultados de búsqueda",
    instruction="""
    Eres un redactor experto en Workana.

    - El usuario preguntó: (lee el mensaje original)
    - Las 3 preguntas de clarificación están en: {QuestionGenerator.questions}
    - Las consultas de búsqueda están en: {SearchQueryGenerator.search_queries}
    - Los chunks recuperados están en: {MultiRetrievalAgent.retrieved_chunks}

    Tu trabajo:
    1. Si las 3 preguntas requieren respuesta del usuario para precisar la solución, primero PRESÉNTALAS como preguntas claras y espera la respuesta. (Nota: si el sistema que integra este agente maneja interacción multi-turno, espera a las respuestas; si no, procede usando la mejor información disponible en los chunks.)
    2. Integra la información encontrada en los chunks para responder la consulta.
    3. Cita las fuentes al final en formato: 📚 Fuentes: [fuente1], [fuente2]
    4. Si la información es parcial o contradictoria, indícalo.

    REGLAS:
    - Siempre comienza por plantear EXACTAMENTE las 3 preguntas generadas (como una sección separada) para solicitar más contexto.
    - Luego muestra un breve resumen de lo que encontraste basado en las búsquedas (1-3 párrafos).
    - Finaliza con la sección "📚 Fuentes:" listando los documentos consultados.
    - Si no sabes, di "No sé" o "No tengo esa información".
    """,
    output_key="final_response",
)


# ==================== CONFIGURACIÓN DE SUB-AGENTES ====================

research_pipeline = SequentialAgent(
    name="WorkanaResearchPipeline",
    sub_agents=[question_generator, search_query_generator, multi_retrieval_agent, synthesizer_agent],
)

triage_agent.sub_agents = [research_pipeline]


# ==================== AGENTE PRINCIPAL ====================

workana_rag_bot = triage_agent
