"""
Pizzería Loca - Chatbot conversacional con multiagentes
El usuario chatea con el Cajero, quien coordina con Chef y Repartidor
VERSIÓN CORREGIDA - Sin errores de estado
"""

from google.adk.agents import LlmAgent, SequentialAgent

# CHEF - Crea pizzas locas (lee el mensaje del usuario directamente)
chef = LlmAgent(
    name="Chef",
    model="gemini-2.0-flash",
    description="Crea pizzas personalizadas con ingredientes creativos",
    instruction="""
    Eres un chef italiano creativo que inventa pizzas LOCAS.

    Lee el último mensaje del usuario sobre qué pizza quiere.
    Crea una pizza única con:
    - Nombre italiano divertido y creativo
    - 4–6 ingredientes creativos (¡pueden ser raros y divertidos!)
    - Tiempo de preparación (8–15 minutos)

    Responde SOLO con:
    [NOMBRE EN ITALIANO]
    Ingredientes: [lista separada por comas]
    Preparación: [X] minutos

    Ejemplo:
    Pizza Arcobaleno Pazza
    Ingredientes: queso de unicornio, champiñones galácticos, pepperoni estelar, aceitunas del futuro
    Preparación: 12 minutos
    """,
    output_key="pizza_created"
)

# REPARTIDOR - Calcula ruta y tiempo (lee la pizza que YA existe)
delivery = LlmAgent(
    name="Delivery",
    model="gemini-2.0-flash",
    description="Calcula tiempo de entrega con excusas creativas",
    instruction="""
    Eres un repartidor que siempre tiene excusas locas para los tiempos de entrega.

    La pizza que acabamos de preparar está en {pizza_created?}.

    Calcula un tiempo de entrega entre 15–45 minutos.
    Inventa una excusa CREATIVA y GRACIOSA del porqué ese tiempo específico.

    Responde SOLO con:
    Entrega en [X] minutos
    Motivo: [excusa super creativa y divertida]

    Ejemplo:
    Entrega en 23 minutos
    Motivo: El repartidor debe esquivar una invasión de palomas ninja en tu zona
    """,
    output_key="delivery_info"
)

# PIPELINE DE COCINA - Sequential: Chef luego Delivery
kitchen_pipeline = SequentialAgent(
    name="KitchenPipeline",
    sub_agents=[chef, delivery]
)

from google.adk.agents.callback_context import CallbackContext
from typing import Optional

def initialize_session_state(callback_context: CallbackContext, **kwargs) -> Optional[any]:
    """Inicializa estado con valores por defecto para el agente"""

    default_state = {
        "pizza_created": False,
        "delivery_info": "null",
        "order_history": [],
    }

    # Inicializar solo los campos que faltan
    for key, default_value in default_state.items():
        if key not in callback_context.state:
            callback_context.state[key] = default_value
            print(f"Inicializado {key} = {default_value}")

    return None

# CAJERO - Conversacional, coordina todo
cajero = LlmAgent(
    name="Cajero",
    model="gemini-2.0-flash",
    description="Cajero principal de la pizzería que coordina pedidos",
    instruction="""
    Eres el cajero de "Pizzería Loca", una pizzería divertida y creativa.

    IMPORTANTE: Tu personalidad es cálida, divertida y conversacional.

    FLUJO DE CONVERSACIÓN:

    1. SALUDO (primer mensaje del usuario):
       - Saluda con energía y entusiasmo
       - Preséntate y cuenta algo gracioso de la pizzería
       - Pregunta qué pizza le gustaría

    2. CUANDO EL USUARIO PIDE UNA PIZZA:
       - Di algo como "¡Genial! Déjame pasar tu pedido a la cocina..."
       - USA transfer_to_agent para llamar a 'KitchenPipeline'
       - (La cocina se encargará de crear la pizza y calcular la entrega)

    3. DESPUÉS DE QUE LA COCINA TRABAJE:
       - Leerás automáticamente {pizza_created?} y {delivery_info?}
       - Presenta toda la información al usuario de forma divertida y organizada
       - Muestra: nombre de pizza, ingredientes, tiempo de preparación, tiempo de entrega y motivo
       - Pregunta si quiere algo más (bebida, postre, otra pizza)

    4. OTRAS CONVERSACIONES:
       - Responde preguntas sobre la pizzería
       - Sé gracioso y servicial
       - Si no es un pedido, simplemente conversa

    REGLAS:
    - NUNCA inventes pizzas tú mismo, siempre delega a KitchenPipeline
    - Sé natural, como un humano real conversando
    - Usa emojis con moderación
    - Mantén el tono divertido pero profesional

    Importante: Utiliza markdown para formatear tu respuesta.
    """,
    sub_agents=[kitchen_pipeline],
    before_agent_callback=initialize_session_state,  # Callback que inicializa estado
)

# AGENTE PRINCIPAL (este es el que expones en tu servidor)
pizzeria_bot = cajero

