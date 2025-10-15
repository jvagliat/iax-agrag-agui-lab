# Import necessary libraries
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For OpenAI support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

import warnings
import logging


load_dotenv()

import logfire
from langfuse import Langfuse

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

logfire.configure(token=os.getenv("LOGFIRE_WRITE_TOKEN"))
# logfire.info('Hello, {place}!', place='World')


# langfuse = Langfuse(
#   secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
#   public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
#   host=os.getenv("LANGFUSE_HOST")
# )
langfuse = Langfuse(
  secret_key="sk-lf-6ce5f05d-8a04-45cf-9515-97c213299fa1",
  public_key="pk-lf-01bf2b5e-1d2e-4aaa-9821-51501f9638a2",
  host="https://us.cloud.langfuse.com"
)
# Convenience libraries for working with Neo4j inside of Google ADK
from data.neo4j_for_adk import graphdb
# Define Model Constants for easier use 
MODEL_GPT = "openai/gpt-4o"

llm = LiteLlm(model=MODEL_GPT)

# Sending a simple query to the database
neo4j_is_ready = graphdb.send_query("RETURN 'Neo4j is Ready!' as message")

# Define a basic tool -- send a parameterized cypher query
def say_hello(person_name: str) -> dict:
    """Formatea un saludo personalizado para una persona.

    Args:
        person_name (str): nombre de la persona a saludar.

    Returns:
        dict: Un diccionario con los resultados de la consulta.
              Incluye la clave 'status' ('success' o 'error').
              Si es 'success', incluye 'query_result' con un arreglo de filas de resultado.
              Si es 'error', incluye 'error_message' con la razón del error.
    """
    return graphdb.send_query("RETURN 'Hello to you, ' + $person_name AS reply",
    {
        "person_name": person_name
    })
# Define the Cypher Agent
hello_agent = Agent(
    name="hello_agent_v1",
    model=llm, # defined earlier in a variable
    description="Mantiene charlas cordiales con el usuario.",
    instruction="""
                Eres un asistente útil y amable. Preséntate brevemente y pregunta el nombre del usuario.

                Si el usuario proporciona su nombre, USA la herramienta 'say_hello' para obtener un saludo personalizado.
                - Si la herramienta devuelve error: informa con cortesía y pide reintentar el nombre.
                - Si es exitosa: muestra la respuesta del saludo en una línea clara.

                Reglas de estilo:
                - Responde en el mismo idioma del usuario (por defecto, español).
                - Sé breve (1–2 párrafos máximo).
                - Formato de salida sugerido:
                  "Saludo: <texto devuelto por la herramienta>"
                """,
    tools=[say_hello], # Pass the function directly
)
