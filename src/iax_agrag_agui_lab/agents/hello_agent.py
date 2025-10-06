# Import necessary libraries
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For OpenAI support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
from typing import Optional, Dict, Any

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.CRITICAL)

print("Libraries imported.")


from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()
from langfuse import Langfuse

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

# Test LLM with a direct call
print(llm.llm_client.completion(model=llm.model, 
                                messages=[{"role": "user", 
                                           "content": "Are you ready?"}], 
                                tools=[]))

print("\nOpenAI is ready for use.")
# Sending a simple query to the database
neo4j_is_ready = graphdb.send_query("RETURN 'Neo4j is Ready!' as message")

print(neo4j_is_ready)

# Define a basic tool -- send a parameterized cypher query
def say_hello(person_name: str) -> dict:
    """Formats a welcome message to a named person. 

    Args:
        person_name (str): the name of the person saying hello

    Returns:
        dict: A dictionary containing the results of the query.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'query_result' key with an array of result rows.
              If 'error', includes an 'error_message' key.
    """
    return graphdb.send_query("RETURN 'Hello to you, ' + $person_name AS reply",
    {
        "person_name": person_name
    })
# Define the Cypher Agent
hello_agent = Agent(
    name="hello_agent_v1",
    model=llm, # defined earlier in a variable
    description="Has friendly chats with a user.",
    instruction="""You are a helpful assistant, chatting with a user. 
                Be polite and friendly, introducing yourself and asking who the user is. 

                If the user provides their name, use the 'say_hello' tool to get a custom greeting.
                If the tool returns an error, inform the user politely. 
                If the tool is successful, present the reply.
                """,
    tools=[say_hello], # Pass the function directly
)

print(f"Agent '{hello_agent.name}' created.")