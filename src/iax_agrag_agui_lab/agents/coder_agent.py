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
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

MODEL = "openai/gpt-4o"
llm = LiteLlm(model=MODEL)

# Crear el agente con el prompt para escribir código
coder_agent = Agent(
    model=llm,
    name="Coder",
    description="I am a coder", 
    instruction="""Eres especialista en progración en python. 
    Tu única función es escribir código cuando te lo soliciten. 
    Responde únicamente con el código solicitado, 
    sin explicaciones adicionales.
    
    Tu salida es siempre codigo Python en un formato estandar.""", 
    output_key="Coder.code"
)
