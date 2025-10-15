from agents.tools.tavily_search_tool import create_adk_tavily_search_tool
from google.adk.agents import LlmAgent

web_search_agent = LlmAgent(
    name="WebSearchAgent",
    model="gemini-2.0-flash",
    description="Agent to answer questions using TavilySearch.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    tools=[create_adk_tavily_search_tool()] 
)