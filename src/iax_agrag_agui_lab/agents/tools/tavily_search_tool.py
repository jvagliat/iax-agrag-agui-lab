from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import TavilySearchResults


def create_adk_tavily_search_tool() -> LangchainTool:
    """
    Creates a Tavily search tool wrapped in a LangchainTool for ADK.

    Returns:
        LangchainTool: The Tavily search tool wrapped for ADK.
    """

    # Instantiate the LangChain tool
    tavily_tool_instance = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True,
    )

    # Wrap it with LangchainTool for ADK
    adk_tavily_tool = LangchainTool(tool=tavily_tool_instance)

    return adk_tavily_tool
