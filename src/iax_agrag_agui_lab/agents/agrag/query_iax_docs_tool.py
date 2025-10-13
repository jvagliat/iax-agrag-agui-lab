
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from google.adk.agents import Agent
from typing import List

async def query_iax_documentation_rag(question: str) -> str:
    """A tool that research the IAX ("la plataforma") documentation to find relevant
    information to answer the user's question.

    Params:
        question: The question to search for.

    Returns:
        A list of documents that contain relevant information.
    """
    pinecone_vector_store = Pinecone.from_existing_index(
        index_name="iax-documentation",
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )
    retrived_documents = pinecone_vector_store.similarity_search(
        question,
        namespace="iax-documentation-namespace",
    )
    print("--------------------------------")
    print(f"se esta buscando la pregunta: {question}")
    print("--------------------------------")
    print(f"se encontraron {len(retrived_documents)} documentos")
    print(retrived_documents)
    print("--------------------------------")

    # Format documents as a string with metadata
    formatted_results = []
    for i, doc in enumerate(retrived_documents, 1):
        source = doc.metadata.get('source', 'Unknown')
        formatted_results.append(f"Document {i} (Source: {source}):\n{doc.page_content}")

    return retrived_documents
    #return "\n\n---\n\n".join(formatted_results)
