
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from google.adk.agents import Agent
from typing import List

async def query_workana_documentation_rag(question: str, top_k: int = 5) -> List[Document]:
    """Busca en el Help Desk de Workana para encontrar información relevante.

    Params:
        question: Pregunta a buscar.
        top_k: Número máximo de documentos a recuperar (por defecto 5).

    Returns:
        List[Document]: Lista de documentos relevantes (con metadata de fuente cuando esté disponible).
    """
    pinecone_vector_store = Pinecone.from_existing_index(
        index_name="iax-workana-discord-doc-files",
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )
    retrived_documents = pinecone_vector_store.similarity_search(
        question,
        k=top_k,
        # namespace="iax-workana-discord-doc-files-namespace",
    )
    print("--------------------------------")
    print(f"Buscando: {question}")
    print(f"Documentos encontrados: {len(retrived_documents)}")
    print("--------------------------------")

    return retrived_documents
