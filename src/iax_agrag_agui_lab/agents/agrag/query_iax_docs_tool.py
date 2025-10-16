
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from google.adk.agents import Agent
from typing import List

async def query_iax_documentation_rag(question: str, top_k: int = 5) -> List[Document]:
    """Busca en la documentación de IAX ("la plataforma") para encontrar
    información relevante que responda la pregunta del usuario.

    Params:
        question: Pregunta a buscar.
        top_k: Número máximo de documentos a recuperar (por defecto 5).

    Returns:
        List[Document]: Lista de documentos relevantes (con metadata de fuente cuando esté disponible).
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
        k=top_k,
        namespace="iax-documentation-namespace",
    )
    print("--------------------------------")
    print(f"Buscando: {question}")
    print(f"Documentos encontrados: {len(retrived_documents)}")
    print("--------------------------------")

    return retrived_documents
