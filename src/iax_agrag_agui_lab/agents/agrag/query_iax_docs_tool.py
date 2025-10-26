
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from typing import List

async def query_iax_documentation_rag(question: str, top_k: int = 5) -> List[dict]:
    """Busca en la documentación de IAX ("la plataforma") para encontrar
    información relevante que responda la pregunta del usuario.

    Params:
        question: Pregunta a buscar.
        top_k: Número máximo de documentos a recuperar (por defecto 5).

    Returns:
        List[dict]: Lista de documentos relevantes como dicts (con metadata de fuente cuando esté disponible).
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

    # Convert Documents to dicts to make them JSON serializable
    return [doc.model_dump() for doc in retrived_documents]
