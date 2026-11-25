
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from document_loader import load_all_documents
from text_chunker import create_chunks


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "fin_kb"

# Free local embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_vector_database():

    print("\nFINANCIAL RAG - VECTOR DATABASE")
    print("=" * 45)

    # Step 1: Load PDF documents
    print("\nStep 1: Loading financial PDFs...")

    documents = load_all_documents()

    if not documents:
        raise ValueError("No PDF documents found.")

    # Step 2: Create chunks
    print("\nStep 2: Creating text chunks...")

    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError("No text chunks were generated.")

    print("Total chunks:", len(chunks))

    # Step 3: Prepare LangChain documents
    print("\nStep 3: Preparing documents...")

    langchain_documents = []
    ids = []

    for chunk in chunks:

        langchain_documents.append(
            Document(
                page_content=chunk["content"],
                metadata=chunk["metadata"]
            )
        )

        metadata = chunk["metadata"]

        ids.append(
            f"{metadata['source']}"
            f"_page_{metadata['page']}"
            f"_chunk_{metadata['chunk_id']}"
        )

    # Step 4: Initialize local embeddings
    print("\nStep 4: Loading local embedding model...")
    print("Model:", MODEL_NAME)

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Local embedding model loaded successfully!")

    # Step 5: Store vectors in ChromaDB
    print("\nStep 5: Creating ChromaDB collection...")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=langchain_documents,
        embedding=embeddings,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR)
    )

    stored_count = vector_store._collection.count()

    print("\nVector database created successfully!")
    print("Collection:", COLLECTION_NAME)
    print("Stored vectors:", stored_count)
    print("Database location:", CHROMA_DIR)

    if stored_count == 0:
        raise ValueError("No vectors were stored in ChromaDB.")

    # Step 6: Test similarity search
    print("\nStep 6: Testing similarity search...")

    query = "What was the Q3 revenue?"

    results = vector_store.similarity_search(
        query,
        k=3
    )

    print("\nSEARCH QUERY:", query)

    for index, result in enumerate(results, start=1):

        print(f"\nRESULT {index}")
        print("Source:", result.metadata["source"])
        print("Page:", result.metadata["page"])
        print("Content:", result.page_content[:250])

    print("\nVECTOR DATABASE TEST COMPLETED!")

    return vector_store


if __name__ == "__main__":

    try:
        create_vector_database()

    except Exception as error:
        print("\nERROR:", error)
