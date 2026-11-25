
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "fin_kb"

# Store the last few conversation exchanges
conversation_history = []


def create_rag_assistant():

    print("\nFINANCIAL RAG ASSISTANT")
    print("=" * 45)

    # Step 1: Load local embedding model
    print("\nLoading local embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Step 2: Connect to ChromaDB
    print("Connecting to ChromaDB...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # Step 3: Create MMR retriever
    print("Creating MMR retriever...")

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )

    # Step 4: Load local Llama model
    print("Loading Llama model...")

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0.1
    )

    print("\nRAG Assistant is ready!")

    return retriever, llm


def rewrite_question(question, llm):

    if not conversation_history:
        return question

    history = ""

    for exchange in conversation_history[-5:]:
        history += (
            f"User: {exchange['question']}\n"
            f"Assistant: {exchange['answer']}\n"
        )

    rewrite_prompt = ChatPromptTemplate.from_template("""
You rewrite follow-up questions into standalone questions.

Use the conversation history to understand references
such as "it", "that", "previous year", and "what about".

Rules:
1. Preserve the user's original meaning.
2. Do not answer the question.
3. Do not add information that is not in the conversation.
4. If the question is already standalone, keep its meaning.
5. Return only the rewritten question.

Conversation history:
{history}

New question:
{question}

Standalone question:
""")

    response = llm.invoke(
        rewrite_prompt.invoke({
            "history": history,
            "question": question
        })
    )

    rewritten = response.content.strip()

    print("\nUnderstanding follow-up question...")
    print("Search question:", rewritten)

    return rewritten


def ask_question(question, retriever, llm):

    if not question.strip():
        print("Please enter a question.")
        return

    print("\nSearching financial documents...")

    # Step 1: Rewrite follow-up question
    search_question = rewrite_question(question, llm)

    # Step 2: Retrieve relevant documents
    documents = retriever.invoke(search_question)

    if not documents:
        answer = "I_DONT_KNOW"

        print("\nASSISTANT ANSWER")
        print("=" * 45)
        print(answer)

        conversation_history.append({
            "question": question,
            "answer": answer
        })
        return

    # Step 3: Prepare document context
    context_parts = []

    for doc in documents:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"Source: {source}, Page: {page}\n"
            f"Content: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Step 4: Generate grounded answer
    prompt = ChatPromptTemplate.from_template("""
You are a Financial Document RAG Assistant.

Answer ONLY using the provided document context.

Rules:
1. Never invent financial figures.
2. Never use outside knowledge.
3. Cite every factual answer.
4. Use citations in this format:
   [Document Name, Page X]
5. If the answer is not supported by the context,
   respond exactly: I_DONT_KNOW
6. Do not provide investment advice.
7. Keep answers clear and concise.

Context:
{context}

Question:
{question}

Answer:
""")

    response = llm.invoke(
        prompt.invoke({
            "context": context,
            "question": search_question
        })
    )

    answer = response.content.strip()

    # Step 5: Display answer
    print("\nASSISTANT ANSWER")
    print("=" * 45)
    print(answer)

    # Step 6: Display unique retrieved sources
    print("\nRETRIEVED SOURCES")
    print("=" * 45)

    unique_sources = set()

    for doc in documents:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        source_key = (source, page)

        if source_key not in unique_sources:
            unique_sources.add(source_key)
            print(f"- {source} | Page {page}")

    # Step 7: Save conversation history
    conversation_history.append({
        "question": question,
        "answer": answer
    })

    # Keep only the latest 5 exchanges
    if len(conversation_history) > 5:
        conversation_history.pop(0)


if __name__ == "__main__":

    try:

        retriever, llm = create_rag_assistant()

        print("\nAsk questions about your financial PDFs.")
        print("Type 'exit' to stop.")

        while True:

            question = input("\nYour Question: ")

            if question.lower().strip() == "exit":
                print("Assistant closed.")
                break

            ask_question(
                question,
                retriever,
                llm
            )

    except Exception as error:
        print("\nERROR:", error)
