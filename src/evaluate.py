
from pathlib import Path
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = BASE_DIR / "output" / "evaluation_questions.csv"
RESULTS_FILE = BASE_DIR / "output" / "evaluation_results.csv"
CHROMA_DIR = BASE_DIR / "chroma_db"

print("Loading evaluation dataset...")

questions_df = pd.read_csv(QUESTIONS_FILE)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = Chroma(
    collection_name="fin_kb",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15}
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_template("""
You are a Financial Document RAG Assistant.

Answer only using the document context.

Rules:
- Never invent financial figures.
- Never use outside knowledge.
- Cite factual answers as [Document Name, Page X].
- If the answer is unsupported, respond exactly I_DONT_KNOW.
- Do not provide investment advice.

Context:
{context}

Question:
{question}

Answer:
""")

results = []

print(f"Starting evaluation of {len(questions_df)} questions...")

for index, row in questions_df.iterrows():

    question = str(row["question"])
    expected = str(row["expected_result"])
    category = str(row["category"])

    print(f"\nTesting {index + 1}/{len(questions_df)}: {question}")

    try:
        documents = retriever.invoke(question)

        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in documents
        )

        response = llm.invoke(
            prompt.invoke({
                "context": context,
                "question": question
            })
        )

        answer = response.content.strip()

        if expected == "I_DONT_KNOW":
            passed = answer.strip().upper() == "I_DONT_KNOW"
        else:
            expected_parts = [
                part.strip().lower()
                for part in expected.split(" and ")
            ]
            answer_lower = answer.lower()
            passed = all(
                part in answer_lower for part in expected_parts
            )

        sources = "; ".join(
            f"{Path(str(doc.metadata.get('source', 'Unknown'))).name}"
            f", Page {doc.metadata.get('page', 'Unknown')}"
            for doc in documents
        )

        status = "PASS" if passed else "FAIL"

        print("Result:", status)
        print("Answer:", answer)

    except Exception as error:
        answer = f"ERROR: {error}"
        sources = ""
        status = "ERROR"
        passed = False

        print(answer)

    results.append({
        "id": row["id"],
        "category": category,
        "question": question,
        "expected_result": expected,
        "actual_answer": answer,
        "retrieved_sources": sources,
        "status": status
    })

    # Save progress after every question
    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)


results_df = pd.DataFrame(results)

print("\n========== EVALUATION SUMMARY ==========")

total = len(results_df)
passed_count = (results_df["status"] == "PASS").sum()
failed_count = (results_df["status"] == "FAIL").sum()
error_count = (results_df["status"] == "ERROR").sum()

print("Total questions:", total)
print("Passed:", passed_count)
print("Failed:", failed_count)
print("Errors:", error_count)

accuracy = (passed_count / total) * 100 if total else 0

print(f"Exact expected-result match rate: {accuracy:.2f}%")
print("Results saved to:", RESULTS_FILE)
