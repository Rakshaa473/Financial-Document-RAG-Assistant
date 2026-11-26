
from pathlib import Path
from datetime import datetime
import csv

import streamlit as st
import os
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
import pandas as pd
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "audit_log.csv"

COLLECTION_NAME = "fin_kb"


# ---------- PAGE CONFIGURATION ----------

st.set_page_config(
    page_title="Financial Document RAG Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Document RAG Assistant")
st.caption("Ask questions about your financial documents using AI.")


# ---------- LOGGING ----------

def save_log(question, answer, sources):

    LOG_DIR.mkdir(exist_ok=True)

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "question",
                "answer",
                "sources"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            question,
            answer,
            "; ".join(sources)
        ])


# ---------- LOAD RAG COMPONENTS ----------


@st.cache_resource
def load_rag():

    # 1. Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 2. Connect to ChromaDB
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # 3. Automatically index PDFs if database is empty
    if vector_store._collection.count() == 0:

        st.info("Building the financial document knowledge base...")

        pdf_files = list(DATA_DIR.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError("No PDF files found in data folder.")

        all_documents = []

        for pdf in pdf_files:
            with pdfplumber.open(pdf) as document:

                for page_number, page in enumerate(document.pages, start=1):
                    text = page.extract_text() or ""

                    if text.strip():
                        all_documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": pdf.name,
                                    "page": page_number,
                                    "doc_type": pdf.stem
                                }
                            )
                        )

        if not all_documents:
            raise ValueError("No readable text found in the PDFs.")

        # 4. Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(all_documents)

        # 5. Store chunks in ChromaDB
        vector_store.add_documents(chunks)

    # 6. Create MMR retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    # 7. Connect to Groq cloud LLM (not local Ollama)
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    return retriever, llm



# ---------- QUESTION REWRITING ----------

def rewrite_question(question, history, llm):

    if not history:
        return question

    recent_history = ""

    for item in history[-5:]:
        recent_history += (
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}\n"
        )

    prompt = ChatPromptTemplate.from_template("""
Rewrite the user's follow-up as a standalone question.

Use the conversation to resolve words such as "it",
"that", and "previous year".

Do not answer the question.
Do not add unsupported details.
Return only the standalone question.

Conversation:
{history}

New question:
{question}

Standalone question:
""")

    response = llm.invoke(
        prompt.invoke({
            "history": recent_history,
            "question": question
        })
    )

    return response.content.strip()


# ---------- ANSWER GENERATION ----------

def get_answer(question, retriever, llm, history):

    search_question = rewrite_question(
        question, history, llm
    )

    documents = retriever.invoke(search_question)

    if not documents:
        return "I_DONT_KNOW", []

    context_parts = []
    sources = []

    for doc in documents:

        source = Path(
            str(doc.metadata.get("source", "Unknown"))
        ).name

        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"Source: {source}, Page: {page}\n"
            f"Content: {doc.page_content}"
        )

        citation = f"{source}, Page {page}"

        if citation not in sources:
            sources.append(citation)

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template("""
You are a Financial Document RAG Assistant.

Answer only using the supplied document context.

Rules:
1. Never invent financial information.
2. Do not use outside knowledge.
3. Cite factual statements using [Document Name, Page X].
4. Use only document names and pages present in the context.
5. If the answer is not supported by the context,
   return exactly I_DONT_KNOW.
6. Do not provide investment advice.
7. Give a clear and concise answer.

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

    if not answer:
        answer = "I_DONT_KNOW"

    return answer, sources


# ---------- SIDEBAR ----------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    ["Chat Assistant", "Document Browser", "Analytics"]
)


# ---------- LOAD MODEL ----------

try:
    with st.spinner("Connecting to Groq AI and loading embeddings..."):
        retriever, llm = load_rag()

except Exception as error:
    st.error(f"Could not load the RAG system: {error}")
    st.info(
    "Check that GROQ_API_KEY is configured in Streamlit Secrets "
    "and that ChromaDB contains the fin_kb collection."
)
    st.stop()


# ---------- CHAT ASSISTANT ----------

if page == "Chat Assistant":

    st.subheader("💬 Ask your financial questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View retrieved sources"):
                    for source in message["sources"]:
                        st.write("📄", source)

    question = st.chat_input(
        "Ask about revenue, assets, profits..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching financial documents..."):

                try:
                    answer, sources = get_answer(
                        question,
                        retriever,
                        llm,
                        st.session_state.history
                    )

                    st.markdown(answer)

                    if sources:
                        with st.expander("View retrieved sources"):
                            for source in sources:
                                st.write("📄", source)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                    st.session_state.history.append({
                        "question": question,
                        "answer": answer
                    })

                    st.session_state.history = (
                        st.session_state.history[-5:]
                    )

                    save_log(question, answer, sources)

                except Exception as error:
                    st.error(f"Error generating answer: {error}")


# ---------- DOCUMENT BROWSER ----------

elif page == "Document Browser":

    st.subheader("📚 Financial Document Browser")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        st.warning(
            "No PDF documents found. Add your PDFs to the data folder."
        )

    else:

        st.write(f"**Total documents:** {len(pdf_files)}")

        for pdf in pdf_files:

            with st.expander(pdf.name):

                st.write(f"File size: {pdf.stat().st_size / 1024:.2f} KB")

                with open(pdf, "rb") as file:
                    st.download_button(
                        label="Download PDF",
                        data=file.read(),
                        file_name=pdf.name,
                        mime="application/pdf",
                        key=pdf.name
                    )


# ---------- ANALYTICS DASHBOARD ----------

elif page == "Analytics":

    st.subheader("📈 Usage Analytics")

    if not LOG_FILE.exists():

        st.info("No questions have been logged yet. Use the chat first.")

    else:

        df = pd.read_csv(LOG_FILE)

        total_questions = len(df)

        unknown_count = (
            df["answer"].astype(str).str.contains(
                "I_DONT_KNOW",
                regex=False
            ).sum()
        )

        answered_count = total_questions - unknown_count

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Questions", total_questions)
        col2.metric("Answered", answered_count)
        col3.metric("Unknown Answers", unknown_count)

        st.write("### Question History")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.write("### Answer Distribution")

        chart_data = pd.DataFrame({
            "Category": ["Answered", "I_DONT_KNOW"],
            "Count": [answered_count, unknown_count]
        })

        st.bar_chart(
            chart_data.set_index("Category")
        )

        st.download_button(
            "Download Audit Log CSV",
            data=df.to_csv(index=False),
            file_name="audit_log.csv",
            mime="text/csv"
        )
