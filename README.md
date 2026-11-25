
# Financial Document RAG Assistant

## Project Overview

A Retrieval-Augmented Generation (RAG) application that answers questions using financial PDF documents.

The system retrieves relevant document passages and generates answers using a locally running Large Language Model.

## Features

- PDF document loading and text extraction
- Text chunking using RecursiveCharacterTextSplitter
- Local Hugging Face embeddings
- ChromaDB vector storage
- Maximum Marginal Relevance (MMR) retrieval
- Llama 3.2 local LLM integration through Ollama
- Conversational follow-up question handling
- Source and page citations
- I_DONT_KNOW fallback for unsupported questions
- Streamlit chat interface
- Financial document browser
- Usage analytics dashboard
- CSV audit logging
- 50-question evaluation dataset

## Technology Stack

- Python 3.11
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Ollama
- Llama 3.2 (3B)
- Streamlit
- Pandas
- pdfplumber

Note: This implementation uses LangChain and ChromaDB rather than the LlamaIndex and FAISS tools mentioned in the task description.

## Project Structure

```text
Financial_RAG_Assistant/
│
├── app.py
├── data/
│   ├── q3_earnings_transcript.pdf
│   ├── annual_report_2025.pdf
│   ├── 10k_filing.pdf
│   └── financial_statements.pdf
│
├── src/
│   ├── create_sample_data.py
│   ├── document_loader.py
│   ├── text_chunker.py
│   ├── vector_store.py
│   ├── rag_assistant.py
│   └── evaluate.py
│
├── output/
│   ├── evaluation_questions.csv
│   └── evaluation_results.csv
│
├── logs/
│   └── audit_log.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Install Python 3.11.
2. Install Ollama.
3. Download the model:

```bash
ollama pull llama3.2:3b
```

4. Install project dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Build the vector database:

```bash
python src/vector_store.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open:

http://localhost:8501

## Evaluation

The project includes 50 evaluation questions:

- 30 in-scope questions
- 10 out-of-scope questions
- 10 adversarial questions

Initial strict expected-result matching:

- Total questions: 50
- Passed: 28
- Failed: 22
- Runtime errors: 0
- Initial match rate: 56%

This is an initial automated string-matching result, not a validated semantic accuracy score.

## Sample Dataset Disclaimer

The included NovaTech Industries financial PDFs are fictional sample documents created for project testing. Their figures are not real-world financial disclosures.

## Limitations

- Local LLM performance depends on available hardware.
- Citation formatting requires further validation.
- The initial evaluation identified unsupported-answer and adversarial-prompt weaknesses.
- Conversation history is maintained within the active application session.
- The application currently runs locally and is not publicly hosted.

## Future Improvements

- Integrate the specified LlamaIndex and FAISS stack.
- Improve citation validation.
- Improve adversarial prompt handling.
- Add persistent conversational memory.
- Expand semantic and citation evaluation.
- Deploy the application online.
