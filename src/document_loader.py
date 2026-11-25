
import pdfplumber
from pathlib import Path
import re

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def detect_doc_type(filename):
    """Identify the type of financial document."""

    name = filename.lower()

    if "earnings" in name:
        return "Earnings Transcript"
    elif "annual_report" in name:
        return "Annual Report"
    elif "10k" in name:
        return "10-K Filing"
    elif "financial_statements" in name:
        return "Financial Statements"

    return "General Document"


def extract_section(text, current_section):
    """Detect simple section headings."""

    lines = text.splitlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Identify short heading-like lines
        if len(line) < 70 and not line.endswith("."):
            if re.search(r"[A-Za-z]", line):
                return line

    return current_section


def load_pdf(file_path):
    """Extract text and metadata from each PDF page."""

    documents = []

    current_section = "General"

    try:
        with pdfplumber.open(file_path) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                text = page.extract_text()

                if not text or not text.strip():
                    print(
                        f"Warning: No text on page {page_number} "
                        f"of {file_path.name}"
                    )
                    continue

                current_section = extract_section(
                    text, current_section
                )

                document = {
                    "content": text,
                    "metadata": {
                        "source": file_path.name,
                        "page": page_number,
                        "section": current_section,
                        "doc_type": detect_doc_type(file_path.name)
                    }
                }

                documents.append(document)

        print(
            f"Successfully loaded {file_path.name}: "
            f"{len(documents)} pages"
        )

    except Exception as error:
        print(f"Error loading {file_path.name}: {error}")

    return documents


def load_all_documents():
    """Load every PDF inside the data folder."""

    all_documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data folder.")
        return []

    for file_path in pdf_files:
        documents = load_pdf(file_path)
        all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":

    print("\nFINANCIAL DOCUMENT LOADER")
    print("=" * 45)

    documents = load_all_documents()

    print("\nTOTAL DOCUMENTS LOADED:", len(documents))

    print("\nEXTRACTED DOCUMENT DETAILS")
    print("=" * 45)

    for document in documents:

        print("\nSource:", document["metadata"]["source"])
        print("Page:", document["metadata"]["page"])
        print("Section:", document["metadata"]["section"])
        print("Document Type:", document["metadata"]["doc_type"])

        print("\nExtracted Text:")
        print(document["content"][:300])

        print("-" * 45)
