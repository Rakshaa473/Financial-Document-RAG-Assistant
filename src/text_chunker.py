
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_all_documents


def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for document in documents:

        text = document["content"]
        metadata = document["metadata"]

        chunks = splitter.split_text(text)

        for index, chunk in enumerate(chunks):

            chunk_data = {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": index
                }
            }

            all_chunks.append(chunk_data)

    return all_chunks


if __name__ == "__main__":

    print("\nFINANCIAL DOCUMENT TEXT CHUNKER")
    print("=" * 45)

    documents = load_all_documents()

    chunks = create_chunks(documents)

    print("\nTOTAL PAGES:", len(documents))
    print("TOTAL CHUNKS:", len(chunks))

    for index, chunk in enumerate(chunks[:5]):

        print(f"\nCHUNK {index + 1}")
        print("Source:", chunk["metadata"]["source"])
        print("Page:", chunk["metadata"]["page"])
        print("Section:", chunk["metadata"]["section"])
        print("Chunk ID:", chunk["metadata"]["chunk_id"])
        print("Text:", chunk["content"][:250])
        print("-" * 45)
