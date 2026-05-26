import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_and_split_data(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks

if __name__ == "__main__":
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Notes.pdf"))
    try:
        my_chunks = load_and_split_data(path)
        print(f"Successfully split the document into {len(my_chunks)} chunks.")
    except Exception as e:
        print(f"Error: {e}")