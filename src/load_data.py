import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


pdf_path = os.path.join(os.getcwd(), "data", "Notes.pdf")


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 600, chunk_overlap: int = 100):
    """Load PDF and split into chunks"""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} pages")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    
    # Clean empty chunks
    clean_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    print(f"✓ Created {len(clean_chunks)} chunks")
    
    return clean_chunks

