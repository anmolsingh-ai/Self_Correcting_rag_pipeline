import os

from src.load_data import load_and_chunk_pdf
from src.vectore_store import create_vector_store
from src.hybrid_retriver import create_hybrid_retriever
from src.reranker import create_reranked_retriever


pdf_path = os.path.join(os.getcwd(), "data", "Notes.pdf")
chunks = load_and_chunk_pdf(pdf_path)
if not chunks:
    raise ValueError("No document chunks were created from the PDF.")
print(f"✓ Created {len(chunks)} chunks")

import json

# Save
data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2, default=str)