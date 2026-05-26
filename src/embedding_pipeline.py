import os
 
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings