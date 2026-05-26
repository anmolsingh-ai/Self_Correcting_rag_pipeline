from langchain_community.vectorstores import FAISS

def load_existing_vectorstore(embeddings):
    vectorstore = FAISS.load_local(
        folder_path="faiss_index", 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True 
    )
    return vectorstore