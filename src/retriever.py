from langchain_community.vectorstores import FAISS


from vectorstore_manager import load_existing_vectorstore


def get_retriever(embeddings):

    vectorstore = load_existing_vectorstore(embeddings)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever