from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document


def load_documents(path: str) -> List[Document]:
    loaders = [
        DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader),
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    print(f"✅ Loaded {len(documents)} documents")
    return documents
