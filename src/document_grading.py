from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from typing import List
import re
import time

def create_grading_chain(llm: ChatGoogleGenerativeAI):
    """Create a chain for grading document relevance"""
    grading_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a grader assessing relevance of a retrieved document to the query. "
            "Score from 1 (not relevant) to 10 (highly relevant). "
            "Respond ONLY with the numeric score."
        ),
        HumanMessagePromptTemplate.from_template(
            "Query: {query}\n"
            "Document: {document}\n"
            "Score:"
        )
    ])
    
    return grading_prompt | llm


def grade_documents(
    grading_chain,
    docs: List[Document],
    query: str,
    threshold: float = 7.0
) -> List[Document]:
    """Grade retrieved documents and keep only relevant ones"""
    filtered_docs = []
    
    for doc in docs:
        response = grading_chain.invoke({
            "query": query,
            "document": doc.page_content
        })
        
        score_match = re.search(r"\d+(\.\d+)?", response.content.strip())
        score = float(score_match.group()) if score_match else 0.0
        
        if score >= threshold:
            filtered_docs.append(doc)
        print("⏳ Waiting 15 seconds to avoid Google's rate limits...")
        time.sleep(15)  # Sleep to respect rate limits
    
    return filtered_docs

