import re
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


grading_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a grader assessing relevance of a retrieved document to the query. "
        "Score from 1 (not relevant) to 10 (highly relevant). Respond ONLY with the score."
    ),
    HumanMessagePromptTemplate.from_template(
        "Query: {query}\nDocument: {document}\nScore:"
    ),
])


def grade_documents(llm, docs: List[Document], query: str, threshold: float):
    filtered = []

    for doc in docs:
        response = llm.invoke(grading_prompt.format(query=query, document=doc.page_content))
        match = re.search(r"\d+", response.content.strip())
        score = float(match.group()) if match else 0.0

        if score >= threshold:
            filtered.append(doc)

    return filtered
