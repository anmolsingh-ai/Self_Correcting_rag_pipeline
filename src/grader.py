from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


def grade_documents(question, docs):
    graded_docs = []

    for doc in docs:
        prompt = f"""
        Question: {question}

        Document:
        {doc.page_content}

        Is this document relevant?
        Answer only YES or NO.
        """

        response = llm.invoke(prompt)

        if "YES" in response.content.upper():
            graded_docs.append(doc)

    return graded_docs