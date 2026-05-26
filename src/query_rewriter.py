from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


def rewrite_query(question):
    prompt = f"""
    Rewrite this query to improve retrieval:

    Query: {question}
    """

    response = llm.invoke(prompt)

    return response.content