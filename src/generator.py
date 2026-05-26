from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv() # This loads the keys from your .env file into the environment
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


def generate_answer(question, docs):
    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    prompt = f"""
    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return response.content