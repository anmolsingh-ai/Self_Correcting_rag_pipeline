from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv() # This loads the keys from your .env file into the environment
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")



def reflect_answer(question, answer, docs):
    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    prompt = f"""
    Question:
    {question}

    Context:
    {context}

    Answer:
    {answer}

    Is the answer supported by context?

    Answer only:
    YES or NO
    """

    response = llm.invoke(prompt)

    return response.content