"""
Conversational RAG chain
------------------------
Flow for every user turn:

    user question + chat history
        -> (1) REWRITER   : turn a follow-up into a standalone question
        -> (2) RETRIEVER  : fetch chunks using the standalone question
        -> (3) GENERATOR  : answer from the chunks (no chat history)
        -> answer, saved to the session history

The generator is built by `build_generator()` so the production chain and the
component-level eval (`generate(query, context)`) use the exact same prompt.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# How many recent messages (human + AI) the rewriter is allowed to see.
MAX_HISTORY_MESSAGES = 6


# --------------------------------------------------------------------------
# Formatting helpers (shared by production and eval)
# --------------------------------------------------------------------------
def format_chunks(chunks: list[str]) -> str:
    """Join plain text chunks into one context string."""
    return "\n\n".join(chunks)


def format_docs(docs) -> str:
    """Join retrieved LangChain Documents into one context string."""
    return format_chunks([doc.page_content for doc in docs])


# --------------------------------------------------------------------------
# Component 1: question rewriter
# --------------------------------------------------------------------------
def build_question_rewriter(llm):
    """Rewrites a follow-up question into a standalone question."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and latest user question, rewrite "
            "the question so it can be understood independently. "
            "Preserve the user's original intent, constraints, and negations. "
            "Do NOT answer the question. Return only the rewritten question. "
            "If it already stands on its own, return it unchanged.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


# --------------------------------------------------------------------------
# Component 3: generator (this is what you evaluate with golden context)
# --------------------------------------------------------------------------
def build_generator(llm):
    """
    Input : {"standalone_question": str, "context": str}
    Output: answer string
    No chat history here, so it can be tested in isolation.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant.\n"
            "Answer the question using the retrieved context. "
            "If the context does not contain the answer, "
            "say no instead of guessing.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{standalone_question}"),
    ])
    return prompt | llm | StrOutputParser()


# --------------------------------------------------------------------------
# Full chain
# --------------------------------------------------------------------------
def create_conversational_rag_chain(retriever, llm):
    """
    Returns (conversational_rag_chain, store).

    store is the dict of session_id -> ChatMessageHistory, handy for
    inspecting or clearing conversations.
    """
    rewriter = build_question_rewriter(llm)
    generator = build_generator(llm)

    # Step 1: rewrite using only the last MAX_HISTORY_MESSAGES messages.
    # First turn (empty history) skips the LLM call entirely.
    def contextualize_input(x: dict) -> dict:
        question = x["input"]
        history = x.get("chat_history", [])

        if history:
            rewritten = rewriter.invoke({
                "input": question,
                "chat_history": history[-MAX_HISTORY_MESSAGES:],
            }).strip()

            if rewritten:  # fall back to the original if the LLM returns nothing
                question = rewritten

        return {**x, "standalone_question": question}

    # Step 2: retrieve with the standalone question and attach it as "context"
    retrieve_context = (
        RunnableLambda(lambda x: x["standalone_question"])
        | retriever
        | format_docs
    )

    rag_chain = (
        RunnableLambda(contextualize_input)
        | RunnablePassthrough.assign(context=retrieve_context)
        | generator
    )

    # Step 3: per-session memory. The ORIGINAL user input and the final answer
    # are what get stored, not the rewritten question.
    store: dict[str, BaseChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("Conversational RAG chain created")
    return conversational_rag_chain, store


