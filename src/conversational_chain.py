from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate


def create_conversational_rag_chain(retriever, llm):
    """
    Create a conversational RAG chain manually.

    Flow:
      1. If chat history exists, rewrite the user's question as a standalone question.
      2. Use that standalone question to retrieve documents.
      3. Answer the user's ORIGINAL question using the retrieved context + chat history.
      4. Save the exchange to per-session message history.

    Returns:
        (conversational_rag_chain, store)
        - invoke with: chain.invoke({"input": "..."}, config={"configurable": {"session_id": "abc"}})
        - the result is an AIMessage; use `response.content` to get the text.
    """

    # ------------------------------------------------------------------
    # Step 1: Prompt that rewrites a follow-up question into a standalone one
    # ------------------------------------------------------------------
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rewrite the "
         "question so it can be understood without the chat history. "
         "Do NOT answer the question. Return only the rewritten question. "
         "If it already stands on its own, return it unchanged."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    contextualize_q_chain = contextualize_q_prompt | llm

    def get_standalone_question(x: dict) -> str:
        """Rewrite the question using chat history, only if history exists."""
        if x.get("chat_history"):
            return contextualize_q_chain.invoke(x).content
        return x["input"]

    # ------------------------------------------------------------------
    # Step 2: QA prompt
    # ------------------------------------------------------------------
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful AI assistant.\n"
         "Use the retrieved context to answer the question. "
         "If the context does not contain the answer, say so instead of guessing.\n\n"
         "Context:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    def format_docs(docs) -> str:
        """Format retrieved documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    # ------------------------------------------------------------------
    # Step 3: Build the RAG chain
    # ------------------------------------------------------------------
    rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(get_standalone_question) | retriever | format_docs
        )
        | qa_prompt
        | llm
    )

    # ------------------------------------------------------------------
    # Step 4: Add per-session message history
    # ------------------------------------------------------------------
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # NOTE: no `output_messages_key` here. The chain ends with `llm`, which
    # returns an AIMessage (not a dict), so there is no key to look up.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("✓ Conversational RAG chain created")
    return conversational_rag_chain, store