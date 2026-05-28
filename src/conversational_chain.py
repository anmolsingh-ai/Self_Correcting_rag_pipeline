from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

def create_conversational_rag_chain(retriever, llm):
    """
    Create conversational RAG chain manually (compatible with LangChain 1.3.2)
    """
    
    # Step 1: Contextualize query based on chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and latest user question, formulate a standalone question."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    contextualize_q_chain = contextualize_q_prompt | llm
    
    # Step 2: History-aware retriever (manual implementation)
    def contextualize_input(x):
        """If chat history exists, contextualize the query"""
        if "chat_history" in x and x["chat_history"]:
            # Reformulate question based on history
            contextualized = contextualize_q_chain.invoke(x)
            return {"input": contextualized.content, "chat_history": x["chat_history"]}
        return x
    
    # Step 3: QA Chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful AI assistant.\n"
         "Use the retrieved context to answer the question.\n\n"
         "Context:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    # Step 4: Combine everything
    def format_docs(docs):
        """Format retrieved documents for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    # Build the chain
    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["input"]) | retriever | format_docs
        )
        | qa_prompt
        | llm
    )
    
    # Step 5: Add message history
    store = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="content"
    )
    
    print("✓ Conversational RAG chain created")
    return conversational_rag_chain, store
