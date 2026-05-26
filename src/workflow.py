from ingestion_pipeline import load_and_split_data
from embedding_pipeline import get_embedding_model
from vectorstore_manager import create_vectorstore,load_existing_vectorstore
from retriever import get_retriever
from grader import grade_documents
from query_rewriter import rewrite_query
from generator import generate_answer
import os
from reflection_engine import reflect_answer
from dotenv import load_dotenv
load_dotenv() 
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Notes.pdf"))



# STEP 1: Load Data
chunks = load_and_split_data(DATA_PATH)


# STEP 2: Embeddings
embeddings = get_embedding_model()


# STEP 3: Vector DB
create_vectorstore(chunks, embeddings)


# STEP 4: Retriever
retriever = get_retriever(embeddings)

# STEP 5: User Question
question = "What is self correcting RAG?"


# STEP 6: Retrieve Documents
docs = retriever.invoke(question)


# STEP 7: Grade Documents
graded_docs = grade_documents(question, docs)

# STEP 8: Rewrite Query if Needed
if len(graded_docs) == 0:
    print("Rewriting Query...")

    better_question = rewrite_query(question)

    docs = retriever.invoke(better_question)

    graded_docs = grade_documents(
        better_question,
        docs
    )

    question = better_question


# STEP 9: Generate Answer
answer = generate_answer(question, graded_docs)

print("\nGenerated Answer:\n")
print(answer)

# STEP 10: Reflection
reflection = reflect_answer(
    question,
    answer,
    graded_docs
)

print("\nReflection Result:\n")
print(reflection)


# STEP 11: Retry if Hallucinated
if "NO" in reflection.upper():
    print("\nAnswer failed validation. Regenerating...\n")

    answer = generate_answer(question, graded_docs)

    print(answer)