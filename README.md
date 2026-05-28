# Self-Correcting RAG Pipeline

A production-ready Retrieval Augmented Generation (RAG) system with intelligent document grading, hybrid retrieval (vector + BM25), and conversational chat history. **Fully compatible with LangChain 1.3.2**.

## 🎯 Features

- ✅ **PDF Document Loading** - Extract and chunk documents intelligently
- ✅ **Vector + BM25 Hybrid Retrieval** - Combines semantic search with keyword matching
- ✅ **Intelligent Document Grading** - Filters low-quality results using LLM grading
- ✅ **Automatic Fallback** - Falls back to BM25 if vector search is low-quality
- ✅ **Conversational RAG** - Maintains chat history and contextualizes follow-up questions
- ✅ **RAGAS Evaluation** - Measure system quality (faithfulness, relevancy, precision, recall)
- ✅ **Latency Testing** - Monitor response times
- ✅ **LangChain 1.3.2 Compatible** - Works with the latest version

## 📋 Architecture

```
PDF Document
    ↓
[Chunking & Splitting]
    ↓
┌─────────────────────────────┐
│   Vector Store (Chroma)     │
│  + BM25 Index (KeywordBased)│
└─────────────────────────────┘
    ↓
[Hybrid Retriever]
    ├─→ Vector Search (k=8)
    ├─→ Grade Results (threshold=7.5)
    ├─→ BM25 Fallback (if low quality)
    ├─→ Combine & Deduplicate (k=6)
    ↓
[Contextualized Query]
    ↓ (with chat history)
[LLM Response]
    ↓
[User]
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Setup Environment

Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

Get your keys from:
- [Google AI Studio](https://aistudio.google.com/app/apikey) - Gemini API
- [Google Cloud Console](https://console.cloud.google.com/) - Google API

### 3. Prepare Your PDF

Place your PDF in:
```
your_project/
├── data/
│   └── Notes.pdf
├── chroma_db/  (auto-created)
└── ...
```

### 4. Run the Pipeline

```python
from self_correcting_rag_pipeline import main

# Initialize pipeline
rag_chain, message_store, vector_store = main("data/Notes.pdf")

# Ask a question
response = rag_chain.invoke(
    {"input": "What is the main topic?"},
    config={"configurable": {"session_id": "user_1"}}
)

print(response.content)
```

## 📚 Detailed Usage

### Single Query

```python
response = rag_chain.invoke(
    {"input": "Your question here"},
    config={"configurable": {"session_id": "user_1"}}
)

answer = response.content if hasattr(response, 'content') else str(response)
print(answer)
```

### Conversational Chat (With History)

```python
session_id = "chat_1"

# First turn
response1 = rag_chain.invoke(
    {"input": "Summarize the document"},
    config={"configurable": {"session_id": session_id}}
)
print("Assistant:", response1.content)

# Follow-up (automatically includes history)
response2 = rag_chain.invoke(
    {"input": "What else should I know?"},
    config={"configurable": {"session_id": session_id}}
)
print("Assistant:", response2.content)

# Different user/conversation
response3 = rag_chain.invoke(
    {"input": "New topic question"},
    config={"configurable": {"session_id": "user_2"}}  # Different session
)
```

### Test Latency

```python
from self_correcting_rag_pipeline import test_latency

avg_latency = test_latency(rag_chain, num_tests=5)
# Output: Average latency: 2.34s
```

### Evaluate with RAGAS

```python
from self_correcting_rag_pipeline import evaluate_rag

test_data = [
    {
        "question": "What is the main topic?",
        "ground_truth": "The document discusses..."
    },
    # ... more test cases
]

scores, results = evaluate_rag(rag_chain, test_data)

# Metrics:
# - faithfulness: How factual is the answer (0-1)
# - answer_relevancy: Is the answer relevant to the question (0-1)
# - context_precision: Are retrieved docs relevant (0-1)
# - context_recall: Does context contain all needed info (0-1)
```

## 🔧 Configuration

### Adjust Retriever Sensitivity

```python
retriever = create_hybrid_retriever(
    chunks,
    vector_store,
    llm,
    grading_chain,
    threshold=7.5,  # Document relevance threshold (1-10)
    k=6             # Number of final documents to return
)
```

**Threshold** (1-10):
- `5.0`: Very permissive (returns more docs, some low-quality)
- `7.5`: Balanced (recommended)
- `9.0`: Strict (returns only highly relevant docs)

### Adjust Chunking

```python
chunks = load_and_chunk_pdf(
    pdf_path,
    chunk_size=600,      # Size of each chunk (characters)
    chunk_overlap=100    # Overlap between chunks
)
```

**Chunk Size**:
- `300`: Small chunks, more specific retrieval
- `600`: Balanced (recommended)
- `1000`: Large chunks, more context

### Change Embedding Model

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Better but slower
)
```

Alternative models:
- `all-MiniLM-L6-v2` - Fast, good for most cases (default)
- `all-mpnet-base-v2` - Better quality, slower
- `multilingual-e5-large` - Multilingual support

## 🧠 How It Works

### 1. Document Grading

The system grades each retrieved document on relevance:
```
Query: "What is machine learning?"
Document: "ML is a subset of AI..."
Score: 8/10 ✓ (Above threshold, kept)

Document: "The weather today is sunny..."
Score: 2/10 ✗ (Below threshold, removed)
```

### 2. Hybrid Retrieval Flow

```
User Query: "How does RAG work?"

Step 1: Vector Search (Semantic)
  └─→ Found 8 documents

Step 2: Grade Results
  └─→ Only 2 scored above 7.5

Step 3: Not Enough Quality Results
  └─→ Activate BM25 Fallback

Step 4: BM25 Search (Keyword)
  └─→ Found 10 documents

Step 5: Grade BM25 Results
  └─→ 5 scored above 7.5

Step 6: Combine & Deduplicate
  └─→ Final 6 documents (2+5-1 duplicate)

Step 7: Generate Answer
  └─→ LLM uses all 6 for context
```

### 3. Conversational Context

When you ask a follow-up question, the system:
1. Takes your new question
2. Combines it with chat history
3. Reformulates into a standalone question
4. Retrieves relevant documents
5. Generates answer using context + history

Example:
```
Chat History:
- User: "What is RAG?"
- Assistant: "RAG is..."

New Question: "How does it compare to fine-tuning?"
Reformulated: "How does RAG compare to fine-tuning?"
  └─→ Much more specific retrieval!
```

## 📊 Expected Performance

### Quality Metrics (RAGAS)
- **Faithfulness**: 0.75-0.85 (how factual answers are)
- **Answer Relevancy**: 0.70-0.80 (how relevant to question)
- **Context Precision**: 0.60-0.75 (are retrieved docs useful)
- **Context Recall**: 0.65-0.80 (does context have all needed info)

### Speed
- **Initial retrieval**: 0.5-1.5 seconds
- **LLM generation**: 2-5 seconds
- **Total latency**: 2.5-6.5 seconds

Factors affecting speed:
- PDF size and document count
- LLM model (gemini-1.5-flash vs gemini-2.0-pro)
- Network latency

## 🐛 Troubleshooting

### Error: "cannot import name 'chains' from 'langchain'"
✅ **FIXED** - This code uses LangChain 1.3.2 compatible imports

### Error: "GEMINI_API_KEY not set"
```python
# Check your .env file has:
# GEMINI_API_KEY=sk_...

# Or set in code:
import os
os.environ["GEMINI_API_KEY"] = "your_key_here"
```

### Slow responses
- Use faster LLM: `ChatGoogleGenerativeAI(model="gemini-1.5-flash")`
- Reduce chunk size (faster retrieval)
- Reduce vector retrieval k value

### Low relevance scores
- Increase chunk size (more context per chunk)
- Lower threshold value in retriever
- Check that your PDF is relevant to questions

### RAGAS evaluation fails
```bash
# Make sure you have both API keys
echo $GEMINI_API_KEY
echo $GOOGLE_API_KEY

# And all dependencies:
pip install ragas datasets
```

## 🔐 Security & Best Practices

1. **Never commit API keys** - Use .env files
2. **Use limited-scope API keys** if possible
3. **Monitor token usage** - LLM calls can be expensive
4. **Cache embeddings** - Reuse vector store for multiple queries
5. **Rate limiting** - Implement throttling for production

## 📈 Optimization Tips

### Improve Answer Quality
```python
# 1. Better embeddings (slower but higher quality)
"sentence-transformers/all-mpnet-base-v2"

# 2. Larger context window (bigger k)
retriever = create_hybrid_retriever(..., k=10)

# 3. Lower grading threshold (more lenient)
threshold=6.0
```

### Improve Speed
```python
# 1. Faster embeddings (lower quality but quick)
"sentence-transformers/all-MiniLM-L6-v2"

# 2. Smaller context window (fewer docs)
retriever = create_hybrid_retriever(..., k=3)

# 3. Faster LLM
ChatGoogleGenerativeAI(model="gemini-1.5-flash")
```

### Reduce Costs
```python
# 1. Use batch processing
responses = [rag_chain.invoke(...) for q in questions]

# 2. Cache vector search results
# 3. Use smaller LLM when possible
# 4. Monitor and limit evaluations
```

## 📖 Understanding the Code

### Main Components

**`load_and_chunk_pdf()`**
- Loads PDF using PyMuPDFLoader
- Splits into overlapping chunks
- Cleans empty chunks

**`create_vector_store()`**
- Uses HuggingFace embeddings
- Stores in Chroma vector DB
- Persists to disk

**`create_grading_chain()`**
- LLM-based document relevance grader
- Scores 1-10 based on query match
- Filters low-quality results

**`GradedFallbackRetriever`**
- Custom retriever class
- Tries vector search first
- Falls back to BM25 if needed
- Deduplicates results

**`create_conversational_rag_chain()`**
- Chains together all components
- Handles chat history
- Contextualizes queries
- Generates answers

## 🎓 Learning Resources

- [LangChain Docs](https://python.langchain.com/)
- [RAG Best Practices](https://github.com/langchain-ai/langchain)
- [RAGAS Evaluation](https://github.com/explodinggradients/ragas)
- [Chroma Vector DB](https://docs.trychroma.com/)

## 📝 License

MIT License - Feel free to use and modify

## 🤝 Contributing

Improvements welcome! Some ideas:
- Support for more document types (Word, PowerPoint)
- Multi-document retrieval
- Custom grading prompts
- Streaming responses
- Web UI interface

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review LangChain 1.3.2 documentation
3. Check API key configuration

---

**Happy RAGging! 🚀**
