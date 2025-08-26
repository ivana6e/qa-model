from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader

import time
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

class UTF8TextLoader(TextLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path, encoding="utf-8")

# 1. Init multilingual embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)

# 2. Your documents
print('Start Generating RAG')
loader = DirectoryLoader(
    "docs",  # or "./docs_backup"
    glob="**/*.md",
    loader_cls=UTF8TextLoader,
    show_progress=True,
)
docs = loader.load()

# 3. Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# 4. Create FAISS vector store
vector_db = FAISS.from_documents(split_docs, embedding_model)
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
print('Generate RAG Completed')

# 5. Connect to local Ollama model
print('Start Connecting to LLM')
try:
    llm = OllamaLLM(
        model="llama3",
        # base_url="http://localhost:11435"
    )
except Exception as e:
    raise Exception(f"Failed to connect to LLM: {str(e)}")

# 6. MCP-style Prompt Template
mcp_template = """
You are a customer service assistant answering users' questions based on the following knowledge base.
If the knowledge base doesn't have enough information, please state "Unable to answer based on current information.
Respond according to the user's language. If multiple languages are mixed, Chinese takes priority.

【Knowledge Base】
{context}

【Question】
{question}

【Answer】
""".strip()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=mcp_template,
)

# 7. Build RAG pipeline with MCP-style prompt
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
print('Connect LLM Completed')

# 8. Embed fastapi
# input
class QueryRequest(BaseModel):
    question: str

# APIRouter
router = APIRouter(prefix="/api/v1", tags=["query"])

# 9. Ask questions
@router.post("/query")
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        start_time = time.perf_counter()
        result = qa.invoke({"query": request.question})
        elapsed_time = time.perf_counter() - start_time

        return {
            "answer": result['result'],
            "sources": [doc.page_content for doc in result['source_documents']],
            "time taken in sec": round(elapsed_time, 2),
        }
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(ex)}")

# while True:
#     query = input("🔍 問題 / Question: ")
#     if query.lower() in {"exit", "quit"}:
#         break
#
#     start_time = time.perf_counter()  # Start timing
#     result = qa.invoke({"query": query})
#     elapsed_time = time.perf_counter() - start_time  # End timing
#
#     print("\n📣 回答 / Answer:\n", result['result'])
#     print("\n📚 來源 / Sources:\n", [doc.page_content for doc in result['source_documents']])
#     print(f"\n⏱️ 耗時 / Time taken: {elapsed_time:.2f} 秒")
#     print("\n------------------------------------------------------------------------------------\n")
