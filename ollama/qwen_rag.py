from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Init multilingual embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}  # change to "cuda" if GPU
)

# 2. Create some Chinese-English documents
docs = [
    Document(page_content="The capital of France is Paris."),
    Document(page_content="訂單狀態請至[首頁/商戶系統/訂單]查詢。"),
    Document(page_content="Python 是一種高級程式語言。"),
    Document(page_content="LangChain is useful for building LLM apps."),
    Document(page_content="You can return products within 30 days of purchase with a receipt for a full refund."),
    Document(page_content="Standard shipping takes 5-7 business days, while express shipping takes 2-3 business days."),
    Document(page_content="To reset your device, hold the power button for 10 seconds until it restarts."),
    Document(page_content="Our customer service is available 24/7 via chat or phone."),
    Document(page_content="Opened items can be refunded within 15 days if they are in good condition."),
]

# 3. Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# 4. Build FAISS retriever
vector_db = FAISS.from_documents(split_docs, embedding_model)
retriever = vector_db.as_retriever()

# 5. Connect to local Ollama model (qwen or llama3)
llm = OllamaLLM(model="qwen:7b-chat")  # you must have pulled it already

# 6. Build RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Ask questions
while True:
    query = input("🔍 問題 / Question: ")
    if query.lower() in {"exit", "quit"}:
        break
    result = qa.invoke(query)
    print("\n📣 回答 / Answer:\n", result['result'])
    print("\n📚 來源 / Sources:\n", [doc.page_content for doc in result['source_documents']])
    print("\n------------------------------------------------------------------------------------\n")
