from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Init multilingual embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)

# 2. Your documents
print('generate rag')
loader = DirectoryLoader(
    "docs",  # or "./docs_backup"
    glob="**/*.md",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
    show_progress=True
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
print('generate rag done')

# 5. Connect to local Ollama model
print('connect llm')
llm = OllamaLLM(
    model="llama3",
    base_url="http://localhost:11435"
)
# llm = OllamaLLM(model="gpt-oss")

# 6. MCP-style Prompt Template
mcp_template = """
你是一個客服助理，根據以下知識庫內容回答使用者問題。
如果知識庫中沒有足夠資訊，請說「根據目前資訊無法回答」。
若我是用繁體中文輸入問題，請用繁體中文回覆。

【知識庫】
{context}

【問題】
{question}

【回答】
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
print('Connect LLM finished')

# 8. Ask questions
while True:
    query = input("🔍 問題 / Question: ")
    if query.lower() in {"exit", "quit"}:
        break
    result = qa.invoke({"query": query})
    print("\n📣 回答 / Answer:\n", result['result'])
    print("\n📚 來源 / Sources:\n", [doc.page_content for doc in result['source_documents']])
    print("\n------------------------------------------------------------------------------------\n")
