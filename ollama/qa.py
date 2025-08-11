from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Init embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)

# 2. Load documents
loader = DirectoryLoader(
    "docs",
    glob="**/*.md",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
    show_progress=True
)
docs = loader.load()

# 3. Split docs_backup
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# 4. Create vector DB
vector_db = FAISS.from_documents(split_docs, embedding_model)
retriever = vector_db.as_retriever()

# 5. Load Ollama LLM
llm = OllamaLLM(model="llama3")

# 6. MCP Prompt template
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

# 7. Build QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# 8. 多語言關鍵字列表
unsatisfied_keywords = [
    "不對", "不滿意", "沒用", "找真人", "客服", "不會", "錯誤",
    "wrong", "unsatisfied", "useless", "human", "agent", "support", "error"
]

# 9. 載入SBERT模型（會花點時間）
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
unsatisfied_examples = [
    "不滿意", "沒用", "找真人", "這答案錯了",
    "this answer is wrong", "I want a human agent",
    "not helpful", "unsatisfied"
]
example_embeddings = sbert_model.encode(unsatisfied_examples, convert_to_tensor=True)

def is_unsatisfied(query, keyword_list, sbert_model, example_embeds, threshold=0.7):
    # 1. 關鍵字快速判斷
    query_lower = query.lower()
    if any(word in query_lower for word in keyword_list):
        return True
    # 2. SBERT語意相似度判斷
    query_emb = sbert_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, example_embeds)
    max_score = cos_scores.max().item()
    return max_score >= threshold

# 10. 問答互動與判斷
unsatisfied_count = 0

print("智能客服系統啟動，輸入 exit 離開。")
while True:
    query = input("🔍 問題 / Question: ")
    if query.lower() in {"exit", "quit"}:
        print("感謝使用，再見！")
        break

    # 判斷用戶是否表達不滿意
    if is_unsatisfied(query, unsatisfied_keywords, sbert_model, example_embeddings):
        print("系統：您似乎需要真人客服，是否要轉接？(y/n)")
        choice = input().strip().lower()
        if choice == "y":
            print("系統：正在為您轉接真人客服，請稍後...")
            break
        else:
            print("系統：繼續為您提供AI客服服務。")

    # AI 回答
    result = qa.invoke({"query": query})
    print("\n📣 回答 / Answer:\n", result['result'])
    print("\n📚 來源 / Sources:\n", [doc.page_content for doc in result['source_documents']])
    print("\n------------------------------------------------------------------------------------\n")

    # 簡單不滿意判斷 (例如回應中含「根據目前資訊無法回答」)
    if "根據目前資訊無法回答" in result['result']:
        unsatisfied_count += 1
    else:
        unsatisfied_count = 0

    # 連續兩次回答無法解決，自動詢問是否轉真人
    if unsatisfied_count >= 2:
        print("系統：看起來我無法回答您的問題，是否需要轉接真人客服？(y/n)")
        choice = input().strip().lower()
        if choice == "y":
            print("系統：正在為您轉接真人客服，請稍後...")
            break
        else:
            unsatisfied_count = 0
            print("系統：好的，請繼續提問。")
