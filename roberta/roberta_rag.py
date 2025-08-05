from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch


# STEP 1: Load models
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model
question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2")

# STEP 2: Define your knowledge base (e.g., chunks of help articles or docs)
documents = [
    "You can return products within 30 days of purchase with a receipt for a full refund.",
    "Standard shipping takes 5-7 business days, while express shipping takes 2-3 business days.",
    "To reset your device, hold the power button for 10 seconds until it restarts",
    "Our customer service is available 24/7 via chat or phone",
    "Opened items can be refunded within 15 days if they are in good condition.",
    "ChatGPT is an AI language model developed by OpenAI. It is capable of natural language understanding and generation.",
    "The OpenAI API allows developers to integrate language models into their own apps and services.",
    "Fine-tuning enables users to customize the model's behavior for specific tasks or domains.",
    "System messages can guide how ChatGPT responds during conversations.",
]

# STEP 3: Embed all documents for retrieval
corpus_embeddings = retriever_model.encode(documents, convert_to_tensor=True)

# STEP 4: Ask a question
question_list = [
    "What’s the refund policy?",
    "How long does shipping take?",
    "How do I reset my device?",
    "What are your customer service hours?",
    "Do you offer refunds for opened items?",
    "How can developers integrate ChatGPT into their applications?",
]
question = question_list[0]

# Encode question and retrieve top relevant document
question_embedding = retriever_model.encode(question, convert_to_tensor=True)
cosine_scores = util.cos_sim(question_embedding, corpus_embeddings)

# Get top result (index of most relevant document)
top_result_idx = torch.argmax(cosine_scores).item()
retrieved_context = documents[top_result_idx]

# STEP 5: Run RoBERTa QA on retrieved context
answer = question_answerer(question=question, context=retrieved_context)

# Print result
print(f"Question: {question}")
print(f"Retrieved Context: {retrieved_context}")
print(f"Answer: {answer['answer']}")
