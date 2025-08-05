from transformers import pipeline


question_answerer1 = pipeline("question-answering", model="deepset/roberta-base-squad2")
question_answerer2 = pipeline("question-answering", model="deepset/tinyroberta-squad2")

context = """"
You can return products within 30 days of purchase with a receipt for a full refund.
Standard shipping takes 5-7 business days, while express shipping takes 2-3 business days.
To reset your device, hold the power button for 10 seconds until it restarts.
Our customer service is available 24/7 via chat or phone.
Opened items can be refunded within 15 days if they are in good condition.
"""
question_list = [
    "What’s the refund policy?",
    "How long does shipping take?",
    "How do I reset my device?",
    "What are your customer service hours?",
    "Do you offer refunds for opened items?",
]

result = question_answerer1(question=question_list[0], context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

result = question_answerer2(question=question_list[0], context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
