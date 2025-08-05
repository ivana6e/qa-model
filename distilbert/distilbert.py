from transformers import pipeline


# cased: sensitive to capitalization, better for formal texts, names, technical docs
# uncased: ignores capitalization, better for informal, chat, or social media-like text
cased = pipeline("question-answering", model='distilbert/distilbert-base-cased-distilled-squad')
uncased = pipeline("question-answering", model="distilbert/distilbert-base-uncased-distilled-squad")

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

question = question_list[0]

result_cased = cased(question=question, context=context)
print(f"cased answer: '{result_cased['answer']}', score: {round(result_cased['score'], 4)}, start: {result_cased['start']}, end: {result_cased['end']}")
result_uncased = uncased(question=question, context=context)
print(f"uncased answer: '{result_uncased['answer']}', score: {round(result_uncased['score'], 4)}, start: {result_uncased['start']}, end: {result_uncased['end']}")
# score is in the range of (0.0, 1.0), the higher the score, the higher the confidence
