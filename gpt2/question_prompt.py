from transformers import GPT2Tokenizer, GPT2LMHeadModel


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# prompt = "Q: What is the capital of France?"
# inputs = tokenizer(prompt, return_tensors="pt")
#
# output_ids = model.generate(
#     input_ids=inputs["input_ids"],
#     attention_mask=inputs["attention_mask"],
#     max_new_tokens=10,
#     do_sample=False,  # Greedy decoding for factual answers
#     pad_token_id=tokenizer.eos_token_id,
# )
#
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(output_text)


# List of questions
questions = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "What is the largest planet in the Solar System?",
    "What year did the Titanic sink?"
]

# Function to generate answer
def answer_question(question):
    prompt = f"Q: {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output.strip()

# Loop through and print answers
for q in questions:
    result = answer_question(q)
    print(result)
    print("-" * 40)