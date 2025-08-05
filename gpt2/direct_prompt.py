from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Prompt
prompt = "The capital of France is"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text with controlled randomness
output_ids = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=20,
    do_sample=True,             # Enables sampling
    top_k=50,                   # Sample from top 50 tokens
    top_p=0.95,                 # Or use nucleus sampling
    temperature=0.8,            # Controls randomness (lower = more greedy)
    # num_beams=5,                # Explore multiple possible sequences to find the best one
    # early_stopping=True,        # Stop once the best beams agree
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode and print result
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
