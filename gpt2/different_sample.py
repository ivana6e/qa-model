from transformers import GPT2Tokenizer, GPT2LMHeadModel


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = {}

# Greedy decoding
# Pros: Fast, deterministic
# Cons: Can repeat or get stuck in loops
outputs["Greedy"] = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=20,
    do_sample=False,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)

# Sampling
# Pros: More variety
# Cons: May produce incoherent or less factual output
outputs["Sampling"] = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=20,
    do_sample=True,
    temperature=1.0, # 0.7 to 1.0 is common, <1.0 more conservative, >1.0 more random
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)

# Top-k
# Pros: More coherent than pure sampling
# Cons: Less control than beam search
outputs["Top-k Sampling"] = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=20,
    do_sample=True,
    top_k=50,
    temperature=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)

# Top-p
outputs["Top-p Sampling"] = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=20,
    do_sample=True,
    top_p=0.95,
    temperature=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)

# Beam search
# Pros: Often better quality than greedy
# Cons: Slower, still can be repetitive
outputs["Beam Search"] = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=20,
    num_beams=5,         # number of beams (paths)
    early_stopping=True, # stop when beams converge
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)

# Print all results
for method, output in outputs.items():
    print(f"\n{method}:\n{tokenizer.decode(output[0], skip_special_tokens=True)}")
