"""
Chat with the changed model built by vocabexperiments_thesis.py.
    /switch   toggle between the new tokenizer and the original one 
    quit      stop
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_FOLDER = "my-model"  
MAX_REPLY_TOKENS = 478


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_FOLDER, torch_dtype=torch.bfloat16).to(device)

tokenizers = {
    "new": AutoTokenizer.from_pretrained(MODEL_FOLDER),
    "original": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct"),  # same as the unmodified tokenizer; must match MODEL in vocabexperiments_thesis.py
}

mode = "new"
history = []
print(f"Using the {mode} tokenizer. Type /switch to toggle, quit to stop.")

while True:
    message = input(f"\nUser [{mode}]: ").strip()

    if message.lower() in ("quit", "exit"):
        break
    if message == "/switch":
        mode = "original" if mode == "new" else "new"
        history = []   
        print(f"Switched to the {mode} tokenizer. Conversation cleared.")
        continue
    if not message:
        continue

    tokenizer = tokenizers[mode]
    print("  tokens:", tokenizer.tokenize(message))   # shows how tokenization changes between the two 

    # Put the conversation in the model's chat format
    history.append({"role": "user", "content": message})
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(**inputs, max_new_tokens=MAX_REPLY_TOKENS)
    reply_ids = output[0, inputs.input_ids.shape[1]:]   # drop the prompt, keep only the reply
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True)

    print("Qwen:", reply)
    history.append({"role": "assistant", "content": reply})
