import csv
import os
import random
import sys
import unicodedata
import torch
from transformers import AutoModel, AutoTokenizer
from vocabexperiments_thesis import MODEL, load_lookalikes

N = 5000
# which model to test: the original qwen by default, or a folder with a changed model
# e.g. python embedding_similarity.py ../models/script_copy
model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL
out_file = "embedding_similarity.csv" if len(sys.argv) == 1 else f"embedding_similarity_{os.path.basename(model_path.rstrip('/'))}.csv"
random.seed(0)

# we want the perturbed version not to be affected by normalization 
lookalikes = {}
for root, chars in load_lookalikes().items():
    lookalikes[root] = []
    for c in chars:
        if unicodedata.normalize("NFC", c) == c:
            lookalikes[root].append(c) 


def get_sentences():
    sentences = []
    for s in open("sentences.txt", encoding="utf-8").read().splitlines():
        if 8 <= len(s.split()) <= 30:
            start, word = s.rstrip(".").rsplit(" ", 1)
            sentences.append((start + " ", word))
    random.shuffle(sentences)
    return sentences


def swap_letter(word):
    # pick a random letter that has lookalikes and replace it with one of them
    options = []
    for i in range(len(word)):
        if lookalikes.get(word[i]):
            options.append(i)
    if not options:
        return None
    i = random.choice(options) # choose a random letter to swap 
    return word[:i] + random.choice(lookalikes[word[i]]) + word[i + 1:] # choose a random lookalike to swap with 


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()

# get the last token embedding for each layer of the model, outputs a tensor of N layers by embedding_dim
def last_token_layers(text):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return torch.stack([h[0, -1] for h in out.hidden_states]).float()

sentences = get_sentences()
results = []

# swap a letter in the last word of each sentence with a lookalike 
for start, word in sentences:
    new_word = swap_letter(word)
    if new_word is None:
        continue
    # compare the original sentence with the same sentence but with the swapped word
    original = last_token_layers(start + word)
    perturbed = last_token_layers(start + new_word)
    similarity = torch.cosine_similarity(original, perturbed, dim=-1)
    results.append([start + word, new_word] + similarity.tolist())

    if len(results) % 500 == 0:
        print(len(results), "sentences done")
    if len(results) == N:
        break

# save to CSV
num_layers = len(similarity)
with open(out_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["sentence", "perturbed"]
    header += ["sim_L" + str(l) for l in range(num_layers)]
    writer.writerow(header)
    writer.writerows(results)
print("saved", out_file)
