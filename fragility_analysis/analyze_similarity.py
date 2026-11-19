import csv
import sys
import numpy as np


def biggest_change(sim):
    # returns the layer where the similarity goes up the most
    changes = np.diff(sim)
    return np.argmax(changes) + 1

sentences = []
sims = []
# which results to look at, e.g. python analyze_similarity.py embedding_similarity_script_copy.csv
csv_file = sys.argv[1] if len(sys.argv) > 1 else "embedding_similarity.csv"
with open(csv_file, encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    num_layers = len([h for h in header if h.startswith("sim_L")])
    for row in reader:
        # skip sentences where the swapped letter wasnt in the last token (layer 0 is exactly 1 for those)
        if float(row[2]) > 0.9999:
            continue
        sentences.append((row[0], row[1]))
        sims.append([float(x) for x in row[2:]])

# print results in table format, one table per sentence
print(f"{len(sims)} sentences\n")
for (sentence, perturbed), sim in zip(sentences, sims):
    print(sentence)
    print(f"perturbed word: {perturbed}\n")
    print("layer | similarity | change")
    for l in range(num_layers):
        change = f"{sim[l] - sim[l - 1]:+.3f}" if l > 0 else "  -   "
        print(f"{l:5} | {sim[l]:10.3f} | {change}")

    layer = biggest_change(sim)
    print(f"\nbiggest change in similarity: layer {layer - 1} -> {layer}\n\n")
