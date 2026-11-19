import csv
import numpy as np


def biggest_change(avg_sim):
    # returns the layer where the similarity goes up the most 
    changes = np.diff(avg_sim)
    return np.argmax(changes) + 1

sims = []
with open("embedding_similarity.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    num_layers = len([h for h in header if h.startswith("sim_L")])
    for row in reader:
        # skip sentences where the swapped letter wasnt in the last token (layer 0 is exactly 1 for those)
        if float(row[2]) > 0.9999:
            continue
        sims.append([float(x) for x in row[2:]])

avg_sim = np.mean(sims, axis=0)

# print results in table format
print(f"{len(sims)} sentences\n")
print("layer | similarity | change")
for l in range(num_layers):
    change = f"{avg_sim[l] - avg_sim[l - 1]:+.3f}" if l > 0 else "  -   "
    print(f"{l:5} | {avg_sim[l]:10.3f} | {change}")

layer = biggest_change(avg_sim)
print(f"\nbiggest change in similarity: layer {layer - 1} -> {layer}")
