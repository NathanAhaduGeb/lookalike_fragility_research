import csv
import glob
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BASE_NAME = "qwen2.5-7b (base)"


vocab_path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/vocab.json"))[0]
with open(vocab_path, encoding="utf-8") as f:
    vocab = json.load(f)


def is_known(word):
    # the model "knows" a word if the word (with the space before it) is a single token in the vocab
    # qwen's vocab.json writes the space before a word as "Ġ". this only works for plain ascii words
    # since other characters are stored differently, so we make sure there arent any
    assert word.isascii(), f"{word} has non-ascii characters"
    return "Ġ" + word in vocab


# quick check that the vocab lookup works
assert is_known("the") and is_known("California")
assert not is_known("1963") and not is_known("histopathologic")


def load_csv(csv_file):
    words = []
    sims = []
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip the header
        for row in reader:
            words.append(row[0].split(" ")[-1])
            sims.append([float(x) for x in row[2:]])
    return np.array(words), np.array(sims)


def get_model_name(csv_file):
    name = os.path.basename(csv_file).replace(".csv", "").replace("embedding_similarity", "").strip("_")
    if name == "":
        return BASE_NAME
    return name


def plot(groups, out_file):
    # top plot is the similarity at each layer, bottom plot is the change from previous layer
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for i in range(len(groups)):
        label, sims = groups[i]
        color = COLORS[i]
        layers = np.arange(sims.shape[1])
        changes = np.diff(sims, axis=1)

        mean = sims.mean(axis=0)
        std = sims.std(axis=0)
        top.plot(layers, mean, color=color, linewidth=2, marker="o", markersize=4, label=f"{label} (n={len(sims)})")
        top.fill_between(layers, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

        mean = changes.mean(axis=0)
        std = changes.std(axis=0)
        bottom.plot(layers[1:], mean, color=color, linewidth=2, marker="o", markersize=4)
        bottom.fill_between(layers[1:], mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    top.set_ylabel("cosine similarity")
    top.set_title("Last-token similarity: perturbed vs. original sentence (mean ± 1 SD)")
    top.legend(frameon=False)

    bottom.set_xlabel("layer")
    bottom.set_ylabel("change from previous layer")
    bottom.set_title("Layer-to-layer change in similarity (mean ± 1 SD)")
    bottom.set_xticks(layers[::2])
    bottom.axhline(0, color="#888", linewidth=1)

    for ax in [top, bottom]:
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    print("saved", out_file)


def plot_share_increased(groups, out_file):
    # share of sentences whose similarity went up at each layer
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i in range(len(groups)):
        label, sims = groups[i]
        layers = np.arange(1, sims.shape[1])
        changes = np.diff(sims, axis=1)
        share_increased = (changes > 0).mean(axis=0)
        ax.plot(layers, share_increased, color=COLORS[i], linewidth=2, marker="o", markersize=4, label=f"{label} (n={len(sims)})")

    # if a layer did nothing on purpose we would expect about half the sentences to go up
    ax.axhline(0.5, color="#888", linewidth=1, linestyle="--", label="chance (0.5)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("layer")
    ax.set_ylabel("share of sentences")
    ax.set_title("Share of sentences whose similarity increased from the previous layer")
    ax.set_xticks(np.arange(0, layers[-1] + 1, 2))
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    print("saved", out_file)


csv_files = sys.argv[1:]
if len(csv_files) == 0:
    csv_files = ["embedding_similarity.csv", "embedding_similarity_plain_copy.csv"]

all_sentences = []  # base qwen keeps every sentence, the other models drop the ones that start at similarity 1
dropped_sentences = []  # every model drops the ones that start at similarity 1

for csv_file in csv_files:
    model = get_model_name(csv_file)
    words, sims = load_csv(csv_file)

    in_last_token = sims[:, 0] <= 0.9999
    known = np.array([is_known(w) for w in words])

    if model == BASE_NAME:
        all_sentences.append((model, sims))
    else:
        all_sentences.append((model, sims[in_last_token]))
    dropped_sentences.append((model, sims[in_last_token]))

    short_name = model.split(" ")[0]
    plot([("known word (1 token)", sims[known]),
          ("unknown word (split)", sims[~known])],
         f"similarity_known_{short_name}.png")
    plot([("swap in last token", sims[in_last_token]),
          ("swap not in last token", sims[~in_last_token])],
         f"similarity_last_token_{short_name}.png")

plot(all_sentences, "similarity_by_layer.png")
plot(dropped_sentences, "similarity_by_layer_1sim_dropped.png")
plot_share_increased(dropped_sentences, "similarity_share_increased.png")
