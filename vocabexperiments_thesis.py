import unicodedata
from collections import defaultdict
import torch
from tokenizers import AddedToken
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-7B-Instruct"   
OUT = "my-model"


def load_lookalikes(path="chars.txt"):
    lookalikes = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        root = line[0]
        if not (root.isascii() and root.isalnum()):
            continue
        lookalikes[root] = []
        for c in line[1:]:
            if not c.isascii():  # the 0 and 1 rows list O o I i l | as lookalikes; those are normal chars, keep them out
                lookalikes[root].append(c)
    return lookalikes


def plain_copy(tokenizer, lookalikes, embeddings):
    """Each lookalike copies the root word's embedding"""
    copy_from = {}
    for character, look_alikes in lookalikes.items():
        for look_alike in look_alikes:
            copy_from[look_alike] = tokenizer(character, add_special_tokens=False).input_ids
            copy_from[" " + look_alike] = tokenizer(" " + character, add_special_tokens=False).input_ids  #tokenizer counts leading space as char
    new_embeddings = {}
    for token, ids in copy_from.items():
        new_embeddings[token] = embeddings[ids].mean(dim=0)
    return new_embeddings


def mean_copy(tokenizer, lookalikes, embeddings):
    """Each new lookalike copies the mean of the root + every variant that already has an embedding"""
    copy_from = {}
    for character, look_alikes in lookalikes.items():
        for prefix in ("", " "):
            encoded = {}
            for v in [character] + look_alikes:
                encoded[v] = tokenizer(prefix + v, add_special_tokens=False).input_ids
            known = []
            for ids in encoded.values():
                if len(ids) == 1:
                    known.append(ids[0])
            for v in look_alikes:
                if len(encoded[v]) > 1:
                    copy_from[prefix + v] = known or encoded[character]
    new_embeddings = {}
    for token, ids in copy_from.items():
        new_embeddings[token] = embeddings[ids].mean(dim=0)
    return new_embeddings


def script_copy(tokenizer, lookalikes, embeddings, alpha=0.5):
    ids_by_script = defaultdict(list)
    for i in range(len(tokenizer)):
        ch = tokenizer.decode([i]).lstrip(" ")
        if len(ch) == 1 and ch != "\ufffd":  # \ufffd = partial UTF-8 byte token
            ids_by_script[unicodedata.name(ch, "UNKNOWN").split()[0]].append(i)
    copy_from = {}
    for character, look_alikes in lookalikes.items():
        for prefix in ("", " "):
            root_vec = embeddings[tokenizer(prefix + character, add_special_tokens=False).input_ids].float().mean(dim=0)
            for v in look_alikes:
                if len(tokenizer(prefix + v, add_special_tokens=False).input_ids) == 1:
                    continue  # already a token, leave it alone
                script_ids = ids_by_script.get(unicodedata.name(v, "UNKNOWN").split()[0])
                if not script_ids:
                    print(f"no single-char tokens in the script of {v!r}, using plain copy")
                    vec = root_vec
                else:
                    vec = (1 - alpha) * root_vec + alpha * embeddings[script_ids].float().mean(dim=0)
                copy_from[prefix + v] = vec * root_vec.norm() / vec.norm()  # keep the norm the model expects for a real char
    return copy_from


COPY_TYPES = {"plain_copy": plain_copy, "mean_copy": mean_copy, "script_copy": script_copy}
def add_lookalikes(MODEL, OUT, LOOKALIKES, COPY_TYPE):  # copy_kwargs e.g. alpha=0.3 for the script copies

    # make sure that no lookalikes are repeated
    all_spellings = []
    for spellings in LOOKALIKES.values():
        for s in spellings:
            all_spellings.append(s)
    assert len(all_spellings) == len(set(all_spellings)), "lookalike repeated"


    #load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    tokenizer.save_pretrained(f"{OUT}/original-tokenizer") #save the original tokenizer to compare against


    # build each lookalike's new embedding, separately for the input and output tables (must run on the original tokenizer)
    copy = COPY_TYPES[COPY_TYPE]
    print(f"Building {OUT} with {COPY_TYPE}")
    with torch.no_grad():
        input_copy = copy(tokenizer, LOOKALIKES, model.get_input_embeddings().weight)
        output_copy = copy(tokenizer, LOOKALIKES, model.get_output_embeddings().weight)

    old_vocab_size = len(tokenizer)

    # add lookalikes as new tokens with no value yet
    tokenizer.add_tokens([AddedToken(t, normalized=False) for t in input_copy])

    # Qwen has ~400 unused rows but if we add enough words we will need to expand the embedding tables
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))


    # copy the embeddings from the original words to the new lookalike tokens
    input_vectors = model.get_input_embeddings().weight    
    output_vectors = model.get_output_embeddings().weight  
    new_ids = []
    with torch.no_grad():
        for token_text, vec in input_copy.items():
            new_id = tokenizer.convert_tokens_to_ids(token_text)
            if new_id < old_vocab_size:
                print(f"{token_text!r} was already in the vocabulary")
                continue
            input_vectors[new_id] = vec
            output_vectors[new_id] = output_copy[token_text]
            new_ids.append(new_id)

    #supress lookalikes so they don't show up in outputs 
    model.generation_config.suppress_tokens = new_ids 

    # make sure new words are added as words not multiple tokens
    for token_text in input_copy:
        pieces = tokenizer.tokenize(token_text)
        assert len(pieces) == 1, f"{token_text!r} splits into {pieces}"


    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)


if __name__ == "__main__":
    add_lookalikes(MODEL, OUT, load_lookalikes(), "mean_copy")  # "plain_copy", "mean_copy", or "script_copy"
    