"""
Score every tokenizer setup with the Fragility benchmark (github.com/nchambers/fragility)
and write fragility_results.xlsx next to this file.

Run on the CUDA machine, from the folder holding this file, vocabexperiments_thesis.py and chars.txt:
    pip install openpyxl          # the only thing the fragility env is missing
    python run_fragility.py

Safe to re-run: built models and finished setups are skipped, so after a crash just start it again.
"""
import contextlib
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "fragility")             # cloned here if missing
OUTDIR = os.path.join(HERE, "fragility-outputs")   # raw test files, one folder per setup
XLSX = os.path.join(HERE, "fragility_results.xlsx")

QWEN = "Qwen/Qwen2.5-7B-Instruct"   # must match MODEL in vocabexperiments_thesis.py
SETUPS = {
    "original": QWEN,
    "plain_copy": os.path.join(HERE, "models", "plain_copy"),
    "mean_copy": os.path.join(HERE, "models", "mean_copy"),
    "script_copy": os.path.join(HERE, "models", "script_copy"),
}
GRADES = ["OVERALL", "PHONE", "PHONE (no CoD)", "LETTERS", "MMLU", "NLU"]
GRADE_NOTES = {
    "OVERALL": "0.3 x NLU + 0.3 x LETTERS + 0.4 x PHONE",
    "PHONE": "Reading digits / phone numbers written with lookalikes, noise and spelled-out digits (incl. hint prompts)",
    "PHONE (no CoD)": "PHONE without the chain-of-dictionary hint prompts",
    "LETTERS": "Naming lookalike letters and recovering words written with them",
    "MMLU": "MMLU accuracy with 20% of characters perturbed, relative to clean MMLU (capped at 100%)",
    "NLU": "0.6 x MMLU grade + 0.4 x recovering whole sentences containing lookalikes",
}


def build_models():
    """Build each modified Qwen with vocabexperiments_thesis.add_lookalikes (needs ~15 GB disk each)."""
    from vocabexperiments_thesis import add_lookalikes, load_lookalikes
    lookalikes = load_lookalikes(os.path.join(HERE, "chars.txt"))
    for name, path in SETUPS.items():
        if name == "original" or os.path.exists(os.path.join(path, "config.json")):
            continue
        print(f"=== building {name} -> {path}")
        add_lookalikes(QWEN, path, lookalikes, name)


def run_one(name, path):
    """Child process: run the repo's fragility.py unchanged on one setup."""
    os.chdir(REPO)                      # fragility.py reads data/chars.txt relative to cwd
    sys.path.insert(0, REPO)
    import models
    setattr(models, name, lambda qn=None: models.GenericModel(path, device=models.optimal_gpu(), quantization=qn))
    src = open("fragility.py", encoding="utf-8").read()
    # fixed seeds, so every setup is tested on the exact same items
    fixed = src.replace("MIMIC_PAPER_RANDOM = False", "MIMIC_PAPER_RANDOM = True", 1)
    assert fixed != src, "fragility.py changed: MIMIC_PAPER_RANDOM flag not found"
    sys.argv = ["fragility.py", "-outdir", OUTDIR, name]
    exec(compile(fixed, "fragility.py", "exec"), {"__name__": "__main__"})


def score(name):
    """Run the repo's official scorer and parse what it prints."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import scorer
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        scorer.score_dir(os.path.join(OUTDIR, f"model-{name}"))
    tests, grades = {}, {}
    for line in printed.getvalue().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        label, value = parts[0].strip(), float(parts[1])
        if " grade" in label:   # e.g. "PHONE grade (no CoD) for plain_copy"
            grades[label.split(" for ")[0].replace(" grade", "")] = value
        else:
            tests[label] = value
    assert set(GRADES) <= set(grades), f"scorer output changed, got grades {list(grades)}"
    return {"tests": tests, "grades": grades}


def write_spreadsheet(scores, path):
    from openpyxl import Workbook
    names = list(scores)
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Grade (higher = less fragile)"] + names + ["What it measures"])
    for g in GRADES:
        ws.append([g] + [scores[n]["grades"][g] for n in names] + [GRADE_NOTES[g]])
    pt = wb.create_sheet("Per test")
    pt.append(["Test (accuracy)"] + names)
    for t in sorted({t for n in names for t in scores[n]["tests"]}):
        pt.append([t] + [scores[n]["tests"].get(t) for n in names])
    wb.save(path)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--run":
        run_one(sys.argv[2], sys.argv[3])
        sys.exit()

    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "https://github.com/nchambers/fragility.git", REPO], check=True)
    build_models()
    for name, path in SETUPS.items():
        done = os.path.join(OUTDIR, f"model-{name}", "DONE")   # not .txt, so the scorer ignores it
        if os.path.exists(done):
            print(f"=== {name} already scored, skipping")
            continue
        print(f"=== running fragility on {name}")
        # own process per setup so GPU memory is fully freed between models
        subprocess.run([sys.executable, os.path.abspath(__file__), "--run", name, path], check=True)
        open(done, "w").close()

    scores = {name: score(name) for name in SETUPS}
    write_spreadsheet(scores, XLSX)
    print(f"wrote {XLSX}")
