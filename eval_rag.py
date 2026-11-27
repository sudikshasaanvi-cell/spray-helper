"""Evaluate RAG retrieval hit rate for bm25, dense and hybrid."""
import csv
from rag import search, build_index

build_index()

modes = ["bm25", "dense", "hybrid"]
questions = []
with open("rag_questions.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        questions.append((row["question"], row["expected_source"]))

results = {m: 0 for m in modes}
total = len(questions)
for q, exp in questions:
    for m in modes:
        hits = search(q, k=5, mode=m)
        sources = [h["source"].split("/")[-1] for h in hits]
        # expected_source is filename; source URL ends with .html/.pdf not filename.
        # We stored source as URL. Map expected filename -> not directly. Instead compare by doc filename? 
        # Our chunks payload source is URL, not filename. Need map URL->filename.
        pass

# Map source URL to filename
url_to_file = {}
import os
for fn in os.listdir("docs"):
    if fn.endswith(".txt"):
        first = open(os.path.join("docs", fn)).readline().strip()
        url_to_file[first] = fn

for q, exp in questions:
    for m in modes:
        hits = search(q, k=5, mode=m)
        files = [url_to_file.get(h["source"], "") for h in hits]
        if exp in files:
            results[m] += 1

lines = []
for m in modes:
    hr = results[m] / total if total else 0.0
    lines.append(f"{m}: top-5 hit rate = {hr:.3f} ({results[m]}/{total})")
    print(lines[-1])
with open("results/rag_eval.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
    f.write(f"Questions: {total}\n")
