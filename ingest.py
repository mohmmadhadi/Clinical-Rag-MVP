import json, random
from datasets import load_dataset

N_NOTES, SCAN_ROWS = 500, 20000
ds = load_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes", split="train", streaming=True)

seen = set()
for i, row in enumerate(ds):
    if i == 0:
        print("columns:", list(row.keys()))
    text = (row.get("note") or "").strip()   # if the column name differs, fix it here
    if text:
        seen.add(text)
    if i + 1 >= SCAN_ROWS:
        break

texts = sorted(seen)
random.Random(42).shuffle(texts)
texts = texts[:N_NOTES]

import os; os.makedirs("data", exist_ok=True)
with open("data/notes.jsonl", "w", encoding="utf-8") as f:
    for i, t in enumerate(texts):
        f.write(json.dumps({"note_id": f"n{i:04d}", "text": t}, ensure_ascii=False) + "\n")
print(len(seen), "unique notes seen;", len(texts), "written")