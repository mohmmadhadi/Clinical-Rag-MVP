import json
from clinrag.chunking import chunk_note
from clinrag.retrieval import Retriever

chunks = []
for line in open("data/notes.jsonl", encoding="utf-8"):
    r = json.loads(line)
    chunks += chunk_note(r["note_id"], r["text"])
print(len(chunks), "chunks")
Retriever.build(chunks)