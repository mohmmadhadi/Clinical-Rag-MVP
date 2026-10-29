import re
_TOKEN = re.compile(r"\S+\s*")   # keeps newlines when re-joined

def chunk_note(note_id, text, size=180, overlap=40):
    tokens = _TOKEN.findall(text)
    if not tokens:
        return []
    step, out, start, i = size - overlap, [], 0, 0
    while True:
        out.append({"chunk_id": f"{note_id}#{i}", "note_id": note_id,
                    "text": "".join(tokens[start:start + size]).strip()})
        i += 1
        if start + size >= len(tokens):
            break
        start += step
    return out