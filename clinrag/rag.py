import os, re, time
from openai import OpenAI, RateLimitError

MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

_client = None
def get_client():
    global _client
    if _client is None:
        key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY (or LLM_API_KEY).")
        _client = OpenAI(base_url=BASE_URL, api_key=key)
    return _client

def chat(system, user, max_tokens=2500):
    """One chat call. The budget is generous because Nemotron Ultra is a reasoning model:
    if reasoning uses up max_tokens, the final answer comes back empty."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(3):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL, messages=messages, max_tokens=max_tokens)
            break
        except RateLimitError:                      # free tier: back off and retry
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    if not resp.choices:
        raise RuntimeError(f"Provider returned no choices: {getattr(resp, 'error', None)}")
    choice = resp.choices[0]
    text = re.sub(r"<think>.*?</think>", "", choice.message.content or "", flags=re.S).strip()
    if not text:
        raise RuntimeError(f"Model returned no final answer (finish_reason={choice.finish_reason}). "
                           "If it says 'length', raise max_tokens.")
    return text

SYSTEM = """You answer questions about clinical notes for a research demo.
Rules:
1. Use ONLY the numbered passages provided. Never use outside knowledge.
2. After every factual statement, cite the supporting passage ID in square brackets exactly as given, e.g. [n0042#1].
3. If the passages do not contain the answer, reply exactly: NOT_FOUND: <one-sentence reason>.
4. Be concise. Do not give medical advice."""

REWRITE_SYSTEM = ("Rewrite the user's clinical question as a standalone search query. "
                  "Expand abbreviations (e.g. SOB -> shortness of breath). Keep all specific details. "
                  "Output only the query.")

ID_RE = re.compile(r"n\d+#\d+")

def verify(text, hits):
    """Flag any cited ID that was not actually retrieved."""
    valid = {h["chunk_id"] for h in hits}
    cited, bad = [], []
    def fix(m):
        cid = m.group(0)
        if cid in valid:
            if cid not in cited:
                cited.append(cid)
            return cid
        bad.append(cid)
        return f"{cid} (UNVERIFIED)"
    return ID_RE.sub(fix, text), cited, bad

def answer(question, hits):
    passages = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits)
    raw = chat(SYSTEM, f"<passages>\n{passages}\n</passages>\n\nQuestion: {question}")
    text, cited, bad = verify(raw, hits)
    return {"text": text, "cited": cited, "invalid": bad,
            "not_found": raw.startswith("NOT_FOUND"), "hits": hits}

def rewrite(question):
    return chat(REWRITE_SYSTEM, question, max_tokens=1500)

def run_agent(question, retriever, k=6, use_agent=True):
    if not use_agent:
        res = answer(question, retriever.retrieve(question, k))
        res["trace"] = ["Plain RAG (no rewrite, no retry)"]
        return res

    trace = []
    try:
        q2 = rewrite(question)
        trace.append(f"Rewrote query: {q2}")
    except RuntimeError as e:                       # a failed rewrite shouldn't kill the answer
        q2 = question
        trace.append(f"Rewrite failed ({e}); using the original question")
    hits = retriever.retrieve(q2, k)
    res = answer(question, hits)
    if res["not_found"]:
        trace.append("Answer was NOT_FOUND, so retrying with the original question and more passages")
        merged = {h["chunk_id"]: h for h in hits}
        for h in retriever.retrieve(question, 2 * k):
            if h["chunk_id"] not in merged or h["score"] > merged[h["chunk_id"]]["score"]:
                merged[h["chunk_id"]] = h
        hits = sorted(merged.values(), key=lambda h: -h["score"])[:2 * k]
        res = answer(question, hits)
    res["trace"] = trace
    return res