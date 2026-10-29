import html, time
from collections import defaultdict, deque
import gradio as gr
from clinrag.retrieval import Retriever
from clinrag.rag import run_agent

retriever = Retriever()
HOURLY_LIMIT = 15
_calls = defaultdict(deque)

def allowed(ip):
    now, q = time.time(), _calls[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= HOURLY_LIMIT:
        return False
    q.append(now)
    return True

def render_passages(hits, cited):
    parts = []
    for h in hits:
        border = "#f5a623" if h["chunk_id"] in cited else "#cccccc"
        bg = "#fff7e6" if h["chunk_id"] in cited else "#f6f6f6"
        parts.append(
            f'<div style="border-left:4px solid {border};background:{bg};color:#222;'
            f'padding:8px 12px;margin:8px 0;border-radius:4px">'
            f'<b>{h["chunk_id"]}</b> · score {h["score"]:.2f}<br>{html.escape(h["text"])}</div>')
    return "".join(parts)

def ask(question, use_agent, k, request: gr.Request):
    if not question.strip():
        return "Type a question.", "", ""
    fwd = request.headers.get("x-forwarded-for") if request else None
    ip = (fwd or (request.client.host if request else "local")).split(",")[0].strip()
    if not allowed(ip):
        return f"Rate limit reached ({HOURLY_LIMIT} questions/hour). Please try later.", "", ""
    try:
        res = run_agent(question, retriever, k=int(k), use_agent=use_agent)
    except Exception as e:
        return f"Error: {e}", "", ""
    warn = ""
    if res["invalid"]:
        warn = "\n\n**Warning:** citations to passages that were not retrieved: " + ", ".join(res["invalid"])
    trace = "\n".join(f"- {t}" for t in res["trace"])
    return res["text"] + warn, trace, render_passages(res["hits"], set(res["cited"]))

with gr.Blocks(title="Clinical notes RAG (synthetic data)") as demo:
    gr.Markdown("# Clinical notes RAG with citations\n"
                "**Research demo on synthetic notes (Asclepius). Not medical advice.**")
    q = gr.Textbox(label="Question", lines=2)
    with gr.Row():
        agent = gr.Checkbox(True, label="Agent mode (query rewrite + retry)")
        k = gr.Slider(3, 10, value=6, step=1, label="Passages to retrieve")
    btn = gr.Button("Ask", variant="primary")
    ans = gr.Markdown()
    trace = gr.Markdown()
    passages = gr.HTML()
    gr.Examples(["Which patients were treated for pneumonia and with what?",
                 "Which patient had SOB and what was given?",
                 "Who was diagnosed with a stroke?"], inputs=q)
    btn.click(ask, [q, agent, k], [ans, trace, passages])

if __name__ == "__main__":
    demo.launch()