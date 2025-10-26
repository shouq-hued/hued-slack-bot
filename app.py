import os, json, math, asyncio
from typing import List
from datetime import datetime
from fastapi import FastAPI, Request
import httpx
import threading
from dotenv import load_dotenv
load_dotenv()

def run_bg(coro):
    import asyncio, threading
    threading.Thread(target=lambda: asyncio.run(coro), daemon=True).start()

# ==== ENV ====
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")
ALLOWED_CHANNELS = set([c.strip() for c in os.environ.get("ALLOWED_CHANNELS","").split(",") if c.strip()])

USE_SLACK = True
if SLACK_BOT_TOKEN in ("", "test") or SLACK_SIGNING_SECRET in ("", "test") or os.environ.get("USE_SLACK","1") == "0":
    USE_SLACK = False

HF_LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MEMORY_FILE = "memory.jsonl"
EMB_FILE = "embeddings.jsonl"

api = FastAPI()

# ==== Memory helpers ====
def _append_jsonl(path:str, obj:dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _read_jsonl(path:str):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

async def embed_texts(texts:List[str]) -> List[List[float]]:
    if not HF_API_KEY: return [[0.0]]
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMB_MODEL}"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json={"inputs": texts})
        r.raise_for_status()
        data = r.json()
        if isinstance(data[0][0], float): return [data]
        return data

def _cos(a:List[float], b:List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return dot / (na*nb)

async def retrieve_context(query:str, k:int=4) -> str:
    mem = _read_jsonl(MEMORY_FILE); emb_mem = _read_jsonl(EMB_FILE)
    if not mem or not emb_mem: return ""
    q_emb = (await embed_texts([query]))[0]
    scored = []
    for i, e in enumerate(emb_mem):
        sim = _cos(q_emb, e["embedding"])
        scored.append((sim, mem[i]))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [item[1]["text"] for item in scored[:k]]
    return "\n\n".join(f"- {t}" for t in top)

async def add_to_memory(user:str, question:str, answer:str):
    entry = {"ts": datetime.utcnow().isoformat(), "user": user, "text": f"Q: {question}\nA: {answer}"}
    _append_jsonl(MEMORY_FILE, entry)
    emb = (await embed_texts([entry["text"]]))[0]
    _append_jsonl(EMB_FILE, {"embedding": emb})

# ==== LLM ====
SYSTEM_PROMPT = (
    "You are HUED internal assistant. Answer briefly and helpfully. "
    "Use Arabic if the user uses Arabic; otherwise use English. "
    "If you're unsure, ask a brief clarifying question."
)

async def call_llm(prompt:str) -> str:
    if not HF_API_KEY:
        return "⚠️ HF_API_KEY غير مهيأ. هذا رد تجريبي."
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    url = f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}"
    payload = {"inputs": f"{SYSTEM_PROMPT}\n\n{prompt}", "parameters": {"max_new_tokens": 300, "temperature": 0.2, "return_full_text": False}}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return str(data)[:800]

# ==== Slack (guarded) ====
if USE_SLACK:
    from slack_bolt import App as SlackApp
    from slack_bolt.adapter.fastapi import SlackRequestHandler
    bolt_app = SlackApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
    handler = SlackRequestHandler(bolt_app)

    # يرد على المنشن داخل القنوات
    @bolt_app.event("app_mention")
    def on_mention(body, say, client, logger):
        channel = body["event"].get("channel")
        user = body["event"].get("user")
        text = body["event"].get("text","")
        if ALLOWED_CHANNELS and channel not in ALLOWED_CHANNELS:
            return
        placeholder = say("_جارٍ المعالجة..._")
        ts = placeholder["ts"]

        async def worker():
            cleaned = text.split(">", 1)[-1].strip() if ">" in text else text
            ctx = await retrieve_context(cleaned, k=4)
            augmented = f"السياق السابق:\n{ctx}\n\nالسؤال الحالي:\n{cleaned}" if ctx else cleaned
            answer = await call_llm(augmented)
            await add_to_memory(user, cleaned, answer)
            try:
                client.chat_update(channel=channel, ts=ts, text=answer)
            except Exception as e:
                logger.error(f"chat_update failed: {e}")
                say(answer)
        run_bg(worker())

    # يرد على رسائل الخاص (DM) — لازم تضيفي event: message.im و Scope: im:history
    @bolt_app.event("message")
    def on_dm(body, say, client, logger, event):
        if event.get("channel_type") != "im":
            return
        user = event.get("user")
        channel = event.get("channel")
        text = event.get("text","").strip()
        if not text:
            return
        placeholder = say("_جارٍ المعالجة..._")
        ts = placeholder["ts"]

        async def worker():
            ctx = await retrieve_context(text, k=4)
            prompt = f"السياق السابق:\n{ctx}\n\nالسؤال الحالي:\n{text}" if ctx else text
            answer = await call_llm(prompt)
            await add_to_memory(user, text, answer)
            try:
                client.chat_update(channel=channel, ts=ts, text=answer)
            except Exception as e:
                logger.error(f"chat_update failed: {e}")
                say(answer)
        run_bg(worker())

    @api.post("/slack/events")
    async def slack_events(request: Request):
        return await handler.handle(request)

@api.get("/health")
async def health():
    return {"ok": True}
