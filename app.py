import os, json, math, asyncio, threading, traceback
from typing import List
from datetime import datetime
from fastapi import FastAPI, Request
import httpx
import threading
from dotenv import load_dotenv
load_dotenv()

# ========= Helpers to run background =========
def run_bg(coro):
    """Run an async coroutine in background safely on FastAPI."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # No running loop (or called from sync ctx) -> use a thread
        threading.Thread(target=lambda: asyncio.run(coro), daemon=True).start()

# ========= ENV =========
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "") or os.environ.get("HF_TOKEN", "")
ALLOWED_CHANNELS = set(
    c.strip() for c in os.environ.get("ALLOWED_CHANNELS", "").split(",") if c.strip()
)

# allow turning Slack off in local
USE_SLACK = not (
    SLACK_BOT_TOKEN in ("", "test")
    or SLACK_SIGNING_SECRET in ("", "test")
    or os.environ.get("USE_SLACK", "1") == "0"
)

HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3").strip()
HF_EMB_MODEL = os.environ.get("HF_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
MEMORY_FILE = "memory.jsonl"
EMB_FILE = "embeddings.jsonl"

api = FastAPI()

# ========= Memory helpers =========
def _append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Call HF feature-extraction pipeline. Returns list of vectors (one per input)."""
    if not HF_API_KEY:
        return [[0.0] * 5 for _ in texts]
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMB_MODEL}"
    timeout = httpx.Timeout(60.0, read=60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json={"inputs": texts})
        print("HF-EMB status:", r.status_code, r.text[:200])
        if not r.is_success:
            # fallback deterministic vector sizes so the bot keeps running
            return [[0.0] * 5 for _ in texts]
        data = r.json()
        # HF sometimes returns [dim] when single input, or [batch, dim]
        if isinstance(data, list) and data and isinstance(data[0], float):
            return [data]
        # sometimes [[dim], [dim], ...] for multiple inputs
        return data

def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)

async def retrieve_context(query: str, k: int = 4) -> str:
    mem = _read_jsonl(MEMORY_FILE)
    emb_mem = _read_jsonl(EMB_FILE)
    if not mem or not emb_mem:
        return ""
    q_emb = (await embed_texts([query]))[0]
    scored = []
    for i, e in enumerate(emb_mem):
        vec = e.get("embedding") or e  # support older shape
        sim = _cos(q_emb, vec)
        scored.append((sim, mem[i]))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [item[1]["text"] for item in scored[:k]]
    return "\n\n".join(f"- {t}" for t in top)

async def add_to_memory(user: str, question: str, answer: str):
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "user": user,
        "text": f"Q: {question}\nA: {answer}",
    }
    _append_jsonl(MEMORY_FILE, entry)
    emb = (await embed_texts([entry["text"]]))[0]
    _append_jsonl(EMB_FILE, {"embedding": emb})

# ========= LLM =========
SYSTEM_PROMPT = (
    "You are HUED internal assistant. Answer briefly and helpfully. "
    "Use Arabic if the user uses Arabic; otherwise use English. "
    "If you're unsure, ask a brief clarifying question."
)

async def call_llm(prompt: str) -> str:
    if not HF_API_KEY:
        return "⚠️ HF_API_KEY غير مهيأ. هذا رد تجريبي."
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    payload = {
        "inputs": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }
    timeout = httpx.Timeout(90.0, read=90.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        print("HF-LLM:", HF_MODEL, "status:", r.status_code, r.text[:300])
        if r.status_code == 404:
            # يحدث لو اسم الموديل قديم (v0.2) أو غير متاح
            return "تعذّر الوصول إلى نموذج الذكاء (404). تم ضبط الموديل الافتراضي على v0.3—تأكدي من المتغير HF_MODEL."
        if r.status_code == 401:
            return "‏(401) التوكن غير صحيح أو بلا صلاحية. أعيدي توليد HF Access Token بصلاحية Read."
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return (str(data) or "").strip()[:800]

# ========= Slack (guarded) =========
if USE_SLACK:
    from slack_bolt import App as SlackApp
    from slack_bolt.adapter.fastapi import SlackRequestHandler
    from slack_sdk.errors import SlackApiError

    bolt_app = SlackApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
    handler = SlackRequestHandler(bolt_app)

    def _allowed_channel(channel_id: str) -> bool:
        return (not ALLOWED_CHANNELS) or (channel_id in ALLOWED_CHANNELS)

    async def _process_and_reply(channel: str, ts: str, text: str, user: str, client, say, logger):
        try:
            cleaned = text.split(">", 1)[-1].strip() if ">" in text else text
            ctx = await retrieve_context(cleaned, k=4)
            augmented = f"السياق السابق:\n{ctx}\n\nالسؤال الحالي:\n{cleaned}" if ctx else cleaned
            answer = await call_llm(augmented)
            await add_to_memory(user, cleaned, answer)
            try:
                client.chat_update(channel=channel, ts=ts, text=answer)
            except SlackApiError as e:
                logger.error(f"chat_update failed: {e.response.data}")
                say(answer)
            except Exception as e:
                logger.error(f"chat_update failed: {e}")
                say(answer)
        except Exception as e:
            print("Worker error:", e, traceback.format_exc())
            try:
                client.chat_update(channel=channel, ts=ts, text=f"صار خطأ أثناء المعالجة:\n{e}")
            except Exception:
                say(f"صار خطأ أثناء المعالجة:\n{e}")

    # mention in channels
    @bolt_app.event("app_mention")
    def on_mention(body, say, client, logger):
        event = body.get("event", {})
        channel = event.get("channel")
        user = event.get("user")
        text = event.get("text", "")
        if not _allowed_channel(channel):
            return
        placeholder = say("_جارِ المعالجة…_")
        ts = placeholder["ts"]
        run_bg(_process_and_reply(channel, ts, text, user, client, say, logger))

    # direct messages (needs 'message.im' event & scopes im:history + chat:write)
    @bolt_app.event("message")
    def on_dm(body, say, client, logger, event):
        if event.get("channel_type") != "im":
            return
        channel = event.get("channel")
        user = event.get("user")
        text = event.get("text", "").strip()
        if not text:
            return
        placeholder = say("_جارِ المعالجة…_")
        ts = placeholder["ts"]
        run_bg(_process_and_reply(channel, ts, text, user, client, say, logger))

    @api.post("/slack/events")
    async def slack_events(request: Request):
        return await handler.handle(request)

# ========= Health / Test =========
@api.get("/")
async def root():
    return {
        "ok": True,
        "service": "HUED Slack bot",
        "slack_enabled": USE_SLACK,
        "hf_model": HF_MODEL,
    }

@api.get("/health")
async def health():
    return {"ok": True}

@api.get("/hf_test")
async def hf_test():
    key = HF_API_KEY
    model = HF_MODEL
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # POST with a tiny prompt gives صورة أوضح من GET
            r = await client.post(url, headers=headers, json={"inputs": "ping"})
        return {
            "status": r.status_code,
            "ok": r.is_success,
            "model": model,
            "preview": r.text[:200],
        }
    except Exception as e:
        return {"status": "client_error", "error": str(e), "model": model}
