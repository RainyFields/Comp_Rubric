#!/usr/bin/env python3
"""Stateless judge shim: presents both API dialects the FoldAgent repo uses and
forwards everything to the infra vLLM gpt-oss-120b server, coercing model names.

  - POST /v1/chat/completions  (OpenAI dialect; used by local_search.call_openai_raw
    via AsyncOpenAI with OPENAI_BASE_URL, model names gpt-4o-mini / gpt-4.1)
  - POST /chat                 (custom dialect; agents/utils.call_openai POSTs here via
    OPENAI_URL and reads resp["content"]; model name gpt-5-nano)

Run: python judge_shim.py --upstream http://<infra-ip>:8001/v1 --port 8002
Consumers set: OPENAI_BASE_URL=http://127.0.0.1:8002/v1
               OPENAI_URL=http://127.0.0.1:8002/chat
All requests are logged to --log-dir for the judge audit.
"""
import argparse, json, os, time, uuid
import httpx, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

parser = argparse.ArgumentParser()
parser.add_argument("--upstream", default=None, help="judge vLLM base url, e.g. http://IP:8001/v1")
parser.add_argument("--upstream-marker", default=None,
                    help="path to INFRA_READY marker holding the infra IP; re-resolved per request (survives infra relaunch)")
parser.add_argument("--model", default="gpt-oss-120b")
parser.add_argument("--agent-upstream", default=None,
                    help="optional agent vLLM base url; non-gpt-* model requests route here unmodified")
parser.add_argument("--force-agent-temperature", type=float, default=None,
                    help="inject this temperature into agent-upstream requests (repo's CallAPI sends none)")
parser.add_argument("--port", type=int, default=8002)
parser.add_argument("--log-dir", default=None)
args, _ = parser.parse_known_args()

JUDGE_ALIASES = {"gpt-5-nano", "gpt-4o-mini", "gpt-4.1", args.model}

app = FastAPI()
client = httpx.AsyncClient(timeout=600.0)

def audit(kind, req, resp_text):
    if not args.log_dir:
        return
    os.makedirs(args.log_dir, exist_ok=True)
    rec = {"t": time.time(), "kind": kind, "requested_model": req.get("model"),
           "messages": req.get("messages"), "response": resp_text}
    with open(os.path.join(args.log_dir, "judge_calls.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def infra_ip():
    ip = open(args.upstream_marker).read().strip()
    return f"[{ip}]" if ":" in ip else ip

def judge_base():
    if args.upstream_marker:
        return f"http://{infra_ip()}:8001/v1"
    return args.upstream

def search_base():
    return f"http://{infra_ip()}:8000"

def is_judge_model(name):
    return name in JUDGE_ALIASES or (name or "").startswith("gpt-")

async def upstream_chat(body):
    body = dict(body)
    if args.agent_upstream and not is_judge_model(body.get("model")):
        if args.force_agent_temperature is not None:
            body["temperature"] = args.force_agent_temperature
        r = await client.post(f"{args.agent_upstream}/chat/completions", json=body)
    else:
        body["model"] = args.model
        r = await client.post(f"{judge_base()}/chat/completions", json=body)
    r.raise_for_status()
    return r.json()


@app.post("/v1/chat/completions")
async def openai_dialect(request: Request):
    body = await request.json()
    out = await upstream_chat(body)
    audit("openai", body, out["choices"][0]["message"].get("content"))
    return JSONResponse(out)

@app.get("/v1/models")
async def models():
    # advertise every alias the repo might ask for
    now = int(time.time())
    names = [args.model, "gpt-5-nano", "gpt-4o-mini", "gpt-4.1"]
    return {"object": "list", "data": [{"id": n, "object": "model", "created": now, "owned_by": "shim"} for n in names]}

@app.post("/chat")
async def custom_dialect(request: Request):
    body = await request.json()
    out = await upstream_chat(body)
    content = out["choices"][0]["message"].get("content") or ""
    audit("custom", body, content)
    return JSONResponse({"content": content, "id": str(uuid.uuid4())})


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def search_proxy(path: str, request: Request):
    """Catch-all: forward non-judge routes (e.g. /search, /open) to the search server.
    Only active with --upstream-marker; resolves infra IP per request."""
    if not args.upstream_marker:
        return JSONResponse({"error": "no upstream marker configured"}, status_code=502)
    body = await request.body()
    r = await client.request(request.method, f"{search_base()}/{path}",
                             content=body, headers={"Content-Type": request.headers.get("Content-Type", "application/json")})
    return JSONResponse(r.json(), status_code=r.status_code)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
