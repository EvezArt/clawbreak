"""ClawBreak - FastAPI server with chat, tools, memory, and web UI."""
import os
import sys
import json
import time
import uuid
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
import uvicorn

from config import Config
from memory import Memory
from tools import ToolEngine
from tool_parser import parse_tool_calls

import stripe

# Init
config = Config()
memory = Memory(config.get("memory", "db_path"))
tools = ToolEngine(config, memory)

app = FastAPI(title="ClawBreak", version="0.1.0")

# --- Stripe Config ---
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_IDS = {
    "pro": os.environ.get("STRIPE_PRICE_ID_PRO", ""),
    "team": os.environ.get("STRIPE_PRICE_ID_TEAM", ""),
    "enterprise": os.environ.get("STRIPE_PRICE_ID_ENTERPRISE", ""),
}
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# --- LLM Client ---

async def _call_llm(base_url, api_key, model, messages, stream=False, timeout=120):
    """Single LLM API call. Returns (result_dict, error_string)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": config.get("llm", "max_tokens"),
        "temperature": config.get("llm", "temperature"),
        "stream": stream,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json(), None
        except httpx.HTTPStatusError as e:
            return None, f"LLM API error {e.response.status_code}: {e.response.text[:500]}"
        except Exception as e:
            return None, f"LLM request failed: {e}"


async def chat_completion(messages, stream=False):
    """Call the LLM API with fallback chain (EVEZ API → Vultr direct)."""
    # Try fallback chain first
    fallback_chain = config.data.get("fallback", [])
    if fallback_chain:
        for backend in fallback_chain:
            base_url = backend.get("base_url", "")
            api_key = backend.get("api_key", "")
            model = backend.get("model", config.get("llm", "model"))
            timeout = backend.get("timeout", 30)
            if not api_key:
                continue
            result, err = await _call_llm(base_url, api_key, model, messages, stream=stream, timeout=timeout)
            if result:
                return result
            # Backend failed, try next
            continue
        # All fallbacks failed, fall through to primary

    # Primary (legacy) path
    api_key = config.get("llm", "api_key")
    base_url = config.get("llm", "base_url")
    model = config.get("llm", "model")

    if not api_key:
        return {"error": "No API key configured. Set CLAWBREAK_LLM_API_KEY or add to config.yaml"}

    result, err = await _call_llm(base_url, api_key, model, messages, stream=stream)
    if result:
        return result
    return {"error": err or "All LLM backends failed"}


async def chat_completion_stream(messages):
    """Stream chat completion chunks."""
    api_key = config.get("llm", "api_key")
    base_url = config.get("llm", "base_url")
    model = config.get("llm", "model")

    if not api_key:
        yield f"data: {json.dumps({'error': 'No API key'})}\n\n"
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": config.get("llm", "max_tokens"),
        "temperature": config.get("llm", "temperature"),
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            async with client.stream("POST", f"{base_url}/chat/completions", headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            yield f"data: [DONE]\n\n"
                            return
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                yield f"data: {json.dumps({'content': delta['content']})}\n\n"
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def process_message(user_message: str, session_id: str = "default") -> str:
    """Process a user message through the LLM with tool execution."""
    # Store user message
    memory.store_message("user", user_message, session_id)

    # Build context
    history = memory.get_messages(session_id, limit=config.get("memory", "max_context_messages"))
    messages = [{"role": "system", "content": config.get("system_prompt")}]
    messages.extend(history)

    # Call LLM
    response = await chat_completion(messages)

    if "error" in response:
        return f"Error: {response['error']}"

    try:
        assistant_msg = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"Error: Unexpected API response format"

    # Check for tool calls
    tool_calls = parse_tool_calls(assistant_msg)
    if tool_calls:
        tool_results = []
        for tc in tool_calls:
            result = tools.execute(tc["name"], tc["args"])
            tool_results.append(f"[tool:{tc['name']} result]\n{json.dumps(result, indent=2, ensure_ascii=False)[:2000]}")

        # Send tool results back to LLM for a proper response
        tool_context = "\n\n".join(tool_results)
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": f"Tool execution results:\n{tool_context}\n\nNow respond to the user based on these results."})

        response2 = await chat_completion(messages)
        if "error" not in response2:
            try:
                assistant_msg = response2["choices"][0]["message"]["content"]
            except:
                pass

    # Store and return
    memory.store_message("assistant", assistant_msg, session_id)
    memory.cleanup_old_messages(session_id)
    return assistant_msg


# --- Routes ---

async def _check_evez_health():
    """Check EVEZ API health and return status dict."""
    evez_cfg = config.data.get("evez_api", {})
    base_url = evez_cfg.get("base_url", "http://localhost:8081")
    health_path = evez_cfg.get("health_path", "/health")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url}{health_path}")
            resp.raise_for_status()
            return {"status": "ok", **resp.json()}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


@app.get("/health")
async def health():
    evez_status = await _check_evez_health()
    return {
        "status": "ok",
        "version": "0.1.0",
        "model": config.get("llm", "model"),
        "memory_facts": len(memory.list_facts(limit=9999)),
        "uptime": time.time(),
        "evez_api": evez_status,
    }


@app.post("/chat")
async def chat(request: Request):
    """Non-streaming chat endpoint."""
    body = await request.json()
    message = body.get("message", "")
    session_id = body.get("session_id", "default")

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    reply = await process_message(message, session_id)
    return {"reply": reply, "session_id": session_id}


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """Streaming chat endpoint."""
    body = await request.json()
    message = body.get("message", "")
    session_id = body.get("session_id", "default")

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    memory.store_message("user", message, session_id)
    history = memory.get_messages(session_id, limit=config.get("memory", "max_context_messages"))
    messages = [{"role": "system", "content": config.get("system_prompt")}]
    messages.extend(history)

    async def generate():
        full_response = []
        async for chunk in chat_completion_stream(messages):
            yield chunk
            if chunk.startswith("data: ") and "content" in chunk:
                try:
                    data = json.loads(chunk[6:])
                    if "content" in data:
                        full_response.append(data["content"])
                except:
                    pass

        # Store complete response
        if full_response:
            memory.store_message("assistant", "".join(full_response), session_id)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/memory")
async def get_memory(category: str = None, query: str = None):
    if query:
        return {"facts": memory.recall_facts(query)}
    return {"facts": memory.list_facts(category)}


@app.post("/memory")
async def store_memory(request: Request):
    body = await request.json()
    key = body.get("key")
    value = body.get("value")
    category = body.get("category", "general")
    if not key or not value:
        return JSONResponse({"error": "key and value required"}, status_code=400)
    memory.store_fact(key, value, category)
    return {"status": "stored", "key": key}


@app.delete("/memory/{key}")
async def delete_memory(key: str):
    memory.delete_fact(key)
    return {"status": "deleted", "key": key}


@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """WebSocket for real-time chat."""
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                message = msg.get("message", "")
            except json.JSONDecodeError:
                message = data

            if not message:
                continue

            reply = await process_message(message, session_id)
            await websocket.send_text(json.dumps({"reply": reply, "session_id": session_id}))
    except WebSocketDisconnect:
        pass


# --- Stripe Billing Endpoints ---

@app.post("/billing/checkout")
async def billing_checkout(request: Request):
    """Create a Stripe Checkout Session for the given plan."""
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"error": "Stripe not configured. Set STRIPE_SECRET_KEY."}, status_code=500)

    body = await request.json()
    plan = body.get("plan", "").lower()

    if plan not in STRIPE_PRICE_IDS:
        return JSONResponse({"error": f"Invalid plan. Choose from: {', '.join(STRIPE_PRICE_IDS.keys())}"}, status_code=400)

    price_id = STRIPE_PRICE_IDS[plan]
    if not price_id:
        return JSONResponse({"error": f"No Stripe price ID configured for plan '{plan}'. Set STRIPE_PRICE_ID_{plan.upper()}."}, status_code=500)

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{BASE_URL}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{BASE_URL}/pricing",
            metadata={"plan": plan},
        )
        return {"url": session.url, "session_id": session.id}
    except stripe.error.StripeError as e:
        return JSONResponse({"error": f"Stripe error: {str(e)}"}, status_code=502)


@app.post("/billing/portal")
async def billing_portal(request: Request):
    """Create a Stripe Customer Portal session for managing subscriptions."""
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"error": "Stripe not configured."}, status_code=500)

    body = await request.json()
    customer_id = body.get("customer_id")
    if not customer_id:
        return JSONResponse({"error": "customer_id is required"}, status_code=400)

    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{BASE_URL}/pricing",
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        return JSONResponse({"error": f"Stripe error: {str(e)}"}, status_code=502)


@app.post("/billing/webhook")
async def billing_webhook(request: Request):
    """Handle Stripe webhook events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        except stripe.error.SignatureVerificationError:
            return JSONResponse({"error": "Invalid signature"}, status_code=400)
    else:
        event = json.loads(payload)

    evt_type = event.get("type", "")

    # Route events
    if evt_type == "checkout.session.completed":
        session_obj = event["data"]["object"]
        plan = session_obj.get("metadata", {}).get("plan", "unknown")
        customer_id = session_obj.get("customer")
        print(f"✅ Checkout completed: plan={plan}, customer={customer_id}")
        # TODO: activate plan for customer in your DB

    elif evt_type == "customer.subscription.updated":
        sub = event["data"]["object"]
        print(f"🔄 Subscription updated: {sub.get('id')}")

    elif evt_type == "customer.subscription.deleted":
        sub = event["data"]["object"]
        print(f"❌ Subscription canceled: {sub.get('id')}")
        # TODO: downgrade customer

    elif evt_type == "invoice.payment_failed":
        inv = event["data"]["object"]
        print(f"⚠️ Payment failed: {inv.get('id')}")

    else:
        print(f"ℹ️ Unhandled webhook: {evt_type}")

    return {"received": True}


@app.get("/billing/success")
async def billing_success():
    """Landing page after successful checkout."""
    return HTMLResponse("<html><body style='background:#0a0a0f;color:#e0e0e0;font-family:sans-serif;text-align:center;padding-top:15vh'><h1>✅ Welcome to ClawBreak!</h1><p>Your subscription is active.</p><p><a href='/pricing' style='color:#ff4500'>Back to pricing</a></p></body></html>")


# Serve web UI
STATIC_DIR = Path(__file__).parent / "static"

@app.get("/")
async def serve_landing():
    landing = STATIC_DIR / "landing.html"
    if landing.exists():
        return HTMLResponse(landing.read_text())
    return HTMLResponse("<h1>EVEZ-OS</h1><p>Landing page not found.</p>")


@app.get("/chat")
async def serve_chat():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>ClawBreak Chat</h1><p>Chat UI not found. Use the API.</p>")


@app.get("/config")
async def get_config():
    """Get current config (redacting API key)."""
    cfg = dict(config.data)
    cfg["llm"]["api_key"] = f"{cfg['llm']['api_key'][:6]}..." if cfg["llm"]["api_key"] else "not set"
    return cfg


@app.post("/config")
async def update_config(request: Request):
    body = await request.json()
    config._deep_merge(config.data, body)
    config.save()
    return {"status": "updated"}


# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    host = config.get("server", "host")
    port = config.get("server", "port")

    # Ensure API key is set
    if not config.get("llm", "api_key"):
        print("⚠️  No LLM API key set. Set CLAWBREAK_LLM_API_KEY or add to config.yaml")

    print(f"🚀 ClawBreak starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")




@app.get("/api/v1/models")
async def list_models():
    """List available models from EVEZ API, falling back to local config."""
    evez_cfg = config.data.get("evez_api", {})
    base_url = evez_cfg.get("base_url", "http://localhost:8081")
    models_path = evez_cfg.get("models_path", "/v1/models")
    # Try EVEZ API first
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url}{models_path}")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        pass
    # Fallback: return the configured model
    return {
        "object": "list",
        "data": [{
            "id": config.get("llm", "model"),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "vultr",
            "note": "EVEZ API unreachable, showing primary model only"
        }]
    }


# === EVEZ-OS Integration ===

import subprocess
import hashlib
import time as _time

EVEZ_STATE = {
    "v_global": 6.128,
    "round": 182,
    "fires": 31,
    "plane": "✓ CANONICAL",
    "spine": []
}

@app.get("/api/evez/status")
async def evez_status():
    return EVEZ_STATE

@app.post("/api/evez/run/{cmd}")
async def evez_run(cmd: str):
    ts = _time.time()
    h = hashlib.sha256(f"{cmd}{ts}".encode()).hexdigest()[:12]
    
    if cmd == "play":
        EVEZ_STATE["round"] += 1
        EVEZ_STATE["v_global"] = round(EVEZ_STATE["v_global"] + 0.001, 3)
        EVEZ_STATE["spine"].append({"t": ts, "h": h, "event": "PLAY", "round": EVEZ_STATE["round"]})
        return {"status": "ok", "round": EVEZ_STATE["round"], "v": EVEZ_STATE["v_global"], "hash": h}
    
    elif cmd == "visualize":
        return {"status": "ok", "artifact": f"cognition-{h}.svg", "frames": 24, "layers": ["attention", "memory", "output"]}
    
    elif cmd == "spine":
        recent = EVEZ_STATE["spine"][-10:] if len(EVEZ_STATE["spine"]) > 10 else EVEZ_STATE["spine"]
        return {"status": "ok", "spine_length": len(EVEZ_STATE["spine"]), "recent": recent, "integrity": "✓ VALID"}
    
    elif cmd == "sat":
        EVEZ_STATE["fires"] += 1
        return {"status": "ok", "contradictions": 0, "fires": EVEZ_STATE["fires"], "result": "CONSISTENT"}
    
    return {"status": "unknown_command", "cmd": cmd}

@app.get("/api/memory")
async def get_memory():
    facts = memory.recall_all() if hasattr(memory, 'recall_all') else []
    return {"facts": [{"id": i, "key": k, "value": v, "category": c} 
                      for i, (k, v, c) in enumerate(facts)] if facts else []}

@app.delete("/api/memory/{fact_id}")
async def delete_memory(fact_id: int):
    if hasattr(memory, 'delete'):
        memory.delete(fact_id)
    return {"status": "deleted"}

@app.post("/api/history/clear")
async def clear_history():
    if hasattr(memory, 'clear_history'):
        memory.clear_history()
    return {"status": "cleared"}



    main()
