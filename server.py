"""
Legal RAG Chatbot — FastAPI Server
=====================================
Main server tích hợp: Hybrid Retrieval + Ollama Chat + SSE Streaming + Ngrok.

Usage:
    python server.py                      # Local HTTP (port 8000)
    python server.py --ngrok              # Expose qua ngrok
    python server.py --port 8080          # Custom port
"""

import argparse
import asyncio
import json
import sys
import time
import os
from pathlib import Path
from urllib import request as urllib_request, error as urllib_error

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# ── Project setup ──
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import cfg
from src.llm.prompt_builder import build_rag_prompt, build_search_summary

# ── Logger ──
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
log_file = cfg.LOG_DIR / "server.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(log_file), level="DEBUG", rotation="10 MB")

# ── FastAPI App ──
app = FastAPI(title="Legal RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──
_retriever = None
_chat_sessions: dict[str, list] = {}  # session_id → history


def _init_retriever():
    """Khởi tạo retrieval engine."""
    global _retriever
    try:
        from src.retrieval.hybrid import HybridRetriever
        _retriever = HybridRetriever(
            bm25_index_dir=cfg.BM25_INDEX_DIR,
            vector_index_dir=cfg.VECTOR_INDEX_DIR,
            chunks_path=cfg.PROCESSED_DIR / "chunks.jsonl",
            ollama_url=cfg.OLLAMA_BASE_URL,
            embedding_model=cfg.EMBEDDING_MODEL,
            rrf_k=cfg.RRF_K,
        )
        logger.info(f"Retriever initialized: mode={_retriever.mode}")
    except Exception as e:
        logger.error(f"Retriever init failed: {e}")
        _retriever = None


# ── API Routes ──

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve frontend HTML."""
    html_path = _PROJECT_ROOT / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Legal RAG Chatbot</h1><p>web/index.html not found</p>")


@app.get("/api/health")
async def health_check():
    """System health check."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=3)
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "retriever": _retriever.mode if _retriever else "not_loaded",
        "ollama": "online" if ollama_ok else "offline",
        "chat_model": cfg.CHAT_MODEL,
        "embedding_model": cfg.EMBEDDING_MODEL,
    }


@app.post("/api/search")
async def search_endpoint(request: Request):
    """Retrieval-only search (không gọi LLM)."""
    data = await request.json()
    query = data.get("query", "").strip()
    top_k = data.get("top_k", cfg.HYBRID_TOP_K)

    if not query:
        return JSONResponse({"error": "Query is empty"}, status_code=400)

    if not _retriever:
        return JSONResponse({"error": "Retriever not loaded"}, status_code=503)

    t0 = time.time()
    results = _retriever.search(
        query, cfg.BM25_TOP_K, cfg.VECTOR_TOP_K, top_k
    )
    elapsed = time.time() - t0

    return {
        "query": query,
        "results": results,
        "total": len(results),
        "mode": _retriever.mode,
        "elapsed_ms": round(elapsed * 1000),
    }


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Chat endpoint với SSE streaming.
    Trả response dạng text/event-stream.
    """
    data = await request.json()
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default")

    if not query:
        return JSONResponse({"error": "Query is empty"}, status_code=400)

    # Get/create session history
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = []
    history = _chat_sessions[session_id]

    async def event_generator():
        try:
            # Step 1: Retrieve context
            contexts = []
            if _retriever:
                t0 = time.time()
                contexts = _retriever.search(
                    query, cfg.BM25_TOP_K, cfg.VECTOR_TOP_K, cfg.HYBRID_TOP_K
                )
                retrieval_ms = round((time.time() - t0) * 1000)

                # Send sources info
                sources = build_search_summary(contexts)
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'retrieval_ms': retrieval_ms}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'retrieval_ms': 0}, ensure_ascii=False)}\n\n"

            # Step 2: Build prompt
            messages = build_rag_prompt(query, contexts, history)

            # Step 3: Stream from Ollama
            full_response = ""
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{cfg.OLLAMA_BASE_URL}/api/chat",
                        json={
                            "model": cfg.CHAT_MODEL,
                            "messages": messages,
                            "stream": True,
                            "options": {
                                "temperature": 0.3,
                                "top_p": 0.9,
                                "num_predict": 2048,
                            },
                        },
                        timeout=120,
                    ) as response:
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    token = chunk["message"]["content"]
                                    full_response += token
                                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    ConnectionRefusedError, OSError) as e:
                logger.warning(f"Ollama connection failed: {type(e).__name__}: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Ollama server không phản hồi. Hãy chạy: ollama serve && ollama pull ' + cfg.CHAT_MODEL}, ensure_ascii=False)}\n\n"
                return
            except Exception as e:
                logger.error(f"Ollama chat error: {type(e).__name__}: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Lỗi LLM: {type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"
                return

            # Step 4: Save to history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": full_response})

            # Trim history
            if len(history) > cfg.MAX_HISTORY_TURNS * 2:
                _chat_sessions[session_id] = history[-(cfg.MAX_HISTORY_TURNS * 2):]

            # Done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/clear")
async def clear_session(request: Request):
    """Clear chat history."""
    data = await request.json()
    session_id = data.get("session_id", "default")
    _chat_sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    """List active sessions."""
    return {
        "sessions": [
            {"id": sid, "turns": len(h) // 2}
            for sid, h in _chat_sessions.items()
        ]
    }


# ── Ngrok integration (from Meeting-trans) ──

def _normalize_ngrok_addr(addr) -> str:
    if addr is None:
        return ""
    addr = str(addr).strip()
    for prefix in ("http://", "https://"):
        if addr.startswith(prefix):
            addr = addr[len(prefix):]
    return addr.rstrip("/")


def _desired_ngrok_addrs(port: int) -> set:
    return {str(port), f"localhost:{port}", f"127.0.0.1:{port}", f"0.0.0.0:{port}"}


def _list_local_ngrok_tunnels() -> list:
    req = urllib_request.Request("http://127.0.0.1:4040/api/tunnels", method="GET")
    try:
        with urllib_request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("tunnels", []) or []
    except Exception:
        return []


def _delete_local_ngrok_tunnel(uri: str) -> bool:
    if not uri:
        return False
    req = urllib_request.Request(f"http://127.0.0.1:4040{uri}", method="DELETE")
    try:
        with urllib_request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def _find_local_ngrok_tunnel_for_port(port: int):
    desired = _desired_ngrok_addrs(port)
    for tunnel in _list_local_ngrok_tunnels():
        addr = _normalize_ngrok_addr((tunnel.get("config") or {}).get("addr"))
        if addr in desired:
            return tunnel
    return None


def _close_conflicting_tunnels(port: int) -> int:
    desired = _desired_ngrok_addrs(port)
    closed = 0
    for tunnel in _list_local_ngrok_tunnels():
        addr = _normalize_ngrok_addr((tunnel.get("config") or {}).get("addr"))
        if addr in desired:
            continue
        if _delete_local_ngrok_tunnel(tunnel.get("uri", "")):
            closed += 1
    return closed


def start_server(host: str, port: int, use_ngrok: bool = False):
    """Khởi động server."""
    import time as _time

    cfg.ensure_dirs()

    # Init retriever
    _init_retriever()

    print("=" * 60)
    print("  🏛️  LEGAL RAG CHATBOT")
    print(f"  Mode: {'HYBRID (BM25 + Vector)' if _retriever and _retriever.mode == 'hybrid' else 'BM25 ONLY'}")
    print(f"  Chat model: {cfg.CHAT_MODEL}")
    print(f"  Embedding: {cfg.EMBEDDING_MODEL}")
    print(f"  Local: http://{host}:{port}")
    print("=" * 60)

    # Ngrok tunnel
    if use_ngrok:
        try:
            from pyngrok import ngrok, conf as ngrok_conf
            tunnel = None

            # Set authtoken từ .env
            ngrok_token = os.getenv("ngrok_token", "").strip()
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
                print(f"[NGROK] Auth token set từ .env")
            else:
                print("[NGROK] Không tìm thấy ngrok_token trong .env, dùng config mặc định")

            # Đóng tất cả tunnel cũ trên port này để tránh xung đột
            existing = _find_local_ngrok_tunnel_for_port(port)
            if existing:
                print(f"[NGROK] Đóng tunnel cũ trên port {port}...")
                try:
                    ngrok.disconnect(existing.get("public_url", ""))
                except Exception:
                    _close_conflicting_tunnels(port)
                _time.sleep(1)

            # Tạo tunnel mới
            try:
                tunnel = ngrok.connect(port, "http")
            except Exception as e:
                err_msg = str(e)
                if "ERR_NGROK_334" in err_msg or "already online" in err_msg:
                    print("[NGROK] Tunnel bị xung đột, đang dọn dẹp...")
                    _close_conflicting_tunnels(port)
                    _time.sleep(2)
                    tunnel = ngrok.connect(port, "http")
                else:
                    raise

            print(f"\n{'=' * 60}")
            print(f"  🌐 NGROK TUNNEL ACTIVE")
            print(f"  Public URL: {tunnel.public_url}")
            print(f"  Chia sẻ URL này để demo!")
            print(f"{'=' * 60}\n")
        except ImportError:
            print("[LỖI] Cần cài pyngrok: pip install pyngrok")
        except Exception as e:
            print(f"[LỖI] Không thể tạo ngrok tunnel: {e}")
            print("       Kiểm tra ngrok_token trong .env")

    # Filter noisy logs
    import logging

    class _Filter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage() if hasattr(record, 'getMessage') else ""
            return "/api/health" not in msg

    logging.getLogger("uvicorn.access").addFilter(_Filter())

    uvicorn.run(app=app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal RAG Chatbot Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8899,
                        help="Port (mặc định 8899 để tránh xung đột với apps khác)")
    parser.add_argument("--ngrok", action="store_true",
                        help="Tạo ngrok tunnel (truy cap từ Internet)")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port, use_ngrok=args.ngrok)
