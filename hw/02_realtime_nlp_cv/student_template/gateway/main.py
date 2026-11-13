"""Real-time inference gateway — СКЕЛЕТ.

Готовы /health и /metrics. Остальные ручки реализуйте сами по контракту
(API_CONTRACT.md в корне этого каталога):
    /triton/health, /vllm/health, /metrics/triton, /metrics/vllm, /metrics/node,
    /info (Bearer, с полем gpu), /vision (Bearer, модель вашего варианта — make variant),
    /chat (Bearer, SSE).

Как структурировать код (auth, конфиг, инференс) — на ваше усмотрение.
"""
from __future__ import annotations

import os
from io import BytesIO
from time import perf_counter

import httpx
import jwt
import numpy as np
import tritonclient.http as triton
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool


JWT_SECRET = os.environ["JWT_SECRET"]
CV_MODEL = os.environ["CV_MODEL"]
LLM_MODEL = os.environ["LLM_MODEL"]
STUDENT_NAME = os.environ["STUDENT_NAME"]
STUDENT_EMAIL = os.environ["STUDENT_EMAIL"]
GPU_NAME = os.environ["GPU_NAME"]
TRITON_URL = os.environ["TRITON_URL"]
VLLM_URL = os.environ["VLLM_URL"]
METRICS_URLS = {
    "triton": os.environ["TRITON_METRICS_URL"],
    "vllm": os.environ["VLLM_METRICS_URL"],
    "node": os.environ["NODE_METRICS_URL"],
}

with open("./imagenet_classes.txt", "r", encoding="UTF-8") as f:
    LABELS = [line.rstrip('\n') for line in f]

app = FastAPI(title="Real-time Inference Gateway")
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
bearer = HTTPBearer(auto_error=False)
http = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5), trust_env=False)


async def check_token(token: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if token is None:
        raise HTTPException(401, "Bearer token required")
    try:
        jwt.decode(token.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid or expired token") from None


def preprocess_image(data):
    with Image.open(BytesIO(data)) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        size = (256, int(256 * height / width)) if width <= height else (int(256 * width / height), 256)
        image = image.resize(size, Image.Resampling.BICUBIC)
        left, top = round((size[0] - 224) / 2), round((size[1] - 224) / 2)
        image = image.crop((left, top, left + 224, top + 224))
        x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return np.array(((x - mean) / std)[None])


def predict(x):
    with triton.InferenceServerClient(TRITON_URL, connection_timeout=5, network_timeout=30) as client:
        inputs = triton.InferInput("INPUT__0", x.shape, "FP32")
        inputs.set_data_from_numpy(x)
        result = client.infer(CV_MODEL, [inputs], outputs=[
            triton.InferRequestedOutput("OUTPUT__0", binary_data=False)
        ])
    logits = result.as_numpy("OUTPUT__0").reshape(1000)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return [
        {"class_id": int(i), "label": LABELS[i], "score": float(probabilities[i])}
        for i in np.argsort(-probabilities)[:5]
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/triton/health")
async def triton_health():
    try:
        r = await http.get(f"http://{TRITON_URL}/v2/health/ready", timeout=3)
        ready = r.status_code == 200
    except httpx.HTTPError:
        ready = False
    return JSONResponse(
        {"backend": "triton", "ready": ready, "model": CV_MODEL},
        status_code=200 if ready else 503,
    )


@app.get("/vllm/health")
async def vllm_health():
    try:
        r = await http.get(f"{VLLM_URL}/v1/models", timeout=3)
        r.raise_for_status()
        models = [model["id"] for model in r.json()["data"]]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        models = []
    ready = LLM_MODEL in models
    return JSONResponse(
        {"backend": "vllm", "ready": ready, "models": models},
        status_code=200 if ready else 503,
    )


@app.get("/metrics/{backend}")
async def backend_metrics(backend: str):
    if backend not in METRICS_URLS:
        raise HTTPException(404, "Unknown backend")
    try:
        r = await http.get(METRICS_URLS[backend], timeout=5)
        r.raise_for_status()
    except httpx.HTTPError:
        raise HTTPException(502, "Metrics backend unavailable") from None
    return Response(r.content, headers={"Content-Type": r.headers.get("content-type", "text/plain")})


@app.get("/info", dependencies=[Depends(check_token)])
async def info():
    return {
        "student": {"name": STUDENT_NAME, "email": STUDENT_EMAIL},
        "models": {"vision": CV_MODEL, "llm": LLM_MODEL},
        "gpu": GPU_NAME,
        "version": "1.0",
    }


@app.post("/vision", dependencies=[Depends(check_token)])
async def vision(file: UploadFile = File(...)):
    started = perf_counter()
    try:
        x = await run_in_threadpool(preprocess_image, await file.read())
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise HTTPException(400, "Empty or invalid image") from None
    try:
        predictions = await run_in_threadpool(predict, x)
    except Exception:
        raise HTTPException(502, "Triton inference failed") from None
    return {
        "model": CV_MODEL,
        "predictions": predictions,
        "latency_ms": (perf_counter() - started) * 1000,
    }


@app.post("/chat", dependencies=[Depends(check_token)])
async def chat(request: Request):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "Invalid JSON") from None
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        raise HTTPException(400, "messages must be a nonempty list")
    if not isinstance(body.get("stream", True), bool):
        raise HTTPException(400, "stream must be true or false")
    # body.setdefault("model", LLM_MODEL)
    # body.setdefault("max_tokens", 128)
    # body.setdefault("stream", True)

    try:
        r = await http.send(http.build_request(
            "POST", f"{VLLM_URL}/v1/chat/completions", json=body
        ), stream=True)
    except httpx.HTTPError:
        raise HTTPException(502, "vLLM unavailable") from None

    # Проверяем статус до отправки клиенту HTTP 200 и начала SSE.
    if not r.is_success:
        status = 400 if 400 <= r.status_code < 500 else 502
        await r.aclose()
        raise HTTPException(status, "vLLM rejected the request")
    if not body["stream"]:
        try:
            return Response(await r.aread(), media_type="application/json")
        except httpx.HTTPError:
            raise HTTPException(502, "vLLM response interrupted") from None
        finally:
            await r.aclose()

    async def chunks():
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        except httpx.HTTPError:
            yield b'data: {"error":{"message":"vLLM stream interrupted"}}\n\n'

    return StreamingResponse(
        chunks(), media_type="text/event-stream", background=BackgroundTask(r.aclose)
    )