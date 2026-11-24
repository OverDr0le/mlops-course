"""Нагрузка через публичные ручки gateway: httpx==0.27.0, Python 3.10+."""
import argparse
import asyncio
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import httpx


def p95(values):
    return round(sorted(values)[math.ceil(.95 * len(values)) - 1], 2) if values else None


def model_counter(text, metric, model):
    pattern = rf'^{re.escape(metric)}\{{[^}}]*\bmodel="{re.escape(model)}"[^}}]*\}}\s+(\S+)'
    values = re.findall(pattern, text, flags=re.MULTILINE)
    if not values:
        raise ValueError(f"В метриках нет {metric} для {model}")
    return sum(float(value) for value in values)


async def main(args):
    image_path = Path(args.image)
    image = image_path.read_bytes() if args.mode != "chat" else b""
    records = {kind: {"requests": 0, "times": []} for kind in ("vision", "chat")}
    errors = Counter()
    server_errors = 0
    chat_number = 0
    limits = httpx.Limits(max_connections=args.concurrency + 5,
                         max_keepalive_connections=args.concurrency + 5)
    async with httpx.AsyncClient(
        base_url=args.base.rstrip("/"), headers={"Authorization": f"Bearer {args.token}"},
        limits=limits, timeout=httpx.Timeout(60, connect=5), trust_env=False,
    ) as client:

        async def snapshot():
            r = await client.get("/metrics/triton")
            r.raise_for_status()
            return [model_counter(r.text, name, args.cv_model)
                    for name in ("nv_inference_count", "nv_inference_exec_count")]

        async def request(kind):
            nonlocal chat_number
            started = perf_counter()
            if kind == "vision":
                r = await client.post("/vision", files={"file": (image_path.name, image)})
                r.raise_for_status()
                body = r.json()
                if body.get("model") != args.cv_model or len(body.get("predictions", [])) != 5:
                    raise ValueError("Некорректный ответ /vision")
                return (perf_counter() - started) * 1000

            first_token = None
            chunks = 0
            done = False
            chat_number += 1
            async with client.stream("POST", "/chat", json={
                "messages": [{"role": "user", "content":
                              f"Request {chat_number}: explain in two short sentences why the sky is blue."}],
                "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
            }) as r:
                r.raise_for_status()
                if not r.headers.get("content-type", "").startswith("text/event-stream"):
                    raise ValueError("/chat вернул не SSE")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    if not data:
                        continue
                    event = json.loads(data)
                    if "error" in event:
                        raise ValueError("Ошибка внутри SSE")
                    chunks += 1
                    choices = event.get("choices", [])
                    content = choices[0].get("delta", {}).get("content") if choices else None
                    if content and first_token is None:
                        first_token = (perf_counter() - started) * 1000
            if not done or first_token is None or chunks < 2:
                raise ValueError("Нет текста, двух SSE-чанков или [DONE]")
            return first_token

        # Прогрев не входит ни в перцентили, ни в разности счётчиков.
        for kind in ("vision", "chat") if args.mode == "mixed" else (args.mode,):
            for _ in range(3 if kind == "vision" else 1):
                await request(kind)
        before = await snapshot()
        started = perf_counter()
        deadline = started + args.seconds

        async def worker(worker_id):
            nonlocal server_errors
            step = 0
            while perf_counter() < deadline:
                kind = args.mode
                if kind == "mixed":
                    kind = "vision" if (worker_id + step) % 2 == 0 else "chat"
                step += 1
                records[kind]["requests"] += 1
                try:
                    records[kind]["times"].append(await request(kind))
                except httpx.HTTPStatusError as e:
                    errors[f"{kind}: HTTP {e.response.status_code}"] += 1
                    server_errors += 500 <= e.response.status_code < 600
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                    errors[f"{kind}: {type(e).__name__}: {str(e)[:120]}"] += 1

        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
        elapsed = perf_counter() - started
        after = await snapshot()

    inf, executions = (after[i] - before[i] for i in range(2))
    if inf < 0 or executions < 0:
        raise RuntimeError("Счётчики сбросились во время теста: повтори без рестарта Triton")
    report = {
        "mode": args.mode, "concurrency": args.concurrency,
        "load_seconds": args.seconds, "elapsed_with_drain_s": round(elapsed, 2),
    }
    for kind, record in records.items():
        n = record["requests"]
        if n:
            report[kind] = {
                "requests": n, "successful": len(record["times"]),
                "success_rate": len(record["times"]) / n,
                "p95_ms" if kind == "vision" else "ttft_p95_ms": p95(record["times"]),
            }
    n = sum(record["requests"] for record in records.values())
    successes = sum(len(record["times"]) for record in records.values())
    report["overall_success_rate"] = successes / n if n else None
    report["http_5xx"] = server_errors
    report["errors"] = dict(errors)
    report["triton"] = {
        "inferences_delta": inf, "executions_delta": executions,
        "avg_batch": inf / executions if executions else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.getenv("BASE", "http://127.0.0.1:8080"))
    parser.add_argument("--token", default=os.getenv("TOKEN"))
    parser.add_argument("--image", default="sample.jpg")
    parser.add_argument("--cv-model", default="efficientnet_b0")
    parser.add_argument("--mode", choices=["vision", "chat", "mixed"], default="mixed")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if not args.token:
        parser.error("Задай TOKEN в окружении или передай --token")
    if args.seconds <= 0 or args.concurrency < 1 or args.max_tokens < 1:
        parser.error("seconds, concurrency и max-tokens должны быть положительными")
    asyncio.run(main(args))
