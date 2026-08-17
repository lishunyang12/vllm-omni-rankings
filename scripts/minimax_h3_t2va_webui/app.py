from __future__ import annotations

import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response


ROOT = Path(__file__).resolve().parent
BACKEND_URL = os.environ.get("H3_BACKEND_URL", "http://127.0.0.1:18091").rstrip("/")
ACCESS_TOKEN = os.environ.get("H3_UI_TOKEN", "")
COOKIE_NAME = "h3_ui_token"

active_job_id: str | None = None
job_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=1800.0))
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def _valid_token(value: str | None) -> bool:
    return bool(ACCESS_TOKEN and value and secrets.compare_digest(value, ACCESS_TOKEN))


def require_auth(request: Request) -> None:
    if not _valid_token(request.cookies.get(COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="Invalid or missing access token")


def _backend_error(response: httpx.Response) -> Response:
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


async def _request(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> httpx.Response:
    try:
        return await client.request(method, f"{BACKEND_URL}{path}", **kwargs)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"MiniMax-H3 backend unavailable: {exc}") from exc


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, token: str | None = None) -> HTMLResponse:
    cookie_token = request.cookies.get(COOKIE_NAME)
    if not (_valid_token(token) or _valid_token(cookie_token)):
        return HTMLResponse("Access token required.", status_code=401)

    response = HTMLResponse((ROOT / "index.html").read_text(encoding="utf-8"))
    if _valid_token(token):
        response.set_cookie(
            COOKIE_NAME,
            token,
            httponly=True,
            samesite="strict",
            max_age=7 * 24 * 60 * 60,
        )
    return response


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    try:
        response = await _request(request.app.state.client, "GET", "/health")
        return {"ui": "healthy", "backend": response.status_code == 200}
    except HTTPException:
        return {"ui": "healthy", "backend": False}


@app.post("/api/videos", dependencies=[Depends(require_auth)])
async def create_video(
    request: Request,
    prompt: Annotated[str, Form(min_length=1, max_length=4000)],
    seconds: Annotated[int, Form(ge=4, le=15)] = 5,
    seed: Annotated[int | None, Form(ge=0, le=2**31 - 1)] = None,
) -> Response:
    global active_job_id

    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt cannot be empty")

    client: httpx.AsyncClient = request.app.state.client
    async with job_lock:
        if active_job_id is not None:
            status_response = await _request(client, "GET", f"/v1/videos/{active_job_id}")
            if status_response.status_code == 200:
                status = status_response.json().get("status")
                if status in {"queued", "in_progress"}:
                    raise HTTPException(status_code=409, detail=f"Generation already running: {active_job_id}")
            active_job_id = None

        fields = {
            "prompt": prompt,
            "seconds": str(seconds),
            "width": "1344",
            "height": "768",
            "aspect_ratio": "16:9",
            "fps": "24",
            "num_inference_steps": "50",
            "flow_shift": "12",
            "extra_params": json.dumps(
                {"task": "t2va", "duration": seconds, "audio_flow_shift": 3.0}, separators=(",", ":")
            ),
        }
        if seed is not None:
            fields["seed"] = str(seed)
        multipart = {name: (None, value) for name, value in fields.items()}
        response = await _request(client, "POST", "/v1/videos", files=multipart)
        if response.status_code != 200:
            return _backend_error(response)

        payload = response.json()
        active_job_id = payload["id"]
        return Response(content=response.content, media_type="application/json")


@app.get("/api/videos/{video_id}", dependencies=[Depends(require_auth)])
async def video_status(video_id: str, request: Request) -> Response:
    response = await _request(request.app.state.client, "GET", f"/v1/videos/{video_id}")
    return _backend_error(response)


@app.get("/api/videos/{video_id}/content", dependencies=[Depends(require_auth)])
async def video_content(video_id: str, request: Request) -> Response:
    response = await _request(request.app.state.client, "GET", f"/v1/videos/{video_id}/content")
    if response.status_code != 200:
        return _backend_error(response)
    return Response(
        content=response.content,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{video_id}.mp4"',
            "Cache-Control": "private, max-age=3600",
        },
    )


@app.delete("/api/videos/{video_id}", dependencies=[Depends(require_auth)])
async def cancel_video(video_id: str, request: Request) -> Response:
    global active_job_id
    response = await _request(request.app.state.client, "DELETE", f"/v1/videos/{video_id}")
    if response.status_code == 200 and active_job_id == video_id:
        active_job_id = None
    return _backend_error(response)
