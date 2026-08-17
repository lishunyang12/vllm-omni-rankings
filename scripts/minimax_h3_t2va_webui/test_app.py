from __future__ import annotations

import importlib
import json
import os

import httpx
from fastapi.testclient import TestClient


os.environ["H3_UI_TOKEN"] = "test-token"
app_module = importlib.import_module("app")


def test_authenticated_generation_flow() -> None:
    job = {
        "id": "video_gen_test",
        "status": "queued",
        "prompt": "A lighthouse in a storm",
        "model": "MiniMax-H3",
        "object": "video",
        "seconds": "5",
        "quality": "high",
        "progress": 0,
        "media_type": "video/mp4",
        "created_at": 0,
        "stage_durations": {},
        "peak_memory_mb": 0,
    }

    def backend(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json=job)
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"mp4", headers={"content-type": "video/mp4"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": job["id"], "deleted": True})
        return httpx.Response(200, json={**job, "status": "completed", "progress": 100})

    with TestClient(app_module.app) as client:
        replacement = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        app_module.app.state.client = replacement

        assert client.get("/").status_code == 401
        page = client.get("/?token=test-token")
        assert page.status_code == 200
        assert "MiniMax-H3 文生视频" in page.text

        created = client.post(
            "/api/videos",
            data={"prompt": "A lighthouse in a storm", "seconds": "5", "seed": "1101"},
        )
        assert created.status_code == 200
        assert created.json()["id"] == job["id"]
        assert client.get(f"/api/videos/{job['id']}").json()["status"] == "completed"
        assert client.get(f"/api/videos/{job['id']}/content").content == b"mp4"
        assert client.delete(f"/api/videos/{job['id']}").status_code == 200
        assert client.post("/api/videos", data={"prompt": "x", "seconds": "16"}).status_code == 422

        asyncio_client = app_module.app.state.client
        assert asyncio_client is replacement


def test_generation_payload_is_fixed_high_quality() -> None:
    captured = {}

    def backend(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "id": "video_gen_payload",
                "status": "queued",
                "prompt": "test",
                "model": "MiniMax-H3",
                "object": "video",
                "seconds": "7",
                "quality": "high",
                "progress": 0,
                "media_type": "video/mp4",
                "created_at": 0,
                "stage_durations": {},
                "peak_memory_mb": 0,
            },
        )

    app_module.active_job_id = None
    with TestClient(app_module.app) as client:
        app_module.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        client.get("/?token=test-token")
        response = client.post("/api/videos", data={"prompt": "test", "seconds": "7", "seed": "42"})
        assert response.status_code == 200

    body = captured["body"]
    assert b"1344" in body
    assert b"768" in body
    assert b"50" in body
    assert b"audio_flow_shift" in body
    assert json.dumps(7).encode() in body
