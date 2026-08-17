# MiniMax-H3 interactive T2VA Web UI

Small authenticated Web UI for the resident MiniMax-H3 FL2VA asynchronous video API.

The browser can choose a prompt, duration from 4 to 15 seconds, and seed. The server fixes the validated high-quality settings to 1344x768, 24 FPS, 50 steps, video flow shift 12, and audio flow shift 3.

## Start

```bash
export H3_UI_TOKEN='replace-with-a-long-random-token'
nohup bash scripts/minimax_h3_t2va_webui/start_service.sh \
  > /home/zjy/code/lsy/runtime/minimax_h3_t2va_webui/supervisor.log 2>&1 &
```

The backend binds to loopback port 18091. The authenticated UI binds to port 8094. Open:

```text
http://SERVER:8094/?token=YOUR_TOKEN
```

The token is stored in an HttpOnly, SameSite cookie and removed from the visible URL by the page. Generated MP4 files are stored under `/home/zjy/code/lsy/runtime/minimax_h3_t2va_webui/storage`.

Only one generation may be active at a time, preventing concurrent long-duration requests from exhausting GPU memory.

## Test

```bash
PYTHONPATH=scripts/minimax_h3_t2va_webui \
  /home/zjy/code/lsy/vllm-omni/.venv/bin/python -m pytest -q \
  scripts/minimax_h3_t2va_webui/test_app.py
```
