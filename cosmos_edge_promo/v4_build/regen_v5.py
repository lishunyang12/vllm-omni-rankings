#!/usr/bin/env python3
"""Final I2V footage: coastal fp8 (matches dense repro) + a 2nd driving I2V for variety.
Passes the big official prompt/negative JSON via arg lists (no shell quoting)."""
import json, os, subprocess, sys

REPO = "/home/zjy/code/lsy/vllm-omni"
PY = "/home/zjy/code/lsy/cosmos-venv/bin/python"
OFF = "/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_official"
G = "/tmp/claude-1012/-home-zjy-code-lsy/b46991ea-8372-479e-a045-fa5064c4e953/scratchpad/cosmos_edge_gen"
I2V = f"{REPO}/examples/offline_inference/image_to_video/image_to_video.py"

NEG = json.dumps(json.load(open(f"{OFF}/negative_prompt.json")))
OFFPROMPT = json.dumps(json.load(open(f"{OFF}/example_i2v_prompt.json")))
EB = '{"flow_shift": 3.0, "max_sequence_length": 4096, "guardrails": false}'
COMMON = ["--height", "480", "--width", "832", "--num-frames", "121",
          "--num-inference-steps", "50", "--guidance-scale", "5.0", "--fps", "24",
          "--seed", "0", "--negative-prompt", NEG, "--extra-body", EB]

def launch(gpu, extra, out, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [PY, I2V, "--model", "nvidia/Cosmos3-Edge", *COMMON, *extra, "--output", out]
    return subprocess.Popen(cmd, cwd=REPO, env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)

jobs = [
    # coastal fp8 — identical scene to the dense repro, + fp8 (the photoreal fp8 comparison)
    launch(5, ["--image", f"{OFF}/example_i2v_input.jpg", "--prompt", OFFPROMPT, "--quantization", "fp8"],
           f"{G}/v5_coastal_fp8.mp4", f"{G}/v5_coastal_fp8.log"),
    # 2nd driving I2V (different road) for variety — plain descriptive prompt
    launch(6, ["--image", f"{OFF}/av0_frame.png",
               "--prompt", "Driver's POV dashcam footage: the car drives smoothly forward along the road, "
                           "buildings and traffic passing by, realistic daylight, sharp focus, steady continuous motion."],
           f"{G}/v5_av_driving.mp4", f"{G}/v5_av_driving.log"),
]
codes = [j.wait() for j in jobs]
print("exit codes:", codes)
for f in ["v5_coastal_fp8", "v5_av_driving"]:
    p = f"{G}/{f}.mp4"
    print(f, "OK" if os.path.exists(p) else "MISSING")
