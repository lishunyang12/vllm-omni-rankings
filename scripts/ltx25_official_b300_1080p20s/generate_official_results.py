#!/usr/bin/env python3
"""Generate the official LTX-2.5 prompt set at 1080p and 20 seconds.

The prompts are copied verbatim from the Lightricks/LTX-2 repository. Each
prompt is rendered with three deterministic seeds through vLLM-Omni's public
offline example and LTX2DistilledPipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROMPTS = {
    "quickstart-dialogue": (
        "A medium close-up shot features a Caucasian man with a beard, wearing a green and white "
        "baseball cap without any letters on the front, and a light blue shirt over a white t-shirt. "
        "He is positioned in the center of the frame, looking intently directly at the camera, his "
        "eyes focused on camera. His facial expression is one of deep concentration, with his brow "
        "slightly raised. As he looks straight at the camera, a quick sniff sound is heard, and then "
        "he speaks with a deep male voice and a satisfied tone, saying, 'I think it's so good.' The "
        "camera remains static throughout, maintaining a shallow depth of field, which keeps the man "
        "in sharp focus while the background is softly blurred, showing a beige wall behind him. "
        "After a brief pause, another short, audible sniff is heard. The man then continues to speak, "
        "his voice maintaining the same quality, as he states, 'So good. So good.' He elaborates "
        "further, emphasizing his point with a final statement, 'This got to be, it's got to be the "
        "best tool I've ever seen.'"
    ),
    "home-office": (
        "A woman with long brown hair sits at a wooden desk in a cozy home office, typing on a "
        "laptop while occasionally glancing at notes beside her. Soft natural light streams through "
        "a large window, casting warm shadows across the room. She pauses to take a sip from a "
        "ceramic mug, then continues working with focused concentration. The audio captures the "
        "gentle clicking of keyboard keys, the soft rustle of papers, and ambient room tone with "
        "occasional distant bird chirps from outside."
    ),
    "chef": (
        "A chef in a white uniform stands in a professional kitchen, carefully plating a gourmet "
        "dish with precise movements. Steam rises from freshly cooked vegetables as he arranges "
        "them with tweezers. The stainless steel surfaces gleam under bright overhead lights, and "
        "various pots simmer on the stove behind him. The audio features the sizzling of pans, "
        "the clinking of utensils against plates, and the ambient hum of kitchen ventilation."
    ),
}

NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive "
    "noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, "
    "unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, "
    "extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent "
    "perspective, camera shake, incorrect depth of field, background too sharp, background clutter, "
    "distracting reflections, harsh shadows, inconsistent lighting direction, color banding, "
    "cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect "
    "ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, "
    "silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, "
    "incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, "
    "incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

WIDTH = 1920
HEIGHT = 1088
NUM_FRAMES = 481
FPS = 24
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", choices=[*PROMPTS, "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--model",
        default=os.environ.get("LTX25_MODEL", "Lightricks/LTX-2.5-Diffusers"),
    )
    parser.add_argument(
        "--vllm-omni-root",
        type=Path,
        default=Path(os.environ.get("VLLM_OMNI_ROOT", ".")),
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("VLLM_OMNI_PYTHON", sys.executable),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def probe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def main() -> None:
    args = parse_args()
    example = args.vllm_omni_root / "examples/offline_inference/text_to_video/text_to_video.py"
    if not example.is_file():
        raise FileNotFoundError(example)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_ids = PROMPTS if args.prompt_id == "all" else {args.prompt_id: PROMPTS[args.prompt_id]}
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "model": "Lightricks/LTX-2.5-Diffusers",
        "pipeline": "LTX2DistilledPipeline",
        "width": WIDTH,
        "height": HEIGHT,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "duration_seconds": NUM_FRAMES / FPS,
        "prompts": PROMPTS,
        "negative_prompt": NEGATIVE_PROMPT,
        "runs": {},
    }
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        manifest["runs"].update(prior.get("runs", {}))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.vllm_omni_root)
    for prompt_id, prompt in prompt_ids.items():
        for seed in args.seeds:
            run_id = f"{prompt_id}-seed-{seed}"
            output = args.output_dir / f"{run_id}.mp4"
            log = args.output_dir / f"{run_id}.log"
            if output.is_file() and output.stat().st_size > 0 and not args.force:
                manifest["runs"][run_id] = {
                    "status": "passed",
                    "reused": True,
                    "output": output.name,
                    "probe": probe(output),
                }
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                continue

            command = [
                args.python,
                str(example),
                "--model",
                args.model,
                "--model-class-name",
                "LTX2DistilledPipeline",
                "--prompt",
                prompt,
                "--negative-prompt",
                NEGATIVE_PROMPT,
                "--height",
                str(HEIGHT),
                "--width",
                str(WIDTH),
                "--num-frames",
                str(NUM_FRAMES),
                "--num-inference-steps",
                "8",
                "--frame-rate",
                str(FPS),
                "--fps",
                str(FPS),
                "--seed",
                str(seed),
                "--enforce-eager",
                "--output",
                str(output),
            ]
            started = time.time()
            print(f"[{run_id}] starting", flush=True)
            with log.open("w") as stream:
                completed = subprocess.run(
                    command,
                    cwd=args.vllm_omni_root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record = {
                "status": "passed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "elapsed_seconds": time.time() - started,
                "output": output.name,
                "log": log.name,
            }
            if completed.returncode == 0:
                record["probe"] = probe(output)
            manifest["runs"][run_id] = record
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            if completed.returncode:
                raise SystemExit(f"{run_id} failed; inspect {log}")
            print(f"[{run_id}] complete", flush=True)


if __name__ == "__main__":
    main()
