from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np


MODES = (
    "trtllm_dense",
    "sage_fp8",
    "skip_softmax_005_gate097",
    "sage_fp8_skip_005_gate097",
)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        sample_rate = int(stream.codec_context.sample_rate)
        channels = len(stream.codec_context.layout.channels)
        for frame in container.decode(audio=0):
            samples = np.asarray(frame.to_ndarray())
            if samples.ndim == 1:
                samples = samples[None, :]
            if samples.shape[0] != channels and samples.shape[-1] == channels:
                samples = samples.T
            if samples.shape[0] == 1 and channels > 1:
                samples = samples.reshape(-1, channels).T
            if samples.shape[0] != channels:
                raise RuntimeError(
                    f"Unexpected decoded audio shape {samples.shape} in {path}"
                )
            if np.issubdtype(samples.dtype, np.integer):
                samples = samples.astype(np.float64) / max(
                    abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max
                )
            else:
                samples = samples.astype(np.float64)
            chunks.append(samples)
    if not chunks:
        raise RuntimeError(f"No audio decoded from {path}")
    return np.concatenate(chunks, axis=1), sample_rate


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"Decoded audio shapes differ: {reference.shape} != {candidate.shape}"
        )
    error = candidate - reference
    reference_energy = float(np.square(reference).sum())
    error_energy = float(np.square(error).sum())
    correlation = float(np.corrcoef(reference.ravel(), candidate.ravel())[0, 1])
    return {
        "samples_per_channel": reference.shape[1],
        "channels": reference.shape[0],
        "decoded_sha256": hashlib.sha256(
            np.ascontiguousarray(candidate).tobytes()
        ).hexdigest(),
        "correlation": correlation,
        "snr_db": None
        if error_energy == 0
        else 10 * math.log10(reference_energy / error_energy),
        "rms_error": float(np.sqrt(np.mean(np.square(error)))),
        "candidate_rms": float(np.sqrt(np.mean(np.square(candidate)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=MODES)
    args = parser.parse_args()

    videos = {
        mode: args.session_root / mode / "client/response_2.mp4"
        for mode in args.modes
    }
    audio = {mode: read_audio(path) for mode, path in videos.items()}
    reference, reference_rate = audio["trtllm_dense"]
    results: dict[str, object] = {}
    for mode, (candidate, sample_rate) in audio.items():
        if sample_rate != reference_rate:
            raise RuntimeError(
                f"Audio sample rate differs for {mode}: {sample_rate} != {reference_rate}"
            )
        results[mode] = {
            "video": str(videos[mode]),
            **metrics(reference, candidate),
        }

    payload = {
        "reference_mode": "trtllm_dense",
        "sample_rate": reference_rate,
        "comparison": "decoded stereo AAC waveform",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
