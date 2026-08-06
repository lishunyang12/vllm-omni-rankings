"""Capture the second MiniMax-H3 diffuse call with Nsight Systems."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import functools
import os
import sys
from types import ModuleType


_TARGET = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"


def _patch(module: ModuleType) -> None:
    import torch

    pipeline_type = module.MiniMaxH3Pipeline
    original = pipeline_type.diffuse
    warmup_requests = int(os.environ.get("NSYS_CAPTURE_WARMUP_REQUESTS", "1"))

    @functools.wraps(original)
    def diffuse(self, *args, **kwargs):
        count = getattr(self, "_nsys_capture_request", 0)
        is_rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        capture = is_rank_zero and count == warmup_requests
        if capture:
            print(
                f"NSYS_CAPTURE_START pid={os.getpid()} request={count}",
                file=sys.stderr,
                flush=True,
            )
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("steady_diffusion")
        try:
            return original(self, *args, **kwargs)
        finally:
            self._nsys_capture_request = count + 1
            if capture:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
                print(
                    f"NSYS_CAPTURE_STOP pid={os.getpid()} request={count}",
                    file=sys.stderr,
                    flush=True,
                )

    pipeline_type.diffuse = diffuse
    print(
        f"NSYS_CAPTURE_PATCHED pid={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):
        create_module = getattr(self._wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("VLLM_OMNI_NSYS_STEADY_CAPTURE") == "1":
    sys.meta_path.insert(0, _Finder())
