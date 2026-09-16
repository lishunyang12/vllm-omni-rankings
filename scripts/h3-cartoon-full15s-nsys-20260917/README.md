# H3 complete 15-second request Nsight trace

[Download the native Nsight report](h3-cartoon-mxfp8-nvfp4-full15s.nsys-rep) (84,170,793 bytes).
Open with Nsight Systems 2025.3.2 or newer.

This newly captured trace covers one warmed H3 T2VA request for a 15-second video:
request preparation, text encoding, all four DiT forwards on eight ranks, full video/audio
VAE decoding, CPU encoding, audio mux and the complete MP4 response. Model loading and
one warmup request are outside the collection window.

- Full output verified locally: 362 frames, 1280 × 704, 24 fps, 15.083333 seconds, with audio.
- Profiled HTTP request: 28.399 seconds; first-to-last CUDA kernel span: 28.075 seconds.
- 644,469 CUDA kernel events across eight GPU worker processes.
- Four completed denoise-step ranges per rank; 1,600 fine-attention calls, exactly 200 per GPU.
- Runtime: FastH3 four-step VSA, MXFP8 DiT, BF16 RDMA transport, NVFP4 full VAE.
- CUDA/NVTX/OS runtime tracing; CUDA graph node tracing; CPU/context-switch sampling disabled.
- Original backend, gateway and public page returned HTTP 200 after restoration; the playlist was unchanged.

The request latency includes profiling overhead. Other preexisting GPU work was left running,
so contention is possible. This is diagnostic evidence, not an isolated serving benchmark.
NVTX scopes are unsynchronized host ranges; summed GPU time and overlapping stages do not equal wall latency.

Nsight emits warnings that not all CUDA/NVTX/OS-runtime events might have been collected,
as well as no-event warnings for non-worker processes and absent scheduling samples.
All eight workers and every expected request stage were verified; individual-event
losslessness is not guaranteed. No severity-Error diagnostic appears in the report.

Verify the downloaded files:

```bash
sha256sum -c SHA256SUMS.txt
```

The published `.nsys-rep` is the original collected report without event filtering,
splitting or re-encoding.
