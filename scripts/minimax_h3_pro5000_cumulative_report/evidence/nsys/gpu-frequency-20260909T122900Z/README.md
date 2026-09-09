# MiniMax-H3 raw GPU frequency data

Complete recorded GPC/SYS clock samples and NVML readings from the same H3
request on eight RTX PRO 5000-class SM120 GPUs on 2026-09-09. This publishes
an existing capture; no new benchmark was run for this upload.

## GPC summary

[Summary table and methodology](gpc-frequency-summary.md) · [Download summary CSV](gpc-frequency-summary.csv)

| Physical GPU | GPC Max (MHz) | GPC Mean (MHz) | Samples |
|---:|---:|---:|---:|
| 0 | 2377.56 | 2308.16 | 30,972 |
| 1 | 2377.98 | 2317.79 | 30,866 |
| 2 | 2377.55 | 2312.78 | 30,626 |
| 3 | 2377.52 | 2298.01 | 31,021 |
| 4 | 2586.98 | 2383.95 | 31,579 |
| 5 | 2591.22 | 2403.70 | 31,619 |
| 6 | 2650.92 | 2413.58 | 31,414 |
| 7 | 1950.06 | 1946.84 | 28,332 |

These values use all recorded GPC samples, including non-VSA activity and waits.
Mean is the per-sample arithmetic mean, not time-weighted. Max is the observed
sample maximum. The raw ZIP remains unchanged; summary files are separate downloads.

## Downloads

- [Raw frequency data ZIP](h3-frequency-raw-data.zip) — 6.88 MB, including the
  complete GPC and SYS CSV exports, original NVML CSV, device metadata,
  capture/export scripts, and checksums.
- [Native Nsight Systems report](minimax-h3-original-cutedsl-mlp-fp8-bf16-wire-e2e.nsys-rep)
  — 65.66 MB. Open with Nsight Systems 2025.3.2 or newer and select GPU Metrics
  under the relevant PCI device to inspect GPC and SYS Clock Frequency.
- [Metric/device mapping](metric-device-map.csv),
  [capture contract](measurement-contract.json), and
  [source manifest](manifest.json).

The ZIP contains these full-window datasets:

| File inside `raw-export/` | Records | Contents |
|---|---:|---|
| `nsys-gpc-raw.csv` | 246,429 | Every recorded GPC frequency value |
| `nsys-sys-raw.csv` | 246,429 | Every recorded SYS frequency value |
| `nvml-frequency.csv` | 42,896 | 5,362 polls per GPU, including startup and the request |

The native trace is a separate download and is not duplicated in the ZIP.
The complete 520.68 MB source SQLite export is not included; its SHA256 is
recorded in the manifest. Local source paths in the manifest and ZIP describe
the original capture environment, not additional online download locations.

## Capture configuration

- Nsight Systems 2025.3.2; all eight GPUs; requested GPU Metrics sampling rate
  **1,000 Hz**. The 20,000 Hz documentation example was not used for this capture.
- Collection started after the server health check and stopped after the full
  MP4 response. The native trace includes CUDA activity and GPU Metrics.
- NVML sequentially polled eight GPUs, targeting **10 Hz per GPU**. Each row
  retains query-start/query-end wall-clock timestamps and a monotonic timestamp.
- Four FastH3 forwards, original PR #4259 CuTeDSL VSA, tile 64/top-k 162,
  MLP-only FP8, BF16 QKV/O projection and transport, TAEH3 FP16, seed 1101,
  15-second 1280x720 request. No request warmup; existing JIT caches were retained.
- GPU clock and power-limit settings were not changed.

Sampling targets do not imply perfectly uniform intervals. The CSVs retain
every recorded timestamp, including samples outside the VSA kernels. They
contain no VSA-only filtering, resampling, averages, or medians.

## Raw values and units

Both Nsight CSVs preserve the five source `GPU_METRICS` fields exactly:

```text
rawTimestamp,timestamp,typeId,metricId,value
```

`value` is the integer returned by the source SQLite export, including negative
values. It is not converted to uint32 or MHz. This version's source metadata
labels the fields `GPC Clock Frequency [MHz]` and `SYS Clock Frequency [MHz]`;
those labels are also preserved verbatim. The raw integers must not be treated
directly as MHz. The earlier analysis used `(value & 0xffffffff) / 1e6` to obtain
MHz; this raw export does not apply that conversion.

NVIDIA documents the underlying GPC/SYS metrics as average frequencies in
[hertz](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#available-metrics).
NVML's `nvmlDeviceGetClockInfo` returns
[MHz](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html).
The original NVML CSV directly preserves the returned SM, graphics, and memory
clock values. Its power columns were already converted from mW to W by the
capture script. VIDEO clock and supported clock combinations were not collected.

Nsight `timestamp` is in nanoseconds on the capture timeline;
`TARGET_INFO_SESSION_START_TIME.csv` supplies the epoch anchor. The device map
uses PCI addresses and UUIDs to associate Nsight metric IDs with physical GPUs.

This is a different capture from the adjacent
[earlier full-request trace](../original-cutedsl-mlp-fp8-bf16-wire-e2e-20260909/),
even though the native report filenames match. The GPU Metrics capture adds
sampling overhead and is not a new warm-serving latency result.

## Integrity

Each exported Nsight row was compared field-for-field with the original SQLite.
The NVML CSV, ZIP, and native trace retain their source bytes. Check all published
files from this directory with:

```bash
sha256sum -c SHA256SUMS.txt
```

After extracting the ZIP, its `raw-export/SHA256SUMS.txt` validates the individual
CSV, metadata, and script files.
