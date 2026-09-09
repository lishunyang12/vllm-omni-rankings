# GPC frequency summary: all recorded samples

Per-physical-GPU maximum and arithmetic mean from the full recorded GPC data.

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

- **Scope:** all 246,429 rows in `raw-export/nsys-gpc-raw.csv`, including activity and waits outside VSA. No time-window or kernel filter.
- **Window boundaries:** each GPU uses all of its own recorded samples; start/end timestamps differ slightly across devices.
- **Mean:** ordinary arithmetic mean of decoded sample values, with equal weight per recorded sample; not time-weighted.
- **Max:** largest observed decoded sample, not the GPU's advertised maximum clock.
- **Decode:** `frequency_mhz = (int(value) & 0xffffffff) / 1_000_000`.
- **Identity:** map `(typeId, metricId)` through `raw-export/metric-device-map.csv` to the physical GPU number.
- **Precision:** this table displays two decimal places; the CSV preserves six decimal places.
- **Capture:** existing 2026-09-09 capture with Nsight Systems 2025.3.2, requested 1,000 Hz. No new GPU workload was run to produce this summary.

[Download summary CSV](gpc-frequency-summary.csv) · [Download source raw ZIP](h3-frequency-raw-data.zip)

Source ZIP SHA256:

```text
2ca9c9f3833b32e02e4a9916c66e5ee92568f04619ed5d698cc61a56f5e7596b
```
