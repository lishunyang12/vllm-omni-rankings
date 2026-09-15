# FastH3 三版本与 VAE 优化对照

固定 Moon Teahouse 完整提示词，seed 1101，4 NFE，24 fps，实际 1280×704。

| 方案 | 预热后完整生成 |
|---|---:|
| 原版 FastH3 | 46.621 s |
| 历史实时方案 | 14.927 s |
| 当前 VAE 优化方案 | 14.444 s |

## 本轮 VAE 实验

上一版当前最佳：14.973 s。

| VAE | 提前 gather | E2E | 与上一版 MP4 字节一致 |
|---|---|---:|---|
| MXFP8 | True | 14.938 s | True |
| NVFP4 | False | 14.472 s | False |
| NVFP4 | True | 14.444 s | False |

本轮展示 NVFP4 + 提前 gather 的组合视频：单次完整请求 14.444 秒，较上一版 14.973 秒少 0.529 秒（3.53%）。NVFP4 单独为 14.472 秒，新增提前 gather 只再少 28 毫秒；独立 VAE 交错测试未显示稳定额外收益。NVFP4 是有损量化，全部 362 帧平均 SSIM 为 0.9721、PSNR 为 40.06 dB；音频 PCM 一致。组合视频与 NVFP4 单独版本逐字节一致，上一版 MXFP8 和各候选均保留下载。

## 配置、验证与边界

- 三组使用同一完整 prompt、seed 和 4 次 DiT forward。原版为官方完整 VSA Data-Free checkpoint；另两组使用此前发布的 VSA Data-Free LoRA。不同权重形式、实现和精度可能改变生成轨迹。
- 三组实际输出均为 1280×704。原版明确请求高度 704，以满足上游 32 倍数检查；另两组保留历史请求高度 720，由原有路径向下对齐到 704。
- 三组均请求 362 帧。原版沿用上游 ffmpeg -shortest 封装，实际视频轨为 361 帧（15.042 秒），含音频的 MP4 为 15.075 秒；另两组为 362 帧（15.083 秒）。原始文件直接展示，未补帧；已用相同音轨长度的 CPU 合成输入复现这一个末帧差异。
- 原版基于上游 FastVideo 提交 3196835，仅修正输入检查：让 15 秒上限接受 VAE 对齐后的 362 帧。权重、推理算法和 kernel 未修改。使用原生 Triton VSA、关闭 FA4、非 VSA 注意力使用 Torch SDPA，启用 FSDP 权重分片和 all profile 融合。
- 原版请求了区域编译，但上游规则会对 Triton VSA 自动关闭 DiT 编译，因此本次 DiT 实际为 eager；VAE 编译正常启用。
- 当前最佳沿用既有优化开关，在独立副本中适配本次提示词的输入与 RDMA 输出分块形状；400 次 QK/V 拆分及 400 次输出打包检查均逐字节一致。
- 历史组对应指定页面的 FP8/E4M3 + TAEH3 FP16 (real-time stack)。该组的 E4M3 QKV 传输曾在历史主观质量评审中退回，此处保留用于直接比较。
- 每个耗时来自本次预热后的单次完整生成。原版边界为 VideoGenerator.generate() 至保存视频完成；vLLM-Omni 边界为本地 HTTP 请求至完整 MP4 返回。
- 这里同时涉及运行框架、注意力、传输和 VAE 的变化；这些耗时不用于归因纯 DiT per-tensor → MXFP8 收益。历史 14.561 秒来自另一条 prompt，未套用于本次视频。
- 历史组重新生成的全部解码画面与指定页面逐字节一致；解码音频未逐字节复现，波形相关系数 0.999299、SNR 28.53 dB。页面播放本次新生成的视频。
- 本轮只变更视频 VAE。DiT、提示词、seed、音频路径、362 帧和分块拼接数学保持相同；VAE 精度变化属于有损量化，不视为逐字节等价。
- 提前 gather 保留 B4/B3 解码顺序和 FP32 通信数据，每轮先传前 4 个 tile，再计算后 3 个 tile；21 次分批 gather 覆盖原来的 11 轮，全部 588 个 tile 不变。VAE 使用现有 NCCL gather；DiT 保持 FlashInfer RDMA。
- 本轮完整生成耗时均为 1 次预热后的单次正式 HTTP 请求，不能据此认定小幅差值是稳定收益。独占校验不通过的尝试不纳入表格。
- 独立 VAE 测试使用合成 latent、8 卡和原始 FP32 输出，按基线/提前/提前/基线交错计时；GPU 顺序为 0–7，完整模型沿用 0,4,1,5,2,6,3,7。该测试用于调度正确性和机制诊断，不代替完整请求耗时。
- 独立 VAE 基线中位数 3.087443 秒，提前 gather 为 3.086844 秒，差约 0.6 毫秒。两种调度的 trace 均有 NCCL SendRecv 与 GEMM 同时执行；profile 中 NCCL kernel 时长包含等待，不能等同于纯传输时间或加速收益。

## 原始像素和 GPU trace 校验

```json
{
  "status": "PASS",
  "scope": "VAE-only synthetic latent, original FP32 pixel equality, rank-zero diagnostic trace; separate from full video E2E",
  "raw_fp32_byteexact_all_ranks": true,
  "timing": {
    "baseline": {
      "samples_seconds": [
        3.0888195037841797,
        3.0860660150647163
      ],
      "median_seconds": 3.087442759424448
    },
    "early": {
      "samples_seconds": [
        3.0869020465761423,
        3.0867851292714477
      ],
      "median_seconds": 3.086843587923795
    }
  },
  "traces": {
    "baseline": {
      "gather_kernels": 11,
      "gemm_kernels": 4995,
      "gather_gpu_ms": 241.95046484375,
      "gather_overlapping_gemm_ms": 134.183919921875,
      "gather_overlapping_compute_ms": 228.4510595703125,
      "gather_streams": [
        20
      ],
      "gemm_streams": [
        7
      ],
      "sample_gather_kernel": "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)",
      "sample_gemm_kernel": "void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_nn_align8>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_nn_align8::Params)"
    },
    "early": {
      "gather_kernels": 21,
      "gemm_kernels": 4995,
      "gather_gpu_ms": 1935.6035107421876,
      "gather_overlapping_gemm_ms": 1090.1461611328125,
      "gather_overlapping_compute_ms": 1777.62193359375,
      "gather_streams": [
        20
      ],
      "gemm_streams": [
        7
      ],
      "sample_gather_kernel": "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)",
      "sample_gemm_kernel": "void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_nn_align8>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_nn_align8::Params)"
    }
  }
}
```

### mxfp8-early-r1

视频 SHA256：`b4eb88a362c68af80084c1d08ca1cddf4adc73b616e9b3fbd74e2e5e688555a2`。音频 PCM 与上一版一致，完整解码与运行路径校验 PASS。

### nvfp4-r3

视频 SHA256：`061b79b418dbae623ede09a203f4b0f83fb231f6f7ab381d50cb310ea7865fab`。音频 PCM 与上一版一致，完整解码与运行路径校验 PASS。

```json
{
  "status": "PASS",
  "scope": "Decoded 8-bit YUV frames from final H264 MP4; same DiT/prompt/seed; metrics are not perceptual acceptance",
  "reference": "previous current best MXFP8 VAE",
  "frames": 362,
  "ssim": 0.972081,
  "ssim_min": 0.962034,
  "ssim_min_frame": 316,
  "psnr_db": 40.057057
}
```

### nvfp4-early-r1

视频 SHA256：`061b79b418dbae623ede09a203f4b0f83fb231f6f7ab381d50cb310ea7865fab`。音频 PCM 与上一版一致，完整解码与运行路径校验 PASS。

```json
{
  "status": "PASS",
  "scope": "Decoded 8-bit YUV frames from final H264 MP4; same DiT/prompt/seed; metrics are not perceptual acceptance",
  "reference": "previous current best MXFP8 VAE",
  "frames": 362,
  "ssim": 0.972081,
  "ssim_min": 0.962034,
  "ssim_min_frame": 316,
  "psnr_db": 40.057057
}
```
