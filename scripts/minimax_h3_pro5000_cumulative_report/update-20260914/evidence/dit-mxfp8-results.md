# FastH3 MXFP8 与 blockwise FP8 快速 A/B

固定 FastH3 VSA 四步、CAKE、BF16 RDMA、exact AdaLN 和完整 H3 VAE。
teahouse / seed 1101；请求1280×720，输出362帧、1280×704、24fps；每臂1次预热+3次正式。

| 主干量化 | 三次正式秒数 | 均值秒 | 中位数秒 | 样本标准差秒 |
| --- | --- | ---: | ---: | ---: |
| blockwise | 17.928 / 17.904 / 17.925 | 17.9190 | 17.9250 | 0.0131 |
| mxfp8 | 17.959 / 17.926 / 17.887 | 17.9240 | 17.9260 | 0.0360 |

MXFP8 相对 blockwise 均值增加 0.0050 s，耗时降低 -0.03%；速度比 0.9997×。

两臂 controller 完成、before/after 源码一致，运行期独占与量化/VSA/RDMA/VAE/MP4门禁通过。实际每rank16NFE、200模块、5600次量化GEMM（O每层分4段）。
MXFP8 为 A1×32/W1×32 E8M0；blockwise 为 A1×128/W128×128 FP32 scale。
计时为客户端 POST 到完整 MP4 写入文件，精度1ms；加载、预热、后置媒体校验与哈希排除。worker内PyAV/libx264与VAE重叠，API mp4_encode_s并不是x264耗时。
本轮未开阶段profiler；原TSV中的“-”保留为缺失，不能填0或推算encoder/DiT/VAE独立耗时。
源码编码参数为CRF18、ultrafast、threads=0（自动）；实际8个MP4的x264 SEI签名相同，AAC双声道32kHz。源码preset与码流签名分别记录。
共用custom-op边界并关闭BF16 VSPLIT，不能与历史21.7712秒直接作量化归因。三次顺序样本仅作快速筛查，未完成人工音视频质量评估，不称无损。

[全部原始数据与证据](comparison.json) · [评测配置](../study-plan.json)
