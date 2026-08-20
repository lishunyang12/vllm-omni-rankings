# MiniMax-H3 Ulysses SP=8 Profiling 幻灯片

- `MiniMax-H3_Ulysses_U4_学术解析.pptx`：7 页、16:9、白底绿色调中文技术演示稿。
- `build_deck.py`：可维护的本地生成源文件。

内容基于原始 profile 分支 `agent/minimax-h3-pillar1@50cf90da7` 和
`minimax_h3_sp8_2step_b300_20260820.nsys-rep`，重点解释：

- SP=8 下 Q/K/V 与 Output All-to-All 的数据布局；
- NCCL 前 `direct_copy_kernel_cuda` 的源码归因；
- 由 `_rope_combine_kernel` 直接生成 Ulysses packed Q/K 的优化；
- 预期收益、适用边界和 before/after 验证标准。

重新生成：

```bash
python3 -m pip install -r requirements.txt
python3 build_deck.py
```

校验：

```bash
python3 validate_deck.py MiniMax-H3_Ulysses_U4_学术解析.pptx
```
