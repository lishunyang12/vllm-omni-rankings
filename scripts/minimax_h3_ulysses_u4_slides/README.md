# MiniMax-H3 Ulysses U4 学术幻灯片

- `MiniMax-H3_Ulysses_U4_学术解析.pptx`：23 页、16:9、中文学术演示稿。
- `build_deck.py`：可维护的本地生成源文件。

内容基于当前分支 `agent/minimax-h3-online-fp8@f76f8e58fb`、本地 B300 U4 实验记录，以及后续严格 Ulysses 边界优化提交 `7b76b6446`。

重新生成：

```bash
python3 -m pip install -r requirements.txt
python3 build_deck.py
```

校验：

```bash
python3 validate_deck.py MiniMax-H3_Ulysses_U4_学术解析.pptx
```
