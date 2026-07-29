"""Convert a saved frame array (…,F,H,W,3) to a compressed H.264 mp4 (<10 MB) for shipping."""
import sys, numpy as np, imageio.v2 as imageio

src, dst = sys.argv[1], sys.argv[2]
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
a = np.load(src)
while a.ndim > 4:
    a = a[0]
if a.dtype != np.uint8 and float(a.max()) <= 1.0 + 1e-3:
    a = a * 255.0
a = a.clip(0, 255).astype("uint8")
w = imageio.get_writer(dst, fps=fps, codec="libx264", quality=6,
                       macro_block_size=8, ffmpeg_params=["-crf", "26", "-pix_fmt", "yuv420p"])
for f in a:
    w.append_data(f)
w.close()
print(f"{src} {a.shape} -> {dst}")
