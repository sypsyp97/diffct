"""Render the 1080p film and matching 800px README GIF with installed Manim/PyAV."""
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / ".validation/video-review/final"
ASSETS = ROOT / "docs/assets"


def main():
    subprocess.run([sys.executable, "-m", "manim", "-qh", "--fps", "30", "--disable_caching",
                    "--media_dir", str(MEDIA), str(Path(__file__).with_name("diffct_intro.py")),
                    "DiffCTIntro"], cwd=ROOT, check=True)
    source = MEDIA / "videos/diffct_intro/1080p30/DiffCTIntro.mp4"
    export(source)


def export(source):
    # Normalize segment-boundary timestamps to constant 30 fps.
    with av.open(str(source)) as src, av.open(str(ASSETS / "diffct_intro.mp4"), "w",
                                             options={"movflags": "+faststart"}) as dst:
        stream = dst.add_stream("libx264", rate=30)
        stream.width, stream.height = 1920, 1080
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for index, frame in enumerate(src.decode(video=0)):
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    frames = []
    next_time = 0.0
    with av.open(str(ASSETS / "diffct_intro.mp4")) as src:
        for frame in src.decode(video=0):
            if float(frame.time) + 1e-6 < next_time:
                continue
            image = frame.to_image().resize((800, 450), Image.Resampling.LANCZOS)
            frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=64))
            next_time += 0.125
    # GIF delays have 10 ms resolution; alternating 120/130 ms preserves 8 fps.
    frames[0].save(ASSETS / "diffct_intro.gif", save_all=True, append_images=frames[1:],
                   duration=[120 if i % 2 == 0 else 130 for i in range(len(frames))],
                   loop=0, disposal=2, optimize=False)
    print(f"Exported MP4 and GIF ({len(frames)} frames) to {ASSETS}")


if __name__ == "__main__":
    main()
