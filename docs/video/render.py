"""Render the 1080p film and matching 800px README GIF with installed Manim/PyAV.

`python docs/video/render.py --gif-only` rebuilds only the GIF from the committed MP4.
"""
import argparse
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / ".validation/video-review/final"
ASSETS = ROOT / "docs/assets"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif-only", action="store_true", help="rebuild the GIF from docs/assets/diffct_intro.mp4")
    if parser.parse_args().gif_only:
        export_gif()
        return
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
    export_gif()


def export_gif():
    """8 fps README GIF from the exported MP4.

    Every frame uses one 255-colour palette taken from frames across the whole film, and each pixel
    whose palette index equals the previous frame's becomes transparent (index 255, disposal 1), so
    the GIF only stores what changes. This keeps the file near 12 MB instead of 41 MB with
    per-frame palettes, at a smaller error against the MP4.
    """
    frames = []
    next_time = 0.0
    with av.open(str(ASSETS / "diffct_intro.mp4")) as src:
        for frame in src.decode(video=0):
            if float(frame.time) + 1e-6 < next_time:
                continue
            frames.append(frame.to_image().resize((800, 450), Image.Resampling.LANCZOS))
            next_time += 0.125
    step = max(1, len(frames) // 64)
    sample = frames[::step][:64]
    mosaic = Image.new("RGB", (800, 450 * len(sample)))
    for index, image in enumerate(sample):
        mosaic.paste(image, (0, 450 * index))
    palette = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    colours = palette.getpalette()[:255 * 3] + [0, 0, 0]
    transparent = 255
    previous = None
    images = []
    for frame in frames:
        indices = np.asarray(frame.quantize(palette=palette, dither=Image.Dither.NONE))
        stored = indices.copy()
        if previous is not None:
            stored[indices == previous] = transparent
        previous = indices
        image = Image.fromarray(stored, mode="P")
        image.putpalette(colours)
        images.append(image)
    # GIF delays have 10 ms resolution; alternating 120/130 ms preserves 8 fps.
    images[0].save(ASSETS / "diffct_intro.gif", save_all=True, append_images=images[1:],
                   duration=[120 if i % 2 == 0 else 130 for i in range(len(images))],
                   loop=0, disposal=1, transparency=transparent, optimize=False)
    print(f"Exported GIF ({len(images)} frames) to {ASSETS}")


if __name__ == "__main__":
    main()
