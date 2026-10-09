# Intro video

`diffct_intro.py` is the Manim Community (v0.21) source of `docs/assets/diffct_intro.gif`
and `docs/assets/diffct_intro.mp4`. Rendering needs Manim, LaTeX and FFmpeg.
The scenes read three files in `docs/video/`:

- `walnut_measured.npz`: measured projections and reconstruction slices.
- `data2d.npz`: a walnut slice and a display sinogram made with SciPy.
- `calib_history.json`: saved results from a separate geometry calibration run.

These inputs are included in the checkout. Rendering them does not need CUDA.
The scenes do not read `recon_slices.npz` or `recon_psnr.json`.

To rebuild the inputs, install SciPy in the environment that runs `make_inputs.py`.
SciPy is not a core diffct dependency.
Run these commands from the repository root:

```bash
python -m pip install scipy
python docs/video/make_inputs.py --gpu
```

`--gpu` needs a CUDA-enabled diffct installation. It writes `walnut_measured.npz`,
`recon_slices.npz`, `recon_psnr.json` and `data2d.npz`.
Without `--gpu`, the script writes only `data2d.npz` and reads the existing `walnut_measured.npz`.
Neither mode writes `calib_history.json`. The calibration example does not export this JSON file.

Render and join the scenes:

```bash
cd docs/video
rm -f scenes.txt
for s in S00Intro S01Trajectory S02Autograd S03Geometry S04MultiGPU S05Measured S06End; do
  manim -qh diffct_intro.py $s
  echo "file 'media/videos/diffct_intro/1080p60/$s.mp4'" >> scenes.txt
done
ffmpeg -f concat -safe 0 -i scenes.txt -c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p -r 30 \
  -movflags +faststart -an ../assets/diffct_intro.mp4
ffmpeg -i ../assets/diffct_intro.mp4 -vf "fps=8,scale=800:-1:flags=lanczos,palettegen=max_colors=64:stats_mode=full" palette.png
ffmpeg -i ../assets/diffct_intro.mp4 -i palette.png \
  -lavfi "fps=8,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none" ../assets/diffct_intro.gif
```

The reconstructions come from the measured walnut in `examples/data/walnut_cone.npz`
(Meaney 2022, CC BY 4.0; see `examples/data/NOTICE`). The timings in the multi-GPU scene
are measured diffct results on A100 64 GB GPUs; see `docs/assets/scaling.json`.
The scene stores these timings as constants. Rebuilding the inputs does not update them.
