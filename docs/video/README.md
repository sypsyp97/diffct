# Intro video

`diffct_intro.py` is the Manim Community (v0.21) source of `docs/assets/diffct_intro.gif`
and `docs/assets/diffct_intro.mp4`. Make the inputs on a CUDA GPU, then render and join the scenes:

```bash
python docs/video/make_inputs.py --gpu      # walnut reconstructions; without --gpu only data2d.npz
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
