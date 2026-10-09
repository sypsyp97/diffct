# Intro video

`diffct_intro.py` is the Manim Community (v0.21) source of `docs/assets/diffct_intro.gif`
and `docs/assets/diffct_intro.mp4`. Make the inputs on a CUDA GPU, then render and join the scenes:

```bash
python docs/video/make_inputs.py --gpu      # walnut reconstructions; without --gpu only data2d.npz
cd docs/video
for s in S1Title S2Projection S3Siddon S4Trajectories S5MultiGPU S6Measured S7Helical S8Geometry S9End; do
  manim -qh diffct_intro.py $s
done
```

The reconstructions come from the measured walnut in `examples/data/walnut_cone.npz`
(Meaney 2022, CC BY 4.0; see `examples/data/NOTICE`). The timings in the multi-GPU scene
are measured diffct results on A100 64 GB GPUs (Leonardo Booster); see `docs/assets/scaling.json`.
