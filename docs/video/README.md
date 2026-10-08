# Intro video

`diffct_intro.py` is the Manim Community (v0.21) source of `docs/assets/diffct_intro.gif`
and `docs/assets/diffct_intro.mp4`. Render and join the scenes:

```bash
python docs/video/make_inputs.py            # add --gpu to recompute the reconstructions
cd docs/video
for s in S1Title S2Projection S3Siddon S4Trajectories S5MultiGPU S6Reconstruction S7Geometry S8End; do
  manim -qh diffct_intro.py $s
done
```

The timings in the multi-GPU scene and the reconstructions are measured diffct results
(A100 64 GB, Leonardo Booster); see `docs/VALIDATION.md`.
