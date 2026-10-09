"""diffct intro video (Manim CE 0.21). Render every scene, then join them (see README.md).

The video introduces the project: what diffct is, what it adds (any trajectory, autograd,
geometry gradients, many GPUs) and measured results. Data files in this directory are
written by make_inputs.py: data2d.npz (walnut slice and its parallel sinogram),
walnut_measured.npz (measured walnut reconstructions), calib_history.json (a real
geometry calibration run). All reconstructions and timings are diffct output on
A100 GPUs.

Layout rules are enforced at render time: every header and caption sits on a fixed
baseline, connectors are horizontal or vertical (`ortho`), labels must fit their boxes
(`fits`), stay inside the safe area (`safe`) and keep apart (`apart`), and captions are
one line.
"""
import json
import math
from pathlib import Path

import numpy as np
from manim import *

HERE = Path(__file__).parent

BG = "#0E1013"
INK = "#ECECEC"
SUB = "#A0A6AE"
GRID = "#3C4650"
PANEL = "#161C22"
BLUE = "#58C4DD"
GREEN = "#83C167"
YELLOW = "#F9E04C"
GOLD = "#F0AC5F"
RED = "#FC6255"
VIOLET = "#9A72AC"
GPU_COLORS = [BLUE, GREEN, YELLOW, GOLD]

CHAPTERS = ["Any trajectory", "Autograd", "Geometry gradients", "Many GPUs", "Measured data"]

XL, XR = -6.3, 6.3
HEAD_BASE = 3.17               # every chapter header sits on this baseline
CAP_BASE = -3.33               # every caption sits on this baseline
BAR_Y = -3.78
TOP, BOTTOM = 2.65, -2.9       # content area between header and caption

config.background_color = BG


# ---------------------------------------------------------------- helpers

def T(text, size=34, color=INK):
    return Tex(text, font_size=size, color=color)


def M(tex, size=36, color=INK):
    return MathTex(tex, font_size=size, color=color)


def baseline(mob):
    """Baseline of a text object: the median bottom of its glyphs (descenders are the minority)."""
    return float(np.median([g.get_bottom()[1] for g in mob.family_members_with_points()]))


def on_baseline(mob, y):
    return mob.shift(UP * (y - baseline(mob)))


def left_at(mob, x):
    return mob.shift(RIGHT * (x - mob.get_left()[0]))


def right_at(mob, x):
    return mob.shift(RIGHT * (x - mob.get_right()[0]))


def fits(inner, outer, pad=0.08):
    il, ir, ib, it = inner.get_left()[0], inner.get_right()[0], inner.get_bottom()[1], inner.get_top()[1]
    ol, orr, ob, ot = outer.get_left()[0], outer.get_right()[0], outer.get_bottom()[1], outer.get_top()[1]
    assert il >= ol + pad and ir <= orr - pad and ib >= ob + pad and it <= ot - pad, \
        f"does not fit: x[{il:.2f},{ir:.2f}] y[{ib:.2f},{it:.2f}] in x[{ol:.2f},{orr:.2f}] y[{ob:.2f},{ot:.2f}]"
    return inner


def apart(*mobs, gap=0.12):
    for i, a in enumerate(mobs):
        for b in mobs[i + 1:]:
            sep_x = max(b.get_left()[0] - a.get_right()[0], a.get_left()[0] - b.get_right()[0])
            sep_y = max(b.get_bottom()[1] - a.get_top()[1], a.get_bottom()[1] - b.get_top()[1])
            assert max(sep_x, sep_y) >= gap, \
                f"too close: {getattr(a, 'tex_string', type(a).__name__)} vs {getattr(b, 'tex_string', type(b).__name__)}"
    return mobs


def safe(mob, top=TOP, bottom=BOTTOM):
    l, r, b, t = mob.get_left()[0], mob.get_right()[0], mob.get_bottom()[1], mob.get_top()[1]
    assert l >= XL - 1e-6 and r <= XR + 1e-6 and b >= bottom - 1e-6 and t <= top + 1e-6, \
        f"outside the safe area: x[{l:.2f},{r:.2f}] y[{b:.2f},{t:.2f}]"
    return mob


def ortho(*points, color=SUB, sw=3, tip=0.16, arrow=True):
    """A connector made only of horizontal and vertical segments."""
    pts = [np.array(p, dtype=float) for p in points]
    for a, b in zip(pts[:-1], pts[1:]):
        assert abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6, f"diagonal segment {a} -> {b}"
    g = VGroup(*[Line(a, b, color=color, stroke_width=sw) for a, b in zip(pts[:-2], pts[1:-1])])
    if arrow:
        g.add(Arrow(pts[-2], pts[-1], buff=0, color=color, stroke_width=sw, tip_length=tip,
                    max_tip_length_to_length_ratio=0.6, max_stroke_width_to_length_ratio=100))
    else:
        g.add(Line(pts[-2], pts[-1], color=color, stroke_width=sw))
    return g


def gray_image(array, height, vmax=None, width=None):
    a = np.asarray(array, dtype=np.float32)
    img = np.clip(a / (vmax or a.max()), 0, 1)
    mob = ImageMobject((np.stack([img] * 3, axis=-1) * 255).astype(np.uint8))
    mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
    mob.height = height
    if width:
        mob.stretch_to_fit_width(width)
    return mob


def framed(mob, color=GRID, sw=1.5):
    return Group(mob, Rectangle(width=mob.width, height=mob.height, stroke_color=color, stroke_width=sw).move_to(mob))


def code(text, size=26, color=INK, width=None):
    """A code card: monospace text in a dark rounded panel."""
    t = T(r"\texttt{" + text + "}", size, color)
    b = RoundedRectangle(width=width or t.width + 0.6, height=t.height + 0.42, corner_radius=0.12,
                         stroke_color=GRID, stroke_width=1.5, fill_color=PANEL, fill_opacity=1)
    t.move_to(b)
    if width:
        left_at(t, b.get_left()[0] + 0.3)
    fits(t, b, pad=0.1)
    return VGroup(b, t)


def header_mob(index):
    header = Tex(r"{\footnotesize " + f"{index:02d}" + "}", r"\enspace " + CHAPTERS[index - 1], font_size=44,
                 color=INK)
    header[0].set_color(RED)
    left_at(header, XL)
    return on_baseline(header, HEAD_BASE)


def bar(done):
    track = Line([XL, BAR_Y, 0], [XR, BAR_Y, 0], color=GRID, stroke_width=3)
    x = XL + (XR - XL) * done / len(CHAPTERS)
    return track, Line([XL, BAR_Y, 0], [max(x, XL + 1e-3), BAR_Y, 0], color=RED, stroke_width=3)


def chapter(scene, index):
    """Start from the exact frame the previous scene ended on (its header and bar), then morph
    the old title into the new one while the bar fills one step."""
    track, filled = bar(index - 1)
    old = header_mob(max(index - 1, 1))
    scene.add(track, filled, old)
    _, target = bar(index)
    if index == 1:
        scene.play(Transform(filled, target), run_time=0.6)
        return VGroup(old, track, filled)
    new = header_mob(index)
    scene.play(TransformMatchingShapes(old, new), Transform(filled, target), run_time=0.9)
    return VGroup(new, track, filled)


class Captions:
    """One caption line on the fixed caption baseline; each new one replaces the last."""
    def __init__(self, scene):
        self.scene, self.cur = scene, None

    def __call__(self, text, hold=2.4, color=SUB, size=32):
        new = safe(on_baseline(T(text, size, color).set_x(0), CAP_BASE), top=-2.9, bottom=-3.6)
        assert new.height < 0.45, f"caption wraps to two lines: {text}"
        if self.cur is None:
            self.scene.play(FadeIn(new, shift=0.08 * UP), run_time=0.5)
        else:
            self.scene.play(FadeOut(self.cur, shift=0.08 * UP), FadeIn(new, shift=0.08 * UP), run_time=0.5)
        self.cur = new
        if hold:
            self.scene.wait(hold)
        return new


def fade_all(scene, run_time=0.8):
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=run_time)


def fade_except(scene, keep, run_time=0.8):
    """Clear the frame but leave `keep` (the chapter header and progress bar) untouched."""
    kept = set(keep.get_family())
    scene.play(*[FadeOut(m) for m in scene.mobjects if m not in kept and not (set(m.get_family()) & kept)],
               run_time=run_time)


def chip(index, name, width=3.4):
    b = RoundedRectangle(width=width, height=0.66, corner_radius=0.14, stroke_color=GRID, stroke_width=1.5,
                         fill_color=PANEL, fill_opacity=1)
    n = T(f"{index:02d}", 28, RED)
    t = T(name, 28, SUB)
    left_at(n, b.get_left()[0] + 0.28).set_y(b.get_center()[1])
    left_at(t, n.get_right()[0] + 0.22)
    on_baseline(t, baseline(n))
    fits(VGroup(n, t), b, pad=0.1)
    return VGroup(b, n, t)


# ---------------------------------------------------------------- 3D drawing on a 2D canvas

class View3D:
    """Orthographic view of 3D points: rotate about z by the azimuth, tilt by a fixed elevation."""
    def __init__(self, centre, scale, elevation=22, azimuth=-35):
        self.centre, self.scale = np.array(centre, dtype=float), scale
        self.el = math.radians(elevation)
        self.az = ValueTracker(azimuth)

    def __call__(self, p):
        a = math.radians(self.az.get_value())
        x, y, z = p
        u = math.cos(a) * x - math.sin(a) * y
        d = math.sin(a) * x + math.cos(a) * y
        v = math.cos(self.el) * z + math.sin(self.el) * d
        return self.centre + self.scale * np.array([u, v, 0.0])


R_ORBIT = 2.4
HALF = 0.85                    # half edge of the volume cube


def orbit(kind, t):
    """Source position at parameter t in [0, 1) for each named trajectory."""
    a = 2 * math.pi * t
    if kind == "helical":
        a = 3 * math.pi * t
        return np.array([R_ORBIT * math.cos(a), R_ORBIT * math.sin(a), -0.95 + 1.9 * t])
    z = {"circular": 0.0, "saddle": 0.6 * math.cos(2 * a), "sinusoidal": 0.45 * math.sin(3 * a)}[kind]
    return np.array([R_ORBIT * math.cos(a), R_ORBIT * math.sin(a), z])


CALIBRATED = None


def calibrated_poses(n=36, seed=4):
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.normal(0, 0.05, n)
    r = R_ORBIT + rng.normal(0, 0.08, n)
    z = rng.normal(0, 0.18, n)
    return [np.array([ri * math.cos(ai), ri * math.sin(ai), zi]) for ai, ri, zi in zip(a, r, z)]


def cube_edges(view, color=VIOLET):
    c = [np.array([sx, sy, sz]) * HALF for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(np.abs(c[i] - c[j]) > 0) == 1]
    return VGroup(*[Line(view(c[i]), view(c[j]), color=color, stroke_width=1.8) for i, j in edges])


def detector_and_rays(view, p, color=BLUE):
    """Flat detector opposite the source, facing it, and the four corner rays."""
    radial = p / np.linalg.norm(p)
    tangent = np.cross(np.array([0, 0, 1.0]), radial)
    tangent /= np.linalg.norm(tangent)
    up = np.cross(radial, tangent)
    centre = -radial * 1.55
    corners = [centre + 0.95 * su * tangent + 0.75 * sv * up for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    panel = Polygon(*[view(q) for q in corners], color=SUB, stroke_width=1.5, fill_color=SUB, fill_opacity=0.18)
    rays = VGroup(*[Line(view(p), view(q), color=color, stroke_width=1.2, stroke_opacity=0.55) for q in corners])
    return VGroup(rays, panel, Dot(view(p), radius=0.08, color=RED))


# ---------------------------------------------------------------- scenes

class S00Intro(Scene):
    """Opening: measured walnut data, the name, and the chapter map."""
    def construct(self):
        data = np.load(HERE / "walnut_measured.npz")
        tiles = [gray_image(np.flipud(p.T), 1.7, vmax=data["projections"].max()) for p in data["projections"]]
        tiles += [gray_image(data[f"{k}_axial"], 1.7, vmax=1.0) for k in ("fdk", "sirt", "cgls", "tv")]
        wall = Group(*[framed(t) for t in tiles]).arrange_in_grid(rows=2, cols=4, buff=0.16).move_to(ORIGIN)
        self.play(LaggedStart(*[FadeIn(c, scale=0.96) for c in wall], lag_ratio=0.1), run_time=2.0)
        self.wait(0.6)
        veil = Rectangle(width=config.frame_width, height=config.frame_height, fill_color=BG, fill_opacity=0.86,
                         stroke_width=0)
        title = T(r"\textbf{diffct}", 120, INK)
        sub = T("differentiable CUDA projectors for CT, built on PyTorch", 38, SUB)
        line = Line(LEFT * 4.8, RIGHT * 4.8, color=RED, stroke_width=3)
        VGroup(title, sub, line).arrange(DOWN, buff=0.35).move_to(UP * 1.05)
        self.play(FadeIn(veil), run_time=0.7)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=0.1 * UP), Create(line), run_time=0.9)
        chips = VGroup(*[chip(k + 1, name) for k, name in enumerate(CHAPTERS)])
        top = VGroup(*chips[:3]).arrange(RIGHT, buff=0.2)
        bottom = VGroup(*chips[3:]).arrange(RIGHT, buff=0.2)
        VGroup(top, bottom).arrange(DOWN, buff=0.2).move_to(DOWN * 1.75)
        safe(chips)
        self.play(LaggedStart(*[FadeIn(c, shift=0.1 * UP) for c in chips], lag_ratio=0.12), run_time=1.4)
        self.wait(2.0)
        keep = chips[0]
        self.play(keep[0].animate.set_stroke(YELLOW, width=3), keep[2].animate.set_color(YELLOW), run_time=0.5)
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not keep and not (set(m.get_family()) & set(keep.get_family()))], run_time=0.8)
        track, filled = bar(0)
        head = header_mob(1)
        # The chip box travels with its text and dissolves on arrival.
        box_end = keep[0].copy().set_stroke(opacity=0).set_fill(opacity=0) \
            .stretch_to_fit_width(head.width + 0.5).stretch_to_fit_height(head.height + 0.3).move_to(head)
        self.play(ReplacementTransform(keep[1], head[0]), ReplacementTransform(keep[2], head[1]),
                  Transform(keep[0], box_end), FadeIn(track), FadeIn(filled), run_time=1.1)
        self.remove(keep[0])
        self.wait(0.2)


class S01Trajectory(Scene):
    """Any scan geometry: one source and detector pose per view, the same Projector call."""
    def construct(self):
        hdr = chapter(self, 1)
        cap = Captions(self)
        view = View3D(centre=[-3.1, -0.1, 0], scale=1.12)
        kinds = ["circular", "helical", "saddle", "sinusoidal", "calibrated"]
        colors = [BLUE, GREEN, GOLD, VIOLET, RED]
        t = ValueTracker(0.0)
        state = {"k": 0}
        poses = calibrated_poses()

        def source():
            kind = kinds[state["k"]]
            if kind == "calibrated":
                return poses[int(t.get_value() * len(poses)) % len(poses)]
            return orbit(kind, t.get_value() % 1.0)

        def path():
            kind, color = kinds[state["k"]], colors[state["k"]]
            if kind == "calibrated":
                return VGroup(*[Dot(view(p), radius=0.05, color=color) for p in poses])
            pts = [view(orbit(kind, s)) for s in np.linspace(0, 1, 160)]
            return VMobject(color=color, stroke_width=3.5).set_points_as_corners(pts)

        cube = always_redraw(lambda: cube_edges(view))
        curve = always_redraw(path)
        rig = always_redraw(lambda: detector_and_rays(view, source(), colors[state["k"]]))

        # Right panel: the list of trajectories and the call that takes any of them.
        rows = VGroup()
        for kind, color in zip(kinds, colors):
            swatch = Line(ORIGIN, RIGHT * 0.5, color=color, stroke_width=5)
            label = T(kind, 32, SUB)
            left_at(label, 0.75)
            on_baseline(label, 0)
            swatch.move_to([0.25, label.get_center()[1], 0])
            rows.add(VGroup(swatch, label))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        left_at(rows, 1.0).set_y(1.0)
        call = VGroup(code(r"traj = (src\_pos, det\_center, det\_u, det\_v)", 22, width=5.3),
                      code(r"A = Projector(traj, (D, H, W), (U, V))", 22, width=5.3)).arrange(DOWN, buff=0.12)
        left_at(call, 1.0).set_y(-1.85)
        safe(rows), safe(call), apart(rows, call)

        self.play(FadeIn(cube), Create(curve), FadeIn(rig), FadeIn(rows), run_time=1.2)
        cap("each view has its own source, detector centre and detector axes", hold=0)
        self.play(rows[0][1].animate.set_color(colors[0]), run_time=0.4)
        self.play(t.animate.set_value(1.0), view.az.animate.increment_value(25), run_time=3.2, rate_func=linear)
        for k in range(1, len(kinds)):
            state["k"] = k
            t.set_value(0.0)
            self.play(rows[k - 1][1].animate.set_color(SUB), rows[k][1].animate.set_color(colors[k]),
                      run_time=0.4)
            if k == 2:
                cap("circular, helical, saddle, sinusoidal, or any list of per-view poses", hold=0)
            self.play(t.animate.set_value(1.0), view.az.animate.increment_value(20), run_time=2.4,
                      rate_func=linear)
        self.play(FadeIn(call, shift=0.1 * UP), run_time=0.7)
        cap("one Projector call for every scan geometry", hold=2.4)
        fade_except(self, hdr)


class S02Autograd(Scene):
    """project() and backproject() are a matched pair of autograd functions."""
    def construct(self):
        hdr = chapter(self, 2)
        cap = Captions(self)
        data = np.load(HERE / "data2d.npz")
        side = 2.6
        x_img = framed(gray_image(data["phantom"], side, vmax=1.0)).move_to([-4.2, 0.95, 0])
        y_img = framed(gray_image(data["sinogram"], side, width=side)).move_to([4.2, 0.95, 0])
        x_lab = on_baseline(T("volume $x$", 30, SUB).set_x(x_img.get_x()), x_img.get_bottom()[1] - 0.45)
        y_lab = on_baseline(T("sinogram $y$", 30, SUB).set_x(y_img.get_x()), y_img.get_bottom()[1] - 0.45)
        gap_l, gap_r = x_img.get_right()[0], y_img.get_left()[0]
        fwd_y, back_y = 1.55, 0.35
        fwd = ortho([gap_l + 0.15, fwd_y, 0], [gap_r - 0.15, fwd_y, 0], color=BLUE)
        back = ortho([gap_r - 0.15, back_y, 0], [gap_l + 0.15, back_y, 0], color=GOLD)
        fwd_lab = T(r"\texttt{A.project(x)}", 28, BLUE).next_to(fwd, UP, buff=0.14)
        back_lab = T(r"\texttt{A.backproject(y)}", 28, GOLD).next_to(back, DOWN, buff=0.14)
        adj = M(r"\langle A x,\; y\rangle \;=\; \langle x,\; A^{\top} y\rangle", 40, INK).move_to([0, -1.3, 0])
        grad = code(r"loss = ((A.project(x) - y)**2).sum() / 2;\ loss.backward()", 22)
        grad.move_to([0, -2.3, 0])
        result = M(r"\nabla_x\,\mathrm{loss} = A^{\top}(A x - y)", 30, GREEN)
        safe(Group(x_img, y_img, x_lab, y_lab, fwd_lab, back_lab, adj, grad))
        apart(fwd_lab, back_lab, gap=0.3)
        apart(adj, grad, gap=0.2)

        self.play(FadeIn(x_img), FadeIn(x_lab), run_time=0.7)
        self.play(GrowArrow(fwd[-1]), FadeIn(fwd_lab), run_time=0.8)
        self.play(FadeIn(y_img), FadeIn(y_lab), run_time=0.7)
        cap("project() traces every ray through the volume", hold=1.6)
        self.play(GrowArrow(back[-1]), FadeIn(back_lab), run_time=0.8)
        cap("backproject() walks the same rays in reverse", hold=1.2)
        self.play(Write(adj), run_time=1.0)
        cap("the two form an exact adjoint pair", hold=1.8)
        self.play(FadeIn(grad, shift=0.1 * UP), run_time=0.7)
        result.next_to(grad, RIGHT, buff=0.3)
        if result.get_right()[0] > XR:
            result.next_to(grad, UP, buff=0.12)
        cap("both are PyTorch autograd functions, so any loss can be differentiated", hold=2.4)
        fade_except(self, hdr)


class S03Geometry(Scene):
    """Trajectory tensors can require gradients: calibrate the scan from its projections."""
    def construct(self):
        hdr = chapter(self, 3)
        cap = Captions(self)
        history = json.loads((HERE / "calib_history.json").read_text())
        centre = np.array([-3.3, -0.05, 0])
        R = 2.05
        n = 24
        errors = np.random.default_rng(5).normal(0, math.radians(8), n)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = Circle(radius=R, color=GRID, stroke_width=1.5).move_to(centre)
        truth = VGroup(*[Dot(centre + R * np.array([math.cos(a), math.sin(a), 0]), radius=0.06, color=SUB)
                         for a in angles])
        slice_img = gray_image(np.load(HERE / "data2d.npz")["phantom"], 1.5, vmax=1.0).move_to(centre)
        volume = Group(slice_img, Square(1.5, stroke_color=VIOLET, stroke_width=1.8).move_to(centre))
        progress = ValueTracker(0.0)

        def estimate():
            s = 1 - progress.get_value()
            return VGroup(*[Dot(centre + R * np.array([math.cos(a + s * e), math.sin(a + s * e), 0]), radius=0.09,
                                color=RED) for a, e in zip(angles, errors)])

        dots = always_redraw(estimate)
        key_true = VGroup(Dot(radius=0.06, color=SUB), T("true poses", 28, SUB)).arrange(RIGHT, buff=0.15)
        key_est = VGroup(Dot(radius=0.09, color=RED), T("estimate (schematic)", 28, RED)).arrange(RIGHT, buff=0.15)
        legend = VGroup(key_true, key_est).arrange(RIGHT, buff=0.5)
        on_baseline(legend, -2.75).set_x(centre[0])

        steps = [h["step"] for h in history]
        logs = [math.log10(h["loss"]) for h in history]
        axes = Axes(x_range=[0, 150, 50], y_range=[1, 7, 2], x_length=4.6, y_length=3.0, tips=False,
                    axis_config={"color": GRID, "stroke_width": 1.5}).move_to([3.45, 0.05, 0])
        ylab = M(r"\log_{10}\,\mathrm{loss}", 26, SUB).rotate(PI / 2).next_to(axes, LEFT, buff=0.18)
        xlab = T("Adam step", 26, SUB).next_to(axes, DOWN, buff=0.16)
        flag = code(r"src\_pos.requires\_grad\_(True)", 22).next_to(axes, UP, buff=0.3)
        result = T(r"angle error $0.55^\circ \rightarrow 0.006^\circ$ in 150 steps", 28, INK)
        on_baseline(result, -2.75).set_x(axes.get_x())
        safe(Group(ring, legend, axes, ylab, xlab, flag, result))
        apart(legend, result, gap=0.3)

        self.play(Create(ring), FadeIn(volume), FadeIn(truth), FadeIn(dots), FadeIn(legend), run_time=1.0)
        cap("trajectory tensors can require gradients, like any other parameter", hold=0)
        self.play(FadeIn(flag, shift=0.1 * UP), run_time=0.6)
        self.wait(1.2)
        self.play(Create(axes), FadeIn(ylab), FadeIn(xlab), run_time=0.8)
        curve = always_redraw(lambda: axes.plot_line_graph(
            steps[:max(2, 1 + int(round(progress.get_value() * (len(steps) - 1))))],
            logs[:max(2, 1 + int(round(progress.get_value() * (len(steps) - 1))))],
            line_color=RED, add_vertex_dots=False, stroke_width=3.5))
        self.add(curve)
        cap("gradient descent recovers per-view angle errors from the projections", hold=0)
        self.play(progress.animate.set_value(1.0), run_time=5.0, rate_func=smooth)
        self.play(FadeIn(result), run_time=0.6)
        cap(r"real calibration run: $64^3$ volume, 360 views, detector shift recovered too", hold=2.4)
        fade_except(self, hdr)


class S04MultiGPU(Scene):
    """Views are split across GPUs in one process, or across processes and nodes."""
    def construct(self):
        hdr = chapter(self, 4)
        cap = Captions(self)
        per, n_gpu = 4, 8
        palette = GPU_COLORS + [interpolate_color(ManimColor(c), ManimColor(INK), 0.4) for c in GPU_COLORS]
        blocks = VGroup(*[Rectangle(width=0.27, height=0.55, stroke_width=0, fill_opacity=1,
                                    fill_color=palette[k // per]) for k in range(n_gpu * per)])
        blocks.arrange(RIGHT, buff=0.06).move_to([0.55, 2.0, 0])
        views = T("views", 30, SUB)
        right_at(views, blocks.get_left()[0] - 0.3)
        on_baseline(views, blocks.get_bottom()[1] + 0.12)

        nodes = VGroup()
        for node in range(2):
            gpus = VGroup()
            for k in range(4):
                c = palette[4 * node + k]
                box = RoundedRectangle(width=1.15, height=1.25, corner_radius=0.12, stroke_color=c, stroke_width=2,
                                       fill_color=c, fill_opacity=0.08)
                lab = on_baseline(T(f"GPU {k}", 26, c).set_x(box.get_x()), box.get_top()[1] - 0.36)
                gpus.add(VGroup(box, fits(lab, box)))
            gpus.arrange(RIGHT, buff=0.12)
            frame = SurroundingRectangle(gpus, buff=0.16, corner_radius=0.14, color=SUB, stroke_width=1.5)
            label = T(f"node {node + 1}", 28, SUB)
            left_at(label, frame.get_left()[0])
            on_baseline(label, frame.get_top()[1] + 0.14)
            nodes.add(VGroup(gpus, frame, label))
        nodes.arrange(RIGHT, buff=1.1).move_to([0, -0.05, 0])
        y_link = nodes[0][1].get_center()[1]
        link = ortho([nodes[0][1].get_right()[0], y_link, 0], [nodes[1][1].get_left()[0], y_link, 0],
                     color=RED, arrow=False)
        nccl = T("NCCL", 26, RED).next_to(link, UP, buff=0.1)
        one = code(r"Projector(..., devices=[0, 1, 2, 3])", 22)
        many = code(r"torchrun ...\ \ +\ \ Projector(..., distributed=True)", 22)
        left_at(one, nodes[0][1].get_left()[0]).set_y(-1.95)
        right_at(many, nodes[1][1].get_right()[0]).set_y(-1.95)
        safe(Group(blocks, views, nodes, one, many))
        apart(one, many, gap=0.2)

        self.play(LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.02), FadeIn(views), run_time=1.2)
        cap("projections are split by view; every GPU holds the full volume", hold=0)
        self.play(FadeIn(nodes[0]), run_time=0.7)

        def move_to_node(node):
            moves = []
            for g in range(4 * node, 4 * node + 4):
                group = VGroup(*blocks[g * per:(g + 1) * per])
                target = group.copy().arrange(RIGHT, buff=0.05).scale(0.85)
                target.move_to(nodes[node][0][g - 4 * node][0].get_center() + DOWN * 0.18)
                moves.append(Transform(group, target))
            return moves

        self.play(*move_to_node(0), FadeIn(one, shift=0.1 * UP), run_time=1.4)
        cap("one process can drive several GPUs", hold=1.4)
        self.play(FadeIn(nodes[1]), Create(link), FadeIn(nccl), run_time=0.9)
        self.play(*move_to_node(1), FadeOut(views), FadeIn(many, shift=0.1 * UP), run_time=1.4)
        cap("or one process per GPU, on one node or across nodes", hold=1.8)
        fade_except(self, hdr)

        # Measured: one CGLS iteration, 128^3 volume, 360 views, A100 64 GB (docs/assets/scaling.json).
        rows = [("1 GPU", 30.078, ""), ("4 GPUs, 1 node", 8.082, r"\enspace (3.7\texttimes)"),
                ("8 GPUs, 2 nodes", 5.591, r"\enspace (5.4\texttimes)")]
        scale = 6.2 / 30.078
        x0 = -2.2
        chart = VGroup()
        for k, (name, ms, speed) in enumerate(rows):
            y = 1.15 - 1.25 * k
            bar_mob = Rectangle(width=ms * scale, height=0.75, stroke_width=0, fill_color=GPU_COLORS[k],
                                fill_opacity=0.85)
            left_at(bar_mob, x0).set_y(y)
            label = T(name, 32, SUB)
            right_at(label, x0 - 0.3)
            on_baseline(label, y - 0.12)
            value = T(f"{ms:.1f} ms" + speed, 32, INK)
            left_at(value, bar_mob.get_right()[0] + 0.25)
            on_baseline(value, y - 0.12)
            chart.add(VGroup(label, bar_mob, value))
        title = T(r"one CGLS iteration, $128^3$ volume, 360 views, A100 64 GB", 30, SUB)
        on_baseline(title, 2.15).set_x(0)
        safe(Group(chart, title))
        self.play(FadeIn(title), run_time=0.5)
        for row in chart:
            self.play(FadeIn(row[0]), GrowFromEdge(row[1], LEFT), run_time=0.8)
            self.play(FadeIn(row[2]), run_time=0.35)
        cap("measured on A100 64 GB GPUs; small volumes scale less", hold=2.6)
        fade_except(self, hdr)


class S05Measured(Scene):
    """A real walnut scan, reconstructed four ways from the same scan geometry."""
    def construct(self):
        hdr = chapter(self, 5)
        cap = Captions(self)
        data = np.load(HERE / "walnut_measured.npz")
        names = [("fdk", "FDK"), ("sirt", "SIRT"), ("cgls", "CGLS"), ("tv", "TV")]
        size = 2.25
        columns = Group()
        titles = VGroup()
        for key, label in names:
            cells = Group(*[framed(gray_image(data[f"{key}_{s}"], size, vmax=1.0)) for s in ("axial", "coronal")])
            cells.arrange(DOWN, buff=0.14)
            columns.add(cells)
        columns.arrange(RIGHT, buff=0.18).move_to([0.25, -0.2, 0])
        for (key, label), cells in zip(names, columns):
            titles.add(on_baseline(T(label, 32, INK).set_x(cells.get_x()), cells.get_top()[1] + 0.2))
        rows = VGroup(*[T(s, 28, SUB).rotate(PI / 2).next_to(columns[0][k], LEFT, buff=0.18)
                        for k, s in enumerate(("axial", "coronal"))])
        safe(Group(columns, titles, rows))

        self.play(FadeIn(rows), FadeIn(columns[0]), FadeIn(titles[0]), run_time=0.8)
        cap("240 measured cone-beam views of a walnut (Meaney 2022, CC BY 4.0)", hold=1.2)
        for k in range(1, 4):
            self.play(FadeIn(columns[k], shift=0.1 * UP), FadeIn(titles[k]), run_time=0.7)
        cap("analytical FDK and iterative SIRT, CGLS and TV on the same scan geometry", hold=3.0)
        fade_except(self, hdr)


class S06End(Scene):
    """Install and links."""
    def construct(self):
        track, filled = bar(len(CHAPTERS))
        self.add(track, filled, header_mob(len(CHAPTERS)))
        fade_all(self, run_time=0.7)
        name = T(r"\textbf{diffct}", 110, INK)
        install = code(r'pip install "diffct[cu12]"', 34)
        links = VGroup(T(r"\texttt{sypsyp97.github.io/diffct}", 32, SUB),
                       T(r"\texttt{github.com/sypsyp97/diffct}", 32, SUB)).arrange(DOWN, buff=0.22)
        VGroup(name, install, links).arrange(DOWN, buff=0.5).move_to(ORIGIN)
        safe(VGroup(name, install, links), top=3.6, bottom=-3.6)
        self.play(FadeIn(name), run_time=0.8)
        self.play(FadeIn(install, shift=0.1 * UP), run_time=0.7)
        self.play(FadeIn(links, shift=0.1 * UP), run_time=0.7)
        self.wait(3.0)
        fade_all(self, run_time=1.0)
