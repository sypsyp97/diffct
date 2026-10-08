"""diffct intro video (Manim CE). Render each scene, then concatenate.

Data files (same directory): data2d.npz (2D phantom + parallel sinogram),
calib_history.json (real calibration run), recon_slices.npz + recon_psnr.json
(real diffct reconstructions on Leonardo).
"""
import json
import math
from pathlib import Path

import numpy as np
from manim import *

HERE = Path(__file__).parent
BG = "#101317"
INK = "#E6E4E0"
SUB = "#98A0AA"
GRID = "#3A4048"
BLUE = "#7FA7D9"
TEAL = "#6CC3B8"
VIOLET = "#A99BD8"
ROSE = "#E9A8C1"
ROSE_DEEP = "#C76E91"
GPU_COLORS = ["#7FA7D9", "#6CC3B8", "#A99BD8", "#D9A17F"]

config.background_color = BG
SANS = TexTemplate()
SANS.add_to_preamble(r"\usepackage[scaled=1.0]{helvet}\renewcommand{\familydefault}{\sfdefault}")


def T(text, size=32, color=INK):
    return Tex(text, tex_template=SANS, font_size=size, color=color)


def M(tex, size=36, color=INK):
    return MathTex(tex, font_size=size, color=color)


def gray_image(array, vmax, height):
    img = np.clip(array / vmax, 0, 1)
    rgb = (np.stack([img] * 3, axis=-1) * 255).astype(np.uint8)
    mob = ImageMobject(rgb)
    mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    mob.height = height
    return mob


def caption(scene, text):
    label = T(text, 34, INK).to_edge(DOWN, buff=0.45)
    scene.play(FadeIn(label, shift=0.15 * UP), run_time=0.6)
    return label


class S1Title(Scene):
    def construct(self):
        name = T(r"\textbf{diffct}", 96)
        tag = T("Differentiable CT projectors on GPUs", 40, SUB).next_to(name, DOWN, buff=0.35)
        line = Line(LEFT * 3.2, RIGHT * 3.2, color=ROSE, stroke_width=3).next_to(tag, DOWN, buff=0.35)
        self.play(Write(name), run_time=1.2)
        self.play(FadeIn(tag, shift=0.2 * UP), Create(line), run_time=1.0)
        self.wait(1.2)
        self.play(FadeOut(VGroup(name, tag, line)), run_time=0.6)


class S2Projection(Scene):
    def construct(self):
        data = np.load(HERE / "data2d.npz")
        phantom, sino, angles = data["phantom"], data["sinogram"], data["angles"]
        size = 3.1
        img = gray_image(phantom, 0.45, size).move_to(LEFT * 3.2 + UP * 0.1)
        frame = Square(size, color=GRID, stroke_width=1.5).move_to(img)
        sino_img = gray_image(sino, sino.max(), 3.6)
        sino_img.stretch_to_fit_width(3.0).move_to(RIGHT * 3.6 + UP * 0.6)
        sino_frame = Rectangle(width=3.0, height=3.6, color=GRID, stroke_width=1.5).move_to(sino_img)
        cover = Rectangle(width=3.04, height=3.64, fill_color=BG, fill_opacity=1, stroke_width=0).move_to(sino_img)
        head2 = T(r"sinogram (angle $\downarrow$, detector $\rightarrow$)", 26, SUB).next_to(sino_frame, UP, buff=0.2)
        self.add(img, frame, sino_img, cover, sino_frame)
        self.play(FadeIn(head2), run_time=0.5)

        theta = ValueTracker(0.0)
        centre = img.get_center()
        half = size / 2
        n_rays = 11

        def geometry():
            a = math.radians(theta.get_value())
            d = np.array([math.sin(a), -math.cos(a), 0.0])     # ray direction
            u = np.array([math.cos(a), math.sin(a), 0.0])      # detector axis
            return d, u

        def rays():
            d, u = geometry()
            group = VGroup()
            for k in np.linspace(-0.85, 0.85, n_rays):
                p = centre + k * half * u
                group.add(Line(p - 1.25 * half * d, p + 1.25 * half * d, color=BLUE, stroke_width=1.6,
                               stroke_opacity=0.75))
            return group

        def detector():
            d, u = geometry()
            base = centre + 1.35 * half * d
            row = sino[min(int(round(theta.get_value())), len(angles) - 1)]
            prof = row / sino.max()
            xs = np.linspace(-1, 1, len(prof)) * half
            pts = [base + x * u + 0.55 * p * d for x, p in zip(xs, prof)]
            line = Line(base - half * u, base + half * u, color=GRID, stroke_width=2)
            curve = VMobject(color=TEAL, stroke_width=2.6).set_points_smoothly(pts[::3])
            return VGroup(line, curve)

        ray_group = always_redraw(rays)
        det_group = always_redraw(detector)
        cover.add_updater(lambda m: m.stretch_to_fit_height(max(3.64 * (1 - theta.get_value() / 180), 1e-3))
                          .align_to(sino_frame, DOWN))
        self.play(Create(ray_group), FadeIn(det_group), run_time=0.8)
        formula = M(r"y_i = \sum_k f_k\,\ell_{ik}", 40).next_to(sino_frame, DOWN, buff=0.45)
        self.play(Write(formula), run_time=0.8)
        self.play(theta.animate.set_value(180), run_time=5.5, rate_func=linear)
        cover.clear_updaters()
        self.wait(0.4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)


class S3Siddon(Scene):
    def construct(self):
        n = 7
        cell = 0.62
        rng = np.random.default_rng(3)
        values = rng.uniform(0.15, 0.85, (n, n))
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                sq = Square(cell, stroke_color=GRID, stroke_width=1.2,
                            fill_color=interpolate_color(ManimColor(BG), ManimColor("#B9C2CC"), values[i, j] * 0.55),
                            fill_opacity=1)
                sq.move_to(np.array([(j - (n - 1) / 2) * cell, ((n - 1) / 2 - i) * cell, 0]))
                cells.add(sq)
        cells.shift(LEFT * 2.6)
        a = cells.get_corner(DL) + np.array([-0.6, 0.55, 0])
        b = cells.get_corner(UR) + np.array([0.6, -1.15, 0])
        ray = Line(a, b, color=BLUE, stroke_width=3)
        self.play(FadeIn(cells), run_time=0.6)
        self.play(Create(ray), run_time=0.8)

        # Crossed cells: sample the ray finely and collect cells in order.
        crossed, pieces = [], []
        ts = np.linspace(0, 1, 2000)
        pts = [a + t * (b - a) for t in ts]
        origin = cells.get_corner(UL)
        last, start = None, None
        for p in pts:
            j = int((p[0] - origin[0]) // cell)
            i = int((origin[1] - p[1]) // cell)
            key = (i, j) if 0 <= i < n and 0 <= j < n else None
            if key != last:
                if last is not None:
                    crossed.append(last); pieces.append((start, p))
                last, start = key, p
        highlights = VGroup()
        segs = VGroup()
        for (i, j), (p, q) in zip(crossed, pieces):
            sq = cells[i * n + j]
            highlights.add(sq.copy().set_fill(ROSE, 0.35).set_stroke(ROSE, 2))
            segs.add(Line(p, q, color=ROSE, stroke_width=6))
        eq = M(r"y = \sum_k f_k\,\ell_k", 46).next_to(cells, RIGHT, buff=1.6).shift(UP * 1.4)
        self.play(Write(eq), run_time=0.7)
        self.play(LaggedStart(*[AnimationGroup(FadeIn(h), Create(s)) for h, s in zip(highlights, segs)],
                              lag_ratio=0.35), run_time=2.6)
        notes = VGroup(
            T(r"exact cell-constant Siddon ray tracing", 28, INK),
            T(r"backprojection = the same traversal: exact $A^{\top}$", 28, INK),
            T(r"one CUDA thread per ray", 28, SUB),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).next_to(cells, RIGHT, buff=0.7).shift(DOWN * 0.6)
        self.play(LaggedStart(*[FadeIn(t, shift=0.1 * RIGHT) for t in notes], lag_ratio=0.4), run_time=1.6)
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)


class S4Trajectories(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-50 * DEGREES, zoom=1.2)
        cube = Cube(side_length=1.6, fill_color=VIOLET, fill_opacity=0.12, stroke_color=VIOLET, stroke_width=1.2)
        R = 2.6

        def path(kind):
            def f(t):
                a = 2 * np.pi * t
                z = {"circular": 0.0, "helical": -1.0 + 2.0 * t, "saddle": 0.7 * np.cos(2 * a),
                     "sinusoidal": 0.55 * np.sin(3 * a)}[kind]
                return np.array([R * np.cos(a), R * np.sin(a), z])
            return f

        names = ["circular", "helical", "saddle", "sinusoidal"]
        colors = [BLUE, TEAL, "#D9A17F", VIOLET]
        title = T("any trajectory, one view at a time", 34, INK).to_edge(UP, buff=0.4)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(cube), FadeIn(title), run_time=0.7)
        t = ValueTracker(0.0)
        current = {"f": path("circular")}

        def source_and_cone():
            p = current["f"](t.get_value() % 1.0)
            centre = -p * 0.75
            radial = p / np.linalg.norm(p)
            tangent = np.cross(np.array([0, 0, 1.0]), radial)
            corners = [centre + 0.9 * su * tangent + 0.7 * sv * np.array([0, 0, 1.0])
                       for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            detector = Polygon(*corners, color=SUB, stroke_width=1.5, fill_color=SUB, fill_opacity=0.15)
            lines = VGroup(*[Line(p, c, color=BLUE, stroke_width=1.2, stroke_opacity=0.6) for c in corners])
            dot = Dot3D(p, radius=0.07, color=ROSE)
            return VGroup(detector, lines, dot)

        cone = always_redraw(source_and_cone)
        curve = ParametricFunction(path("circular"), t_range=[0, 1], color=BLUE, stroke_width=3)
        label = T("circular", 36, BLUE).to_corner(DL, buff=0.7)
        self.add_fixed_in_frame_mobjects(label)
        self.play(Create(curve), FadeIn(cone), FadeIn(label), run_time=1.0)
        self.begin_ambient_camera_rotation(rate=0.12)
        self.play(t.animate.set_value(1.0), run_time=1.6, rate_func=linear)
        for name, color in zip(names[1:], colors[1:]):
            new_curve = ParametricFunction(path(name), t_range=[0, 1], color=color, stroke_width=3)
            new_label = T(name, 36, color).to_corner(DL, buff=0.7)
            current["f"] = path(name)
            self.remove(label)
            self.add_fixed_in_frame_mobjects(new_label)
            self.play(Transform(curve, new_curve), FadeIn(new_label), run_time=0.8)
            label = new_label
            self.play(t.animate.increment_value(1.0), run_time=1.4, rate_func=linear)
        self.stop_ambient_camera_rotation()
        self.wait(0.3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)


class S5MultiGPU(Scene):
    def construct(self):
        title = T("views split across GPUs and nodes", 34, INK).to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)
        n_gpu, per = 8, 4
        palette = GPU_COLORS + [interpolate_color(ManimColor(c), ManimColor(INK), 0.35) for c in GPU_COLORS]
        blocks = VGroup(*[Rectangle(width=0.2, height=0.45, stroke_width=0, fill_opacity=1,
                                    fill_color=palette[k // per]) for k in range(n_gpu * per)])
        blocks.arrange(RIGHT, buff=0.05).move_to(UP * 2.0)
        views = T("360 views", 26, SUB).next_to(blocks, LEFT, buff=0.3)
        self.play(LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.02), FadeIn(views), run_time=1.0)

        nodes = VGroup()
        for node in range(2):
            gpus = VGroup()
            for k in range(4):
                c = palette[4 * node + k]
                box = RoundedRectangle(width=1.3, height=0.95, corner_radius=0.1, stroke_color=c,
                                       stroke_width=2, fill_color=c, fill_opacity=0.08)
                box.add(T(f"GPU {k}", 22, c).move_to(box.get_top() + DOWN * 0.2))
                gpus.add(box)
            gpus.arrange(RIGHT, buff=0.14)
            frame = SurroundingRectangle(gpus, buff=0.16, corner_radius=0.14, color=SUB, stroke_width=1.5)
            label = T(f"node {node + 1}", 24, SUB).next_to(frame, UP, buff=0.1).align_to(frame, LEFT)
            nodes.add(VGroup(gpus, frame, label))
        nodes.arrange(RIGHT, buff=1.0).move_to(DOWN * 0.1)
        link = Line(nodes[0][1].get_right(), nodes[1][1].get_left(), color=ROSE, stroke_width=3)
        nccl = T("NCCL", 22, ROSE).next_to(link, UP, buff=0.08)
        self.play(FadeIn(nodes[0]), run_time=0.6)
        moves = []
        for g in range(4):
            group = VGroup(*blocks[g * per:(g + 1) * per])
            target = group.copy().arrange(RIGHT, buff=0.04).move_to(nodes[0][0][g].get_center() + DOWN * 0.15)
            moves.append(Transform(group, target))
        note1 = T(r"one process: \texttt{Projector(..., devices=[0, 1, 2, 3])}", 26, INK).to_edge(DOWN, buff=1.2)
        self.play(*moves, FadeIn(note1), run_time=1.1)
        self.play(FadeIn(nodes[1]), Create(link), FadeIn(nccl), run_time=0.7)
        moves = []
        for g in range(4, 8):
            group = VGroup(*blocks[g * per:(g + 1) * per])
            target = group.copy().arrange(RIGHT, buff=0.04).move_to(nodes[1][0][g - 4].get_center() + DOWN * 0.15)
            moves.append(Transform(group, target))
        note2 = T(r"one process per GPU: \texttt{torchrun ...}\ + \texttt{Projector(..., distributed=True)}", 26, INK)
        note2.next_to(note1, DOWN, buff=0.25)
        self.play(*moves, FadeOut(views), FadeIn(note2), run_time=1.1)
        self.wait(0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)

        # Measured CGLS iteration time, 128^3 Shepp-Logan, 360 views, A100 64 GB (Leonardo).
        head = T(r"CGLS iteration, $128^3$ volume, 360 views, A100", 32, INK).to_edge(UP, buff=0.5)
        rows = [("1 GPU", 30.07, ""), ("4 GPUs, 1 node", 8.09, r"(3.7\texttimes)"),
                ("8 GPUs, 2 nodes", 5.54, r"(5.4\texttimes)")]
        scale = 7.0 / 30.07
        bars = VGroup()
        for k, (name, ms, speed) in enumerate(rows):
            y = 1.2 - 1.3 * k
            label = T(name, 28, SUB).move_to(np.array([-4.6, y, 0])).align_to(np.array([-3.1, 0, 0]), RIGHT)
            bar = Rectangle(width=ms * scale, height=0.62, stroke_width=0, fill_color=GPU_COLORS[min(k, 3)],
                            fill_opacity=0.85).move_to(np.array([-2.9, y, 0]), aligned_edge=LEFT)
            value = T(f"{ms:.1f} ms " + speed, 28, INK).next_to(bar, RIGHT, buff=0.2)
            bars.add(VGroup(label, bar, value))
        self.play(FadeIn(head), run_time=0.5)
        for row in bars:
            self.play(FadeIn(row[0]), GrowFromEdge(row[1], LEFT), run_time=0.7)
            self.play(FadeIn(row[2]), run_time=0.3)
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)


class S6Reconstruction(Scene):
    def construct(self):
        slices = np.load(HERE / "recon_slices.npz")
        psnr = json.loads((HERE / "recon_psnr.json").read_text())
        title = T(r"helical cone beam, $128^3$ Shepp-Logan, real diffct output", 32, INK).to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)
        panels = [("phantom", "phantom", None), ("fdk", "FDK", "fdk"), ("sirt", "SIRT", "sirt"),
                  ("cgls", "CGLS", "cgls"), ("tv", "TV + Adam", "tv")]
        group = Group()
        for key, name, metric in panels:
            img = gray_image(slices[key], 0.4, 2.3)
            label = T(name, 28, INK).next_to(img, UP, buff=0.18)
            parts = [img, label]
            caption_text = f"{psnr[metric]:.1f} dB" if metric else "ground truth"
            parts.append(T(caption_text, 26, SUB).next_to(img, DOWN, buff=0.18))
            group.add(Group(*parts))
        group.arrange(RIGHT, buff=0.3).move_to(DOWN * 0.1)
        for g in group:
            self.play(FadeIn(g, shift=0.1 * UP), run_time=0.55)
        note = T("analytical and iterative reconstruction from the same projector", 28, SUB).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)


class S7Geometry(Scene):
    def construct(self):
        history = json.loads((HERE / "calib_history.json").read_text())
        title = T("gradients through the geometry: calibrate the trajectory", 32, INK).to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)
        centre = LEFT * 3.2 + DOWN * 0.3
        R = 2.1
        n = 24
        rng = np.random.default_rng(5)
        errors = rng.normal(0, math.radians(7), n)
        true_pts = [centre + R * np.array([math.cos(a), math.sin(a), 0]) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
        ring = Circle(radius=R, color=GRID, stroke_width=1.5).move_to(centre)
        ghosts = VGroup(*[Dot(p, radius=0.05, color=SUB) for p in true_pts])
        progress = ValueTracker(0.0)

        def dots():
            s = 1 - progress.get_value()
            group = VGroup()
            for k, a in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
                b = a + s * errors[k]
                p = centre + R * np.array([math.cos(b), math.sin(b), 0])
                group.add(Dot(p, radius=0.075, color=ROSE))
                if s > 0.05:
                    group.add(Arrow(p, true_pts[k], buff=0.08, stroke_width=2.5, color=ROSE_DEEP,
                                    max_tip_length_to_length_ratio=0.35, tip_length=0.12))
            return group

        moving = always_redraw(dots)
        legend = VGroup(
            VGroup(Dot(radius=0.06, color=SUB), T("true source positions", 24, SUB)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(radius=0.07, color=ROSE), T("current estimate", 24, ROSE)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(ring, DOWN, buff=0.35)
        self.play(Create(ring), FadeIn(ghosts), FadeIn(moving), FadeIn(legend), run_time=0.8)

        steps = [h["step"] for h in history]
        logs = [math.log10(h["loss"]) for h in history]
        axes = Axes(x_range=[0, 150, 50], y_range=[1, 7, 2], x_length=4.6, y_length=3.2,
                    axis_config={"color": GRID, "stroke_width": 1.5, "include_ticks": True},
                    tips=False).move_to(RIGHT * 3.3 + DOWN * 0.2)
        ylab = M(r"\log_{10}\,\mathrm{loss}", 26, SUB).next_to(axes, LEFT, buff=0.15).rotate(PI / 2)
        xlab = T("Adam step", 24, SUB).next_to(axes, DOWN, buff=0.2)
        self.play(Create(axes), FadeIn(ylab), FadeIn(xlab), run_time=0.6)
        curve = always_redraw(lambda: axes.plot_line_graph(
            steps[:max(2, 1 + int(progress.get_value() * (len(steps) - 1)))],
            logs[:max(2, 1 + int(progress.get_value() * (len(steps) - 1)))],
            line_color=ROSE, add_vertex_dots=False, stroke_width=3))
        self.add(curve)
        eq = M(r"\frac{\partial\, \|A(\theta)\,x - y\|^2}{\partial \theta}", 34, INK).next_to(axes, UP, buff=0.25)
        self.play(Write(eq), run_time=0.6)
        self.play(progress.animate.set_value(1.0), run_time=4.0, rate_func=smooth)
        result = T(r"angle error $0.55^\circ \rightarrow 0.006^\circ$ in 150 steps (64$^3$, real run)", 26, INK)
        result.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(result), run_time=0.5)
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)


class S8End(Scene):
    def construct(self):
        name = T(r"\textbf{diffct}", 80)
        lines = VGroup(
            T(r"parallel · fan · cone beam\quad{}\textbar\quad{}any trajectory\quad{}\textbar\quad{}exact adjoint", 30, SUB),
            T("PyTorch autograd for volumes, sinograms and geometry", 30, SUB),
            T("one GPU · many GPUs · many nodes", 30, SUB),
            T(r"\texttt{github.com/sypsyp97/diffct}", 32, ROSE),
        ).arrange(DOWN, buff=0.28).next_to(name, DOWN, buff=0.5)
        self.play(FadeIn(name), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(l, shift=0.1 * UP) for l in lines], lag_ratio=0.3), run_time=1.6)
        self.wait(1.8)
