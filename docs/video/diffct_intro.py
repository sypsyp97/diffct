"""diffct intro video (Manim CE). Render each scene, then concatenate.

Data files (same directory), written by make_inputs.py: data2d.npz (walnut slice and
its parallel sinogram), walnut_measured.npz (measured walnut reconstructions),
recon_slices.npz + recon_psnr.json (simulated helical scan of the walnut volume),
calib_history.json (real calibration run). All reconstructions are diffct output
on Leonardo A100 GPUs.
"""
import json
import math
from pathlib import Path

import numpy as np
from manim import *

HERE = Path(__file__).parent
# 3Blue1Brown palette: near-black background, Manim's BLUE_C / GREEN_C / YELLOW / GOLD_C / RED_C.
BG = "#0E1013"
INK = "#ECECEC"
SUB = "#A0A6AE"
GRID = "#3C4650"
BLUE = "#58C4DD"
TEAL = "#83C167"
VIOLET = "#9A72AC"
CLAY = "#F0AC5F"
YELLOW = "#F9E04C"
ROSE = "#FC6255"
ROSE_DEEP = "#C55F73"
GPU_COLORS = [BLUE, TEAL, YELLOW, CLAY]
CHAPTERS = ["Projection", "Ray tracing", "Trajectories", "Multi-GPU", "Measured data", "Helical scan",
            "Geometry gradients"]

config.background_color = BG


def T(text, size=34, color=INK):
    return Tex(text, font_size=size, color=color)


def M(tex, size=36, color=INK):
    return MathTex(tex, font_size=size, color=color)


def gray_image(array, vmax, height):
    img = np.clip(np.asarray(array, dtype=np.float32) / vmax, 0, 1)
    rgb = (np.stack([img] * 3, axis=-1) * 255).astype(np.uint8)
    mob = ImageMobject(rgb)
    mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
    mob.height = height
    return mob


def chapter(scene, index, title, fixed=False):
    """Chapter header at the top left and a progress bar along the bottom edge."""
    number = T(f"{index:02d}", 32, ROSE)
    name = T(title, 44, INK)
    header = VGroup(number, name).arrange(RIGHT, buff=0.3, aligned_edge=DOWN).to_corner(UL, buff=0.5)
    left, right = -config.frame_width / 2 + 0.5, config.frame_width / 2 - 0.5
    y = -config.frame_height / 2 + 0.22
    track = Line([left, y, 0], [right, y, 0], color=GRID, stroke_width=3)
    done = left + (right - left) * (index - 1) / len(CHAPTERS)
    stop = left + (right - left) * index / len(CHAPTERS)
    filled = Line([left, y, 0], [max(done, left + 1e-3), y, 0], color=ROSE, stroke_width=3)
    target = Line([left, y, 0], [stop, y, 0], color=ROSE, stroke_width=3)
    if fixed:
        scene.add_fixed_in_frame_mobjects(header, track, filled)
    scene.play(FadeIn(header, shift=0.15 * RIGHT), FadeIn(track), FadeIn(filled), run_time=0.8)
    scene.play(Transform(filled, target), run_time=0.8)
    return VGroup(header, track, filled)


def note(text, size=30, color=SUB):
    return T(text, size, color).to_edge(DOWN, buff=0.6)


def fade_all(scene, run_time=0.8):
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=run_time)


class S1Title(Scene):
    def construct(self):
        name = T(r"\textbf{diffct}", 132)
        tag = T("Differentiable CT projectors on GPUs", 48, SUB).next_to(name, DOWN, buff=0.4)
        line = Line(LEFT * 4.2, RIGHT * 4.2, color=ROSE, stroke_width=3).next_to(tag, DOWN, buff=0.45)
        chips = VGroup(*[
            VGroup(RoundedRectangle(width=4.3, height=0.85, corner_radius=0.2, stroke_color=c, stroke_width=2,
                                    fill_color=c, fill_opacity=0.08), T(text, 34, c))
            for text, c in (("any trajectory", BLUE), ("many GPUs, many nodes", TEAL),
                            ("geometry gradients", VIOLET))
        ]).arrange(RIGHT, buff=0.35).next_to(line, DOWN, buff=0.6)
        for chip in chips:
            chip[1].move_to(chip[0])
        VGroup(name, tag, line, chips).move_to(ORIGIN)
        self.play(Write(name), run_time=1.6)
        self.play(FadeIn(tag, shift=0.2 * UP), Create(line), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(c, shift=0.15 * UP) for c in chips], lag_ratio=0.35), run_time=1.6)
        self.wait(2.4)
        fade_all(self)


class S2Projection(Scene):
    def construct(self):
        chapter(self, 1, "A forward projection is a line integral")
        data = np.load(HERE / "data2d.npz")
        phantom, sino = data["phantom"], data["sinogram"]
        size = 4.2
        img = gray_image(phantom, 1.0, size).move_to(LEFT * 3.3 + DOWN * 0.45)
        frame = Square(size, color=GRID, stroke_width=1.5).move_to(img)
        sino_h, sino_w = 4.6, 4.0
        sino_img = gray_image(sino, sino.max(), sino_h).stretch_to_fit_width(sino_w).move_to(RIGHT * 3.9 + DOWN * 0.15)
        sino_frame = Rectangle(width=sino_w, height=sino_h, color=GRID, stroke_width=1.5).move_to(sino_img)
        cover = Rectangle(width=sino_w + 0.04, height=sino_h + 0.04, fill_color=BG, fill_opacity=1,
                          stroke_width=0).move_to(sino_img)
        head1 = T("walnut slice $f$", 34, SUB).next_to(frame, DOWN, buff=0.3)
        head2 = T(r"sinogram $y$ (angle $\downarrow$, detector $\rightarrow$)", 34, SUB).next_to(sino_frame, DOWN, buff=0.3)
        self.play(FadeIn(img), Create(frame), FadeIn(head1), run_time=1.0)
        self.add(sino_img, cover, sino_frame)
        self.play(FadeIn(head2), Create(sino_frame), run_time=0.8)

        theta = ValueTracker(0.0)
        centre = img.get_center()
        half = size / 2
        n_rays = 15

        def geometry():
            a = math.radians(theta.get_value())
            return np.array([math.sin(a), -math.cos(a), 0.0]), np.array([math.cos(a), math.sin(a), 0.0])

        def rays():
            d, u = geometry()
            group = VGroup()
            for k in np.linspace(-0.9, 0.9, n_rays):
                p = centre + k * half * u
                group.add(Line(p - 1.12 * half * d, p + 1.12 * half * d, color=BLUE, stroke_width=1.8,
                               stroke_opacity=0.7))
            return group

        def detector():
            d, u = geometry()
            base = centre + 1.2 * half * d
            row = sino[min(int(round(theta.get_value())), len(sino) - 1)]
            prof = row / sino.max()
            xs = np.linspace(-1, 1, len(prof)) * half
            pts = [base + x * u + 0.45 * p * d for x, p in zip(xs, prof)]
            line = Line(base - half * u, base + half * u, color=GRID, stroke_width=2)
            curve = VMobject(color=TEAL, stroke_width=3).set_points_smoothly(pts[::3])
            return VGroup(line, curve)

        ray_group = always_redraw(rays)
        det_group = always_redraw(detector)
        cover.add_updater(lambda m: m.stretch_to_fit_height(max((sino_h + 0.04) * (1 - theta.get_value() / 180), 1e-3))
                          .align_to(sino_frame, DOWN))
        self.play(Create(ray_group), FadeIn(det_group), FadeOut(head1), run_time=1.2)
        formula = M(r"y_i = \sum_k f_k\,\ell_{ik}", 48).to_corner(UR, buff=0.5)
        self.play(Write(formula), run_time=1.0)
        self.play(theta.animate.set_value(180), run_time=9.0, rate_func=linear)
        cover.clear_updaters()
        self.play(FadeIn(head1), run_time=0.6)
        self.wait(1.5)
        fade_all(self)


class S3Siddon(Scene):
    def construct(self):
        chapter(self, 2, "Exact ray tracing, exact adjoint")
        n = 9
        cell = 0.66
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
        cells.move_to(LEFT * 3.3 + DOWN * 0.3)
        a = cells.get_corner(DL) + np.array([-0.7, 0.75, 0])
        b = cells.get_corner(UR) + np.array([0.7, -1.4, 0])
        ray = Line(a, b, color=BLUE, stroke_width=3.5)
        self.play(FadeIn(cells), run_time=1.0)
        self.play(Create(ray), run_time=1.2)

        crossed, pieces = [], []
        origin = cells.get_corner(UL)
        last, start = None, None
        for t in np.linspace(0, 1, 4000):
            p = a + t * (b - a)
            j = int((p[0] - origin[0]) // cell)
            i = int((origin[1] - p[1]) // cell)
            key = (i, j) if 0 <= i < n and 0 <= j < n else None
            if key != last:
                if last is not None:
                    crossed.append(last)
                    pieces.append((start, p))
                last, start = key, p
        highlights = VGroup()
        segs = VGroup()
        for (i, j), (p, q) in zip(crossed, pieces):
            highlights.add(cells[i * n + j].copy().set_fill(ROSE, 0.35).set_stroke(ROSE, 2))
            segs.add(Line(p, q, color=ROSE, stroke_width=7))
        eq = M(r"y = \sum_k f_k\,\ell_k", 56).move_to(RIGHT * 3.6 + UP * 1.7)
        self.play(Write(eq), run_time=1.0)
        self.play(LaggedStart(*[AnimationGroup(FadeIn(h), Create(s)) for h, s in zip(highlights, segs)],
                              lag_ratio=0.4), run_time=4.5)
        notes = VGroup(
            T(r"$\ell_k$: exact length of the ray in cell $k$", 36, INK),
            T(r"backprojection walks the same cells: $A^{\top}$", 36, INK),
            T(r"one CUDA thread per ray, float32", 36, SUB),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to(RIGHT * 3.6 + DOWN * 0.7)
        for line in notes:
            self.play(FadeIn(line, shift=0.15 * RIGHT), run_time=0.9)
            self.wait(0.5)
        check = M(r"\langle A x,\, y\rangle = \langle x,\, A^{\top} y\rangle", 40, TEAL).next_to(notes, DOWN, buff=0.55)
        self.play(Write(check), run_time=1.2)
        self.wait(2.0)
        fade_all(self)


class S4Trajectories(ThreeDScene):
    def construct(self):
        chapter(self, 3, "Any trajectory, one pose per view", fixed=True)
        self.set_camera_orientation(phi=66 * DEGREES, theta=-50 * DEGREES, zoom=1.35)
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
        colors = [BLUE, TEAL, CLAY, VIOLET]
        menu = VGroup(*[T(name, 36, SUB) for name in names],
                      T("calibrated poses", 36, SUB)).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        menu.to_edge(RIGHT, buff=0.6).shift(UP * 0.2)
        api = T(r"\texttt{Projector((source, detector, u, v), ...)}", 32, SUB).to_edge(DOWN, buff=0.45)
        self.add_fixed_in_frame_mobjects(menu, api)
        self.play(FadeIn(cube), FadeIn(menu), FadeIn(api), run_time=1.0)
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
            return VGroup(detector, lines, Dot3D(p, radius=0.07, color=ROSE))

        def highlight(k):
            return [menu[j].animate.set_color(colors[k] if j == k else SUB) for j in range(len(names))]

        cone = always_redraw(source_and_cone)
        curve = ParametricFunction(path("circular"), t_range=[0, 1], color=BLUE, stroke_width=3.5)
        self.play(Create(curve), FadeIn(cone), *highlight(0), run_time=1.2)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.play(t.animate.set_value(1.0), run_time=3.0, rate_func=linear)
        for k, name in enumerate(names[1:], start=1):
            current["f"] = path(name)
            new_curve = ParametricFunction(path(name), t_range=[0, 1], color=colors[k], stroke_width=3.5)
            self.play(Transform(curve, new_curve), *highlight(k), run_time=1.0)
            self.play(t.animate.increment_value(1.0), run_time=3.0, rate_func=linear)
        self.play(menu[-1].animate.set_color(ROSE), menu[3].animate.set_color(SUB), run_time=0.8)
        self.wait(1.5)
        self.stop_ambient_camera_rotation()
        fade_all(self)


class S5MultiGPU(Scene):
    def construct(self):
        chapter(self, 4, "Views split across GPUs and nodes")
        n_gpu, per = 8, 4
        palette = GPU_COLORS + [interpolate_color(ManimColor(c), ManimColor(INK), 0.35) for c in GPU_COLORS]
        blocks = VGroup(*[Rectangle(width=0.28, height=0.6, stroke_width=0, fill_opacity=1,
                                    fill_color=palette[k // per]) for k in range(n_gpu * per)])
        blocks.arrange(RIGHT, buff=0.06).move_to(UP * 2.0 + RIGHT * 0.8)
        views = T("360 views", 36, SUB).next_to(blocks, LEFT, buff=0.35)
        self.play(LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.03), FadeIn(views), run_time=1.6)

        nodes = VGroup()
        for node in range(2):
            gpus = VGroup()
            for k in range(4):
                c = palette[4 * node + k]
                box = RoundedRectangle(width=1.2, height=1.3, corner_radius=0.12, stroke_color=c,
                                       stroke_width=2, fill_color=c, fill_opacity=0.08)
                box.add(T(f"GPU {k}", 30, c).move_to(box.get_top() + DOWN * 0.25))
                gpus.add(box)
            gpus.arrange(RIGHT, buff=0.12)
            frame = SurroundingRectangle(gpus, buff=0.18, corner_radius=0.14, color=SUB, stroke_width=1.5)
            label = T(f"node {node + 1}", 32, SUB).next_to(frame, UP, buff=0.12).align_to(frame, LEFT)
            nodes.add(VGroup(gpus, frame, label))
        nodes.arrange(RIGHT, buff=1.2).move_to(DOWN * 0.15)
        link = Line(nodes[0][1].get_right(), nodes[1][1].get_left(), color=ROSE, stroke_width=3)
        nccl = T("NCCL", 30, ROSE).next_to(link, UP, buff=0.1)
        self.play(FadeIn(nodes[0]), run_time=0.9)

        def move_to_node(node, first):
            moves = []
            for g in range(first, first + 4):
                group = VGroup(*blocks[g * per:(g + 1) * per])
                target = group.copy().arrange(RIGHT, buff=0.05).scale(0.8)
                target.move_to(nodes[node][0][g - first].get_center() + DOWN * 0.27)
                moves.append(Transform(group, target))
            return moves

        note1 = T(r"one process: \texttt{Projector(..., devices=[0, 1, 2, 3])}", 30, INK).to_edge(DOWN, buff=1.35)
        self.play(*move_to_node(0, 0), FadeIn(note1), run_time=1.6)
        self.wait(1.2)
        self.play(FadeIn(nodes[1]), Create(link), FadeIn(nccl), run_time=1.0)
        note2 = T(r"one process per GPU: \texttt{torchrun ...}\ + \texttt{Projector(..., distributed=True)}", 34, INK)
        note2.next_to(note1, DOWN, buff=0.28)
        self.play(*move_to_node(1, 4), FadeOut(views), FadeIn(note2), run_time=1.6)
        self.wait(2.2)
        self.play(*[FadeOut(m) for m in (blocks, nodes, link, nccl, note1, note2)], run_time=0.8)

        # Measured CGLS iteration time, 128^3, 360 views, A100 64 GB (Leonardo Booster).
        head = T(r"one CGLS iteration, $128^3$ volume, 360 views, A100 64 GB", 38, SUB).move_to(UP * 2.2)
        rows = [("1 GPU", 30.07, ""), ("4 GPUs, 1 node", 8.09, r"\quad 3.7\texttimes"),
                ("8 GPUs, 2 nodes", 5.54, r"\quad 5.4\texttimes")]
        scale = 8.0 / 30.07
        bars = VGroup()
        for k, (name, ms, speed) in enumerate(rows):
            y = 0.9 - 1.45 * k
            label = T(name, 38, SUB).move_to(np.array([-4.3, y, 0])).align_to(np.array([-3.2, 0, 0]), RIGHT)
            bar = Rectangle(width=ms * scale, height=0.85, stroke_width=0, fill_color=GPU_COLORS[k],
                            fill_opacity=0.85).move_to(np.array([-3.0, y, 0]), aligned_edge=LEFT)
            value = T(f"{ms:.1f} ms" + speed, 38, INK).next_to(bar, RIGHT, buff=0.25)
            bars.add(VGroup(label, bar, value))
        self.play(FadeIn(head), run_time=0.7)
        for row in bars:
            self.play(FadeIn(row[0]), GrowFromEdge(row[1], LEFT), run_time=1.0)
            self.play(FadeIn(row[2]), run_time=0.5)
        self.wait(2.8)
        fade_all(self)


def slice_grid(columns, rows, size, vmax=1.0):
    """Columns of (name, caption, {row: image}); row labels on the left."""
    grid = Group()
    for name, caption_text, images in columns:
        cells = [gray_image(images[row], vmax, size) for row in rows]
        column = Group(*cells).arrange(DOWN, buff=0.12)
        title = T(name, 36, INK).next_to(column, UP, buff=0.18)
        parts = [column, title]
        if caption_text:
            parts.append(T(caption_text, 32, SUB).next_to(column, DOWN, buff=0.16))
        grid.add(Group(*parts))
    grid.arrange(RIGHT, buff=0.22, aligned_edge=UP)
    labels = VGroup(*[T(row, 30, SUB).rotate(PI / 2).next_to(grid[0][0][k], LEFT, buff=0.15)
                      for k, row in enumerate(rows)])
    return grid, labels


class S6Measured(Scene):
    def construct(self):
        chapter(self, 5, "Measured walnut, 240 real projections")
        data = np.load(HERE / "walnut_measured.npz")
        projections = data["projections"]
        strip = Group(*[gray_image(np.flipud(p.T), projections.max(), 2.9) for p in projections])
        strip.arrange(RIGHT, buff=0.25).move_to(UP * 0.1)
        source = T("4 of 240 measured cone-beam projections (Meaney 2022, CC-BY 4.0)", 32, SUB)
        source.next_to(strip, DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(p, shift=0.1 * UP) for p in strip], lag_ratio=0.3), run_time=1.6)
        self.play(FadeIn(source), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(strip), FadeOut(source), run_time=0.7)

        names = [("fdk", "FDK"), ("sirt", "SIRT"), ("cgls", "CGLS"), ("tv", "TV + Adam")]
        columns = [(label, None, {"axial": data[f"{key}_axial"], "coronal": data[f"{key}_coronal"]})
                   for key, label in names]
        grid, labels = slice_grid(columns, ("axial", "coronal"), 2.3)
        Group(grid, labels).move_to(DOWN * 0.05)
        self.play(FadeIn(labels), run_time=0.5)
        for column in grid:
            self.play(FadeIn(column, shift=0.1 * UP), run_time=0.9)
        foot = note("one Projector for analytical and iterative reconstruction", 32).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(foot), run_time=0.6)
        self.wait(3.2)
        fade_all(self)


class S7Helical(Scene):
    def construct(self):
        chapter(self, 6, r"Simulated helical scan, 720 views, 1\% noise")
        data = np.load(HERE / "recon_slices.npz")
        psnr = json.loads((HERE / "recon_psnr.json").read_text())
        names = [("phantom", "walnut volume", None), ("fdk", "FDK", "fdk"), ("sirt", "SIRT", "sirt"),
                 ("cgls", "CGLS", "cgls"), ("tv", "TV + Adam", "tv")]
        columns = [(label, f"{psnr[metric]:.1f} dB" if metric else "ground truth",
                    {"axial": data[f"{key}_axial"], "coronal": data[f"{key}_coronal"]})
                   for key, label, metric in names]
        grid, labels = slice_grid(columns, ("axial", "coronal"), 2.2)
        Group(grid, labels).move_to(DOWN * 0.2)
        self.play(FadeIn(labels), run_time=0.5)
        for column in grid:
            self.play(FadeIn(column, shift=0.1 * UP), run_time=0.9)
        self.wait(3.5)
        fade_all(self)


class S8Geometry(Scene):
    def construct(self):
        chapter(self, 7, "Gradients through the geometry")
        history = json.loads((HERE / "calib_history.json").read_text())
        centre = LEFT * 3.5 + DOWN * 0.1
        R = 2.3
        n = 24
        rng = np.random.default_rng(5)
        errors = rng.normal(0, math.radians(7), n)
        true_pts = [centre + R * np.array([math.cos(a), math.sin(a), 0]) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
        ring = Circle(radius=R, color=GRID, stroke_width=1.5).move_to(centre)
        ghosts = VGroup(*[Dot(p, radius=0.06, color=SUB) for p in true_pts])
        progress = ValueTracker(0.0)

        def dots():
            s = 1 - progress.get_value()
            group = VGroup()
            for k, a in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
                b = a + s * errors[k]
                p = centre + R * np.array([math.cos(b), math.sin(b), 0])
                group.add(Dot(p, radius=0.09, color=ROSE))
                if s > 0.05:
                    group.add(Arrow(p, true_pts[k], buff=0.08, stroke_width=2.5, color=ROSE_DEEP,
                                    max_tip_length_to_length_ratio=0.35, tip_length=0.12))
            return group

        moving = always_redraw(dots)
        legend = VGroup(
            VGroup(Dot(radius=0.07, color=SUB), T("true source positions", 32, SUB)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(radius=0.08, color=ROSE), T("current estimate", 32, ROSE)).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.5).next_to(ring, DOWN, buff=0.4)
        self.play(Create(ring), FadeIn(ghosts), FadeIn(moving), FadeIn(legend), run_time=1.2)

        steps = [h["step"] for h in history]
        logs = [math.log10(h["loss"]) for h in history]
        axes = Axes(x_range=[0, steps[-1], 50], y_range=[1, 7, 2], x_length=5.6, y_length=3.4,
                    axis_config={"color": GRID, "stroke_width": 1.5, "include_ticks": True},
                    tips=False).move_to(RIGHT * 3.4 + DOWN * 0.1)
        ylab = M(r"\log_{10}\,\mathrm{loss}", 30, SUB).rotate(PI / 2).next_to(axes, LEFT, buff=0.2)
        xlab = T("Adam step", 32, SUB).next_to(axes, DOWN, buff=0.2)
        eq = M(r"\frac{\partial\, \|A(\theta)\,x - y\|^2}{\partial \theta}", 42, INK).next_to(axes, UP, buff=0.35)
        self.play(Create(axes), FadeIn(ylab), FadeIn(xlab), run_time=0.9)
        self.play(Write(eq), run_time=1.0)
        curve = always_redraw(lambda: axes.plot_line_graph(
            steps[:max(2, 1 + int(progress.get_value() * (len(steps) - 1)))],
            logs[:max(2, 1 + int(progress.get_value() * (len(steps) - 1)))],
            line_color=ROSE, add_vertex_dots=False, stroke_width=3.5))
        self.add(curve)
        self.play(progress.animate.set_value(1.0), run_time=6.5, rate_func=smooth)
        result = VGroup(T(r"per-view angle error $0.55^\circ \rightarrow 0.006^\circ$", 34, INK),
                        T(r"150 steps, $64^3$ volume, real run", 30, SUB)).arrange(DOWN, buff=0.12)
        result.next_to(xlab, DOWN, buff=0.25)
        self.play(FadeIn(result), run_time=0.7)
        self.wait(2.5)
        fade_all(self)


class S9End(Scene):
    def construct(self):
        name = T(r"\textbf{diffct}", 110)
        lines = VGroup(
            T(r"parallel · fan · cone beam\quad{}\textbar\quad{}any trajectory\quad{}\textbar\quad{}exact adjoint", 40, SUB),
            T("PyTorch autograd for volumes, sinograms and geometry", 40, SUB),
            T("one GPU · many GPUs · many nodes", 40, SUB),
            T(r"\texttt{github.com/sypsyp97/diffct}", 44, ROSE),
        ).arrange(DOWN, buff=0.36).next_to(name, DOWN, buff=0.6)
        VGroup(name, lines).move_to(ORIGIN)
        self.play(FadeIn(name), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(l, shift=0.1 * UP) for l in lines], lag_ratio=0.35), run_time=2.2)
        self.wait(3.0)
