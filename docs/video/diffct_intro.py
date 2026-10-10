"""diffct 2.0 film: full feature story, restrained typography, measured walnut data.

See README.md for input provenance and render commands. The 3D walnut is an
isosurface of a real FDK reconstruction. Scan rigs and pose errors are schematics.
"""
import json
import av
from pathlib import Path

import numpy as np
from manim import *
from scan_diagram import ScanDiagram

HERE = Path(__file__).parent
BG, INK, SUB, LINE = "#F7F7F2", "#222D29", "#5F6B64", "#C9D0C9"
GREEN, PALE, WARM = "#28765C", "#E3EBE4", "#A46D45"
config.background_color = BG
MATH_TEMPLATE = TexTemplate()
MATH_TEMPLATE.add_to_preamble(r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage{sansmath}\sansmath")
MathTex.set_default(tex_template=MATH_TEMPLATE)


def T(value, size=28, color=INK):
    return Text(value, font="Segoe UI", font_size=size, color=color,
                line_spacing=.6, disable_ligatures=True)


def code(value, size=22, color=INK):
    return Text(value, font="Consolas", font_size=size, color=color,
                line_spacing=.6, disable_ligatures=True)


def text_lines(lines, size=28, color=SUB):
    # Shape one paragraph so every row shares a fixed typographic baseline step.
    paragraph = T('\n'.join(lines), size, color)
    rows, offset = VGroup(), 0
    for line in lines:
        rows.add(VGroup(*paragraph.chars[offset:offset+len(line)]))
        offset += len(line)+1
    return rows


def left(mob, x):
    return mob.shift(RIGHT * (x - mob.get_left()[0]))


def label_baseline(mob, x, y):
    # These row labels start with a capital or digit. Descenders elsewhere in
    # the label must not move its baseline when centering the whole string.
    return mob.shift(RIGHT*(x-mob.get_center()[0]) + UP*(y-mob.chars[0].get_bottom()[1]))


def gray(array, height, vmax=1):
    pixels = (np.clip(np.asarray(array) / vmax, 0, 1) * 255).astype(np.uint8)
    image = ImageMobject(np.repeat(pixels[..., None], 3, axis=-1))
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
    return image.scale_to_fit_height(height)


def safe(mob):
    assert mob.get_left()[0] >= -6.65 and mob.get_right()[0] <= 6.65, f"horizontal overflow: {mob}"
    assert mob.get_bottom()[1] >= -3.7 and mob.get_top()[1] <= 3.7, f"vertical overflow: {mob}"
    return mob


def orbit(kind, t):
    angle = TAU * t
    if kind == "Helical":
        angle *= 1.5
        z = -1.1 + 2.2*t
    elif kind == "Saddle":
        z = .75*np.cos(2*angle)
    elif kind == "Sinusoidal":
        z = .6*np.sin(3*angle)
    else:
        z = 0
    return np.array([2.25*np.cos(angle), 2.25*np.sin(angle), z])


class DiffCTIntro(Scene):
    def show(self, *objects, duration=.65):
        for obj in objects:
            safe(obj)
        if self.outgoing:
            incoming = list(objects) + self.pending
            self.pending = []
            old = self.outgoing
            self.outgoing = []
            start = self.mark(f'transition_{len(self.transitions)}_start')
            # Finish the outgoing text before introducing the next composition.
            # A wipe splices unrelated titles together midway through the frame.
            self.play(Succession(
                AnimationGroup(*[FadeOut(obj) for obj in old]),
                AnimationGroup(*[FadeIn(obj) for obj in incoming]),
            ), run_time=.85)
            end = self.mark(f'transition_{len(self.transitions)}_end')
            self.transitions.append([start,end])
        else:
            self.play(*[FadeIn(obj) for obj in objects], run_time=duration)

    def mark(self, label):
        # Encoded-frame time remains accurate with fractional animation durations.
        parts=getattr(self.renderer.file_writer,'partial_movie_files',[])
        for path in parts[self.seen_parts:]:
            if path is not None and Path(path).exists():
                with av.open(path) as movie:
                    self.encoded_frames += movie.streams.video[0].frames
        self.seen_parts=len(parts)
        time=self.encoded_frames/config.frame_rate if parts else self.time
        self.timeline[label]=round(time,3)
        return time

    def reset(self):
        # Keep the current composition until the next section's first visual is ready.
        for mob in self.mobjects:
            mob.clear_updaters()
        self.outgoing = [m for m in self.mobjects if m is not self.divider]
        self.caption = None

    def heading(self, number, title):
        header = [left(T(title,36),-6.25).set_y(3.14),
                  T(f'{number:02d} / DIFFCT 2.0',20,GREEN).move_to([5.35,3.13,0])]
        assert header[0].get_right()[0]+.2 < header[1].get_left()[0]
        for item in header: safe(item)
        if self.divider is None:
            self.divider = Line([-6.25,2.72,0],[6.25,2.72,0],color=LINE,stroke_width=1)
            header.append(self.divider)
        self.pending.extend(header)

    def note(self, value):
        new = T(value,24,SUB).move_to(DOWN*3.34)
        safe(new)
        if self.outgoing:
            self.pending.append(new)
        elif self.caption is None:
            self.show(new,duration=.4)
        else:
            self.play(Succession(FadeOut(self.caption,shift=UP*.1),FadeIn(new,shift=UP*.1)),run_time=.4)
        self.caption = new

    def walnut(self, height, centre, turn=None):
        image = ImageMobject(self.surface[0]).scale_to_fit_height(height).move_to(centre)
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        if turn is not None:
            def update(mob):
                index = int(turn.get_value()) % len(self.surface)
                mob.pixel_array = self.surface[index].copy()
            image.add_updater(update)
        return image

    def construct(self):
        surface_data = np.load(HERE/'walnut_surface.npz')
        self.surface = surface_data['frames']
        self.surface_depth = surface_data['depth']
        self.data = dict(np.load(HERE/'walnut_measured.npz'))
        self.timeline = {}
        self.encoded_frames, self.seen_parts = 0, 0
        self.transitions = []
        self.outgoing, self.pending = [], []
        self.divider = None
        self.caption = None
        self.intro()
        self.trajectories()
        self.autograd()
        self.geometry()
        self.multigpu()
        self.measured()
        self.ending()
        Path(config.media_dir).mkdir(parents=True, exist_ok=True)
        (Path(config.media_dir)/'timeline.json').write_text(json.dumps(self.timeline, indent=2))
        (Path(config.media_dir)/'transitions.json').write_text(json.dumps(self.transitions))

    def intro(self):
        self.mark('intro_start')
        turn = ValueTracker(0)
        walnut = self.walnut(6.1, [-3.5, .35, 0], turn)
        self.show(walnut, duration=1)
        name = left(T('diffct', 80), .2).set_y(1.55)
        version = T('2.0', 28, GREEN).next_to(name, RIGHT, buff=.3).align_to(name, UP)
        self.show(name, version)
        self.show(left(T('Differentiable CT,\nbuilt on PyTorch.', 36), .25).set_y(.15))
        details = text_lines(['Any trajectory', 'Geometry gradients', 'One GPU to many nodes'])
        left(details, .25).set_y(-1.45)
        self.show(details)
        self.show(T('Measured walnut · 3D surface from CT', 20, SUB).move_to([-3.6,-2.8,0]))
        self.mark('intro_full')
        self.play(turn.animate.set_value(180), run_time=5, rate_func=linear)
        self.reset()

    def trajectories(self):
        self.mark('trajectory_start')
        self.heading(1, 'One object. Any trajectory.')
        kinds = ['Circular','Helical','Saddle','Sinusoidal','Per-view poses']
        labels = text_lines(kinds)
        left(labels,1.3).set_y(.85)
        samples=np.linspace(0,1,121)
        def pose(t):
            a=TAU*t
            return np.array([(2.25+.10*np.sin(5*a))*np.cos(a),
                             (2.25+.10*np.sin(5*a))*np.sin(a),.38*np.sin(3*a)+.12*np.cos(7*a)])
        paths=[np.array([orbit(k,t) for t in samples]) for k in kinds[:-1]]
        paths.append(np.array([pose(t) for t in samples]))
        state={'index':0,'previous':paths[0], 'start':paths[0][0]}
        phase,blend=ValueTracker(0),ValueTracker(1)
        renderer=ScanDiagram(self.surface[0],self.surface_depth)
        def diagram_frame():
            i=state['index'];a=blend.get_value()
            points=(1-a)*state['previous']+a*paths[i]
            if a<1:
                source=(1-a)*state['start']+a*paths[i][0]
            elif i==4:
                discrete_points=paths[i][::3]
                source=discrete_points[min(int(phase.get_value()*(len(discrete_points)-1)),len(discrete_points)-1)]
            else:
                source=orbit(kinds[i],phase.get_value())
            discrete=i==4 and a>.999
            return renderer.draw(points[::3] if discrete else points,source,discrete=discrete)
        diagram=ImageMobject(diagram_frame()).scale_to_fit_width(6.72).move_to([-2.85,-.1,0])
        self.show(diagram,labels)
        diagram.add_updater(lambda image: setattr(image,'pixel_array',np.dstack([
            diagram_frame(),np.full((renderer.height,renderer.width),255,dtype=np.uint8)])))
        self.show(T('Source  •',20,GREEN).move_to([-5.2,2.4,0]),
                  T('Flat detector',20,SUB).move_to([-.6,2.4,0]))
        self.note('Each view has a source, a detector centre and detector axes')
        for i,kind in enumerate(kinds):
            animations=[labels[i].animate.set_color(GREEN)]
            if i:
                state['previous']=paths[i-1]
                state['start']=orbit(kinds[i-1],1)
                state['index']=i;phase.set_value(0);blend.set_value(0)
                animations.extend([labels[i-1].animate.set_color(SUB),blend.animate.set_value(1)])
            self.play(*animations,run_time=.7 if i else .3)
            self.play(phase.animate.set_value(1),run_time=2.4,rate_func=linear)
            self.mark('trajectory_'+kind)
        diagram.clear_updaters()
        api=VGroup(code('traj = (source, centre, u, v)',22),
                   code('A = Projector(traj,',22,GREEN),
                   code('    (D, H, W), (U, V))',22,GREEN)).arrange(DOWN,aligned_edge=LEFT,buff=.18)
        left(api,1.3).set_y(-1.75)
        self.show(api)
        self.show(T('Scan geometry schematic',20,SUB).move_to([-2.95,-2.65,0]))
        self.note('The same Projector interface accepts every list of per-view poses')
        self.wait(2.1)
        self.reset()

    def autograd(self):
        self.mark('autograd_start')
        self.heading(2,'Projection, adjoint, autograd')
        illustration=T('Illustration: simulated parallel-beam sinogram',20,SUB).move_to([0,2.35,0])
        data=np.load(HERE/'data2d.npz')
        volume=gray(data['phantom'],2.7).move_to([-4.45,.75,0])
        sino=gray(data['sinogram'],2.7,float(data['sinogram'].max())).stretch_to_fit_width(2.7).move_to([4.45,.75,0])
        self.show(volume,T('Volume x',28).move_to([-4.45,-.95,0]),illustration)
        forward=Arrow([-2.8,1.35,0],[2.8,1.35,0],buff=0,color=GREEN,stroke_width=3)
        adjoint=Arrow([2.8,.25,0],[-2.8,.25,0],buff=0,color=INK,stroke_width=3)
        self.play(GrowArrow(forward),FadeIn(code('A.project(x)',22,GREEN).move_to([0,1.8,0])),run_time=.8)
        self.show(sino,T('Sinogram y',28).move_to([4.45,-.95,0]))
        self.note('Siddon ray tracing sums physical path lengths through the volume')
        projection_formula=MathTex(r'y_i=\sum_k x_k\,\ell_{ik}',font_size=36,color=INK).move_to([0,-1.45,0])
        self.show(projection_formula)
        self.wait(1.5)
        self.play(GrowArrow(adjoint),FadeIn(code('A.backproject(y)',22).move_to([0,-.2,0])),run_time=.8)
        self.note('The matched adjoint uses the same ray traversal; it is not an inverse')
        equation=MathTex(r'\langle Ax,y\rangle = \langle x,A^{\mathsf T}y\rangle',font_size=36,color=INK).move_to([0,-1.45,0])
        self.play(Succession(FadeOut(projection_formula),FadeIn(equation)),run_time=.7)
        self.wait(1.7)
        loss=MathTex(r'\mathcal L(x)=\frac12\lVert Ax-y\rVert_2^2',font_size=29,color=INK).move_to([-3.1,-2.2,0])
        gradient=MathTex(r'\nabla_x\mathcal L=A^{\mathsf T}(Ax-y)',font_size=29,color=GREEN).move_to([3.1,-2.2,0])
        backward=code('loss.backward()',22,GREEN).move_to([0,-2.77,0])
        self.show(loss,gradient,backward)
        self.note('Autograd for volumes and sinograms, including second derivatives')
        self.mark('autograd_full')
        self.wait(2.6)
        self.reset()

    def geometry(self):
        self.mark('geometry_start')
        self.heading(3,'Learn the acquisition geometry')
        centre=np.array([-3.6,-.1,0]); radius=1.95
        ring=Circle(radius,color=LINE,stroke_width=1.8).move_to(centre)
        walnut=self.walnut(2.8,centre)
        angles=np.linspace(0,TAU,24,endpoint=False)
        errors=np.deg2rad(8)*np.sin(angles*3+.4)
        targets=[centre+radius*np.array([np.cos(a),np.sin(a),0]) for a in angles]
        reference=VGroup(*[Dot(p,radius=.05,color=SUB) for p in targets])
        progress=ValueTracker(0)
        def estimates():
            points=[centre+radius*np.array([np.cos(a+(1-progress.get_value())*e),np.sin(a+(1-progress.get_value())*e),0])
                    for a,e in zip(angles,errors)]
            return VGroup(*[Dot(p,radius=.073,color=GREEN) for p in points])
        moving=always_redraw(estimates)
        legend=VGroup(VGroup(Dot(radius=.05,color=SUB),T('Reference',20,SUB)).arrange(RIGHT,buff=.12),
                      VGroup(Dot(radius=.07,color=GREEN),T('Estimate',20,GREEN)).arrange(RIGHT,buff=.12))
        legend.arrange(RIGHT,buff=.4).move_to([-3.6,-2.38,0])
        self.show(ring,walnut,reference,moving,legend)
        self.show(T('Pose correction schematic',20,SUB).move_to([-3.6,-2.83,0]))
        flag=code('source.requires_grad_(True)',22,GREEN).move_to([3.15,2.1,0])
        self.show(flag)
        self.note('Source, detector centre and detector axes can all receive gradients')
        history=json.loads((HERE/'calib_history.json').read_text())
        steps=np.array([row['step'] for row in history]); logs=np.log10([row['loss'] for row in history])
        axes=Axes(x_range=[0,150,50],y_range=[1,7,2],x_length=4.5,y_length=2.65,tips=False,
                  axis_config={'color':LINE,'stroke_width':1.5,'include_ticks':False}).move_to([3.25,.0,0])
        axis_labels=VGroup(MathTex(r'\log_{10}\,\mathrm{loss}',font_size=23,color=SUB).next_to(axes,UP,buff=.14),
                          T('Adam step',20,SUB).next_to(axes,DOWN,buff=.25))
        for n in [0,50,100,150]:axis_labels.add(T(str(n),20,SUB).next_to(axes.c2p(n,1),DOWN,buff=.07))
        for n in [1,3,5,7]:axis_labels.add(T(str(n),20,SUB).next_to(axes.c2p(0,n),LEFT,buff=.1))
        axis_labels[1].shift(DOWN*.2)
        curve=axes.plot_line_graph(steps,logs,line_color=GREEN,add_vertex_dots=False,stroke_width=3)
        self.show(axes,axis_labels)
        self.play(Create(curve),progress.animate.set_value(1),run_time=4,rate_func=linear)
        result=MathTex(r'\mathrm{Angle\ RMSE}:\ '+f"{history[0]['angle_rmse_deg']:.2f}"+r'^\circ\longrightarrow '+f"{history[-1]['angle_rmse_deg']:.3f}"+r'^\circ',font_size=30,color=GREEN).move_to([3.25,-2.25,0])
        self.show(result,T('Phantom run: 64³, 360 views, 150 steps',20,SUB).move_to([3.25,-2.83,0]))
        self.note('First-order geometry gradients; the recorded run also recovers detector shift')
        self.mark('geometry_full')
        self.wait(2.8)
        self.reset()

    def multigpu(self):
        self.mark('multigpu_start')
        self.heading(4,'Distribute views, keep the volume')
        self.note('Each GPU holds the full volume; projection views are partitioned')
        blocks=VGroup(*[Rectangle(width=.25,height=.4,stroke_width=0,fill_color=GREEN,
                                   fill_opacity=.5+.5*(i%4)/3) for i in range(32)]).arrange(RIGHT,buff=.06).move_to([0,1.85,0])
        self.show(blocks,T('Projection views',24,SUB).move_to([0,2.36,0]))
        nodes=VGroup();gpu_centres=[]
        for node,x in enumerate([-3.25,3.25]):
            gpus=VGroup()
            for gpu in range(4):
                p=np.array([x+(gpu-1.5)*1.13,-.1,0]);gpu_centres.append(p)
                frame=RoundedRectangle(width=1.02,height=1.35,corner_radius=.08,color=LINE,stroke_width=1.5,fill_color=PALE,fill_opacity=.45).move_to(p)
                label=T(f'GPU {gpu}',20,SUB).move_to(p+UP*.4)
                # Keep GPU identifiers readable while view blocks enter the card.
                label.add_background_rectangle(color=interpolate_color(ManimColor(BG),ManimColor(PALE),.45),opacity=1,buff=.025)
                label.set_z_index(2)
                # Every GPU visibly receives the same volume icon.
                cube=Cube(side_length=.25,fill_color=GREEN,fill_opacity=.15,stroke_color=GREEN,stroke_width=1)
                cube.rotate(PI/6,axis=RIGHT).rotate(PI/5,axis=UP).move_to(p+DOWN*.1)
                gpus.add(VGroup(frame,label,cube))
            label=T(f'Node {node+1}',28).move_to([x,-1.03,0])
            nodes.add(VGroup(gpus,label))
        self.show(nodes[0])
        def move_groups(start,end):
            result=[]
            for g in range(start,end):
                group=VGroup(*blocks[g*4:(g+1)*4]);target=group.copy().scale(.61).arrange(RIGHT,buff=.035)
                target.move_to(gpu_centres[g]+DOWN*.44)
                result.append(Transform(group,target))
            return result
        one=code('Projector(..., devices=[0, 1, 2, 3])',22,GREEN).move_to([-3.25,-1.65,0])
        self.play(*move_groups(0,4),FadeIn(one),run_time=1.4)
        self.wait(1.1)
        self.show(nodes[1])
        link=Line([-1,.0,0],[1,.0,0],color=GREEN,stroke_width=2)
        link_label=T('NCCL',20,GREEN).next_to(link,UP,buff=.15)
        self.play(Create(link),FadeIn(link_label),*move_groups(4,8),run_time=1.4)
        many=VGroup(code('torchrun ...',22,GREEN),code('Projector(..., distributed=True)',22)).arrange(DOWN,buff=.17).move_to([3.25,-1.75,0])
        self.show(many)
        self.note('One process with several GPUs, or one process per GPU across nodes')
        self.mark('multigpu_distribution')
        self.wait(2.1)
        self.reset()
        self.heading(4,'Measured multi-GPU scaling')
        self.note('Circular cone beam · 128³ volume · 360 views · A100-SXM-64GB')
        data=json.loads((HERE.parent/'assets/scaling.json').read_text()); timings=data['timings']['128']
        baseline=timings['gpus_1']['cgls_ms_per_iter']
        chart_title=left(T('One CGLS iteration',28,SUB),-6.05).set_y(2.08)
        for i,cfg in enumerate(data['configurations']):
            value=timings[cfg['key']]['cgls_ms_per_iter'];y=1.05-i*1.13
            label=T(cfg['label'],28,SUB).move_to([-4.3,y,0])
            bar=Rectangle(width=6.15*value/baseline,height=.5,stroke_width=0,fill_color=GREEN,fill_opacity=.65 if i==0 else 1)
            left(bar,-2.05).set_y(y)
            value_label=T(f'{value:.2f} ms',28).next_to(bar,RIGHT,buff=.2)
            if i == 0:
                self.show(chart_title,label,bar,value_label)
            else:
                self.show(label,duration=.25)
                self.play(GrowFromEdge(bar,LEFT),FadeIn(value_label),run_time=.7)
            if i:
                speed=T(f'{baseline/value:.1f}×',36,GREEN).move_to([4.75,y,0]);self.show(speed,duration=.3)
        self.show(T('Speedup depends on workload and communication.',24,SUB).move_to([0,-2.4,0]))
        self.mark('scaling_full')
        self.wait(3)
        self.reset()

    def measured(self):
        self.mark('measured_start')
        self.heading(5,'One measured scan. Four reconstructions.')
        self.note('240 measured cone-beam views · 256³ reconstruction · shared display range')
        # Show the real acquisition before revealing the method comparison.
        projections=Group(*[gray(np.flipud(p.T),2.5,float(self.data['projections'].max())) for p in self.data['projections']])
        projections.arrange(RIGHT,buff=.4).move_to([0,.35,0])
        self.show(projections)
        scan_label=T('Four measured projection views of the same walnut',28,SUB).move_to([0,-1.5,0])
        self.show(scan_label)
        self.wait(2.3)
        columns=Group();titles=VGroup();details=VGroup()
        methods=[('fdk','FDK','Analytical'),('sirt','SIRT','200 iterations'),('cgls','CGLS','20 iterations'),('tv','TV-regularized','Adam · 300 iterations')]
        for i,(key,title,detail) in enumerate(methods):
            x=-4.2+i*2.8
            pair=Group(gray(self.data[key+'_axial'],2.15).move_to([x,1.04,0]),
                       gray(self.data[key+'_coronal'],2.15).move_to([x,-1.25,0]))
            columns.add(pair)
            titles.add(label_baseline(T(title,28,INK),x,2.32))
            details.add(label_baseline(T(detail,20,SUB),x,-2.78))
        row_labels=VGroup(T('Axial',20,SUB).rotate(PI/2).move_to([-5.68,1.04,0]),
                          T('Coronal',20,SUB).rotate(PI/2).move_to([-5.68,-1.25,0]))
        for obj in [*columns,*titles,*details,row_labels]:safe(obj)
        self.play(Succession(
            AnimationGroup(FadeOut(projections),FadeOut(scan_label)),
            AnimationGroup(FadeIn(columns[0]),FadeIn(titles[0]),FadeIn(details[0]),FadeIn(row_labels)),
        ),run_time=.85)
        for i in range(1,4):self.show(columns[i],titles[i],details[i],duration=.65)
        self.mark('reconstruction_grid')
        self.wait(4.5)
        self.note('A. Meaney (2022) · CC BY 4.0 · doi:10.5281/zenodo.6986012')
        self.wait(2.2)
        self.reset()

    def ending(self):
        self.mark('ending_start')
        if self.divider is not None:
            self.outgoing.append(self.divider)
            self.divider=None
        turn=ValueTracker(0)
        walnut=self.walnut(5.8,[3.7,.2,0],turn)
        self.show(walnut,left(T('diffct',80),-5.85).set_y(1.65))
        self.show(left(T('From your geometry\nto your reconstruction.',36,SUB),-5.8).set_y(.32))
        self.show(left(code('pip install "diffct[cu12]"',22,GREEN),-5.8).set_y(-1.1),
                  left(T('Install PyTorch first · cu13 extra also available',20,SUB),-5.8).set_y(-1.66))
        self.show(left(T('sypsyp97.github.io/diffct',24),-5.8).set_y(-2.38),
                  left(T('github.com/sypsyp97/diffct',24,SUB),-5.8).set_y(-2.87))
        self.mark('ending_full')
        self.play(turn.animate.set_value(180),run_time=5.5,rate_func=linear)
        walnut.clear_updaters()
        self.wait(.8)
        self.mark('end')
