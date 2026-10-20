"""diffct film: tiled execution, rank ownership and measured walnut data.

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
BG, INK, SUB, LINE = "#101820", "#F2EDE3", "#ABB9C6", "#3B5060"
GREEN, PALE, WARM = "#69C5CD", "#1B2A36", "#E7BA7B"
SERIES = [GREEN, WARM, "#A8A8D4", "#8AB5E1", "#A8C8AE"]
GPU_COLORS = [GREEN, "#8AB5E1", WARM, "#A8C8AE"]
CHAPTERS = ['Any trajectory', 'Autograd', 'Geometry gradients', 'Many GPUs', 'Measured data']
config.background_color = BG


def tex_text(value):
    # Literal UI copy; TeX owns kerning, word spacing and ligatures, as upstream.
    escapes = {'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#', '$': r'\$',
               '{': r'\{', '}': r'\}', '³': r'\textsuperscript{3}',
               '·': r'\textperiodcentered{}', '×': r'$\times$', '•': r'$\bullet$'}
    return ''.join(escapes.get(c, c) for c in value)


def T(value, size=28, color=INK):
    lines = [tex_text(line) for line in value.split('\n')]
    text = lines[0] if len(lines) == 1 else r'\begin{tabular}{@{}l@{}}' + r'\\'.join(lines) + r'\end{tabular}'
    return Tex(text, font_size=size*1.25, color=color)


def code(value, size=22, color=INK, width=None, height=None):
    lines = [r'\texttt{' + tex_text(line).replace(' ', r'\ ') + '}' for line in value.split('\n')]
    text = lines[0] if len(lines) == 1 else r'\begin{tabular}{@{}l@{}}' + r'\\'.join(lines) + r'\end{tabular}'
    label = Tex(text, font_size=size*1.2, color=color)
    panel = RoundedRectangle(width=width or label.width+.42, height=height or label.height+.30,
                             corner_radius=.07, color=LINE, stroke_width=1,
                             fill_color=PALE, fill_opacity=1).move_to(label)
    assert label.width+.4 <= panel.width+.001 and label.height+.28 <= panel.height+.001
    if width is not None:
        left(label, panel.get_left()[0]+.22)
    return VGroup(panel, label)


def baseline(mob):
    return float(np.median([g.get_bottom()[1] for g in mob.family_members_with_points()]))


def on_baseline(mob, y):
    return mob.shift(UP*(y-baseline(mob)))


def left(mob, x):
    return mob.shift(RIGHT * (x - mob.get_left()[0]))


def label_baseline(mob, x, y):
    return on_baseline(mob.set_x(x), y)


def gray(array, height, vmax=1):
    pixels = (np.clip(np.asarray(array) / vmax, 0, 1) * 255).astype(np.uint8)
    image = ImageMobject(np.repeat(pixels[..., None], 3, axis=-1))
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
    image.scale_to_fit_height(height)
    frame = Rectangle(width=image.width,height=image.height,color=LINE,stroke_width=1).move_to(image)
    return Group(image,frame)


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
        header = [on_baseline(left(T(f'{number:02d}',22,GREEN),-6.25),3.17),
                  on_baseline(left(T(CHAPTERS[number-1],36),-5.67),3.17)]
        if self.divider is None:
            self.divider = Line([-6.25,-3.77,0],[6.25,-3.77,0],color=LINE,stroke_width=2)
            header.append(self.divider)
        header.append(Line([-6.25,-3.77,0],[-6.25+12.5*number/5,-3.77,0],color=GREEN,stroke_width=2))
        self.pending.extend(header)

    def note(self, value):
        new = on_baseline(T(value,24,SUB).set_x(0),-3.34)
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
        walnut = self.walnut(3.3, [0, .05, 0], turn)
        self.show(walnut, duration=1)
        name = T('diffct', 72).move_to([0,2.65,0])
        self.show(name)
        self.show(T('Differentiable CT, built on PyTorch.', 28).move_to([0, 1.78, 0]))
        details = VGroup()
        for i,title in enumerate(CHAPTERS):
            x = (i-1)*3.65 if i<3 else (i-3.5)*3.65
            y = -2.23 if i<3 else -2.88
            chip = RoundedRectangle(width=3.4,height=.48,corner_radius=.06,
                                    color=LINE,stroke_width=1,fill_color=PALE,fill_opacity=.55).move_to([x,y,0])
            label = VGroup(T(f'{i+1:02d}',20,GREEN),T(title,20)).arrange(RIGHT,buff=.14).move_to(chip)
            assert label.width+.2<chip.width
            details.add(VGroup(chip,label))
        self.show(details)
        self.show(T('Measured walnut · 3D surface from CT', 20, SUB).move_to([0,-3.48,0]))
        self.mark('intro_full')
        self.play(turn.animate.set_value(180), run_time=5, rate_func=linear)
        self.reset()

    def trajectories(self):
        self.mark('trajectory_start')
        self.heading(1, 'One object. Any trajectory.')
        kinds = ['Circular','Helical','Saddle','Sinusoidal','Per-view poses']
        labels = VGroup()
        swatches = VGroup()
        for i, kind in enumerate(kinds):
            row = on_baseline(left(T(kind,24,SUB),1.3),1.37-i*.56)
            labels.add(row)
            swatches.add(Line([.65,row.get_center()[1],0],[1.08,row.get_center()[1],0],
                              color=SERIES[i],stroke_width=3))
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
            return renderer.draw(points[::3] if discrete else points,source,discrete=discrete,
                                 color=np.array(ManimColor(SERIES[i]).to_rgb())*255)
        diagram=ImageMobject(diagram_frame()).scale_to_fit_width(6.0).move_to([-3.15,-.1,0])
        self.show(diagram,labels,swatches)
        diagram.add_updater(lambda image: setattr(image,'pixel_array',np.dstack([
            diagram_frame(),np.full((renderer.height,renderer.width),255,dtype=np.uint8)])))
        self.show(T('Source  •',20,WARM).move_to([-4.65,2.12,0]),
                  T('Flat detector',20,SUB).move_to([-1.65,2.12,0]),
                  T('Trajectory',20,SUB).move_to([3.15,2.12,0]))
        self.note('Each view has a source, a detector centre and detector axes')
        for i,kind in enumerate(kinds):
            animations=[labels[i].animate.set_color(SERIES[i])]
            if i:
                state['previous']=paths[i-1]
                state['start']=orbit(kinds[i-1],1)
                state['index']=i;phase.set_value(0);blend.set_value(0)
                animations.extend([labels[i-1].animate.set_color(SUB),blend.animate.set_value(1)])
            self.play(*animations,run_time=.7 if i else .3)
            self.play(phase.animate.set_value(1),run_time=2.4,rate_func=linear)
            self.mark('trajectory_'+kind)
        diagram.clear_updaters()
        api=code('traj = (source, centre, u, v)\nA = Projector(traj,\n    (D, H, W), (U, V))',20,GREEN,width=5.4,height=1.22)
        api.move_to([3.15,-2.04,0])
        self.show(api)
        self.show(T('Scan geometry schematic',20,SUB).move_to([-3.15,-2.65,0]))
        self.note('Valid per-view poses share the same Projector interface')
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
        adjoint=Arrow([2.8,.25,0],[-2.8,.25,0],buff=0,color=WARM,stroke_width=3)
        self.play(GrowArrow(forward),FadeIn(code('A.project(x)',22,GREEN).move_to([0,1.8,0])),run_time=.8)
        self.show(sino,T('Sinogram y',28).move_to([4.45,-.95,0]))
        self.note('Siddon ray tracing sums physical path lengths through the volume')
        projection_formula=MathTex(r'y_i=\sum_k x_k\,\ell_{ik}',font_size=36,color=INK).move_to([0,-1.45,0])
        self.show(projection_formula)
        self.wait(1.5)
        self.play(GrowArrow(adjoint),FadeIn(code('A.backproject(y)',22,WARM).move_to([0,-.2,0])),run_time=.8)
        self.note('The matched adjoint uses the same ray traversal; it is not an inverse')
        equation=MathTex(r'\langle Ax,y\rangle = \langle x,A^{\mathsf T}y\rangle',font_size=36,color=INK).move_to([0,-1.45,0])
        self.play(Succession(FadeOut(projection_formula),FadeIn(equation)),run_time=.7)
        self.wait(1.7)
        loss=MathTex(r'\mathcal L(x)=\frac12\lVert Ax-y\rVert_2^2',font_size=29,color=INK).move_to([-3.1,-2.2,0])
        gradient=MathTex(r'\nabla_x\mathcal L=A^{\mathsf T}(Ax-y)',font_size=29,color=GREEN).move_to([3.1,-2.2,0])
        backward=code('loss.backward()',22,GREEN).move_to([0,-2.77,0])
        self.show(loss,gradient,backward)
        self.note('Data autograd: batches and second derivatives')
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
        example=dict(np.load(HERE/'gpu_example.npz'))
        image,projection=example['image'],example['projection']
        partials=example['tile_projection']
        vmax=float(projection.max())

        def photo(array,width,vmax=1,color=LINE):
            result=gray(array,1,vmax).scale_to_fit_width(width)
            result[1].set_color(color)
            return result

        def crop_frame(picture,start,stop,total,color):
            height=picture.height*(stop-start)/total
            y=picture.get_top()[1]-picture.height*(start+stop)/(2*total)
            return Rectangle(width=picture.width,height=height,color=color,
                             stroke_width=2).move_to([picture.get_x(),y,0])

        def chip(index,centre,width=3.5,height=1.6,color=GREEN):
            frame=RoundedRectangle(width=width,height=height,corner_radius=.1,
                                   color=color,stroke_width=1.5,fill_color=PALE,fill_opacity=1)
            frame.move_to(centre)
            pins=VGroup(*[Line(frame.get_edge_center(side*RIGHT)+[0,dy,0],
                              frame.get_edge_center(side*RIGHT)+[side*.12,dy,0],
                              color=color,stroke_width=2)
                          for side in [-1,1] for dy in [-.45,-.15,.15,.45]])
            label=T(f'GPU {index}',22,color).next_to(frame,UP,buff=.1)
            label.add_background_rectangle(color=PALE,opacity=1,buff=.035)
            label.set_z_index(3)
            return VGroup(frame,pins,label)

        # One GPU first: the visible buffer is replaced, while the CPU result grows.
        self.heading(4,'Stream one tile at a time')
        self.note('2D walnut illustration · simulated projections computed with diffct')
        host=photo(image,2.6).move_to([-4.7,.4,0])
        output=photo(np.zeros_like(projection),2.6,vmax).move_to([4.7,.4,0])
        gpu=chip(0,[0,.4,0],width=3,height=2.6)
        on_baseline(gpu[2],1.92)
        self.show(T('1. A large scan, a small GPU buffer',30).move_to([0,2.6,0]),
                  host,output,gpu,
                  label_baseline(T('CPU / disk input',24,SUB),-4.7,1.92),
                  label_baseline(T('CPU / disk output',24,SUB),4.7,1.92),
                  label_baseline(T('Full walnut slice',20,SUB),-4.7,-1.35),
                  label_baseline(T('Views × detector bins',20,SUB),4.7,-1.35),
                  label_baseline(T('Read',22,GREEN),-2.45,1.5),
                  label_baseline(T('Add',22,WARM),2.45,1.5))
        compute_label=T('Project this tile',22,SUB).move_to([0,.4,0])
        compute_label.add_background_rectangle(color=PALE,opacity=1,buff=.035)
        compute_label.set_z_index(3)
        self.show(Arrow([-3.22,.4,0],[-1.65,.4,0],buff=0,color=GREEN),
                  Arrow([1.65,.4,0],[3.22,.4,0],buff=0,color=WARM),
                  compute_label)
        outlines=VGroup(*[crop_frame(host,i*16,(i+1)*16,64,GPU_COLORS[i]) for i in range(4)])
        self.play(Create(outlines),run_time=.6)
        batch=crop_frame(output,0,16,64,GREEN)
        view_label=T('First 16 views',20,GREEN).move_to([4.7,-1.75,0])
        self.show(batch,view_label)
        total=np.zeros_like(projection)
        for tile in range(4):
            colour=GPU_COLORS[tile]
            source=photo(image[tile*16:(tile+1)*16],host.width,color=colour).move_to(outlines[tile])
            resident=source.copy().scale_to_fit_width(2.35).move_to([0,.98,0])
            contribution=photo(partials[tile,:16],2.35,vmax,colour).move_to([0,-.35,0])
            tile_label=T(f'Tile {tile+1} of 4',18,colour).move_to([0,1.45,0])
            tile_label.add_background_rectangle(color=PALE,opacity=1,buff=.02)
            tile_label.set_z_index(3)
            self.play(Indicate(outlines[tile],color=colour,scale_factor=1.04),FadeIn(tile_label),
                      TransformFromCopy(source,resident),run_time=.7)
            self.show(contribution,duration=.3)
            total[:16]+=partials[tile,:16]
            target=photo(total,output.width,vmax).move_to(output)
            packet=contribution.copy()
            self.add(packet)
            destination=photo(partials[tile,:16],output.width,vmax,colour).move_to(batch)
            self.play(Transform(packet,destination),run_time=.55)
            self.remove(packet)
            self.play(Transform(output,target),run_time=.3)
            self.mark(f'stream_tile_{tile}')
            self.play(FadeOut(resident),FadeOut(contribution),FadeOut(tile_label),run_time=.2)
        self.note('The next tile replaces the previous one in the same GPU buffer')
        self.mark('streaming_tiles')
        self.wait(1.6)
        self.play(FadeOut(batch),FadeOut(view_label),run_time=.25)
        self.show(code('volume_chunk_shape=(16, 64), view_chunk_size=16',20,GREEN).move_to([0,-2.35,0]))
        self.note('Repeat for the remaining views; keep the complete result on CPU / disk')
        for view_start in [16,32,48]:
            resident=None
            for tile in range(4):
                next_tile=photo(image[tile*16:(tile+1)*16],2.35,color=GPU_COLORS[tile]).move_to([0,.98,0])
                total[view_start:view_start+16]+=partials[tile,view_start:view_start+16]
                animations=[Indicate(outlines[tile],color=GPU_COLORS[tile],scale_factor=1.02),
                            Transform(output,photo(total,output.width,vmax).move_to(output))]
                animations.append(FadeIn(next_tile) if resident is None else ReplacementTransform(resident,next_tile))
                self.play(*animations,run_time=.2)
                resident=next_tile
            self.play(FadeOut(resident),run_time=.15)
        self.mark('streaming_complete')
        self.show(T('All 64 views',20,GREEN).move_to([4.7,-1.75,0]),duration=.3)
        self.note('GPU working memory holds tiles; caller-owned CUDA tensors are separate')
        self.wait(2)
        self.reset()

        # The same image/result stay at the sides; only the division of work changes.
        for mode in ['views','space']:
            self.heading(4,'Split views' if mode=='views' else 'Split the volume')
            title='2. Two GPUs: different angles' if mode=='views' else '3. Two GPUs: different parts of the walnut'
            self.note('Owned inputs are shown; each GPU still computes in tiles')
            stage_y=(.95-1.1)/2
            host=photo(image,2.5).move_to([-4.7,stage_y,0])
            output=photo(np.zeros_like(projection),2.5,vmax).move_to([4.7,stage_y,0])
            self.show(T(title,30).move_to([0,2.6,0]),host,output,
                      T('Input',24,SUB).move_to([-4.7,1.9,0]),
                      T('Assemble on CPU',24,SUB).move_to([4.7,1.9,0]),
                      code(f'Projector(..., distributed=True, partition="{mode}")',20,GREEN).move_to([0,-2.45,0]))
            results=[]
            for rank,y in enumerate([.95,-1.1]):
                colour=[GREEN,WARM][rank]
                card=chip(rank,[0,y,0],color=colour)
                self.show(card,duration=.3)
                if mode=='views':
                    value=image
                    result=projection[rank*32:(rank+1)*32]
                    source=host
                else:
                    value=image[rank*32:(rank+1)*32]
                    result=partials[rank*2:rank*2+2].sum(0)
                    source=photo(value,host.width,color=colour).move_to(
                        crop_frame(host,rank*32,(rank+1)*32,64,colour))
                    self.show(crop_frame(host,rank*32,(rank+1)*32,64,colour),duration=.2)
                local=photo(value,1.25,color=colour).move_to([-.85,y,0])
                projected=photo(result,1.4,vmax,colour).move_to([.9,y,0])
                self.play(TransformFromCopy(source,local),run_time=.7)
                views_label=f'Views {rank*32+1}–{(rank+1)*32}' if mode=='views' else 'All 64 views'
                self.show(projected,T(views_label,17,colour).move_to([.9,y+.64,0]),duration=.45)
                results.append(projected)
            self.mark(f'partition_{mode}_inputs')
            self.wait(1.2)
            self.note('Same volume, different views: stack the results' if mode=='views' else
                      'Different volume parts, same views: add the results')
            if mode=='views':
                for rank,result in enumerate(results):
                    target=photo(projection[rank*32:(rank+1)*32],output.width,vmax,[GREEN,WARM][rank])
                    target.move_to(crop_frame(output,rank*32,(rank+1)*32,64,[GREEN,WARM][rank]))
                    self.play(TransformFromCopy(result,target),run_time=.8)
                self.show(T('Views 1–32',19,GREEN).move_to([4.7,stage_y+1.12,0]),
                          T('Views 33–64',19,WARM).move_to([4.7,stage_y-1.15,0]),duration=.3)
            else:
                junction=VGroup(Circle(.24,color=WARM,fill_color=BG,fill_opacity=1),
                                MathTex('+',font_size=30,color=WARM)).move_to([2.65,stage_y,0])
                self.show(junction,duration=.3)
                accumulated=np.zeros_like(projection)
                for rank,result in enumerate(results):
                    packet=result.copy();self.add(packet)
                    self.play(packet.animate.scale(.45).move_to(junction),run_time=.65)
                    self.remove(packet)
                    accumulated+=partials[rank*2:rank*2+2].sum(0)
                    self.play(Transform(output,photo(accumulated,output.width,vmax).move_to(output)),
                              Indicate(junction[1],color=INK,scale_factor=1.12),run_time=.45)
            self.mark(f'partition_{mode}_complete')
            self.wait(2.1)
            self.reset()

        # Four readable GPU cards: columns own view groups; rows own real image parts.
        self.heading(4,'Combine both splits')
        self.note('Hybrid: split both inputs; GPU computation stays tiled')
        self.show(T('4. Four GPUs: split both the volume and the views',29).move_to([0,2.6,0]))
        stage_y=(.8-1.2)/2
        host=photo(image,2.1).move_to([-5.05,stage_y,0])
        output=photo(np.zeros_like(projection),2.1,vmax).move_to([5.05,stage_y,0])
        host_label=T('Input',22,SUB).move_to([-5.05,1.5,0])
        output_label=T('Assemble on CPU',22,SUB).move_to([5.05,1.5,0])
        self.show(host,output,host_label,output_label,
                  T('Views 1–32',22,GREEN).move_to([-1.4,1.95,0]),
                  T('Views 33–64',22,WARM).move_to([1.4,1.95,0]),
                  code('Projector(..., distributed=True,\n    partition="hybrid", partition_grid=(2, 2))',20,GREEN).move_to([0,-2.55,0]))
        locals={};computed={}
        for space,y in enumerate([.8,-1.2]):
            colour=[GREEN,WARM][space]
            value=image[space*32:(space+1)*32]
            source=photo(value,host.width,color=colour).move_to(crop_frame(host,space*32,(space+1)*32,64,colour))
            self.show(crop_frame(host,space*32,(space+1)*32,64,colour),duration=.2)
            animations=[]
            for view,x in enumerate([-1.4,1.4]):
                rank=view*2+space
                frame=chip(rank,[x,y,0],width=2.35,height=1.5,color=colour)
                # Compact labels sit inside the card, away from the view headers.
                frame[2].move_to([x,y+.51,0])
                local=photo(value,1.65,color=colour).move_to([x,y-.15,0])
                self.show(frame,duration=.25)
                animations.append(TransformFromCopy(source,local))
                locals[rank]=local
            self.play(*animations,run_time=.8)
        self.mark('hybrid_inputs')
        self.wait(1.5)
        self.note('Each GPU projects its own part; add the two parts within each view group')
        for view,x in enumerate([-1.4,1.4]):
            for space,y in enumerate([.8,-1.2]):
                rank=view*2+space
                result=partials[space*2:space*2+2,view*32:(view+1)*32].sum(0)
                projected=photo(result,1.65,vmax,[GREEN,WARM][space]).move_to(locals[rank])
                self.play(Succession(FadeOut(locals[rank]),FadeIn(projected)),run_time=.55)
                computed[rank]=projected
            self.mark(f'hybrid_partials_{view}')
            target=photo(projection[view*32:(view+1)*32],output.width,vmax,[GREEN,WARM][view])
            target.move_to(crop_frame(output,view*32,(view+1)*32,64,[GREEN,WARM][view]))
            plus=MathTex('+',font_size=30,color=INK).move_to([x,-.2,0])
            self.show(plus,duration=.2)
            packets=[computed[view*2+space].copy() for space in range(2)]
            self.add(*packets)
            self.play(*[packet.animate.scale(.5).move_to(plus) for packet in packets],run_time=.65)
            self.remove(*packets)
            combined=photo(projection[view*32:(view+1)*32],.8,vmax).move_to(plus)
            self.play(TransformFromCopy(combined,target),FadeOut(plus),run_time=.8)
        self.mark('hybrid_forward')
        self.wait(2.2)
        self.note('Backprojection reverses the flow: add view-group contributions for each part')
        self.play(FadeOut(host),
                  Succession(FadeOut(host_label),FadeIn(T('Add on CPU',22,SUB).move_to(host_label))),
                  Succession(FadeOut(output_label),FadeIn(T('Sinogram input',22,SUB).move_to(output_label))),run_time=.3)
        back=example['backprojection'];bmax=float(back.max())
        for space in range(2):
            y=[.8,-1.2][space]
            pieces=[]
            for view,x in enumerate([-1.4,1.4]):
                rank=view*2+space
                part=example['view_backprojection'][view,space*32:(space+1)*32]
                contribution=photo(part,1.65,bmax,[GREEN,WARM][space]).move_to(computed[rank])
                source=photo(projection[view*32:(view+1)*32],output.width,vmax)
                source.move_to(crop_frame(output,view*32,(view+1)*32,64,[GREEN,WARM][view]))
                incoming=photo(projection[view*32:(view+1)*32],1.65,vmax).move_to(computed[rank])
                self.play(FadeOut(computed[rank]),TransformFromCopy(source,incoming),run_time=.55)
                self.play(Succession(FadeOut(incoming),FadeIn(contribution)),run_time=.5)
                pieces.append(contribution)
            final=photo(back[space*32:(space+1)*32],2.1,bmax,[GREEN,WARM][space])
            final.move_to([-5.05,stage_y+(.525 if space==0 else -.525),0])
            plus=MathTex('+',font_size=30,color=INK).move_to([0,y-.15,0])
            self.show(plus,duration=.2)
            packets=[part.copy() for part in pieces];self.add(*packets)
            self.play(*[packet.animate.scale(.5).move_to(plus) for packet in packets],run_time=.6)
            self.remove(*packets)
            combined=photo(back[space*32:(space+1)*32],.8,bmax).move_to(plus)
            self.play(TransformFromCopy(combined,final),FadeOut(plus),run_time=.7)
        self.show(T('Unfiltered backprojection\n(matched adjoint)',17,SUB).move_to([-5.05,stage_y-1.53,0]),duration=.3)
        self.note('Values spread along rays: blur and raised background, not a reconstruction')
        self.mark('hybrid_adjoint')
        self.wait(5)
        self.reset()
        self.heading(4,'Measured multi-GPU scaling')
        self.note('Circular cone beam · 128³ volume · 360 views · A100-SXM-64GB')
        data=json.loads((HERE/'scaling.json').read_text()); timings=data['timings']['128']
        baseline=timings['gpus_1']['cgls_ms_per_iter']
        chart_title=left(T('Archived view-partition benchmark · diffct 2.0.2',24,SUB),-6.05).set_y(2.35)
        metric=left(T('One CGLS iteration',28,SUB),-6.05).set_y(1.8)
        for i,cfg in enumerate(data['configurations']):
            value=timings[cfg['key']]['cgls_ms_per_iter'];y=1.05-i*1.13
            label=T(cfg['label'],28,SUB).move_to([-4.3,y,0])
            bar=Rectangle(width=6.15*value/baseline,height=.5,stroke_width=0,fill_color=GREEN,fill_opacity=.65 if i==0 else 1)
            left(bar,-2.05).set_y(y)
            value_label=T(f'{value:.2f} ms',28).next_to(bar,RIGHT,buff=.2)
            if i == 0:
                self.show(chart_title,metric,label,bar,value_label)
            else:
                self.show(label,duration=.25)
                self.play(GrowFromEdge(bar,LEFT),FadeIn(value_label),run_time=.7)
            if i:
                speed=T(f'{baseline/value:.1f}×',36,GREEN).move_to([4.75,y,0]);self.show(speed,duration=.3)
        self.show(T('These timings do not measure the new hybrid backend.',24,SUB).move_to([0,-2.4,0]))
        self.mark('scaling_full')
        self.wait(3)
        self.reset()

    def measured(self):
        self.mark('measured_start')
        self.heading(5,'One measured scan. Four reconstructions.')
        self.note('240 measured cone-beam views · 256³ reconstruction · shared display range')
        # Show the real acquisition before revealing the method comparison.
        projections=Group(*[gray(p.T,2.5,float(self.data['projections'].max())) for p in self.data['projections']])
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
        walnut=self.walnut(3.3,[0,.05,0],turn)
        self.show(walnut,T('diffct',72).move_to([0,2.65,0]))
        self.show(T('From your geometry to your reconstruction.',28,SUB).move_to([0,1.78,0]))
        self.show(code('pip install diffct',22,GREEN).move_to([-3.15,-1.8,0]),
                  code('pip install "diffct[cu12]"',22,GREEN).move_to([3.15,-1.8,0]),
                  T('CPU / ROCm: backend="torch"',20,SUB).move_to([-3.15,-2.36,0]),
                  T('Native NVIDIA · cu13 extra also available',20,SUB).move_to([3.15,-2.36,0]))
        self.show(T('Install the matching PyTorch build first',20,SUB).move_to([0,-2.7,0]))
        self.show(T('sypsyp97.github.io/diffct',24).move_to([0,-3.03,0]),
                  T('github.com/sypsyp97/diffct',24,SUB).move_to([0,-3.43,0]))
        self.mark('ending_full')
        self.play(turn.animate.set_value(180),run_time=5.5,rate_func=linear)
        walnut.clear_updaters()
        self.wait(.8)
        self.mark('end')
