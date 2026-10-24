"""PROOF: small exact-arithmetic models, not proofs of the compiled kernels.

Source anchors: beam forward/backward kernels (slabs, segment clipping and
partition integration), spacing.py:_axis_pitch/_ray_scale (voxel metrics),
beam analytical gather kernels and analytical.py (coordinates/quadrature).
Paths are relative to the repository root. BOUNDED source guards below
detect selected source drift; they do not establish program equivalence.

Assumptions: exact real arithmetic; nonzero directions use exact division;
zero directions use the parallel branch; an ordered, complete traversal
partition is supplied. Excludes float32/64, fastmath, CUDA compilation,
termination, voxel assignment, _TINY approximations, and _BIG sentinels.
Scalar and tensor-order per-axis pitches are positive. Ray directions are
normalized in voxel coordinates; zero-length rays are excluded from the metric.
"""

import ast
import itertools
import math
from pathlib import Path

import pytest
import sympy as sp
import z3

ROOT = Path(__file__).resolve().parents[1]


def _prove(assumptions, conclusion):
    solver = z3.Solver()
    solver.set(timeout=10000)
    solver.add(*assumptions, z3.Not(conclusion))
    result = solver.check()
    assert result == z3.unsat, (result, solver.model() if result == z3.sat else solver.reason_unknown())


def _min(a, b):
    return z3.If(a <= b, a, b)


def _max(a, b):
    return z3.If(a >= b, a, b)


def test_PROOF_single_slab_including_exact_axis_parallel():
    """The returned closed interval describes precisely one slab's points."""
    o, d, c, t = z3.Reals("o d c t")
    a, b = (-c - o) / d, (c - o) / d
    interval = z3.And(_min(a, b) <= t, t <= _max(a, b))
    spatial = z3.And(-c <= o + t * d, o + t * d <= c)
    _prove([c > 0, d != 0], interval == spatial)
    _prove([c > 0, d == 0], spatial == z3.And(-c <= o, o <= c))


@pytest.mark.parametrize("dimensions", [2, 3])
def test_PROOF_slab_intersection_clipped_source_detector_segment(dimensions):
    """Composition of closed slab intervals, with clipping to [0, L]."""
    L, t = z3.Reals("L t")
    lo, hi = z3.RealVal(0), L
    accepted = z3.BoolVal(True)
    spatial = [0 <= t, t <= L]
    assumptions = [L > 0]
    for axis in range(dimensions):
        o, d, c = z3.Reals(f"o{axis} d{axis} c{axis}")
        assumptions.append(c > 0)
        a, b = (-c - o) / d, (c - o) / d
        lo = z3.If(d == 0, lo, _max(lo, _min(a, b)))
        hi = z3.If(d == 0, hi, _min(hi, _max(a, b)))
        accepted = z3.And(accepted, z3.Or(d != 0, z3.And(-c <= o, o <= c)))
        spatial.extend([-c <= o + t * d, o + t * d <= c])
    _prove(assumptions, z3.And(accepted, lo <= t, t <= hi) == z3.And(*spatial))
    _prove(assumptions + [accepted, lo < hi], z3.And(lo >= 0, hi <= L))
    # Point intersections have zero integral; the source's >= rejection is valid.
    _prove(assumptions + [accepted, lo <= hi], hi - lo >= 0)


def test_PROOF_nearer_endpoint_parameter_shift():
    """Keeping direction fixed changes [0,L] to [-L,0] at detector origin."""
    o, d, L, t = z3.Reals("o d L t")
    detector = o + L * d
    _prove([L > 0], detector + (t - L) * d == o + t * d)
    _prove([L > 0], z3.And(0 <= t, t <= L) == z3.And(-L <= t - L, t - L <= 0))


def test_PROOF_ordered_partition_telescopes_by_induction():
    """Base and arbitrary extension imply the identity for every finite count.

    Let S_n=sum(t[j+1]-t[j], j=0..n-1). Base n=0 is zero.
    The induction hypothesis S_n=t_n-t_0 proves S_(n+1)=t_(n+1)-t_0.
    Repeated crossing times are allowed; their contribution is exactly zero.
    This proves an abstract partition invariant, not traversal completeness.
    """
    first, last, next_t, total, value, scale = z3.Reals("first last next total value scale")
    _prove([], first - first == 0)
    _prove([total == last - first], total + (next_t - last) == next_t - first)
    _prove([first <= last, last <= next_t], next_t - last >= 0)
    _prove([total == last - first], value * scale * (total + next_t - last) == value * scale * (next_t - first))
    # One positive physical-length factor applies to every piece of this ray.
    _prove([scale > 0, first <= last], scale * (last - first) >= 0)
    cotangent = z3.Real("cotangent")
    _prove([], cotangent * (value * (next_t - last) * scale)
           == value * ((cotangent * scale) * (next_t - last)))


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("per_axis", [False, True], ids=["scalar", "per-axis"])
def test_PROOF_coordinate_inverse_and_centering(dimensions, per_axis):
    """Exact symbolic inverse, reflection, centre average and cell containment."""
    i, n, x, offset = sp.symbols("i n x offset", real=True)
    pitches = (sp.symbols(f"s0:{dimensions}", positive=True) if per_axis else
               (sp.symbols("s", positive=True),) * dimensions)
    detector_pitch = sp.symbols("detector_pitch", positive=True)
    detector = (i - (n - 1) / 2) * detector_pitch + offset
    detector_inverse = (x - offset) / detector_pitch + (n - 1) / 2
    assert sp.simplify(detector_inverse.subs(x, detector) - i) == 0
    assert sp.simplify(detector.subs(i, n - 1 - i) + detector - 2 * offset) == 0
    assert sp.simplify((detector.subs(i, 0) + detector.subs(i, n - 1)) / 2 - offset) == 0
    for pitch in reversed(pitches):  # Tensor [z,]y,x pitches map to world x,y[,z].
        voxel = (i + sp.Rational(1, 2) - n / 2) * pitch
        voxel_inverse = x / pitch + n / 2 - sp.Rational(1, 2)
        assert sp.simplify(voxel_inverse.subs(x, voxel) - i) == 0
        assert sp.simplify(voxel.subs(i, n - 1 - i) + voxel) == 0
        assert sp.simplify((voxel.subs(i, 0) + voxel.subs(i, n - 1)) / 2) == 0
        assert sp.simplify(voxel / pitch + n / 2 - i) == sp.Rational(1, 2)
        # Flat detector composition and explicit world points use the same map.
        centre, u, v, a, b = sp.symbols("centre u v a b", real=True)
        world = centre + a * u + (b * v if dimensions == 3 else 0)
        mapped = centre / pitch + a * u / pitch + (b * v / pitch if dimensions == 3 else 0)
        assert sp.simplify(mapped - world / pitch) == 0


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("per_axis", [False, True], ids=["scalar", "per-axis"])
def test_PROOF_normalized_voxel_ray_recovers_physical_length(dimensions, per_axis):
    """For a nonzero ray, the voxel norm times its metric is the physical norm."""
    pitches = (sp.symbols(f"s0:{dimensions}", positive=True) if per_axis else
               (sp.symbols("s", positive=True),) * dimensions)
    displacement = sp.symbols(f"r0:{dimensions}", real=True)
    xyz_pitches = pitches[::-1]
    voxel = [component / pitch for component, pitch in zip(displacement, xyz_pitches)]
    squared_length = sum(component ** 2 for component in voxel)
    direction = [component / sp.sqrt(squared_length) for component in voxel]
    assert sp.simplify(sum(component ** 2 for component in direction) - 1) == 0
    metric_squared = sum((pitch * component) ** 2 for pitch, component in zip(xyz_pitches, direction))
    assert sp.simplify(squared_length * metric_squared - sum(r ** 2 for r in displacement)) == 0
    # Both norms use the nonnegative square root. Uniform tuples reduce to s.
    s = sp.symbols("uniform_pitch", positive=True)
    assert sp.sqrt(sp.simplify(metric_squared.subs(dict.fromkeys(pitches, s)))) == s


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("representation", ["scalar", "uniform", "anisotropic"])
def test_BOUNDED_metric_helper_and_cuda_overload_bodies(simulated_kernels, dimensions, representation):
    """CPU execution of selected overload bodies, without CUDA compilation.

    Enumerate nonzero world displacements in {-1,0,1}^dimensions. Compare
    physical norms and a four-piece partition, including a zero-length piece.
    """
    import numpy as np
    from numba import typeof
    from diffct.kernels import spacing as metrics

    pitches = ((0.7, 1.9) if dimensions == 2 else (0.7, 1.3, 1.9))
    spacing = (np.float32(1.7) if representation == "scalar" else
               tuple(np.float32(s) for s in ((1.7,) * dimensions if representation == "uniform" else pitches)))
    widths = ((float(spacing),) * dimensions if representation == "scalar" else
              tuple(float(s) for s in reversed(spacing)))
    spacing_type = typeof(spacing)
    per_axis_impl = metrics._is_per_axis_overload(spacing_type)
    pitch_impl = metrics._axis_pitch_overload(spacing_type, typeof(0))
    direction_type = typeof(np.float64(1))
    scale_impl = metrics._ray_scale_overload(spacing_type, direction_type, direction_type, direction_type)
    assert metrics._is_per_axis(spacing) == per_axis_impl(spacing) == (representation != "scalar")
    for axis, width in enumerate(widths):
        assert metrics._axis_pitch(spacing, axis) == pitch_impl(spacing, axis) == width
    for displacement in itertools.product((-1.0, 0.0, 1.0), repeat=dimensions):
        if not any(displacement):
            continue
        voxel = tuple(component / width for component, width in zip(displacement, widths))
        length = math.hypot(*voxel)
        direction = tuple(component / length for component in voxel) + (0.0,) * (3 - dimensions)
        scale = metrics._ray_scale(spacing, *direction)
        assert scale == scale_impl(spacing, *direction)
        assert scale > 0
        expected = math.hypot(*displacement)
        assert math.isclose(length * scale, expected, rel_tol=2e-7)
        crossings = (0.0, length / 4, length / 4, length / 2, length)
        integral = sum((end - start) * scale for start, end in zip(crossings, crossings[1:]))
        assert math.isclose(integral, expected, rel_tol=2e-7)


def test_PROOF_angular_quadrature_total_by_induction_and_nonnegativity():
    """For any n>=2, open weights total span; periodic weights total period.

    Base: two samples give g/2+g/2=g. Appending gap h changes the old
    final boundary g/2 into (g+h)/2 and adds h/2, increasing total by h.
    A nonnegative closure contributes c/2 at each end. Period=span+c.
    Classification and clamping are excluded: an overshooting span gives
    closure zero and total span, rather than the nominal period.
    """
    g, h, closure, total = sp.symbols("g h closure total", nonnegative=True)
    assert sp.simplify(g / 2 + g / 2 - g) == 0
    updated = total - g / 2 + (g + h) / 2 + h / 2
    assert sp.simplify(updated - (total + h)) == 0
    assert sp.simplify(closure / 2 + closure / 2 - closure) == 0
    assert (g / 2).is_nonnegative
    assert ((g + h) / 2).is_nonnegative
    assert ((g + closure) / 2).is_nonnegative
    assert sp.simplify((total + closure) / 2 - total / 2 - closure / 2) == 0


def test_PROOF_ideal_parker_complement_with_explicit_plus_gamma_convention():
    """Taper identity only: W(beta,gamma)+W(beta+pi+2gamma,-gamma)=1.

    Domain 0<=beta<=2(delta-gamma), |gamma|<delta, 0<delta<pi/2;
    both points lie in their corresponding taper regions. The source's
    +1e-12 denominators mean this exact identity is NOT a runtime proof.
    It does not prove ray pairing for arbitrary trajectories or coverage.
    Source: analytical.py:207-220, gamma=atan(u/sdd), not its negation.
    """
    beta, gamma, delta = sp.symbols("beta gamma delta", real=True)
    start_arg = sp.pi * beta / (4 * (delta - gamma))
    paired_beta = beta + sp.pi + 2 * gamma
    end_arg = sp.pi * (sp.pi + 2 * delta - paired_beta) / (4 * (delta - gamma))
    assert sp.simplify(end_arg - (sp.pi / 2 - start_arg)) == 0
    theta = sp.symbols("theta", real=True)
    assert sp.trigsimp(sp.sin(theta) ** 2 + sp.sin(sp.pi / 2 - theta) ** 2 - 1) == 0
    # At beta=0 the source is (0,R), and positive u points along +x.
    # The partner source at pi+2*gamma lies on the original fan ray.
    radius = sp.symbols("radius", positive=True)
    partner_displacement = sp.Matrix([radius * sp.sin(2 * gamma), -radius * sp.cos(2 * gamma) - radius])
    ray_direction = sp.Matrix([sp.sin(gamma), -sp.cos(gamma)])
    for component in partner_displacement - 2 * radius * sp.cos(gamma) * ray_direction:
        assert sp.trigsimp(component) == 0


def _source_statements(node):
    return {ast.unparse(item) for item in ast.walk(node)
            if isinstance(item, (ast.Assign, ast.AugAssign, ast.Compare))}


def _spacing_branch_statements(fn, per_axis):
    statements = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_is_per_axis(voxel_spacing)":
            for statement in (node.body if per_axis else node.orelse):
                statements.update(_source_statements(statement))
    return statements


def _check_beam_source_bindings(name, tree):
    normalize = lambda source: ast.unparse(ast.parse(source).body[0])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    traversals = [fn for fn in functions if fn.name.endswith(("_forward_kernel", "_backward_kernel"))]
    assert len(traversals) == 2
    axes = "xyz" if name == "cone_beam" else "xy"
    ray_scale = f"_ray_scale(voxel_spacing, dir_x, dir_y, {'dir_z' if name == 'cone_beam' else '_ZERO'})"
    pixel = "iview, iu, iv" if name == "cone_beam" else "iang, idet"
    view = "iview" if name == "cone_beam" else "iang"
    for fn in traversals:
        statements = _source_statements(fn)
        per_axis = _spacing_branch_statements(fn, True)
        assert "seg_len = t_next - t" in statements, fn.name
        assert "seg_len > _ZERO" in statements, fn.name
        products = {ast.unparse(node) for node in ast.walk(fn) if isinstance(node, ast.BinOp)}
        if fn.name.endswith("_forward_kernel"):
            image = "d_vol[ix, iy, iz]" if name == "cone_beam" else "d_image[iy, ix]"
            assert normalize(f"accum += {image} * seg_len") in statements
            assert f"d_sino[{pixel}] = accum * {ray_scale}" in statements
        else:
            value = "g" if name == "cone_beam" else "val"
            assert f"{value} * seg_len" in products
            assert f"{value} = d_sino[{pixel}] * _axis_pitch(voxel_spacing, 0)" in statements
            assert f"{value} = d_sino[{pixel}] * {ray_scale}" in per_axis
        for index, axis in enumerate(axes):
            update = f"t_min, t_max = max(t_min, min(t{axis}1, t{axis}2)), min(t_max, max(t{axis}1, t{axis}2))"
            assert normalize(update) in statements, fn.name
            if name == "parallel_beam":
                point = f"np.float64(d_detector_positions[{pixel}, {index}])"
                centre = f"np.float64(d_det_origin[{view}, {index}])"
                assert f"pnt_{axis} = {point} / _axis_pitch(voxel_spacing, {index})" in statements
                assert f"det_o{axis} = {centre} / _axis_pitch(voxel_spacing, {index})" in statements
                assert f"dir_{axis} /= _axis_pitch(voxel_spacing, {index})" in per_axis
            else:
                assert f"src_{axis} = d_src_pos[{view}, {index}] / _axis_pitch(voxel_spacing, {index})" in statements
                assert f"det_c{axis} = d_det_center[{view}, {index}] / _axis_pitch(voxel_spacing, {index})" in statements
                assert f"det_{axis} = d_detector_positions[{pixel}, {index}] / _axis_pitch(voxel_spacing, {index})" in statements
                detector = f"det_c{axis} + u_phys * u_vec_{axis} / _axis_pitch(voxel_spacing, {index})"
                if name == "cone_beam":
                    detector += f" + v_phys * v_vec_{axis} / _axis_pitch(voxel_spacing, {index})"
                assert normalize(f"det_{axis} = {detector}") in per_axis
        if name != "parallel_beam":
            assert normalize("t_min, t_max = _ZERO, length") in statements
            assert normalize("t_min, t_max = -length, _ZERO") in statements
            assert "inv_len = _ONE / length" in statements
            direction = ", ".join(f"dir_{axis}" for axis in axes)
            normalized = ", ".join(f"dir_{axis} * inv_len" for axis in axes)
            assert normalize(f"{direction} = {normalized}") in statements
        else:
            assert "inv_length = _ONE / math.sqrt(dir_x * dir_x + dir_y * dir_y)" in per_axis
            assert normalize("dir_x, dir_y = dir_x * inv_length, dir_y * inv_length") in per_axis
        precision = "float64" if name == "parallel_beam" else "float32"
        if name == "cone_beam":
            offsets = ["u_offset = (np.float32(iu) + _HALF - np.float32(n_u) * _HALF) * du / _axis_pitch(voxel_spacing, 0)",
                       "v_offset = (np.float32(iv) + _HALF - np.float32(n_v) * _HALF) * dv / _axis_pitch(voxel_spacing, 0)"]
            physical_offsets = [expression.replace("_offset", "_phys").split(" / _axis_pitch")[0]
                                for expression in offsets]
        else:
            offsets = [f"u_offset = (np.{precision}(idet) + _HALF - np.{precision}(n_det) * _HALF) * det_spacing / _axis_pitch(voxel_spacing, 0)"]
            physical_offsets = [offsets[0].replace("u_offset", "u_phys").split(" / _axis_pitch")[0]]
        assert all(normalize(expression) in statements for expression in offsets)
        if name == "parallel_beam":
            offset_y = offsets[0].replace("u_offset", "offset_y").replace("voxel_spacing, 0", "voxel_spacing, 1")
            assert normalize(offset_y) in per_axis
            assert "pnt_y = det_oy + offset_y * u_vec_y" in per_axis
        else:
            assert all(normalize(expression) in per_axis for expression in physical_offsets)
    gather = [fn for fn in functions if fn.name.endswith("_backproject_kernel")]
    assert len(gather) == 1
    assignments = _source_statements(gather[0])
    per_axis = _spacing_branch_statements(gather[0], True)
    scalar = _spacing_branch_statements(gather[0], False)
    for index, axis in enumerate(axes):
        assert f"{axis}_v = np.float32(i{axis}) + _HALF - c{axis}" in assignments
        assert f"{axis}_v *= _axis_pitch(voxel_spacing, {index})" in per_axis
    assert "coordinate_scale = _ONE" in per_axis
    assert "coordinate_scale = _axis_pitch(voxel_spacing, 0)" in scalar
    detectors = ["n_u", "n_v"] if name == "cone_beam" else ["n_det"]
    for axis, count in zip("uv", detectors):
        assert f"half_{axis} = (np.float32({count}) - _ONE) * _HALF" in assignments
        spacing = f"d{axis}" if name == "cone_beam" else "det_spacing"
        target = f"d{axis}_v" if name == "cone_beam" else "det_spacing_v"
        assert f"{target} = {spacing} / coordinate_scale" in assignments


def _check_spacing_source_bindings(tree):
    expected = {
        "_is_per_axis": "return isinstance(spacing, tuple)",
        "_is_per_axis_overload": """per_axis = isinstance(spacing, types.BaseTuple)
def impl(spacing):
    return per_axis
return impl""",
        "_axis_pitch": """if isinstance(spacing, tuple):
    return spacing[len(spacing) - 1 - axis]
return spacing""",
        "_axis_pitch_overload": """if isinstance(spacing, types.BaseTuple):
    rank = len(spacing)
    def impl(spacing, axis):
        return spacing[rank - 1 - axis]
else:
    def impl(spacing, axis):
        return spacing
return impl""",
        "_ray_scale_2d": """x = np.float64(spacing[-1]) * dx
y = np.float64(spacing[-2]) * dy
squared = x * x + y * y
return np.float32(math.sqrt(squared))""",
        "_ray_scale_3d": """x = np.float64(spacing[-1]) * dx
y = np.float64(spacing[-2]) * dy
squared = x * x + y * y
z = np.float64(spacing[0]) * dz
squared += z * z
return np.float32(math.sqrt(squared))""",
        "_ray_scale": """if isinstance(spacing, tuple):
    if len(spacing) == 3:
        return _ray_scale_3d(spacing, dx, dy, dz)
    return _ray_scale_2d(spacing, dx, dy, dz)
return spacing""",
        "_ray_scale_overload": """if isinstance(spacing, types.BaseTuple):
    return _ray_scale_3d if len(spacing) == 3 else _ray_scale_2d
def impl(spacing, dx, dy, dz):
    return spacing
return impl""",
    }
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, body in expected.items():
        if name.endswith("_overload"):
            helper = name.removesuffix("_overload")
            assert [ast.unparse(node) for node in functions[name].decorator_list] == [
                f"overload({helper}, target='cuda')"], name
        statements = [node for node in functions[name].body
                      if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                              and isinstance(node.value.value, str))]
        assert ast.unparse(ast.Module(body=statements, type_ignores=[])) == ast.unparse(ast.parse(body)), name


def _mutate_source_function(source, name, old, new):
    """Change one occurrence in one function; keep all other bindings intact."""
    tree = ast.parse(source)
    for index, fn in enumerate(tree.body):
        if isinstance(fn, ast.FunctionDef) and fn.name == name:
            function_source = ast.unparse(fn) + "\n"
            assert old in function_source, (name, old)
            tree.body[index] = ast.parse(function_source.replace(old, new, 1)).body[0]
            return tree
    raise AssertionError(f"Missing mutation target: {name}")


@pytest.mark.parametrize("name", ["fan_beam", "cone_beam", "parallel_beam"])
def test_BOUNDED_source_binding_guards_for_partition_and_coordinate_models(name):
    """Selected AST expressions anchor models; finite checks are not proofs."""
    tree = ast.parse((ROOT / "diffct" / "kernels" / f"{name}.py").read_text())
    _check_beam_source_bindings(name, tree)


def test_BOUNDED_source_binding_guards_for_voxel_metric_helpers():
    _check_spacing_source_bindings(ast.parse((ROOT / "diffct" / "kernels" / "spacing.py").read_text()))


@pytest.mark.parametrize("name", ["fan_beam", "cone_beam", "parallel_beam"])
@pytest.mark.parametrize("kernel,mutation", [
    ("forward", "ray-scale"), ("backward", "ray-scale"),
    ("forward", "axis-pitch"), ("backward", "axis-pitch"),
    ("backproject", "gather-pitch"),
    ("forward", "normalization"), ("backward", "normalization"),
])
def test_BOUNDED_beam_source_guards_reject_metric_mutations(name, kernel, mutation):
    """Mutate source text in memory only; every selected drift must be detected."""
    source = (ROOT / "diffct" / "kernels" / f"{name}.py").read_text()
    _check_beam_source_bindings(name, ast.parse(source))
    prefix = "_cone_3d" if name == "cone_beam" else f"_{name.removesuffix('_beam')}_2d"
    if kernel == "backproject":
        kernel = "fdk_backproject" if name == "cone_beam" else "fbp_backproject"
    old, new = {
        "ray-scale": ("_ray_scale(voxel_spacing, dir_x, dir_y,", "_ray_scale(voxel_spacing, dir_x, dir_x,"),
        "axis-pitch": ("/ _axis_pitch(voxel_spacing, 1)", "/ _axis_pitch(voxel_spacing, 0)"),
        "gather-pitch": ("y_v *= _axis_pitch(voxel_spacing, 1)", "y_v *= _axis_pitch(voxel_spacing, 0)"),
        "normalization": (("inv_length = _ONE / math.sqrt(dir_x * dir_x + dir_y * dir_y)",
                           "inv_length = _ONE") if name == "parallel_beam" else
                          ("inv_len = _ONE / length", "inv_len = _ONE")),
    }[mutation]
    mutated = _mutate_source_function(source, f"{prefix}_{kernel}_kernel", old, new)
    with pytest.raises(AssertionError):
        _check_beam_source_bindings(name, mutated)


@pytest.mark.parametrize("function,old,new", [
    ("_is_per_axis", "isinstance(spacing, tuple)", "isinstance(spacing, list)"),
    ("_is_per_axis_overload", "types.BaseTuple", "types.Number"),
    ("_axis_pitch", "len(spacing) - 1 - axis", "axis"),
    ("_axis_pitch_overload", "rank - 1 - axis", "axis"),
    ("_axis_pitch", "return spacing\n", "return spacing * spacing\n"),
    ("_axis_pitch_overload", "return spacing\n", "return spacing * spacing\n"),
    ("_ray_scale_2d", "y * y", "x * x"),
    ("_ray_scale_3d", "squared += z * z", "squared += 0"),
    ("_ray_scale", "return spacing\n", "return spacing * spacing\n"),
    ("_ray_scale_overload", "return spacing\n", "return spacing * spacing\n"),
    ("_ray_scale", "len(spacing) == 3", "len(spacing) == 2"),
    ("_ray_scale_overload", "len(spacing) == 3", "len(spacing) == 2"),
    ("_ray_scale_overload", "target='cuda'", "target='cpu'"),
])
def test_BOUNDED_spacing_source_guards_reject_metric_mutations(function, old, new):
    source = (ROOT / "diffct" / "kernels" / "spacing.py").read_text()
    _check_spacing_source_bindings(ast.parse(source))
    mutated = _mutate_source_function(source, function, old, new)
    with pytest.raises(AssertionError):
        _check_spacing_source_bindings(mutated)


def test_BOUNDED_source_binding_guards_for_angular_and_parker_models():
    tree = ast.parse((ROOT / "diffct" / "analytical.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    coordinates = ast.unparse(functions["detector_coordinates_1d"])
    assert "return (k - 0.5 * (num_detectors - 1)) * detector_spacing + detector_offset" in coordinates
    angular = ast.unparse(functions["angular_integration_weights"])
    assert "0.5 * (diffs[:-1] + diffs[1:])" in angular
    assert "0.5 * (closure + diffs[0])" in angular
    assert "w = w * 0.5" in angular
    assert "out[sort_idx] = w" in angular
    parker = ast.unparse(functions["parker_weights"])
    assert "gamma = torch.atan(u / sdd)" in parker
    assert "beta > math.pi - 2.0 * gamma_b" in parker
    assert "delta + gamma_b + 1e-12" in parker
    constants = ast.parse((ROOT / "diffct" / "constants.py").read_text())
    constants = {node.targets[0].id: node.value for node in constants.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    for name, value in [("_HALF", 0.5), ("_ONE", 1.0), ("_ZERO", 0.0)]:
        assert ast.literal_eval(constants[name].args[0]) == value
