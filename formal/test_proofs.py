"""PROOF: small exact-arithmetic models, not proofs of the compiled kernels.

Source anchors: fan_beam.py:111-140, cone_beam.py:129-177 (slabs and
segment clipping); fan_beam.py:164-186, cone_beam.py:205-231 (partition
integration); analytical.py:51-56,94-129 (coordinates and quadrature).
Paths are relative to the repository root. BOUNDED source guards below
detect selected source drift; they do not establish program equivalence.

Assumptions: exact real arithmetic; nonzero directions use exact division;
zero directions use the parallel branch; an ordered, complete traversal
partition is supplied. Excludes float32/64, fastmath, CUDA compilation,
termination, voxel assignment, _TINY approximations, and _BIG sentinels.
"""

import ast
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
    first, last, next_t, total, value, spacing = z3.Reals("first last next total value spacing")
    _prove([], first - first == 0)
    _prove([total == last - first], total + (next_t - last) == next_t - first)
    _prove([first <= last, last <= next_t], next_t - last >= 0)
    _prove([total == last - first], value * spacing * (total + next_t - last) == value * spacing * (next_t - first))


def test_PROOF_coordinate_inverse_and_centering():
    """Exact symbolic inverse, reflection, centre average and cell containment."""
    i, n, x, offset = sp.symbols("i n x offset", real=True)
    s = sp.symbols("s", positive=True)
    voxel = (i + sp.Rational(1, 2) - n / 2) * s
    detector = (i - (n - 1) / 2) * s + offset
    voxel_inverse = x / s + n / 2 - sp.Rational(1, 2)
    detector_inverse = (x - offset) / s + (n - 1) / 2
    assert sp.simplify(voxel_inverse.subs(x, voxel) - i) == 0
    assert sp.simplify(detector_inverse.subs(x, detector) - i) == 0
    assert sp.simplify(voxel.subs(i, n - 1 - i) + voxel) == 0
    assert sp.simplify(detector.subs(i, n - 1 - i) + detector - 2 * offset) == 0
    assert sp.simplify((voxel.subs(i, 0) + voxel.subs(i, n - 1)) / 2) == 0
    assert sp.simplify((detector.subs(i, 0) + detector.subs(i, n - 1)) / 2 - offset) == 0
    assert sp.simplify(voxel / s + n / 2 - i) == sp.Rational(1, 2)


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


@pytest.mark.parametrize("name", ["fan_beam", "cone_beam", "parallel_beam"])
def test_BOUNDED_source_binding_guards_for_partition_and_coordinate_models(name):
    """Selected AST expressions anchor models; finite checks are not proofs."""
    path = ROOT / "diffct" / "kernels" / f"{name}.py"
    tree = ast.parse(path.read_text())
    normalize = lambda source: ast.unparse(ast.parse(source).body[0])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    traversals = [fn for fn in functions if fn.name.endswith(("_forward_kernel", "_backward_kernel"))]
    assert len(traversals) == 2
    for fn in traversals:
        statements = {ast.unparse(node) for node in ast.walk(fn) if isinstance(node, (ast.Assign, ast.Compare))}
        assert "seg_len = t_next - t" in statements, fn.name
        assert "seg_len > _ZERO" in statements, fn.name
        products = {ast.unparse(node) for node in ast.walk(fn) if isinstance(node, ast.BinOp)}
        if fn.name.endswith("_forward_kernel"):
            assert "accum * voxel_spacing" in products
        else:
            value = "g" if name == "cone_beam" else "val"
            assert f"{value} * seg_len" in products
            assert any(expression.endswith(" * voxel_spacing") for expression in products)
        for axis in ("xyz" if name == "cone_beam" else "xy"):
            update = f"t_min, t_max = max(t_min, min(t{axis}1, t{axis}2)), min(t_max, max(t{axis}1, t{axis}2))"
            assert normalize(update) in statements, fn.name
        if name != "parallel_beam":
            assert normalize("t_min, t_max = _ZERO, length") in statements
            assert normalize("t_min, t_max = -length, _ZERO") in statements
        precision = "float64" if name == "parallel_beam" else "float32"
        if name == "cone_beam":
            offsets = ["u_offset = (np.float32(iu) + _HALF - np.float32(n_u) * _HALF) * du / voxel_spacing",
                       "v_offset = (np.float32(iv) + _HALF - np.float32(n_v) * _HALF) * dv / voxel_spacing"]
        else:
            offsets = [f"u_offset = (np.{precision}(idet) + _HALF - np.{precision}(n_det) * _HALF) * det_spacing / voxel_spacing"]
        assert all(normalize(expression) in statements for expression in offsets)
    gather = [fn for fn in functions if fn.name.endswith("_backproject_kernel")]
    assert len(gather) == 1
    assignments = {ast.unparse(node) for node in ast.walk(gather[0]) if isinstance(node, ast.Assign)}
    for axis in ("xyz" if name == "cone_beam" else "xy"):
        assert f"{axis}_v = np.float32(i{axis}) + _HALF - c{axis}" in assignments
    detectors = ["n_u", "n_v"] if name == "cone_beam" else ["n_det"]
    for axis, count in zip("uv", detectors):
        assert f"half_{axis} = (np.float32({count}) - _ONE) * _HALF" in assignments


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
