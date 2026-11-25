"""The CGLS read-only guard is exact, keeps the iterate local, and stays block bounded.

The guard runs after every CGLS iteration that has a callback (diffct.cgls and
diffct.cgls_into with a tensor output). These tests drive the real solvers and
the real guard (``_solver_state.callback_stop`` with its existing signature);
nothing inside the guard is mocked.
"""

import hashlib
import json
from collections import Counter

import numpy as np
import pytest
import torch
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

import diffct
from diffct import _solver_state
from tests.test_solver_api import _MatrixOperator, _checkpoint, _fixture


READ_ONLY = r"(?i)read[- ]only"
FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
BITS = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
        torch.float32: torch.int32, torch.float64: torch.int64}
WIDTH = {torch.int16: 16, torch.int32: 32, torch.int64: 64}
MANTISSA = {torch.float16: 10, torch.bfloat16: 7, torch.float32: 23, torch.float64: 52}
DEFAULT_BLOCK = 1048576
TRUTH = ((-.3, .7, 1.1), (.2, -.15, .9))
LAYOUTS = ("contiguous", "transposed", "strided", "offset")


def _signed(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value


def _iterate(dtype, layout="contiguous", shape=(3, 4), seed=0, device="cpu"):
    """Distinct raw bit patterns, including NaN payloads, infinities, zeros and subnormals."""
    integer = BITS[dtype]
    width = WIDTH[integer]
    count = shape[0] * shape[1]
    generator = np.random.default_rng(seed)
    patterns = set()
    while len(patterns) < count:
        patterns.add(int(generator.integers(0, 1 << width, dtype=np.uint64)) if width == 64
                     else int(generator.integers(0, 1 << width)))
    # Fixed special values first: +0, -0, one quiet NaN and its payload neighbour.
    mantissa = MANTISSA[dtype]
    nan = ((1 << (width - 1 - mantissa)) - 1) << mantissa | 1 << (mantissa - 1)
    specials = [0, 1 << (width - 1), nan, nan | 1]
    values = specials + sorted(patterns - set(specials))[:count - len(specials)]
    bits = torch.tensor([_signed(value, width) for value in values], dtype=integer, device=device)
    if layout == "contiguous":
        base = bits.reshape(shape)
    elif layout == "transposed":
        base = bits.new_empty(shape[::-1]).t()
        base.copy_(bits.reshape(shape))
    elif layout == "strided":
        base = bits.new_zeros((shape[0], 2 * shape[1]))[:, ::2]
        base.copy_(bits.reshape(shape))
    else:
        base = bits.new_zeros((shape[0] + 2, shape[1] + 3))[1:-1, 2:-1]
        base.copy_(bits.reshape(shape))
    volume = base.view(dtype)
    assert tuple(volume.shape) == tuple(shape)
    assert volume.is_contiguous() == (layout == "contiguous")
    return volume


def _bits(volume):
    return volume.detach().view(BITS[volume.dtype]).clone()


_OPERATOR = _MatrixOperator()


def _guard(volume, callback, block_elements=DEFAULT_BLOCK):
    return _solver_state.callback_stop(_OPERATOR, callback, 1, .5, volume, readonly=True,
                                       block_elements=block_elements)


def _flip(volume, index, mask):
    view = volume.view(BITS[volume.dtype])
    view[index] = view[index] ^ mask


def _rejects(volume, mutate, block_elements=DEFAULT_BLOCK):
    try:
        _guard(volume, lambda iteration, residual, value: mutate(value), block_elements)
    except ValueError as error:
        assert "read-only" in str(error).lower() or "read only" in str(error).lower(), error
        return True
    return False


# ---------------------------------------------------------------------------
# Exact detection through the guard helper (B2)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
def test_guard_rejects_every_single_bit_change_at_every_position(dtype, layout):
    """Each stored bit of each element, across block boundaries and strided layouts."""
    width = WIDTH[BITS[dtype]]
    shape = (3, 4)
    missed = []
    for flat in range(12):
        index = tuple(int(value) for value in np.unravel_index(flat, shape))
        for bit in range(width):
            block = (2, 4, 6, DEFAULT_BLOCK)[(flat + bit) % 4]
            volume = _iterate(dtype, layout, shape)
            mask = _signed(1 << bit, width)
            if not _rejects(volume, lambda value: _flip(value, index, mask), block):
                missed.append((index, bit, block))
    assert not missed, f"{len(missed)} single-bit iterate changes escaped the guard: {missed[:8]}"


@pytest.mark.parametrize("layout", ("contiguous", "transposed"))
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
def test_guard_rejects_swaps_and_compensating_pair_changes(dtype, layout):
    """Order-insensitive, XOR-folded or plain-sum digests miss these changes."""
    width = WIDTH[BITS[dtype]]
    shape = (3, 4)
    missed = []
    positions = [tuple(int(value) for value in np.unravel_index(flat, shape)) for flat in range(12)]
    for first in range(12):
        for second in range(first + 1, 12):
            a, b = positions[first], positions[second]
            block = (2, 5, DEFAULT_BLOCK)[(first + second) % 3]

            def swap(value):
                bits = value.view(BITS[dtype])
                left, right = bits[a].clone(), bits[b].clone()
                bits[a], bits[b] = right, left

            def same_bit(bit):
                mask = _signed(1 << bit, width)
                return lambda value: (_flip(value, a, mask), _flip(value, b, mask))

            def opposite_steps(value):
                bits = value.view(BITS[dtype])
                bits[a] += 1
                bits[b] -= 1

            cases = [("swap", swap), ("opposite integer steps", opposite_steps)]
            cases += [(f"bit {bit} in both", same_bit(bit)) for bit in (0, width // 2, width - 2, width - 1)]
            for name, mutate in cases:
                if not _rejects(_iterate(dtype, layout, shape, seed=first), mutate, block):
                    missed.append((name, a, b, block))
    assert not missed, f"{len(missed)} pair changes escaped the guard: {missed[:8]}"


def _set_bits(value, index, pattern):
    width = WIDTH[BITS[value.dtype]]
    value.view(BITS[value.dtype])[index] = _signed(pattern, width)


def _change(value):
    """One element changed by its lowest stored bit: same sign, nearly the same value."""
    bits = _bits(value)
    bits[0, 1] ^= 1
    return bits.view(value.dtype)


MEANS = {
    "inplace-op": lambda value: value[0, 1].copy_(_change(value)[0, 1]),
    "out-argument": lambda value: torch.add(_change(value), 0, out=value),
    "view-alias": lambda value: value.view(BITS[value.dtype])[0, 1].bitwise_xor_(1),
    "data": lambda value: value.data.view(BITS[value.dtype])[0, 1].bitwise_xor_(1),
    "numpy": lambda value: value.view(BITS[value.dtype]).numpy().__setitem__(
        (0, 1), value.view(BITS[value.dtype]).numpy()[0, 1] ^ 1),
    "dlpack": lambda value: torch.from_dlpack(value.detach().view(BITS[value.dtype]))[0, 1].bitwise_xor_(1),
    "set-storage": lambda value: value.set_(_change(value)),
}


@pytest.mark.parametrize("means", sorted(MEANS))
@pytest.mark.parametrize("layout", ("contiguous", "transposed"))
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
def test_guard_rejects_changes_made_by_any_means(dtype, layout, means):
    volume = _iterate(dtype, layout)
    before = _bits(volume)
    assert _rejects(volume, MEANS[means], 5), f"{means} write escaped the guard"
    assert not torch.equal(_bits(volume), before), "fixture did not actually change the iterate"


@pytest.mark.parametrize("change", ["+0 to -0", "-0 to +0", "NaN payload", "NaN sign", "quiet to signalling NaN"])
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
def test_guard_rejects_value_equal_or_nan_bit_changes(dtype, change):
    """Changes that compare equal (or compare unequal to themselves) are still changes."""
    volume = _iterate(dtype)
    width = WIDTH[BITS[dtype]]
    sign = 1 << (width - 1)
    raw = [value & ((1 << width) - 1) for value in _bits(volume).reshape(-1)[:4].tolist()]
    zero, negative_zero, nan, payload = raw
    assert (zero, negative_zero) == (0, sign)
    target = {"+0 to -0": ((0, 0), sign), "-0 to +0": ((0, 1), 0),
              "NaN payload": ((0, 2), payload),
              "NaN sign": ((0, 2), nan | sign),
              "quiet to signalling NaN": ((0, 3), (payload & ~(1 << (MANTISSA[dtype] - 1))) | 1)}[change]
    index, pattern = target
    original = _bits(volume)[index].item() & ((1 << width) - 1)
    assert pattern != original
    if change.endswith("NaN"):
        assert torch.isnan(volume[index]) and torch.isnan(torch.tensor(_signed(pattern, width),
                                                                       dtype=BITS[dtype]).view(dtype))
    assert _rejects(volume, lambda value: _set_bits(value, index, pattern), 2)


@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("change", ["reshape", "unsqueeze", "flatten", "reinterpret dtype", "convert dtype"])
def test_guard_rejects_shape_or_dtype_replacement_with_identical_bytes(dtype, change):
    volume = _iterate(dtype)
    other = {torch.float16: torch.bfloat16, torch.bfloat16: torch.float16}

    def mutate(value):
        if change == "reshape":
            value.data = value.reshape(4, 3)
        elif change == "unsqueeze":
            value.data = value.unsqueeze(0)
        elif change == "flatten":
            value.data = value.reshape(-1)
        elif change == "reinterpret dtype":
            # float16 <-> bfloat16 keep every stored byte; float32/64 reinterpret as integers.
            value.data = value.view(other.get(dtype, BITS[dtype]))
        else:
            value.data = value.to(torch.float64 if dtype != torch.float64 else torch.float32)

    assert _rejects(volume, mutate, 5)


def test_guard_rejects_shape_change_of_an_empty_iterate():
    volume = torch.empty((0, 3))
    assert not _rejects(volume, lambda value: None, 2)
    assert _rejects(volume, lambda value: setattr(value, "data", value.reshape(3, 0)), 2)


# ---------------------------------------------------------------------------
# No false positives and no side effects (B2)


OBSERVERS = {
    "nothing": lambda value: None,
    "reads": lambda value: (value.float().sum(), value.clone(), value.view(BITS[value.dtype]).numpy().copy(),
                            value.reshape(-1)[:2].tolist()),
    "rewrites identical bytes": lambda value: value.copy_(value.clone()),
    "returns False": lambda value: False,
}


@pytest.mark.parametrize("observer", sorted(OBSERVERS))
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("block", (2, 5, DEFAULT_BLOCK))
def test_unmodified_iterate_never_rejects_and_guard_leaves_bits_and_storage_untouched(dtype, layout, observer, block):
    volume = _iterate(dtype, layout)
    before, pointer, strides = _bits(volume), volume.data_ptr(), volume.stride()
    stopped = _guard(volume, lambda iteration, residual, value: OBSERVERS[observer](value), block)
    assert stopped is (observer == "returns False")
    assert torch.equal(_bits(volume), before), "the guard changed iterate bits"
    assert (volume.data_ptr(), volume.stride(), volume.dtype) == (pointer, strides, dtype)


@pytest.mark.parametrize("layout", ("transposed", "strided"))
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
def test_guard_handles_non_contiguous_iterates_with_single_element_blocks(dtype, layout):
    """block_elements=1 is a valid bound; a column-major or strided iterate still gets an exact check."""
    volume = _iterate(dtype, layout)
    before = _bits(volume)
    assert _guard(volume, lambda iteration, residual, value: None, 1) is False
    assert torch.equal(_bits(volume), before)
    assert _rejects(volume, lambda value: _flip(value, (2, 3), 1), 1)


def test_writes_outside_the_iterate_view_in_the_same_storage_are_not_iterate_changes():
    base = _iterate(torch.float32, "contiguous", shape=(5, 7))
    volume = base[1:4, 2:6]
    assert not volume.is_contiguous()

    def outside(iteration, residual, value):
        base[0].fill_(7.)
        base[:, 6].fill_(-3.)
        base[4, 0] = float("nan")

    assert _guard(volume, outside, 2) is False


# ---------------------------------------------------------------------------
# Public solvers on CPU (B2)


def _problem(dtype):
    operator = _MatrixOperator(dtype=dtype)
    operator.dtype = dtype
    data = operator.project(torch.tensor(TRUTH, dtype=dtype)).detach()
    operator.calls.clear()
    return operator, data


def _run(api, operator, data, directory, iterations=3, **options):
    options = dict(rtol=0., atol=0., **options)
    if api == "cgls":
        return diffct.cgls(operator, data, iterations, **options)
    output = torch.full(operator.volume_shape, torch.nan, dtype=data.dtype)
    return diffct.cgls_into(operator, data, output, iterations, workspace=directory / "workspace",
                            block_elements=3, **options)


SOLVER_DTYPES = [("cgls", dtype) for dtype in FLOATS] + [("cgls_into", torch.float32), ("cgls_into", torch.float64)]
SOLVER_MUTATIONS = ["position-%d" % flat for flat in range(6)] + [
    "swap", "sign", "lowest-bit", "data", "numpy", "dlpack", "reinterpret-dtype", "reshape"]


def _solver_mutation(value, mode):
    integer = BITS[value.dtype]
    bits = value.view(integer)
    if mode.startswith("position-"):
        bits.view(-1)[int(mode.split("-")[1])] ^= 1 << 3
    elif mode == "swap":
        first, last = bits[0, 0].clone(), bits[1, 2].clone()
        assert not torch.equal(first, last)
        bits[0, 0], bits[1, 2] = last, first
    elif mode == "sign":
        bits[1, 1] ^= _signed(1 << (WIDTH[integer] - 1), WIDTH[integer])
    elif mode == "lowest-bit":
        bits[1, 2] ^= 1
    elif mode == "data":
        value.data.view(integer)[0, 2] ^= 1
    elif mode == "numpy":
        array = value.view(integer).numpy()
        array[1, 0] ^= 1
    elif mode == "dlpack":
        torch.from_dlpack(value.detach().view(integer))[0, 0] ^= 1
    elif mode == "reinterpret-dtype":
        value.data = value.view({torch.float16: torch.bfloat16, torch.bfloat16: torch.float16}.get(value.dtype, integer))
    else:
        value.data = value.reshape(3, 2)


@pytest.mark.parametrize("mode", SOLVER_MUTATIONS)
@pytest.mark.parametrize("api,dtype", SOLVER_DTYPES, ids=lambda value: str(value))
@pytest.mark.parametrize("stop", [False, True], ids=["continue", "return-False"])
def test_solver_rejects_any_iterate_change_before_the_next_numerical_call(api, dtype, mode, stop, tmp_path):
    operator, data = _problem(dtype)
    boundary = []

    def callback(iteration, residual, volume):
        if iteration == 2:
            boundary.append(len(operator.calls))
            _solver_mutation(volume, mode)
        return False if stop and iteration == 2 else None

    with pytest.raises(ValueError, match=READ_ONLY):
        _run(api, operator, data, tmp_path, callback=callback)
    assert boundary == [5]
    assert len(operator.calls) == 5, "iterate change crossed the next projection/backprojection"


@pytest.mark.parametrize("observer", ["nothing", "reads", "rewrites identical bytes"])
@pytest.mark.parametrize("api,dtype", SOLVER_DTYPES, ids=lambda value: str(value))
def test_unmodified_solver_callback_is_bit_identical_to_the_callback_free_solve(api, dtype, observer, tmp_path):
    operator, data = _problem(dtype)
    (tmp_path / "baseline").mkdir()
    (tmp_path / "observed").mkdir()
    baseline = _run(api, operator, data, tmp_path / "baseline", iterations=4)
    seen = []

    def callback(iteration, residual, volume):
        seen.append(iteration)
        OBSERVERS[observer](volume)

    observed = _run(api, operator, data, tmp_path / "observed", iterations=4, callback=callback)
    assert seen == [1, 2, 3, 4] and observed.iterations == 4
    assert torch.equal(_bits(observed.volume), _bits(baseline.volume))
    assert observed.residual_history == baseline.residual_history


# ---------------------------------------------------------------------------
# CPU proxy for B1: values derived from the iterate that leave PyTorch


_EGRESS = frozenset(("numpy", "tolist", "item", "cpu", "__array__", "__bool__", "__float__", "__int__",
                     "__index__", "__complex__", "__dlpack__", "__reduce_ex__", "__cuda_array_interface__"))
# These return one host boolean computed from whole tensors (a device sync on CUDA).
_HOST_BOOLEANS = frozenset(("equal", "allclose", "is_nonzero"))


class _Resident(torch.Tensor):
    """A real CPU tensor that records every hand-off of iterate-derived values to the host.

    Every tensor computed from it keeps the subclass, so a NumPy view, a Python
    scalar, a list or an explicit ``.cpu()`` of the iterate or of anything
    derived from it is recorded with its byte count. On CUDA each of these
    is a device-to-host readback.
    """

    log = None

    @classmethod
    def __torch_function__(cls, function, types, args=(), kwargs=None):
        name = getattr(function, "__name__", "")
        if name in ("untyped_storage", "storage", "_typed_storage"):
            storage = super().__torch_function__(function, types, args, kwargs or {})
            if cls.log is not None:
                cls.log[-1].append((name, storage.nbytes()))
            return storage
        if cls.log is not None and args and isinstance(args[0], torch.Tensor):
            if name in _EGRESS:
                scalar = name in ("item", "__bool__", "__float__", "__int__", "__index__", "__complex__")
                cls.log[-1].append((name, args[0].element_size() * (1 if scalar else args[0].numel())))
            elif name in _HOST_BOOLEANS:
                cls.log[-1].append((name, 1))
        return super().__torch_function__(function, types, args, kwargs or {})


@pytest.fixture
def egress():
    _Resident.log = [[]]
    try:
        yield _Resident.log
    finally:
        _Resident.log = None


def _resident(value):
    return value.as_subclass(_Resident)


def _host_budget(events, limit, dtype):
    pair_bytes = 2 * torch.empty((), dtype=dtype).element_size()
    assert all(size <= pair_bytes for _, size in events), f"guard exported more than a value pair: {events}"
    assert len(events) <= limit, f"guard made {len(events)} host extractions; maximum is {limit}: {events}"


@pytest.mark.parametrize("access", ("untyped_storage", "storage", "_typed_storage"))
def test_resident_logs_storage_export_as_oversized(access, egress):
    volume = _resident(torch.zeros(8, dtype=torch.float64))

    def export(value):
        storage = getattr(value, access)()
        return bytes(storage if access == "untyped_storage" else storage.untyped())

    exported = export(volume)
    assert len(exported) == 64
    assert egress[-1] and all(size == 64 for _, size in egress[-1])
    with pytest.raises(AssertionError, match="more than a value pair"):
        _host_budget(egress[-1], 1, volume.dtype)


def test_host_budget_measures_exported_bytes_in_iterate_dtype(egress):
    pair = _resident(torch.zeros(2, dtype=torch.float64))
    pair.view(torch.uint8).numpy()
    assert egress[-1] == [("numpy", 16)]
    _host_budget(egress[-1], 1, pair.dtype)
    _host_budget([], 0, pair.dtype)
    for size in (17, 24):
        egress.append([])
        _resident(torch.zeros(size, dtype=torch.uint8)).numpy()
        assert egress[-1] == [("numpy", size)]
        with pytest.raises(AssertionError, match="more than a value pair"):
            _host_budget(egress[-1], 1, pair.dtype)
    with pytest.raises(AssertionError, match="host extractions"):
        _host_budget([("item", 8)] * 2, 1, pair.dtype)


@pytest.mark.parametrize("layout", ("contiguous", "transposed"))
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("block", (2, 5, DEFAULT_BLOCK))
def test_guard_hands_at_most_one_value_pair_of_the_iterate_to_the_host(dtype, layout, block, egress):
    volume = _resident(_iterate(dtype, layout))
    egress.append([])
    _guard(volume, lambda iteration, residual, value: None, block)
    _host_budget(egress[-1], 1, dtype)
    egress.append([])
    with pytest.raises(ValueError, match=READ_ONLY):
        _guard(volume, lambda iteration, residual, value: _flip(value, (2, 3), 1), block)
    _host_budget(egress[-1], 1, dtype)


def test_guard_without_callback_hands_nothing_to_the_host(egress):
    volume = _resident(_iterate(torch.float32))
    egress.append([])
    assert _guard(volume, None, 5) is False
    assert egress[-1] == []


def _boundaries(operator, name):
    original = getattr(operator, name)

    def marked(*args, **kwargs):
        _Resident.log.append([])
        return original(*args, **kwargs)

    setattr(operator, name, marked)


@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("observed", [False, True], ids=["no-callback", "callback"])
def test_cgls_iteration_hands_at_most_one_iterate_value_pair_to_the_host(dtype, observed, egress):
    operator, data = _problem(dtype)
    _boundaries(operator, "project")
    seen = []
    x0 = _resident(torch.zeros(operator.volume_shape, dtype=dtype))
    result = diffct.cgls(operator, data, 3, x0=x0, rtol=0., atol=0.,
                         callback=(lambda iteration, residual, volume: seen.append(type(volume)))
                         if observed else None)
    assert isinstance(result.volume, _Resident), "the iterate no longer derives from x0"
    assert seen == ([_Resident] * 3 if observed else [])
    # Buckets: request validation, x0's initial projection, then one per iteration.
    iterations = egress[2:]
    assert len(iterations) == result.iterations == 3
    for events in iterations:
        _host_budget(events, 1 if observed else 0, dtype)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=str)
@pytest.mark.parametrize("observed", [False, True], ids=["no-callback", "callback"])
def test_cgls_into_tensor_output_hands_at_most_one_value_pair_per_iteration(dtype, observed, egress, tmp_path):
    operator, data = _problem(dtype)
    _boundaries(operator, "project_into")
    output = _resident(torch.full(operator.volume_shape, torch.nan, dtype=dtype))
    seen = []
    result = diffct.cgls_into(operator, data, output, 3, workspace=tmp_path / "workspace", block_elements=3,
                              rtol=0., atol=0.,
                              callback=(lambda iteration, residual, volume: seen.append(volume is output))
                              if observed else None)
    assert result.volume is output and seen == ([True] * 3 if observed else [])
    iterations = egress[1:]
    assert len(iterations) == result.iterations == 3
    for events in iterations:
        _host_budget(events, 1 if observed else 0, dtype)


# ---------------------------------------------------------------------------
# Bounded guard memory (B3)


class _FreshTensors(TorchDispatchMode):
    """Record every newly allocated tensor of any dtype while ``active``; forward all math."""

    def __init__(self):
        super().__init__()
        self.active, self.sizes = True, []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = {item.untyped_storage()._cdata for item in tree_flatten((args, kwargs or {}))[0]
                  if isinstance(item, torch.Tensor)}
        result = function(*args, **(kwargs or {}))
        if self.active:
            for tensor in tree_flatten(result)[0]:
                if isinstance(tensor, torch.Tensor) and tensor.untyped_storage()._cdata not in inputs:
                    self.sizes.append((str(function), tuple(tensor.shape), str(tensor.dtype), tensor.numel()))
        return result


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("block", (2, 4, 16))
@pytest.mark.parametrize("mutated", [False, True], ids=["observer", "mutation"])
def test_guard_allocates_at_most_block_elements_per_tensor(dtype, layout, block, mutated):
    volume = _iterate(dtype, layout, shape=(8, 8))

    def callback(iteration, residual, value):
        trace.active = False
        if mutated:
            _flip(value, (7, 7), 1)
        trace.active = True

    with _FreshTensors() as trace:
        if mutated:
            with pytest.raises(ValueError, match=READ_ONLY):
                _guard(volume, callback, block)
        else:
            _guard(volume, callback, block)
    oversized = [entry for entry in trace.sizes if entry[-1] > block]
    assert not oversized, f"guard allocated more than block_elements={block} elements: {oversized}"


def test_cgls_into_tensor_output_guard_keeps_every_new_tensor_within_block_elements(tmp_path):
    operator, data = _problem(torch.float32)
    output = torch.empty(operator.volume_shape)
    with _FreshTensors() as trace:
        result = diffct.cgls_into(operator, data, output, 3, workspace=tmp_path / "workspace", block_elements=3,
                                  rtol=0., atol=0., callback=lambda *args: None)
    assert result.volume is output and result.iterations == 3
    oversized = [entry for entry in trace.sizes if entry[-1] > 3]
    assert not oversized, f"cgls_into allocated beyond block_elements: {oversized}"


class _Diagonal:
    """A real diagonal map whose volume is one default block plus 1024 voxels."""

    volume_shape = projection_shape = (1025, 1024)

    def __init__(self):
        self.weights = torch.linspace(.5, 1.5, 1025 * 1024).reshape(self.volume_shape)
        self.calls = []

    def project(self, volume):
        self.calls.append("project")
        return volume * self.weights

    def backproject(self, sinogram):
        self.calls.append("backproject")
        return sinogram * self.weights


def test_cgls_guard_does_not_clone_a_volume_larger_than_one_default_block():
    operator = _Diagonal()
    assert operator.weights.numel() > DEFAULT_BLOCK
    data = operator.weights * .75
    seen = []

    with _FreshTensors() as baseline_trace:
        baseline = diffct.cgls(operator, data, 2, rtol=0., atol=0.)

    def callback(iteration, residual, volume):
        trace.active = False
        try:
            seen.append((iteration, tuple(volume.shape), torch.isfinite(volume).all().item()))
        finally:
            trace.active = True

    with _FreshTensors() as trace:
        result = diffct.cgls(operator, data, 2, rtol=0., atol=0., callback=callback)
    assert seen == [(iteration, operator.volume_shape, True) for iteration in (1, 2)]
    assert result.iterations == baseline.iterations == 2
    assert torch.equal(_bits(result.volume), _bits(baseline.volume))
    oversized = Counter(entry[:3] for entry in trace.sizes if entry[-1] > DEFAULT_BLOCK)
    baseline_oversized = Counter(entry[:3] for entry in baseline_trace.sizes if entry[-1] > DEFAULT_BLOCK)
    extra = oversized - baseline_oversized
    assert not extra, f"callback added full-volume temporaries: {extra}"


# ---------------------------------------------------------------------------
# Unchanged checkpoint fingerprints (B4)


def _sha256(value):
    digest = hashlib.sha256(json.dumps((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.detach().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("block", (2, 5, DEFAULT_BLOCK))
def test_checkpoint_fingerprint_stays_sha256_of_shape_dtype_and_row_major_bytes(dtype, layout, block):
    volume = _iterate(dtype, layout)
    assert _solver_state.fingerprint(volume, block) == _sha256(volume)
    assert _solver_state.fingerprint(diffct.TensorStore(volume), block) == _sha256(volume)


@pytest.mark.parametrize("api", ["cgls", "cgls_into"])
def test_checkpoint_manifest_records_sha256_of_each_saved_state(api, tmp_path):
    operator, data, _ = _fixture()
    _run(api, operator, data, tmp_path, iterations=1, checkpoint_directory=tmp_path / "checkpoint",
         callback=lambda *args: None)
    saved = _checkpoint(tmp_path / "checkpoint", 1)
    manifest = json.loads((saved / "metadata.json").read_text())["states"]
    assert set(manifest) == {"x", "s", "p", "r", "q"}
    for name, specification in manifest.items():
        array = torch.from_numpy(np.load(saved / f"{name}.npy"))
        assert specification == dict(shape=list(array.shape), dtype=str(array.dtype), fingerprint=_sha256(array))
