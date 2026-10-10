"""Host-memory ceilings for the bounded transfer pipeline."""

from itertools import product
import math

import torch

# These are memory policy ceilings, independent of a card's device capacity.
# They do not claim an optimal transfer size on every PCIe/storage system.
_HOST_SLOT_BYTES = 8 * 1024 ** 2
_HOST_SCRATCH_BYTES = 8 * 1024 ** 2
_GEOMETRY_WORK_BYTES = 8 * 1024 ** 2


def _fragments(shape, elements):
    """Rectangular fragments, including strided caller data, without flatten copies."""
    parts, remaining = [1] * len(shape), max(1, elements)
    for axis in reversed(range(len(shape))):
        parts[axis] = min(shape[axis], remaining)
        remaining = max(1, remaining // max(1, parts[axis]))
    if any(size == 0 for size in shape):
        return
    for starts in product(*(range(0, size, part) for size, part in zip(shape, parts))):
        yield tuple(slice(start, min(start + part, size))
                    for start, part, size in zip(starts, parts, shape))


def _selected(outer, inner):
    return tuple(slice(first.start + second.start, first.start + second.stop)
                 for first, second in zip(outer, inner))


def _copy(target, source, stats):
    target.copy_(source, non_blocking=True)
    if target.device != source.device:
        direction = ("h2d" if source.device.type == "cpu" else
                     "d2h" if target.device.type == "cpu" else "p2p")
        stats[direction + "_bytes"] += target.numel() * target.element_size()


class _HostSlot:
    def __init__(self, elements):
        self.tensor = torch.empty(elements, dtype=torch.float32, device="cpu", pin_memory=True)
        self.event = None
        self.drain = None

    def finish(self):
        if self.event is not None:
            self.event.synchronize()
            self.event = None
        drain, self.drain = self.drain, None
        if drain is not None:
            drain()


class _DevicePipeline:
    """Two reusable host slots; events own all asynchronous transfer lifetimes."""
    def __init__(self, device, elements, stats, slots=2):
        self.device, self.stats = device, stats
        self.upload = torch.cuda.Stream(device=device)
        self.compute = torch.cuda.Stream(device=device)
        self.download = torch.cuda.Stream(device=device)
        self.caller = torch.cuda.current_stream(device)
        for stream in (self.upload, self.compute, self.download):
            stream.wait_stream(self.caller)
        self.capacity = max(1, min(elements, _HOST_SLOT_BYTES // 4))
        self.host = [_HostSlot(self.capacity) for _ in range(slots)]
        self.cursor = 0
        self.pending = []
        self.output_streams = {}
        self.outputs = []

    def reclaim_output(self, keep=None):
        """Keep one previous drain in flight when the two-epoch plan permits it."""
        self.outputs[:] = [event for event in self.outputs if not event.query()]
        keep = len(self.host) - 1 if keep is None else keep
        while len(self.outputs) > keep:
            self.outputs.pop(0).synchronize()
        self.pending[:] = [entry for entry in self.pending if not entry[0].query()]

    def retain(self, event, source, target):
        self.pending[:] = [entry for entry in self.pending if not entry[0].query()]
        if len(self.pending) >= len(self.host):
            self.pending.pop(0)[0].synchronize()
        self.pending.append((event, source, target))

    def slot(self):
        slot = self.host[self.cursor % len(self.host)]
        self.cursor += 1
        slot.finish()
        return slot

    @staticmethod
    def event(stream):
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def upload_tensor(self, source, target):
        origin = torch.cuda.current_stream(source.device) if source.is_cuda else None
        with torch.cuda.device(self.device), torch.cuda.stream(self.upload):
            if source.is_cuda:
                if source.device != self.device:
                    self.upload.wait_stream(origin)
                _copy(target, source, self.stats)
                if source.device == self.device:
                    source.record_stream(self.upload)
                else:
                    self.retain(self.event(self.upload), source, target)
            else:
                limit = min(self.capacity, _HOST_SCRATCH_BYTES // source.element_size())
                for selected in _fragments(tuple(source.shape), limit):
                    value = source[selected]
                    slot = self.slot()
                    pinned = slot.tensor[:value.numel()].view(value.shape)
                    pinned.copy_(value)
                    _copy(target[selected], pinned, self.stats)
                    slot.event = self.event(self.upload)
            return self.event(self.upload)

    def upload_store(self, store, selected, target):
        source_device = torch.device(getattr(store, "device", "cpu"))
        itemsize = torch.empty((), dtype=store.dtype, device="cpu").element_size()
        limit = math.prod(target.shape) if source_device.type == "cuda" else min(
            self.capacity, _HOST_SCRATCH_BYTES // itemsize,
        )
        ready = None
        for fragment in _fragments(tuple(target.shape), limit):
            value = store.read(_selected(selected, fragment)).detach()
            ready = self.upload_tensor(value, target[fragment])
            del value
        return ready

    def drain_tensor(self, tensor, ready, store, selected, *, accumulate=False):
        """Queue fragment drains; CPU stores are called only after D2H completes."""
        self.reclaim_output()
        output_device = torch.device(getattr(store, "device", "cpu"))
        if output_device.type == "cuda":
            if output_device == self.device:
                stream, caller = self.download, self.caller
            else:
                if output_device not in self.output_streams:
                    caller = torch.cuda.current_stream(output_device)
                    stream = torch.cuda.Stream(device=output_device)
                    stream.wait_stream(caller)
                    self.output_streams[output_device] = stream, caller
                stream, caller = self.output_streams[output_device]
            with torch.cuda.device(output_device), torch.cuda.stream(stream):
                stream.wait_event(ready)
                value = tensor.to(device=output_device, non_blocking=True)
                if value.device != tensor.device:
                    self.stats["p2p_bytes"] += value.numel() * value.element_size()
                store.write(selected, value, accumulate=accumulate)
                done = self.event(stream)
                self.retain(done, tensor, value)
                value.record_stream(caller)
            caller.wait_event(done)
            self.outputs.append(done)
            return done
        last = ready
        for fragment in _fragments(tuple(tensor.shape), min(self.capacity, _HOST_SCRATCH_BYTES // 4)):
            slot = self.slot()
            source = tensor[fragment]
            pinned = slot.tensor[:source.numel()].view(source.shape)
            with torch.cuda.device(self.device), torch.cuda.stream(self.download):
                self.download.wait_event(ready)
                _copy(pinned, source, self.stats)
                source.record_stream(self.download)
                slot.event = self.event(self.download)
            output_slice = _selected(selected, fragment)
            slot.drain = lambda pinned=pinned, output_slice=output_slice: store.write(
                output_slice, pinned, accumulate=accumulate,
            )
            last = slot.event
            if getattr(store, "synchronous", False):
                slot.finish()
        self.outputs.append(last)
        return last

    def close(self):
        error = None
        try:
            for slot in self.host:
                try:
                    slot.finish()
                except BaseException as caught:
                    if error is None:
                        error = caught
        finally:
            # Also cover a backend exception before a normal completion marker.
            for stream in (self.upload, self.compute, self.download):
                self.event(stream).synchronize()
            for event, _, _ in self.pending:
                event.synchronize()
            for stream, caller in self.output_streams.values():
                self.event(stream).synchronize()
                caller.wait_stream(stream)
            self.caller.wait_stream(self.download)
            self.pending.clear()
            self.outputs.clear()
            self.output_streams.clear()
            self.host.clear()
        if error is not None:
            raise error


class _TransferPipeline:
    def __init__(self, devices, elements, stats, slots=2):
        self.devices, self.elements, self.stats, self.slots = devices, elements, stats, slots
        self.pipes = {}

    def __enter__(self):
        return self

    def for_device(self, device):
        if device not in self.pipes:
            pipe = _DevicePipeline(device, self.elements, self.stats, self.slots)
            self.pipes[device] = pipe
            live = sum(len(value.host) * value.capacity * 4 for value in self.pipes.values())
            self.stats["pinned_bytes_peak"] = max(self.stats["pinned_bytes_peak"], live)
        return self.pipes[device]

    def __exit__(self, kind, error, traceback):
        close_error = None
        for pipe in self.pipes.values():
            try:
                pipe.close()
            except BaseException as caught:
                if close_error is None:
                    close_error = caught
        self.pipes.clear()
        if error is None and close_error is not None:
            raise close_error
