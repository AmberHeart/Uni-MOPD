# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""One-window teacher prefetch; student forward/backward stays on the caller.

The next teacher chunk may run while the current student chunk accumulates
gradients. This module never invokes a model, collective or optimizer from its
I/O thread. The caller retains ONE native full-window Megatron schedule, so its
microbatch divisor and end-of-window gradient synchronization remain unchanged.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
import math
import time
from typing import Callable, Generic, List, Sequence, TypeVar

from .remote_teacher_protocol import require


Request = TypeVar('Request')
Response = TypeVar('Response')


class TeacherInputPrefetch:
    """One CPU encoder and one lookahead input, never a CUDA/collective worker.

    The consumer retains the current input while the next one is encoded.
    Only one callback is active, so a mutable template is not used concurrently
    by multiple encoders. The caller must also keep template use on its main
    thread outside this context and never reuse a failed encoder.
    """

    def __init__(self, items, prepare, *, timeout_seconds):
        require(math.isfinite(timeout_seconds) and timeout_seconds > 0,
                'CPU input preparation requires a positive finite timeout')
        self.items = iter(items)
        self.prepare = prepare
        self.timeout_seconds = timeout_seconds
        self.executor = None
        self.pending = None
        self.wait_seconds = 0.0
        self.closed = False
        self.failed = False

    def _submit(self):
        item = next(self.items, None)
        self.pending = None if item is None else self.executor.submit(self.prepare, item)

    def __enter__(self):
        require(self.executor is None and not self.closed, 'CPU input prefetch was already started')
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mopd-teacher-input')
        self._submit()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        require(self.executor is not None and not self.closed, 'CPU input prefetch is not active')
        if self.pending is None:
            raise StopIteration
        stamp = time.perf_counter()
        try:
            result = self.pending.result(timeout=self.timeout_seconds)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.wait_seconds += time.perf_counter() - stamp
        # Prepare k+1 before the owner thread transfers/forwards input k.
        self._submit()
        return result

    def close(self):
        if self.closed:
            return
        try:
            if (self.pending is not None and not self.pending.cancel()
                    and not (self.failed and not self.pending.done())):
                # Do not leave a template-mutating worker behind on an ordinary
                # error. On timeout the owner marks its engine unusable.
                self.pending.result(timeout=self.timeout_seconds)
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=False, cancel_futures=True)
            self.closed = True
            self.pending = None

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except Exception as cleanup:
            if error is None:
                raise
            error.add_note(f'CPU input prefetch cleanup: {type(cleanup).__name__}: {cleanup}')
        return False


@dataclass(frozen=True)
class TeacherChunk(Generic[Request]):
    index: int
    sample_ids: tuple
    request: Request
    max_hidden_bytes: int


@dataclass(frozen=True)
class PlannedSample(Generic[Request]):
    sample_id: str
    teacher_key: str
    input_tokens: int
    payload: Request


def balance_teacher_samples(samples, teacher_order, dp_size):
    """Redistribute whole occurrences without changing the global batch.

    Consecutive blocks contain one sample per DP rank, alternating direction.
    Thus every teacher-prefix count differs by at most one across ranks; the
    global chunk stream never has to revisit an already-drained teacher.
    Sorting lengths within each teacher reduces systematic long-rank skew.
    Payload includes the complete student input/R3 metadata and moves as a unit.
    """
    require(type(dp_size) is int and dp_size > 0 and samples and len(samples) % dp_size == 0,
            'Global teacher window must divide the student DP size')
    require(len(set(teacher_order)) == len(teacher_order), 'Duplicate teacher order')
    order = {key: index for index, key in enumerate(teacher_order)}
    require(len({sample.sample_id for sample in samples}) == len(samples), 'Duplicate sample occurrence')
    require(all(sample.teacher_key in order and type(sample.input_tokens) is int
                and sample.input_tokens > 0 for sample in samples), 'Invalid routed sample work estimate')
    sorted_samples = sorted(samples, key=lambda sample: (order[sample.teacher_key], -sample.input_tokens,
                                                       sample.sample_id))
    ranks = [[] for _ in range(dp_size)]
    for index, sample in enumerate(sorted_samples):
        block, offset = divmod(index, dp_size)
        rank = offset if block % 2 == 0 else dp_size - 1 - offset
        ranks[rank].append(sample)
    require(all(len(rank) == len(samples) // dp_size for rank in ranks), 'Unbalanced DP sample counts')
    return ranks


class StreamingTeacherWindow(Generic[Request, Response]):
    """Ordered one-chunk lookahead with a fixed full-window sample denominator.

    `score` must be a bounded-time remote I/O call. It may decode CPU tensors but
    must not call torch.distributed or launch CUDA work in this executor. Returned
    chunks are owned until `release` AFTER the student's chunk backward completes.
    """

    def __init__(
        self,
        chunks: Sequence[TeacherChunk],
        score: Callable[[TeacherChunk], Response],
        *,
        expected_samples: int,
        max_inflight_bytes: int,
        timeout_seconds: float,
        response_bytes: Callable[[Response], int],
    ):
        require(bool(chunks) and [chunk.index for chunk in chunks] == list(range(len(chunks))),
                'Teacher chunks must be contiguous and ordered')
        sample_ids = [sample for chunk in chunks for sample in chunk.sample_ids]
        require(len(sample_ids) == expected_samples and len(set(sample_ids)) == expected_samples,
                'Streaming window must contain every sample occurrence exactly once')
        require(all(chunk.sample_ids and type(chunk.max_hidden_bytes) is int
                    and chunk.max_hidden_bytes > 0 for chunk in chunks), 'Invalid teacher chunk budget')
        require(type(max_inflight_bytes) is int and max_inflight_bytes > 0
                and math.isfinite(timeout_seconds) and timeout_seconds > 0, 'Invalid streaming resource limits')
        for index, chunk in enumerate(chunks):
            next_bytes = chunks[index + 1].max_hidden_bytes if index + 1 < len(chunks) else 0
            require(chunk.max_hidden_bytes + next_bytes <= max_inflight_bytes,
                    'Current and prefetched teacher chunks exceed the hidden-memory limit')
        self.chunks = list(chunks)
        self.expected_samples = expected_samples
        self.max_inflight_bytes = max_inflight_bytes
        self.timeout_seconds = timeout_seconds
        self.score = score
        self.response_bytes = response_bytes
        self.executor = None
        self.pending = None
        self.current_index = None
        self.current_bytes = 0
        self.next_index = 0
        self.released_samples = 0
        self.closed = False
        self.failed = False
        self.wait_seconds = 0.0
        self.score_seconds: List[float] = []
        self.peak_reserved_bytes = 0

    def _score(self, chunk):
        start = time.perf_counter()
        response = self.score(chunk)
        used = self.response_bytes(response)
        require(type(used) is int and 0 <= used <= chunk.max_hidden_bytes,
                'Teacher returned more hidden bytes than the admitted chunk budget')
        return response, used, time.perf_counter() - start

    def _submit(self, index):
        chunk = self.chunks[index]
        self.peak_reserved_bytes = max(
            self.peak_reserved_bytes, self.current_bytes + chunk.max_hidden_bytes)
        self.pending = self.executor.submit(self._score, chunk)

    def begin(self):
        require(self.executor is None and not self.closed, 'Teacher window was already started')
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mopd-teacher-io')
        self._submit(0)
        return self

    def acquire(self):
        require(self.executor is not None and not self.closed and not self.failed, 'Teacher window is not active')
        require(self.current_index is None, 'Release the current student chunk before advancing')
        require(self.next_index < len(self.chunks), 'Teacher window has no remaining chunk')
        start = time.perf_counter()
        try:
            response, used, score_seconds = self.pending.result(timeout=self.timeout_seconds)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.wait_seconds += time.perf_counter() - start
        self.pending = None
        self.current_index = self.next_index
        self.current_bytes = used
        self.score_seconds.append(score_seconds)
        self.next_index += 1
        # Launch k+1 BEFORE returning k for the student's forward/backward.
        if self.next_index < len(self.chunks):
            self._submit(self.next_index)
        return self.chunks[self.current_index], response

    def release(self, chunk_index):
        require(not self.closed and not self.failed and self.current_index == chunk_index,
                'Out-of-order or duplicate teacher chunk release')
        self.released_samples += len(self.chunks[chunk_index].sample_ids)
        self.current_index = None
        self.current_bytes = 0

    def finish(self):
        """Called before the native optimizer step, not after a partial update."""
        require(not self.failed and not self.closed and self.executor is not None,
                'Cannot finalize an inactive/failed teacher window')
        require(self.next_index == len(self.chunks) and self.pending is None
                and self.current_index is None and self.released_samples == self.expected_samples,
                'Cannot update weights before the complete streaming window is consumed')
        self.executor.shutdown(wait=True)
        self.closed = True
        return {
            'samples': self.released_samples, 'chunks': len(self.chunks),
            'teacher_io_wait_s': self.wait_seconds,
            'teacher_score_s': sum(self.score_seconds),
            'hidden_reserved_bytes_max': self.peak_reserved_bytes,
        }

    def abort(self):
        """Settle the one in-flight request before the caller aborts remote state."""
        if self.closed:
            return
        self.failed = True
        try:
            if self.pending is not None and not self.pending.cancel():
                try:
                    self.pending.result(timeout=self.timeout_seconds)
                except FutureTimeout as error:
                    # Do not pretend remote GPU work has stopped or start a new
                    # window; the owner must tear down the failed runtime.
                    raise RuntimeError('Teacher I/O did not settle; remote window cannot be safely reused') from error
                except Exception:
                    pass  # the original request failure is propagated by acquire
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=False, cancel_futures=True)
            self.closed = True
            self.pending = None
            self.current_index = None
            self.current_bytes = 0
