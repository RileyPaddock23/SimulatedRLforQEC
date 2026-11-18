"""
Real-Time (Streaming) QEC Environment
=====================================
`QECCodeEnv` evaluates a policy on many independent shots of a short memory experiment.
`StreamingQECEnv` instead runs the whole experiment as ONE shot: a single long memory
experiment whose QEC cycles are appended as time goes on, the way a controller sees
syndrome data stream out of real hardware.

Simulation
----------
The env's memory circuit is split into its first round (the prefix, with the data
resets), one QEC cycle (the body of its REPEAT block), and the final data readout (the
suffix). The single shot is simulated by a `stim.FlipSimulator` that keeps its Pauli
frame and measurement record between calls: every few cycles a fresh noisy copy of the
cycle circuit is built and appended, so the error rates (and the control parameters) can
change from one block of cycles to the next while the quantum state carries over. Every
`drift_update_cycles` cycles the drift is re-evaluated at the current time, so the error
rates are piecewise constant on that grid. The simulator is periodically rebased onto a
fresh copy that only inherits the Pauli frame and the last few measurement records
(whatever the cycle's DETECTORs look back to), so memory doesn't grow with the length of
the shot.

Time: cycle c happens at t = c / cycles_per_epoch reference epochs. Physical drift is
many orders of magnitude slower than a ~1 us QEC cycle, so cycles_per_epoch is a time
compression knob, not a hardware constant.

Observations
------------
`run_cycles(params, n)` returns the detection events of those n cycles, (n x streams x
D) for the D detectors of one cycle. The env's `M` is the factor graph over those D
detectors in steady state (a cycle's detectors see errors from that cycle and the one
before), so an agent built from `num_params`, `num_detectors` and `M` works unchanged
with per-detector EDRs averaged over a sampling window. Parameters whose sources never
appear in the steady-state cycle (e.g. data-qubit readout, which only happens in the
suffix) have empty columns in M; `active_params` marks the rest.

Decoding
--------
Decoding is real time too: `SlidingWindowDecoder` receives each cycle's detection
events as they are produced. Once it holds `window` cycles it decodes them, commits the
corrections of the oldest `commit` cycles (accumulating their effect on the logical
observables), carries the defects that committed edges leave on later cycles forward,
and slides on. It exposes hooks for coupling it to the controller, e.g. `set_priors` to
re-weight the matching graph from the controller's current error-rate estimate, and its
running `stats`.

Logical errors in a single shot
-------------------------------
A single memory experiment only reads its logical observable once, at the end
(`finish`). `probe` reads it without disturbing the experiment: it forks the simulator
and the decoder, runs a (noiseless by default) final readout on the fork, finalizes the
forked decoder, and reports whether that readout would have been a logical failure.
With `probe_interval` set, the env probes every that many cycles and counts how often
the answer toggles: `logical_failures` per stream. The interval trades two biases:
probing every cycle over-counts (a readout right after an error can flicker before the
decoder has seen enough cycles to settle it), while long intervals under-count (two
failures between probes cancel). About one decoder window (~2d cycles) works: on d=3 at
p=0.004 a 5-cycle interval gives 0.0084 failures per cycle against 0.0087 from batch
decoding, while 1 cycle gives 0.0105 and 25 cycles 0.0067.

`num_streams > 1` runs independent copies of the shot side by side in one simulator
(same device, drift, and control schedule; independent noise), e.g. for logical error
statistics. Detection events are returned for every stream, so a controller that should
only see one device should use stream 0.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import stim
import pymatching

from code_envs.qec_env import QECCodeEnv

Params = Union[np.ndarray, Callable[[float], np.ndarray]]  # a parameter vector, or t (epochs) -> parameter vector

_MEAS_BASIS = {"M": "Z", "MZ": "Z", "MR": "Z", "MRZ": "Z", "MX": "X", "MRX": "X", "MY": "Y", "MRY": "Y"}


# ---------------------------------------------------------------------- circuit helpers

def split_memory_circuit(circuit: stim.Circuit) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
    """
    Splits a memory circuit at its top-level REPEAT block into (prefix, body, suffix): the
    first round, one QEC cycle, and the final readout. Circuits from
    `stim.Circuit.generated(..., rounds=r)` have one for r >= 3.
    """
    items = list(circuit.without_noise())
    reps = [i for i, op in enumerate(items) if isinstance(op, stim.CircuitRepeatBlock)]
    if len(reps) != 1:
        raise ValueError(f"expected exactly one top-level REPEAT block (the QEC cycle), found {len(reps)}; "
                         "build the env with rounds >= 3, or pass segments=(prefix, body, suffix)")
    prefix, suffix = stim.Circuit(), stim.Circuit()
    for op in items[:reps[0]]:
        prefix.append(op)
    for op in items[reps[0] + 1:]:
        suffix.append(op)
    return prefix, items[reps[0]].body_copy(), suffix


def measurement_lookback(segment: stim.Circuit) -> int:
    """How many measurement records from before the segment its rec[-k] targets reach back to."""
    count, need = 0, 0
    for op in segment.flattened():
        for t in op.targets_copy():
            if t.is_measurement_record_target:
                need = max(need, -t.value - count)
        count += stim.Circuit(str(op)).num_measurements if _MEAS_BASIS.get(op.name) is None else len(op.targets_copy())
    return need


def logical_operators(suffix: stim.Circuit, num_qubits: int, num_observables: int) -> List[stim.PauliString]:
    """
    The Pauli operator each OBSERVABLE_INCLUDE of the final readout measures, from the
    basis and qubit of the measurements it includes (single-qubit measurements in the
    suffix only).
    """
    meas: List[Tuple[int, str]] = []
    ops = [stim.PauliString(num_qubits) for _ in range(num_observables)]
    for op in suffix.flattened():
        if op.name in _MEAS_BASIS:
            meas += [(t.value, _MEAS_BASIS[op.name]) for t in op.targets_copy()]
        elif op.name == "OBSERVABLE_INCLUDE":
            k = int(op.gate_args_copy()[0])
            for t in op.targets_copy():
                idx = len(meas) + t.value if t.is_measurement_record_target else -1
                if idx < 0:
                    raise NotImplementedError("observables must only include single-qubit measurements of the final readout")
                q, basis = meas[idx]
                single = stim.PauliString(num_qubits)
                single[q] = basis
                ops[k] *= single
        elif stim.Circuit(str(op)).num_measurements:
            raise NotImplementedError(f"unsupported measurement '{op.name}' in the final readout")
    return ops


def _mpp_readout(ops: List[stim.PauliString]) -> str:
    """Noiseless Pauli-product measurements of the logical operators, as the observables of a window circuit."""
    lines = []
    for k, ps in enumerate(ops):
        factors = [f"{'_XYZ'[p]}{q}" for q, p in enumerate(ps) if p]
        if factors:
            lines.append(f"MPP {'*'.join(factors)}\nOBSERVABLE_INCLUDE({k}) rec[-1]\n")
    return "".join(lines)


# ---------------------------------------------------------------------- real-time decoder

@dataclass
class _WindowModel:
    matching: pymatching.Matching
    chunk_of: np.ndarray  # window chunk of each local detector
    chunk_start: np.ndarray  # first local detector of each window chunk
    edge_obs: Dict[Tuple[int, int], np.ndarray]  # (d1, d2) or (d, -1) -> observables the edge flips


class SlidingWindowDecoder:
    """
    Real-time sliding-window minimum-weight perfect matching over a stream of QEC cycles.

    Chunks (one per cycle; the first is the prefix round, the last the final readout)
    arrive with `push`. When `window` chunks are pending, the window is matched and every
    matched edge whose earliest detector lies in the oldest `commit` chunks is committed:
    its logical effect is added to the running correction, and its endpoints in later
    chunks are toggled, so those defects are carried into the next window. The committed
    chunks are then dropped. Windows after the first have a smooth lower edge: error
    mechanisms touching already-committed detectors are left out, since their effect on
    the window was carried forward when they were committed.

    Window matching graphs come from `circuit_for(n_body, final)`, a noisy circuit of the
    prefix, n_body cycles, and either the final readout (final=True) or noiseless
    measurements of the logical operators, so every edge knows which observables it flips.
    `set_priors` swaps that function, e.g. to re-weight the graph from the controller's
    current error-rate estimate.
    """

    def __init__(self, circuit_for: Callable[[int, bool], stim.Circuit], chunk_sizes: Tuple[int, int, int],
                 num_observables: int, num_streams: int = 1, window: int = 10, commit: int = 5):
        if not 1 <= commit <= window:
            raise ValueError(f"need 1 <= commit <= window, got commit={commit}, window={window}")
        self.prefix_size, self.cycle_size, self.readout_size = chunk_sizes
        self.num_observables = num_observables
        self.num_streams = num_streams
        self.window = window
        self.commit = commit
        self.set_priors(circuit_for)
        self.reset()

    def set_priors(self, circuit_for: Callable[[int, bool], stim.Circuit]):
        """Uses new window circuits (e.g. new error-rate priors) from the next window on."""
        self._circuit_for = circuit_for
        self._models: Dict[Tuple[int, bool, bool], _WindowModel] = {}

    def reset(self):
        self.pending: List[np.ndarray] = []  # (streams x chunk detectors) per uncommitted chunk, defects carried in
        self.correction = np.zeros((self.num_streams, self.num_observables), dtype=bool)
        self.committed_chunks = 0  # chunks (prefix included) whose corrections are final
        self.stats = {"windows": 0, "decode_seconds": 0.0, "defects": 0, "chunks": 0, "committed_edges": 0}

    # -- matching graphs

    def _model(self, n_body: int, final: bool, drop_prefix: bool) -> _WindowModel:
        key = (n_body, final, drop_prefix)
        if key not in self._models:
            dem = self._circuit_for(n_body, final).detector_error_model(decompose_errors=True)
            offset = self.prefix_size if drop_prefix else 0
            sizes = ([] if drop_prefix else [self.prefix_size]) + [self.cycle_size] * n_body + ([self.readout_size] if final else [])
            num_local = sum(sizes)
            lines = []
            for inst in dem.flattened():
                if inst.type != "error":
                    continue
                targets = inst.targets_copy()
                dets = [t.val for t in targets if t.is_relative_detector_id()]
                if not dets or min(dets) < offset:  # touches a committed detector, or no detector at all
                    continue
                text = " ".join("^" if t.is_separator() else
                                f"D{t.val - offset}" if t.is_relative_detector_id() else f"L{t.val}" for t in targets)
                lines.append(f"error({inst.args_copy()[0]}) {text}")
            lines += [f"detector D{d}" for d in range(num_local)]
            lines += [f"logical_observable L{k}" for k in range(self.num_observables)]
            matching = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel("\n".join(lines)))

            edge_obs = {}
            for u, v, data in matching.edges():
                obs = np.zeros(self.num_observables, dtype=bool)
                for f in data.get("fault_ids", ()):
                    obs[f] = True
                edge_obs[(u, -1) if v is None else (min(u, v), max(u, v))] = obs
            starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
            self._models[key] = _WindowModel(matching, np.repeat(np.arange(len(sizes)), sizes), starts, edge_obs)
        return self._models[key]

    # -- decoding

    def _decode(self, pending: List[np.ndarray], correction: np.ndarray, first: bool, readout: Optional[np.ndarray],
                commit: int) -> int:
        """Matches pending (+ readout) chunks, commits edges starting in the first `commit` chunks in place."""
        chunks = pending + ([readout] if readout is not None else [])
        model = self._model(len(pending) - int(first), readout is not None, not first)
        t0 = time.perf_counter()
        edges_committed = 0
        for s in range(self.num_streams):
            syndrome = np.concatenate([c[s] for c in chunks]).astype(np.uint8)
            if not syndrome.any():
                continue
            for a, b in model.matching.decode_to_edges_array(syndrome):
                lo = model.chunk_of[a] if b < 0 else min(model.chunk_of[a], model.chunk_of[b])
                if lo >= commit:
                    continue
                edges_committed += 1
                correction[s] ^= model.edge_obs[(a, -1) if b < 0 else (min(a, b), max(a, b))]
                for d in (a, b):
                    if d >= 0 and model.chunk_of[d] >= commit:
                        c = model.chunk_of[d]
                        chunks[c][s, d - model.chunk_start[c]] ^= True
        self.stats["decode_seconds"] += time.perf_counter() - t0
        self.stats["windows"] += 1
        return edges_committed

    def push(self, events: np.ndarray) -> int:
        """Adds one chunk's detection events (streams x detectors); returns how many chunks got committed."""
        self.pending.append(np.array(events, dtype=bool, copy=True))
        self.stats["chunks"] += 1
        self.stats["defects"] += int(events.sum())
        committed = 0
        while len(self.pending) >= self.window:
            self.stats["committed_edges"] += self._decode(self.pending, self.correction,
                                                          self.committed_chunks == 0, None, self.commit)
            del self.pending[:self.commit]
            self.committed_chunks += self.commit
            committed += self.commit
        return committed

    def finalize(self, readout_events: np.ndarray, dry_run: bool = False) -> np.ndarray:
        """
        Decodes everything still pending together with the final readout's detection
        events; returns the predicted observable flips (streams x observables). With
        dry_run the decoder's state is left untouched (used to probe mid-experiment).
        """
        pending = [c.copy() for c in self.pending] if dry_run else self.pending
        correction = self.correction.copy() if dry_run else self.correction
        readout = np.array(readout_events, dtype=bool, copy=True)
        self._decode(pending, correction, self.committed_chunks == 0, readout, len(pending) + 1)
        if not dry_run:
            self.committed_chunks += len(pending) + 1
            self.pending = []
        return correction


# ---------------------------------------------------------------------- environment

@dataclass
class StreamBlock:
    """What `run_cycles` produced."""
    detection_events: np.ndarray  # (cycles x streams x detectors per cycle), bool
    first_cycle: int  # index of the first of these cycles (the prefix round is cycle 0)
    t_start: float  # epochs
    t_end: float
    mean_error_rate: float  # mean source error rate over the block, for diagnostics

    @property
    def edr(self) -> np.ndarray:
        """Per-detector detection rate over the block and all streams, shape (detectors,)."""
        return self.detection_events.mean(axis=(0, 1))


class StreamingQECEnv:
    """
    A `QECCodeEnv` run as one real-time memory experiment (see module docstring). The
    drift, sensitivities, noise placement and parameter layout are the wrapped env's, so
    batch and streaming runs on the same env share one drift realization. Attributes not
    defined here (get_drift_opt, compute_physical_error_rates, param_sensitivities,
    gate_param_slice, ...) are the wrapped env's.

    Typical loop:
        stream = StreamingQECEnv(env); stream.reset()
        block = stream.run_cycles(params, 200)      # sample
        stream.run_cycles(params, 20)               # wait while the update is computed
        failed = stream.probe()                     # would a readout now fail?
        failed = stream.finish(params)              # the real final readout
    """

    def __init__(
        self,
        env: QECCodeEnv,
        cycles_per_epoch: float = 5000.0,
        drift_update_cycles: int = 2,  # error rates are re-evaluated from the drift every this many cycles
        window: Optional[int] = None,  # decoder window in cycles (default 2 * distance, else 10)
        commit: Optional[int] = None,  # cycles committed per window (default window // 2)
        num_streams: int = 1,
        seed: Optional[int] = None,
        rebase_cycles: int = 256,  # move the shot to a fresh simulator every this many cycles
        segments: Optional[Tuple[stim.Circuit, stim.Circuit, stim.Circuit]] = None,  # (prefix, body, suffix)
        probe_interval: Optional[int] = None,  # count logical failures by probing every this many cycles (None: off)
        decode: bool = True,  # False: skip the real-time decoder (detection events only; no probes or logical readout)
    ):
        self.env = env
        self.cycles_per_epoch = float(cycles_per_epoch)
        self.drift_update_cycles = max(1, int(drift_update_cycles))
        self.decode = bool(decode)
        if probe_interval and not self.decode:
            raise ValueError("probing needs the decoder (decode=True)")
        self.probe_interval = int(probe_interval) if probe_interval else 0
        self.num_streams = int(num_streams)
        self.rebase_cycles = max(1, int(rebase_cycles))
        self._seed_rng = np.random.default_rng(env.seed + 7919 if seed is None else seed)
        backend = env.backend
        if not backend.has_observables:
            raise ValueError("circuit has no OBSERVABLE_INCLUDEs, so there is nothing to decode")

        prefix, body, suffix = segments if segments is not None else split_memory_circuit(backend.base_circuit)
        self._noiseless = {"prefix": prefix, "body": body, "suffix": suffix}
        self._templates = {name: backend.segment_template(seg) for name, seg in self._noiseless.items()}
        # Each template split around its rate slots, so a noisy copy formats each source's rate once
        self._formats = {name: ("".join(line for line, _ in lines).split("{:.8g}"), [s for _, s in lines if s >= 0])
                         for name, lines in self._templates.items()}
        self.num_qubits = backend.base_circuit.num_qubits
        self.num_observables = backend.num_observables
        self.lookback = max(measurement_lookback(body), measurement_lookback(suffix))

        d0 = prefix.num_detectors
        db = (prefix + body).num_detectors - d0
        ds = (prefix + body + suffix).num_detectors - d0 - db
        self.chunk_sizes = (d0, db, ds)
        self.num_detectors = db  # detectors per cycle
        self.logical_operators = logical_operators(suffix, self.num_qubits, self.num_observables)
        self._mpp = _mpp_readout(self.logical_operators)

        # Steady-state factor graph: the third round's detectors see errors from rounds two and three
        lines = self._templates["prefix"] + self._templates["body"] + self._templates["body"]
        footprint = backend._compute_footprint(backend.num_sources, lines, rows=slice(d0 + db, d0 + 2 * db),
                                               fill_unobserved=False)
        self.M = ((footprint @ (env.source_weights() > 0)) > 0).astype(np.float32)
        self.active_params = self.M.sum(axis=0) > 0

        window = int(window) if window is not None else 2 * int(getattr(env, "distance", 5))
        commit = int(commit) if commit is not None else max(1, window // 2)
        self.decoder = SlidingWindowDecoder(self._window_circuit_fn(self.nominal_error_rates()), self.chunk_sizes,
                                            self.num_observables, self.num_streams, window, commit)
        self.sim: Optional[stim.FlipSimulator] = None
        self.cycle = 0
        self.done = False

    def __getattr__(self, name):
        if name == "env":  # not set yet
            raise AttributeError(name)
        return getattr(self.env, name)

    # -- circuits and error rates

    def nominal_error_rates(self) -> np.ndarray:
        """Error rates of the static policy (params = 0) at t = 0, the default decoder priors."""
        return self.env.compute_physical_error_rates(np.zeros(self.env.num_params, dtype=np.float32), 0.0)

    def _fill(self, segment: str, eps: np.ndarray) -> str:
        pieces, ids = self._formats[segment]
        rates = [f"{x:.8g}" for x in eps.tolist()]
        out = [pieces[0]]
        for sid, piece in zip(ids, pieces[1:]):
            out.append(rates[sid])
            out.append(piece)
        return "".join(out)

    def _window_circuit_fn(self, eps: np.ndarray) -> Callable[[int, bool], stim.Circuit]:
        prefix, body, suffix = (self._fill(name, eps) for name in ("prefix", "body", "suffix"))
        return lambda n_body, final: stim.Circuit(prefix + body * n_body + (suffix if final else self._mpp))

    def set_decoder_priors(self, eps: np.ndarray):
        """Re-weights the real-time decoder's matching graphs with per-source error rates eps from the next window on."""
        self.decoder.set_priors(self._window_circuit_fn(np.asarray(eps, dtype=np.float32)))

    def time_of(self, cycle: int) -> float:
        return cycle / self.cycles_per_epoch

    @property
    def t(self) -> float:
        """Current time in epochs."""
        return self.time_of(self.cycle)

    def _eps(self, params: Params, t: float) -> np.ndarray:
        p = params(t) if callable(params) else params
        return self.env.compute_physical_error_rates(np.asarray(p, dtype=np.float32), t)

    # -- simulator

    def _new_sim(self) -> stim.FlipSimulator:
        # Detectors are deterministic, so stabilizer randomization isn't needed; it would also
        # put random Z flips on the data qubits of a rebased simulator
        return stim.FlipSimulator(batch_size=self.num_streams, num_qubits=self.num_qubits,
                                  disable_stabilizer_randomization=True,
                                  seed=int(self._seed_rng.integers(2 ** 63)))

    def _take_detectors(self) -> np.ndarray:
        """Detection events produced since the last call, (detectors x streams)."""
        new = self.sim.get_detector_flips()[self._det_read:]
        self._det_read = self.sim.num_detectors
        return new

    def _rebase(self):
        """Moves the shot to a fresh simulator: the Pauli frame and the measurement records still looked back to."""
        xs, zs, meas, _, _ = self.sim.to_numpy(output_xs=True, output_zs=True, output_measure_flips=True)
        sim = self._new_sim()
        sim.broadcast_pauli_errors(pauli="X", mask=xs)
        sim.broadcast_pauli_errors(pauli="Z", mask=zs)
        if self.lookback:
            sim.append_measurement_flips(np.ascontiguousarray(meas[-self.lookback:]))
        self.sim, self._det_read, self._since_rebase = sim, 0, 0

    # -- the experiment

    def reset(self, params: Optional[Params] = None) -> np.ndarray:
        """Starts a new shot with the first round (cycle 0) at t = 0; returns its detection events (streams x prefix detectors)."""
        params = np.zeros(self.env.num_params, dtype=np.float32) if params is None else params
        self.sim, self._det_read, self._since_rebase = self._new_sim(), 0, 0
        self.decoder.reset()
        self.sim.do(stim.Circuit(self._fill("prefix", self._eps(params, 0.0))))
        events = self._take_detectors().T
        if self.decode:
            self.decoder.push(events)
        self.cycle, self.done = 1, False
        self.failed_state = np.zeros((self.num_streams, self.num_observables), dtype=bool)  # at the last probe
        self.logical_failures = np.zeros(self.num_streams, dtype=int)  # probe-to-probe toggles so far
        return events

    def _track(self, failed: np.ndarray):
        self.logical_failures += (failed != self.failed_state).any(axis=1)
        self.failed_state = failed

    def run_cycles(self, params: Params, n: int) -> StreamBlock:
        """
        Runs n more QEC cycles with control parameters `params` (a vector, or a function of
        t for a schedule such as the drift oracle), streaming their detection events to
        the decoder. The drift is re-evaluated at the start of each `drift_update_cycles`
        block, and the logical readout is probed every `probe_interval` cycles.
        """
        if self.sim is None or self.done:
            raise RuntimeError("call reset() first" if self.sim is None else "the shot has been read out; call reset()")
        n = int(n)
        events = np.empty((n, self.num_streams, self.num_detectors), dtype=bool)
        first, eps_sum, i = self.cycle, 0.0, 0
        while i < n:
            k = min(n - i, self.drift_update_cycles - self.cycle % self.drift_update_cycles,
                    self.rebase_cycles - self._since_rebase,
                    self.probe_interval - self.cycle % self.probe_interval if self.probe_interval else n)
            eps = self._eps(params, self.t)
            body = self._fill("body", eps)
            self.sim.do(stim.Circuit(body if k == 1 else f"REPEAT {k} {{\n{body}}}\n"))
            block = self._take_detectors().reshape(k, self.num_detectors, self.num_streams).transpose(0, 2, 1)
            events[i:i + k] = block
            if self.decode:
                for chunk in block:
                    self.decoder.push(chunk)
            eps_sum += k * float(eps.mean())
            i += k
            self.cycle += k
            self._since_rebase += k
            if self._since_rebase >= self.rebase_cycles:
                self._rebase()
            if self.probe_interval and self.cycle % self.probe_interval == 0:
                self._track(self.probe())
        return StreamBlock(events, first, self.time_of(first), self.t, eps_sum / max(1, n))

    def _readout(self, sim: stim.FlipSimulator, circuit: stim.Circuit, dry_run: bool) -> np.ndarray:
        if not self.decode:
            raise RuntimeError("logical readout needs the decoder (decode=True)")
        det_before = sim.num_detectors if sim is not self.sim else self._det_read
        sim.do(circuit)
        events = sim.get_detector_flips()[det_before:].T
        actual = sim.get_observable_flips().T  # only the readout has OBSERVABLE_INCLUDEs
        predicted = self.decoder.finalize(events, dry_run=dry_run)
        return predicted != actual

    def probe(self, noisy: bool = False, params: Optional[Params] = None) -> np.ndarray:
        """
        Would the logical readout fail if the experiment were read out now? Runs the final
        readout on a fork of the shot (noiseless unless `noisy`, then at `params`' current
        error rates) and finalizes a fork of the decoder; the shot itself is not disturbed.
        Returns (streams x observables) bool.
        """
        if self.sim is None or self.done:
            raise RuntimeError("no shot in progress")
        circuit = (stim.Circuit(self._fill("suffix", self._eps(params, self.t))) if noisy
                   else self._noiseless["suffix"])
        return self._readout(self.sim.copy(), circuit, dry_run=True)

    def finish(self, params: Params) -> np.ndarray:
        """Reads the shot out (noisy final readout at `params`) and finishes decoding; returns logical failures (streams x observables)."""
        if self.sim is None or self.done:
            raise RuntimeError("no shot in progress")
        failed = self._readout(self.sim, stim.Circuit(self._fill("suffix", self._eps(params, self.t))), dry_run=False)
        self._track(failed)
        self.done = True
        return failed
