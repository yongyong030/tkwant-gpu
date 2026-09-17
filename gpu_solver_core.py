"""GPU-batched time evolution for `tkwant.manybody.WaveFunction`.

This module provides a drop-in replacement for the CPU-sequential
`WaveFunction.evolve` method: all one-body scattering states are stacked
into a single matrix and propagated together with a batched, GPU-resident
Dormand-Prince (dopri5) integrator. It does not depend on orbital count,
lattice symmetry, or dimensionality -- it only uses tkwant's public
per-state API (`state.kernel.rhs`, `state.psibar`, `state.psi_st`) and
`fsys.hamiltonian_submatrix`.

Recovering H_eff (`reconstruct_H`) has two paths, tried in order:

* fast path -- if the wavefunction was built with `kernel_type=kernels.Scipy`
  (a pure-Python tkwant kernel), the effective Hamiltonian is already a
  plain attribute of the object `kernel.rhs()` returns (`rhs_obj.H0`), so
  it is read directly, no probing required;
* fallback path -- tkwant's *default* kernel (`kernels.Simple`) is a
  compiled, opaque `cdef class` that does not expose H_eff. In that case
  H_eff is recovered by column-probing the kernel's own RHS function
  (feeding unit vectors, reading off `dpsi/dt`). This still works for any
  closed-source/compiled simulation kernel, at the cost of `n` RHS
  evaluations instead of one attribute read.

Two time-dependence regimes are supported, selected in `attach_fsys`:

* purely sinusoidal drives admit an *exact* closed-form decomposition of
  H(t) into three static matrices, so the effective Hamiltonian never
  needs to be reassembled during integration (see `attach_fsys`);
* general drives fall back to periodic numeric reassembly of H(t), either
  frozen over each sub-interval (`update_H`) or linearly interpolated
  between the sub-interval's endpoints (`update_H_linear`), which is
  usually the better trade-off between accuracy and reassembly cost.

Examples
--------
>>> from gpu_solver_core import BatchGPUSolver
>>> solver = BatchGPUSolver(model.psi)
>>> solver.attach_fsys(model.fsys, omega=omega)  # omega=None: numeric fallback
>>> model.psi.evolve = solver.evolve
>>> model.psi.evolve(t)
"""

import logging
import os
import threading
import time as _time_module
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse

__all__ = ['BatchGPUSolver']

logger = logging.getLogger(__name__)

# Dormand-Prince (dopri5) Butcher tableau.
_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], dtype=np.float64)
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_E = np.array([71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200,
               22 / 525, -1 / 40], dtype=np.float64)


class BatchGPUSolver:
    """GPU-batched integrator for a tkwant many-body wavefunction.

    All one-body states share the same effective Hamiltonian H_eff(t);
    they differ only in their static scattering state and energy. This
    class stacks their `psibar` correction vectors into one matrix
    ``Psi`` (n x N_states) and integrates them together with a single
    batched dopri5 step, replacing tkwant's N sequential CPU solves with
    one GPU sparse matrix-matrix multiply per substep.

    Parameters
    ----------
    manybody_wf : `tkwant.manybody.WaveFunction`
        The wavefunction whose local one-body states will be evolved.
        Only states already present on this MPI rank are handled; other
        ranks' states are unaffected.
    atol, rtol : float, optional
        Absolute and relative error tolerance for the adaptive dopri5
        step, matching `tkwant.onebody.solvers.Scipy`'s defaults
        (1e-9, 1e-9) so that CPU and GPU integration target the same
        accuracy.
    nsteps : int, optional
        Maximum number of adaptive steps per `evolve` call before giving
        up.

    Notes
    -----
    Call `attach_fsys` once after construction to enable the fast H(t)
    update path; without it, `evolve` reconstructs H_eff once (assuming
    it is time-independent) and integrates with that frozen operator.
    """

    def __init__(self, manybody_wf, atol=1e-9, rtol=1e-9, nsteps=int(1e9)):
        self.wf = manybody_wf
        self.atol = atol
        self.rtol = rtol
        self.nsteps = nsteps

        self._cp = None
        self._csp = None
        self._use_gpu = False

        self._states = list(manybody_wf.psi.local_data().values())
        if not self._states:
            raise RuntimeError('BatchGPUSolver: no local one-body states found')

        self.n = self._states[0].solver.size
        self.N = len(self._states)

        energies = np.array([s.energy for s in self._states], dtype=np.float64)
        self._delta_E = energies - energies[0]

        self._H_gpu = None
        self._H_cpu = None
        self._fsys = None
        self._src_gpu = None
        self._pert_extractor = None
        self._continuous_local_ready = False
        self.max_step = np.inf

        try:
            import cupy as cp
            import cupyx.scipy.sparse as csp
            self._cp = cp
            self._csp = csp
            self._use_gpu = True
            dev = cp.cuda.Device(0)
            logger.info('GPU ready: n=%d, N_states=%d, CUDA SMs=%d',
                        self.n, self.N, dev.attributes['MultiProcessorCount'])
        except Exception as error:
            logger.warning('cupy unavailable (%s), falling back to CPU', error)

    # -- H_eff reconstruction (column-probing) -------------------------------

    def _make_rhs(self):
        """Return a fresh RHS closure for state 0, in the correct energy convention."""
        state0 = self._states[0]
        if getattr(state0, '_inplace', False):
            return state0.kernel.rhs(state0.energy)
        return state0.kernel.rhs()

    def _try_fast_H_eff(self):
        """Read H_eff directly if the kernel exposes it as a plain attribute.

        `kernels.Scipy` (a pure-Python tkwant kernel, as opposed to the
        default compiled `kernels.Simple`) returns a `_CalcRHS` object from
        `.rhs()` whose `.H0` attribute *is* the assembled effective
        Hamiltonian `H0 - E*I` -- see `tkwant/onebody/kernels.pyx`, class
        `Scipy.rhs()` (builds it) and class `_CalcRHS.__init__` (stores it,
        as a plain public attribute since `_CalcRHS` is a regular Python
        class, not a `cdef class`). No probing needed in that case.

        Returns
        -------
        H_sparse : `scipy.sparse.csr_matrix` or `None`
            The effective Hamiltonian if the fast path is available,
            `None` if the kernel is opaque (e.g. the default
            `kernels.Simple`) and column-probing is required instead.
        """
        rhs_obj = self._make_rhs()
        H0_attr = getattr(rhs_obj, 'H0', None)
        if H0_attr is None:
            return None
        return H0_attr.tocsr().astype(complex)

    def reconstruct_H(self, t=0.0, n_workers=None, cache_path=None):
        """Recover the sparse effective Hamiltonian H_eff(t).

        Tries the fast path first (`_try_fast_H_eff`: a direct attribute
        read, available when the wavefunction uses `kernels.Scipy`). Falls
        back to column-probing when the kernel is opaque (tkwant's default,
        `kernels.Simple`, a compiled `cdef class` that does not expose
        H_eff): evaluating one state's right-hand-side function on each
        unit vector, where column k of H_eff equals ``1j * rhs(e_k, t)``.
        This fallback costs `n` RHS evaluations, parallelized over threads,
        and remains useful for any similarly closed-source/compiled
        simulation kernel beyond tkwant.

        Parameters
        ----------
        t : float, optional
            Time at which to reconstruct H_eff. Only meaningful for the
            column-probing fallback when `attach_fsys` has not been called
            (which handles the time-dependent part separately). Ignored by
            the fast path (the fast path reads the same t=0, W=0 H_eff that
            probing at t=0 would recover).
        n_workers : int, optional
            Thread count for the probing fallback. Defaults to
            ``min(32, os.cpu_count())``. Unused by the fast path.
        cache_path : str, optional
            Path to a ``.npz`` file. If it exists, H_eff is loaded from
            it instead of being recomputed; otherwise it is computed and
            then saved there. Falls back to ``self.H_cache_path`` if set
            and this argument is omitted. Only consulted by the
            column-probing fallback -- the fast path is already O(1), so
            caching it would add overhead rather than save it.
        """
        n = self.n

        H_fast = self._try_fast_H_eff()
        if H_fast is not None:
            logger.info('H_eff read directly from kernel.rhs().H0 (kernels.Scipy '
                        'fast path), nnz=%d -- no column-probing needed', H_fast.nnz)
            self._upload_H(H_fast)
            return

        logger.info('kernel does not expose H_eff (opaque/compiled kernel, e.g. '
                    'kernels.Simple) -- falling back to column-probing')

        if n_workers is None:
            n_workers = min(32, os.cpu_count() or 1)

        cache = cache_path or getattr(self, 'H_cache_path', None)
        if cache and os.path.isfile(cache):
            logger.info('Loading H_eff from cache: %s', cache)
            t0 = _time_module.perf_counter()
            H_sparse = scipy.sparse.load_npz(cache).astype(complex).tocsr()
            logger.info('Cache loaded in %.2fs (nnz=%d)',
                        _time_module.perf_counter() - t0, H_sparse.nnz)
            self._upload_H(H_sparse)
            return

        logger.info('Reconstructing H_eff (%dx%d) via %d probes at t=%.3f, workers=%d',
                    n, n, n, t, n_workers)
        t0 = _time_module.perf_counter()

        thresh = 1e-14
        thread_local = threading.local()

        def get_rhs():
            if not hasattr(thread_local, 'rhs_fn'):
                thread_local.rhs_fn = self._make_rhs()
                thread_local.psi_buf = np.zeros(n, dtype=complex)
                thread_local.dpsi_buf = np.zeros(n, dtype=complex)
            return thread_local.rhs_fn, thread_local.psi_buf, thread_local.dpsi_buf

        def probe(k):
            rhs_fn, psi, dpsi = get_rhs()
            psi[:] = 0.0
            psi[k] = 1.0
            dpsi[:] = 0.0
            rhs_fn(psi, dpsi, float(t))
            column = 1j * dpsi
            nonzero = np.nonzero(np.abs(column) > thresh)[0]
            return nonzero, column[nonzero].copy()

        row_list, col_list, val_list = [], [], []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for k, (nonzero, values) in enumerate(executor.map(probe, range(n))):
                row_list.append(nonzero)
                col_list.append(np.full(len(nonzero), k, dtype=np.intp))
                val_list.append(values)

        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
        vals = np.concatenate(val_list)
        H_sparse = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex)

        logger.info('Done in %.1fs (nnz=%d, density=%.4f)',
                    _time_module.perf_counter() - t0, H_sparse.nnz, H_sparse.nnz / n ** 2)
        self._upload_H(H_sparse)

        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            scipy.sparse.save_npz(cache, H_sparse)
            logger.info('H_eff cached to: %s', cache)

    def _upload_H(self, H_sparse):
        """Store H_eff on the CPU and, if available, upload it to the GPU."""
        self._H_cpu = H_sparse
        if self._use_gpu:
            cp = self._cp
            self._H_gpu = self._csp.csr_matrix(H_sparse)
            self._delta_E_gpu = cp.asarray(self._delta_E, dtype=cp.float64).reshape(1, self.N)
            logger.info('H_eff uploaded to GPU')

    # -- Time-dependent H(t) ---------------------------------------------------

    def attach_fsys(self, fsys, omega=None):
        r"""Enable fast H(t) updates and exact source-term injection.

        Each one-body state's correction obeys

        .. math::
            \frac{d \bar\psi}{dt} = -i H_\mathrm{eff}(t)\,\bar\psi
                                    - i W(t)\, \psi_\mathrm{st}

        Parameters
        ----------
        fsys : `kwant.system.FiniteSystem`
            The finalized, time-dependent system underlying `manybody_wf`.
        omega : float, optional
            Drive frequency. If given (and the perturbation is purely
            sinusoidal, i.e. ``H_TB(t) = H_TB(0) + H_A(cos(wt)-1) +
            H_B sin(wt)``), both H(t) and the source term are evaluated
            *exactly* at every integration substep from two pre-computed
            matrices -- no runtime reassembly at all. If omitted, `evolve`
            instead reassembles H(t) periodically via `update_H` or
            `update_H_linear`, controlled by `dt_reconstruct` and
            `_use_linear_update`.
        """
        self._fsys = fsys
        if self._H_gpu is None and self._H_cpu is None:
            self.reconstruct_H(0.0)

        H_TB0 = fsys.hamiltonian_submatrix(
            params={'time': 0.0}, sparse=True).astype(complex).tocsr()
        n = H_TB0.shape[0]
        n_total = self.n

        self._H_TB0 = H_TB0
        self._n_central = n
        self._pert_extractor = self._try_make_perturbation_extractor(fsys)
        self._H_eff0_cpu = self._H_cpu.copy()
        self._fast_update_ready = True
        if self._use_gpu:
            self._H_eff0_gpu_base = self._csp.csr_matrix(self._H_eff0_cpu)

        psi_st_columns = []
        for state in self._states:
            column = np.zeros(n_total, dtype=complex)
            if getattr(state, 'psi_st', None) is not None:
                column[:n] = state.psi_st
            psi_st_columns.append(column)
        self._psi_st_matrix = np.column_stack(psi_st_columns)
        self._psi_st_central = self._psi_st_matrix[:n, :]
        self._src_gpu = None
        self._exact_source_ready = False
        self.dt_reconstruct = getattr(self, 'dt_reconstruct', 1.0)

        if omega is not None and omega > 0:
            self._attach_exact_fourier(fsys, omega, H_TB0, n, n_total)
        else:
            self._omega = None
            logger.info('omega not provided: source term comes from update_H only')

        logger.info('Fast H(t) update + source term enabled (n_central=%d, n_total=%d)',
                    n, n_total)

    def _attach_exact_fourier(self, fsys, omega, H_TB0, n, n_total):
        """Precompute the exact cos/sin decomposition of H(t) and its source term."""
        self._omega = float(omega)
        t_half = np.pi / (2.0 * omega)
        t_pi = np.pi / omega
        logger.info('Pre-computing H_A, H_B at t=%.2f and t=%.2f', t_half, t_pi)
        H_half = fsys.hamiltonian_submatrix(
            params={'time': float(t_half)}, sparse=True).astype(complex).tocsr()
        H_pi = fsys.hamiltonian_submatrix(
            params={'time': float(t_pi)}, sparse=True).astype(complex).tocsr()
        delta_pi = H_pi - H_TB0
        delta_half = H_half - H_TB0
        H_A = (-delta_pi * 0.5).tocsr()
        H_B = (delta_half - delta_pi * 0.5).tocsr()
        self._H_A_cpu = H_A
        self._H_B_cpu = H_B

        S_A = np.zeros((n_total, self.N), dtype=complex)
        S_B = np.zeros((n_total, self.N), dtype=complex)
        S_A[:n, :] = -1j * H_A.dot(self._psi_st_central)
        S_B[:n, :] = -1j * H_B.dot(self._psi_st_central)
        if self._use_gpu:
            self._src_A_gpu = self._cp.asarray(S_A)
            self._src_B_gpu = self._cp.asarray(S_B)
        self._exact_source_ready = True
        logger.info('Exact source ready (H_A nnz=%d, H_B nnz=%d)', H_A.nnz, H_B.nnz)

        H_A_full = self._embed_central(H_A, n_total - n)
        H_B_full = self._embed_central(H_B, n_total - n)
        if self._use_gpu:
            self._H_eff0_gpu_base = self._csp.csr_matrix(self._H_eff0_cpu)
            self._H_A_full_gpu = self._csp.csr_matrix(H_A_full)
            self._H_B_full_gpu = self._csp.csr_matrix(H_B_full)
            self._exact_H_ready = True
            logger.info('Exact H(t) ready on GPU: no update_H needed')

    def _embed_central(self, matrix, n_buffer):
        """Embed a central-region operator into the full (central + absorbing-buffer) space."""
        if n_buffer <= 0:
            return matrix.tocsr()
        zero_buffer = scipy.sparse.csr_matrix((n_buffer, n_buffer), dtype=complex)
        return scipy.sparse.block_diag([matrix, zero_buffer]).tocsr()

    def _try_make_perturbation_extractor(self, fsys):
        """Build a `tkwant.onebody.kernels.PerturbationExtractor` for fast delta(t),
        but only adopt it if a calibration call shows it is actually faster.

        `fsys.hamiltonian_submatrix()` reassembles every site/hopping value
        function in the system on every call. `PerturbationExtractor` (built
        once here) tracks which elements are time-dependent and its
        `.evaluate(t)` recomputes only those, returning
        ``H_TB(t) - H_TB(time_start)`` directly -- the same quantity
        `update_H`/`update_H_linear` used to get by taking two full
        `hamiltonian_submatrix` snapshots and subtracting. This is a real win
        when the drive is spatially localized (confirmed 2.7x faster on a
        1D-chain test with a narrow driven window, `nnz` far below the
        system's total nonzero count), but a *loss* when the drive touches
        most of the lattice: on the production phonon-OAM model, where the
        drive modulates hoppings broadly rather than in a small window,
        `PerturbationExtractor`'s own per-call overhead made it ~14% *slower*
        than just calling `hamiltonian_submatrix` twice, despite tracking
        fewer nominally-"time-dependent" entries. There is no reliable static
        proxy for this (`nnz` alone does not predict it), so the two paths
        are timed once here and whichever is faster is kept. Falls back to
        `hamiltonian_submatrix` (returning `None`) if the import or
        construction fails for any reason.
        """
        try:
            from tkwant.onebody import kernels
            state0 = self._states[0]
            extractor = kernels.PerturbationExtractor(
                fsys, state0.time_name, state0.time_start, state0.params)
        except Exception as error:
            logger.info('PerturbationExtractor unavailable (%s), update_H/'
                        'update_H_linear will use full hamiltonian_submatrix', error)
            return None

        t_probe = state0.time_start + 1.0
        t0 = _time_module.perf_counter()
        extractor.evaluate(t_probe)
        t_extractor = _time_module.perf_counter() - t0

        t0 = _time_module.perf_counter()
        fsys.hamiltonian_submatrix(params={'time': t_probe}, sparse=True)
        t_full = _time_module.perf_counter() - t0

        logger.info('PerturbationExtractor calibration: nnz=%d of %d total '
                    'elements, extractor=%.4fs, hamiltonian_submatrix=%.4fs',
                    extractor.nnz, self._n_central ** 2, t_extractor, t_full)
        if t_extractor >= t_full:
            logger.info('PerturbationExtractor not faster here, keeping '
                        'hamiltonian_submatrix for update_H/update_H_linear')
            return None
        return extractor

    def _perturbation_delta(self, t):
        """Return H_TB(t) - H_TB(time_start) over the central region, fast path preferred."""
        if self._pert_extractor is not None:
            return self._pert_extractor.evaluate(float(t)).tocsr().astype(complex)
        H_TBt = self._fsys.hamiltonian_submatrix(
            params={'time': float(t)}, sparse=True).astype(complex).tocsr()
        return (H_TBt - self._H_TB0).tocsr()

    def update_H(self, t):
        """Reassemble H(t) and the source term, frozen over the next sub-interval.

        This is the 0th-order (piecewise-constant) numeric fallback: H(t)
        is sampled once at `t` and held constant until the next call,
        giving O(dt_reconstruct) local error. See `update_H_linear` for a
        1st-order alternative with the same per-interval cost structure
        but O(dt_reconstruct^2) error.

        Parameters
        ----------
        t : float
            Time at which to sample H_TB(t) for the upcoming sub-interval.
        """
        delta = self._perturbation_delta(t)
        n = self._n_central
        H_new = (self._H_eff0_cpu + self._embed_central(delta, self.n - n)).tocsr()

        self._H_cpu = H_new
        if self._use_gpu:
            self._H_gpu = self._csp.csr_matrix(H_new)
            self._H_lin_ready = False

        source_top = -1j * delta.dot(self._psi_st_central)
        source_full = np.zeros((self.n, self.N), dtype=complex)
        source_full[:n, :] = source_top
        if self._use_gpu:
            self._src_gpu = self._cp.asarray(source_full)

    def update_H_linear(self, t0, t1):
        """Reassemble H(t), linearly interpolated across the next sub-interval.

        Samples H_TB at both endpoints of ``[t0, t1]`` (one extra
        `hamiltonian_submatrix` call compared to `update_H`) and
        interpolates linearly in time within the GPU right-hand side --
        the same idea as the exact cos/sin decomposition in
        `attach_fsys`, but with a local linear model instead of a global
        sinusoid. Local error is O(dt_reconstruct^2) rather than
        `update_H`'s O(dt_reconstruct), so it typically lets
        `dt_reconstruct` be set much larger for the same accuracy target.

        Parameters
        ----------
        t0, t1 : float
            Start and end time of the upcoming sub-interval.
        """
        delta0 = self._perturbation_delta(t0)
        delta1 = self._perturbation_delta(t1)
        slope = (delta1 - delta0) / (t1 - t0)
        n = self._n_central
        n_buffer = self.n - n

        delta0_full = self._embed_central(delta0, n_buffer)
        slope_full = self._embed_central(slope, n_buffer)

        source0_top = -1j * delta0.dot(self._psi_st_central)
        source_slope_top = -1j * slope.dot(self._psi_st_central)
        source0_full = np.zeros((self.n, self.N), dtype=complex)
        source_slope_full = np.zeros((self.n, self.N), dtype=complex)
        source0_full[:n, :] = source0_top
        source_slope_full[:n, :] = source_slope_top

        if self._use_gpu:
            cp = self._cp
            self._H_lin_delta0_gpu = self._csp.csr_matrix(delta0_full)
            self._H_lin_slope_gpu = self._csp.csr_matrix(slope_full)
            self._src_lin_0_gpu = cp.asarray(source0_full)
            self._src_lin_slope_gpu = cp.asarray(source_slope_full)
            self._lin_t0 = float(t0)
            self._H_lin_ready = True

    # -- GPU batch right-hand side and integrator -------------------------------

    def _rhs_gpu(self, Psi_gpu, t):
        """Evaluate d(Psi)/dt for the batch, dispatching on the active H(t) mode."""
        if getattr(self, '_exact_H_ready', False):
            return self._rhs_exact(Psi_gpu, t)
        if getattr(self, '_continuous_local_ready', False):
            return self._rhs_continuous_local(Psi_gpu, t)
        if getattr(self, '_H_lin_ready', False):
            return self._rhs_linear(Psi_gpu, t)
        return self._rhs_frozen(Psi_gpu)

    def enable_continuous_local_drive(self):
        """Evaluate W(t) at every dopri5 substep via `kernels.PerturbationExtractor`,
        instead of periodically reassembling H(t) via `hamiltonian_submatrix` on a
        `dt_reconstruct` grid decoupled from the ODE's own adaptive step size.

        `update_H`/`update_H_linear` cost is dominated by `hamiltonian_submatrix`,
        which reassembles the whole system regardless of how localized the
        perturbation is. When the drive is spatially local (e.g. a single
        driven site/dot, as opposed to a perturbation spread across the whole
        lattice), `PerturbationExtractor.data(t)` recomputes only the nnz
        time-dependent matrix elements every call, cheaply enough to call at
        every substep -- removing `dt_reconstruct` (and its overhead/accuracy
        trade-off) entirely, matching how tkwant's own native per-state kernel
        already evaluates W(t). This is a poor fit for a broadly time-dependent
        drive (see the calibration in `_try_make_perturbation_extractor`) --
        useful specifically for narrow, localized, and/or ultrashort-in-time
        drives where `dt_reconstruct` would otherwise need to be pushed very
        small, e.g. a voltage pulse through a small quantum dot.

        Requires `attach_fsys` to have been called first (with any `omega`;
        this mode replaces the non-monochromatic fallback path).
        """
        if not getattr(self, '_fast_update_ready', False):
            raise RuntimeError('enable_continuous_local_drive requires attach_fsys(...) first')
        from tkwant.onebody import kernels
        state0 = self._states[0]
        extractor = kernels.PerturbationExtractor(
            self._fsys, state0.time_name, state0.time_start, state0.params)
        row, col = extractor.row_col()
        n = self._n_central

        self._cont_extractor = extractor
        self._cont_data_buf = np.zeros(len(row), dtype=complex)
        if self._use_gpu:
            cp = self._cp
            self._cont_row_gpu = cp.asarray(row, dtype=cp.int32)
            self._cont_col_gpu = cp.asarray(col, dtype=cp.int32)
            self._cont_psi_st_gpu = cp.asarray(self._psi_st_central[:n, :])
            self._cont_shape = (n, n)
        self._continuous_local_ready = True
        logger.info('Continuous local-drive mode enabled (nnz=%d, n_central=%d)',
                    len(row), n)

    def _rhs_continuous_local(self, Psi_gpu, t):
        cp = self._cp
        self._cont_extractor.data(float(t), out=self._cont_data_buf)
        data_gpu = cp.asarray(self._cont_data_buf)
        delta = self._csp.coo_matrix(
            (data_gpu, (self._cont_row_gpu, self._cont_col_gpu)),
            shape=self._cont_shape).tocsr()

        n = self._cont_shape[0]
        dPsi = self._H_eff0_gpu_base.dot(Psi_gpu)
        dPsi[:n, :] += delta.dot(Psi_gpu[:n, :])
        dPsi *= -1j
        dPsi += 1j * Psi_gpu * self._delta_E_gpu
        dPsi[:n, :] += -1j * delta.dot(self._cont_psi_st_gpu)
        return dPsi

    def _rhs_exact(self, Psi_gpu, t):
        cos_t = float(np.cos(self._omega * t))
        sin_t = float(np.sin(self._omega * t))
        dPsi = self._H_eff0_gpu_base.dot(Psi_gpu)
        if cos_t != 1.0:
            dPsi += (cos_t - 1.0) * self._H_A_full_gpu.dot(Psi_gpu)
        if sin_t != 0.0:
            dPsi += sin_t * self._H_B_full_gpu.dot(Psi_gpu)
        dPsi *= -1j
        dPsi += 1j * Psi_gpu * self._delta_E_gpu
        dPsi += (cos_t - 1.0) * self._src_A_gpu + sin_t * self._src_B_gpu
        return dPsi

    def _rhs_linear(self, Psi_gpu, t):
        tau = t - self._lin_t0
        dPsi = self._H_eff0_gpu_base.dot(Psi_gpu)
        dPsi += self._H_lin_delta0_gpu.dot(Psi_gpu)
        if tau != 0.0:
            dPsi += tau * self._H_lin_slope_gpu.dot(Psi_gpu)
        dPsi *= -1j
        dPsi += 1j * Psi_gpu * self._delta_E_gpu
        dPsi += self._src_lin_0_gpu
        if tau != 0.0:
            dPsi += tau * self._src_lin_slope_gpu
        return dPsi

    def _rhs_frozen(self, Psi_gpu):
        dPsi = self._H_gpu.dot(Psi_gpu)
        dPsi *= -1j
        dPsi += 1j * Psi_gpu * self._delta_E_gpu
        if self._src_gpu is not None:
            dPsi += self._src_gpu
        return dPsi

    def _dopri5_step_gpu(self, Psi_gpu, t, dt):
        """Single adaptive dopri5 step for the batched state Psi, on GPU."""
        A = _A
        k = [None] * 7
        k[0] = self._rhs_gpu(Psi_gpu, t)
        k[1] = self._rhs_gpu(Psi_gpu + dt * A[1][0] * k[0], t + _C[1] * dt)
        k[2] = self._rhs_gpu(Psi_gpu + dt * (A[2][0] * k[0] + A[2][1] * k[1]), t + _C[2] * dt)
        k[3] = self._rhs_gpu(Psi_gpu + dt * (A[3][0] * k[0] + A[3][1] * k[1]
                             + A[3][2] * k[2]), t + _C[3] * dt)
        k[4] = self._rhs_gpu(Psi_gpu + dt * (A[4][0] * k[0] + A[4][1] * k[1]
                             + A[4][2] * k[2] + A[4][3] * k[3]), t + _C[4] * dt)
        k[5] = self._rhs_gpu(Psi_gpu + dt * (A[5][0] * k[0] + A[5][1] * k[1]
                             + A[5][2] * k[2] + A[5][3] * k[3] + A[5][4] * k[4]), t + _C[5] * dt)
        Psi_new = Psi_gpu + dt * (A[6][0] * k[0] + A[6][2] * k[2] + A[6][3] * k[3]
                                  + A[6][4] * k[4] + A[6][5] * k[5])
        k[6] = self._rhs_gpu(Psi_new, t + dt)

        E = _E
        error = dt * (E[0] * k[0] + E[2] * k[2] + E[3] * k[3] + E[4] * k[4]
                      + E[5] * k[5] + E[6] * k[6])

        cp = self._cp
        scale = self.atol + self.rtol * cp.abs(Psi_new)
        per_state_rms = cp.sqrt(cp.mean((cp.abs(error) / scale) ** 2, axis=0))
        error_norm = float(cp.max(per_state_rms))
        return Psi_new, error_norm

    def _integrate_gpu(self, Psi, t0, t1):
        """Adaptive dopri5 integration of the batched state from `t0` to `t1`.

        `max_step` bounds the trial step size regardless of the error
        controller's own judgment. This matters for drives with sharp
        features much narrower than the interval being integrated (e.g. an
        ultrashort pulse): the *initial* step guess here is `(t1-t0)/10`,
        which can be far wider than such a feature, and the error estimate
        of a single step that entirely straddles a narrow bump is not
        guaranteed to flag it (most Runge-Kutta stages can land on either
        side of the bump and agree with each other while simply missing
        it). `max_step` is the standard fix (see e.g. `scipy.integrate.
        solve_ivp`'s `max_step`): cap the step at something smaller than the
        known feature width and the controller can no longer step over it
        blindly.
        """
        cp = self._cp
        Psi_gpu = cp.asarray(Psi)
        t = t0
        dt = min((t1 - t0) / 10.0, self.max_step)
        dt_min = 1e-12

        for _ in range(self.nsteps):
            if t >= t1:
                break
            dt = min(dt, t1 - t)
            Psi_new, error = self._dopri5_step_gpu(Psi_gpu, t, dt)
            if error <= 1.0:
                Psi_gpu = Psi_new
                t += dt
                dt *= min(5.0, 0.9 * error ** (-0.2)) if error > 0 else 5.0
                dt = min(dt, self.max_step)
            else:
                dt = max(dt_min, dt * max(0.1, 0.9 * error ** (-0.25)))
        else:
            raise RuntimeError(f'BatchGPUSolver: max_steps reached (t={t:.4f}, t1={t1})')

        return cp.asnumpy(Psi_gpu)

    # -- Public evolve interface -------------------------------------------------

    def evolve(self, time):
        """Evolve all local one-body states to `time`.

        Drop-in replacement for `tkwant.manybody.WaveFunction.evolve`.
        Falls back to tkwant's native sequential CPU evolution if the GPU
        integration raises an exception.

        Parameters
        ----------
        time : float
            Target time. No-op if equal to the current time.
        """
        t_current = self._states[0].time
        if time == t_current:
            return

        if self._H_gpu is None and self._H_cpu is None:
            self.reconstruct_H(t_current)

        if self._use_gpu and self._H_gpu is not None:
            try:
                Psi = self._evolve_on_gpu(t_current, time)
                for i, state in enumerate(self._states):
                    state.psibar = np.ascontiguousarray(Psi[:, i])
                    state.time = time
                return
            except Exception as error:
                logger.warning('GPU error (%s), falling back to CPU', error)
                self._use_gpu = False

        for state in self._states:
            state.evolve(time)

    def _evolve_on_gpu(self, t_current, time):
        Psi = np.stack([state.psibar for state in self._states], axis=1)

        if getattr(self, '_exact_H_ready', False):
            return self._integrate_gpu(Psi, t_current, time)

        if getattr(self, '_continuous_local_ready', False):
            return self._integrate_gpu(Psi, t_current, time)

        if not getattr(self, '_fast_update_ready', False):
            return self._integrate_gpu(Psi, t_current, time)

        dt_reconstruct = getattr(self, 'dt_reconstruct', time - t_current)
        use_linear = getattr(self, '_use_linear_update', False)
        t = t_current
        while t < time - 1e-14:
            t_next = min(t + dt_reconstruct, time)
            if use_linear:
                self.update_H_linear(t, t_next)
            else:
                self.update_H(t)
            Psi = self._integrate_gpu(Psi, t, t_next)
            t = t_next
        return Psi
