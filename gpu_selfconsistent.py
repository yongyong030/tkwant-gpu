"""Self-consistent (mean-field) evolution on top of `BatchGPUSolver`.

`tkwant.interaction.SelfConsistentState` adds a mean-field potential q(t) on
top of a wavefunction's existing time-dependent perturbation W(t), via
`WaveFunction.add_perturbation(qt)`. That call mutates each one-body state's
*kernel* (`kernel.set_W(...)`), not `fsys`'s own onsite/hopping functions --
so `BatchGPUSolver.attach_fsys`'s exact-Fourier and `update_H`/
`update_H_linear` paths, which all read H(t) straight from
`fsys.hamiltonian_submatrix()`, never see q(t) at all, and re-probing
`state.kernel.rhs()` (`reconstruct_H`) is *not* a safe substitute: the
kernel's RHS bakes in state 0's own `psi_st` as an additive source term, so
column-probing it after `add_perturbation` contaminates every column of the
reconstructed H_eff with a spurious rank-1 correction from state 0's
`psi_st` alone -- wrong for every other state's `psi_st`.

Instead, `q(t)` is applied directly: `mf_potential.evaluate(t)` already
*is* q(t) as a sparse diagonal matrix (no probing needed), so it is added
to the batch's right-hand side the same way `attach_fsys` adds the base
drive's own H(t) contribution -- one extra diagonal term plus its
`-i q(t) psi_st` source, both embedded into the full (central + buffer)
space with `BatchGPUSolver._embed_central`, held frozen over each
self-consistent window `[time_sc - tau, time_sc]` (0th order in q(t)
within a window; the base drive can still use `attach_fsys`'s exact or
1st-order paths independently). This requires `attach_fsys` to already be
attached (so `_psi_st_central`/`_embed_central`/`_n_central` exist), and
works by wrapping `gpu_solver._rhs_gpu`, not by touching
`gpu_solver_core.py`.
"""

import logging

import numpy as np

from tkwant.interaction import AdaptiveStepsize, Extrapolate

logger = logging.getLogger(__name__)

__all__ = ['GPUSelfConsistentState']


class GPUSelfConsistentState:
    """Self-consistent mean-field evolution, GPU-batched between updates.

    Same constructor/evolve semantics as `tkwant.interaction.SelfConsistentState`,
    but `wave_function.evolve` must already be routed through a
    `gpu_solver_core.BatchGPUSolver` with `attach_fsys` already called (i.e.
    ``wave_function.evolve = gpu_solver.evolve`` after
    ``gpu_solver.attach_fsys(fsys, omega=...)``), and `gpu_solver` must be
    passed explicitly so the mean-field term can be injected into its
    right-hand side.

    Parameters
    ----------
    wave_function : `tkwant.manybody.WaveFunction`
        The manybody wavefunction, GPU-batched (see above).
    gpu_solver : `gpu_solver_core.BatchGPUSolver`
        The solver driving `wave_function.evolve`. Must have `attach_fsys`
        already called.
    mf_operator, mf_potential, atol, rtol, tau, extrapolator_type :
        Same as `tkwant.interaction.SelfConsistentState`.
    """

    def __init__(self, wave_function, gpu_solver, mf_operator, mf_potential,
                 atol=1e-6, rtol=1e-6, tau=AdaptiveStepsize,
                 extrapolator_type=Extrapolate):
        if not getattr(gpu_solver, '_fast_update_ready', False):
            raise RuntimeError(
                'GPUSelfConsistentState requires gpu_solver.attach_fsys(...) '
                'to be called first (needed for _psi_st_central/_embed_central)')

        self.wavefunction = wave_function
        self.gpu_solver = gpu_solver
        self._mf_operator = mf_operator
        self._mf_potential = mf_potential
        self.steps = 0
        self.time = wave_function.time

        # wrap the solver's RHS once; `_base_rhs_gpu` re-dispatches on
        # gpu_solver's own state (exact/linear/frozen) at every call, so the
        # base drive keeps whatever accuracy `attach_fsys` set up
        self._base_rhs_gpu = gpu_solver._rhs_gpu
        gpu_solver._rhs_gpu = self._rhs_with_meanfield
        self._Q_gpu = None
        self._Qsrc_gpu = None

        if isinstance(tau, (int, float)):
            if tau <= 0:
                raise ValueError('stepsize tau must be > 0')
            self._tau = tau

            def const_tau(*args, **kwargs):
                return tau
            self._estimate_tau = const_tau
        else:
            self._estimate_tau = tau(rtol, atol)
            self._tau = self._estimate_tau.tau_min
        self._time_sc = self.time + self._tau

        y = self._evaluate_mf()
        self._yt = extrapolator_type(y, self._tau, x0=self.time)
        self._mf_potential.prepare(self._yt, self._time_sc)
        self.wavefunction.add_perturbation(self._mf_potential.evaluate)
        self._update_meanfield_term(self.time)

    def _evaluate_mf(self):
        try:
            return self.wavefunction.evaluate(self._mf_operator, root=None)
        except TypeError:
            return self.wavefunction.evaluate(self._mf_operator)

    def _update_meanfield_term(self, t):
        """Recompute q(t)'s GPU RHS contribution, frozen for the upcoming window."""
        solver = self.gpu_solver
        Q_central = self._mf_potential.evaluate(float(t)).tocsr().astype(complex)
        n_buffer = solver.n - solver._n_central
        Q_full = solver._embed_central(Q_central, n_buffer)

        src_central = -1j * Q_central.dot(solver._psi_st_central)
        src_full = np.zeros((solver.n, solver.N), dtype=complex)
        src_full[:solver._n_central, :] = src_central

        if solver._use_gpu:
            self._Q_gpu = solver._csp.csr_matrix(Q_full)
            self._Qsrc_gpu = solver._cp.asarray(src_full)

    def _rhs_with_meanfield(self, Psi_gpu, t):
        dPsi = self._base_rhs_gpu(Psi_gpu, t)
        if self._Q_gpu is not None:
            dPsi += -1j * self._Q_gpu.dot(Psi_gpu)
            dPsi += self._Qsrc_gpu
        return dPsi

    def evolve(self, time):
        """Evolve the self-consistent manybody state up to `time`."""
        while self._time_sc <= time:

            self.wavefunction.evolve(self._time_sc)

            y = self._evaluate_mf()
            dyt = self._yt.add_point(self._time_sc, y)

            dt = self._estimate_tau(y, dyt, self._tau)
            logger.debug('tmin, tmax=[%s, %s], new tau=%s',
                         self._time_sc - self._tau, self._time_sc, dt)
            self._tau = dt
            time_next = self._time_sc + self._tau

            self._yt.set_stepsize(self._tau)
            self._mf_potential.prepare(self._yt, time_next)
            self.wavefunction.add_perturbation(self._mf_potential.evaluate)
            self._update_meanfield_term(self._time_sc)
            self._time_sc = time_next

            self.steps += 1

        self.wavefunction.evolve(time)
        self.time = time

    def evaluate(self, observable, root=0):
        """Evaluate the expectation value of an operator at the current time."""
        try:
            return self.wavefunction.evaluate(observable, root)
        except TypeError:
            return self.wavefunction.evaluate(observable)
