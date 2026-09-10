"""Smoke test: self-consistent (mean-field) evolution through `BatchGPUSolver`.

Compares `GPUSelfConsistentState` (GPU-batched between self-consistent
updates, H_eff re-probed via `reconstruct_H` whenever the mean-field
potential changes) against native CPU `tkwant.interaction.SelfConsistentState`.
The system and drive (`create_system`/`gaussian`) are copied from tkwant's
own `tests/test_interaction.py::test_self_consistent_state` -- a voltage
pulse on one lead drives the density transiently away from equilibrium, so
the Hartree feedback actually does something (a purely static system would
never leave its equilibrium density, making the comparison vacuous).
"""

import functools

import numpy as np
import pytest
import scipy.sparse
import kwant
import kwantspectrum
from scipy.special import erf
from tkwant import interaction, manybody, leads

from gpu_solver_core import BatchGPUSolver
from gpu_selfconsistent import GPUSelfConsistentState
from test_gpu_solver_core import make_wavefunction, ordered_psibar

cupy = pytest.importorskip('cupy')

INTERACTION_STRENGTH = 2.0
# NOTE: at this interaction strength the self-consistent feedback is fairly
# stiff -- even native CPU tkwant's own trajectory is sensitive to `tau`
# (verified by hand: tau=1.0/0.5/0.2 all disagree with each other on the
# CPU side alone; tau=0.1 is where CPU-vs-CPU and CPU-vs-GPU both converge).
# This is a property of the test system/potential, not of the GPU wrapper.
TAU = 0.1
TIMES = [10.0, 40.0, 80.0, 100.0]


def create_system(a=1, L=20, W=1):
    """Copied from tkwant's tests/test_interaction.py::create_system."""
    gamma = 1 / a ** 2

    def onsite(time):
        return 4 * gamma

    lat = kwant.lattice.square(a=a, norbs=1)
    syst = kwant.Builder()
    syst[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    syst[lat.neighbors()] = -gamma

    sym = kwant.TranslationalSymmetry((-a, 0))
    lead_left = kwant.Builder(sym)
    lead_left[(lat(0, y) for y in range(W))] = 4 * gamma
    lead_left[lat.neighbors()] = -gamma

    syst.attach_lead(lead_left)
    syst.attach_lead(lead_left.reversed())
    return syst


def gaussian(time):
    """Copied from tkwant's tests/test_interaction.py::gaussian."""
    t0, A, tau = 50, 0.31415926535, 12.01122
    return A * (1 + erf((time - t0) / tau))


def make_driven_wavefunction(fsys, chemical_potential=3):
    """Like `test_gpu_solver_core.make_wavefunction`, but with an explicit chemical potential."""
    occupations = manybody.lead_occupation(chemical_potential=chemical_potential)
    spectra = kwantspectrum.spectra(fsys.leads)
    boundaries = leads.automatic_boundary(spectra, tmax=max(TIMES))
    interval_type = functools.partial(manybody.Interval, order=4, quadrature='gausslegendre')
    intervals = manybody.calc_intervals(spectra, occupations, interval_type=interval_type)
    tasks = manybody.calc_tasks(intervals, spectra, occupations)
    psi0 = manybody.calc_initial_state(fsys, tasks, boundaries)
    psi = manybody.WaveFunction(psi0, tasks)
    psi.evolve(0.0)
    return psi


class HartreePotential:
    """Diagonal onsite Hartree potential, following tkwant's own interaction test."""

    def __init__(self, interaction_strength, density0):
        self._interaction_strength = interaction_strength
        self._density0 = density0

    def prepare(self, density_func, tmax):
        self._density = density_func

    def evaluate(self, time):
        diag = (self._density(time) - self._density0) * self._interaction_strength
        return scipy.sparse.diags([diag], [0], dtype=complex)


def test_gpu_self_consistent_matches_cpu():
    syst = create_system()
    leads.add_voltage(syst, 0, gaussian)
    fsys = syst.finalized()

    density_operator = kwant.operator.Density(fsys)
    density_operator_sum = kwant.operator.Density(fsys, sum=True)

    # -- CPU reference --------------------------------------------------
    psi_cpu = make_driven_wavefunction(fsys)
    density0_cpu = psi_cpu.evaluate(density_operator, root=None)
    mf_cpu = interaction.SelfConsistentState(
        psi_cpu, density_operator, HartreePotential(INTERACTION_STRENGTH, density0_cpu),
        tau=TAU)

    densities_cpu = []
    for t in TIMES:
        mf_cpu.evolve(t)
        densities_cpu.append(mf_cpu.evaluate(density_operator_sum))

    # -- GPU --------------------------------------------------------------
    psi_gpu = make_driven_wavefunction(fsys)
    density0_gpu = psi_gpu.evaluate(density_operator, root=None)
    solver = BatchGPUSolver(psi_gpu)
    solver.dt_reconstruct = 0.1
    solver._use_linear_update = True
    solver.attach_fsys(fsys, omega=None)
    psi_gpu.evolve = solver.evolve
    mf_gpu = GPUSelfConsistentState(
        psi_gpu, solver, density_operator, HartreePotential(INTERACTION_STRENGTH, density0_gpu),
        tau=TAU)

    densities_gpu = []
    for t in TIMES:
        mf_gpu.evolve(t)
        densities_gpu.append(mf_gpu.evaluate(density_operator_sum))

    cpu_psibar = ordered_psibar(psi_cpu)
    gpu_psibar = ordered_psibar(psi_gpu)
    rel_err = np.linalg.norm(cpu_psibar - gpu_psibar) / np.linalg.norm(cpu_psibar)
    print(f'psibar rel_err = {rel_err:.3e}')
    print(f'densities_cpu = {densities_cpu}')
    print(f'densities_gpu = {densities_gpu}')

    # sanity check: the drive must actually move the density, or the
    # comparison below is vacuous (both sides would trivially agree at
    # the frozen equilibrium value)
    assert np.ptp(densities_cpu) > 1e-3

    assert rel_err < 1e-2
    np.testing.assert_allclose(densities_gpu, densities_cpu, rtol=1e-2)
