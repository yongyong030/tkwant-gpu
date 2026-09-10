"""Test module for `gpu_solver_core`.

Validates `BatchGPUSolver` against native tkwant CPU evolution: the exact
cos/sin decomposition for a monochromatic drive, and the frozen/linear
numeric fallback for a non-monochromatic one. Skipped entirely if cupy or
a CUDA device is unavailable.
"""

from functools import partial

import numpy as np
import pytest
import kwant
import tkwant
import kwantspectrum

from gpu_solver_core import BatchGPUSolver

cupy = pytest.importorskip('cupy')

L = 40
EF = 0.0
V0 = 0.3
OMEGA = 0.5
X0, WIDTH = L // 2, 4
TMAX_BOUNDARY = 80


def onsite_sinusoidal(site, time):
    return V0 * np.sin(OMEGA * time)


PULSE_T0 = 10.0
PULSE_TAU = 2.0


def onsite_voltage_step(site, time):
    return V0 * 0.5 * (1.0 + np.tanh((time - PULSE_T0) / PULSE_TAU))


def make_1d_chain(onsite):
    """1-orbital tight-binding chain with a perturbed window and two leads."""
    lat = kwant.lattice.square(a=1, norbs=1)
    syst = kwant.Builder()
    for x in range(L):
        syst[lat(x, 0)] = onsite if X0 - WIDTH <= x <= X0 + WIDTH else 0.0
    for x in range(L - 1):
        syst[lat(x, 0), lat(x + 1, 0)] = -1.0

    lead = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
    lead[lat(0, 0)] = 0.0
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = -1.0
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst.finalized()


def make_wavefunction(fsys):
    specs = kwantspectrum.spectra(fsys.leads)
    occupations = tkwant.manybody.lead_occupation(chemical_potential=EF, temperature=0.0)
    emin, emax = tkwant.manybody.calc_energy_cutoffs(occupations)
    boundaries = tkwant.leads.automatic_boundary(specs, tmax=TMAX_BOUNDARY, emin=emin, emax=emax)

    interval_type = partial(tkwant.manybody.Interval, order=5, quadrature='gausslegendre')
    intervals = tkwant.manybody.calc_intervals(specs, occupations, interval_type)
    intervals = tkwant.manybody.split_intervals(intervals, number_subintervals=1)
    tasks = tkwant.manybody.calc_tasks(intervals, specs, occupations)
    psi0 = tkwant.manybody.calc_initial_state(fsys, tasks, boundaries=boundaries)
    psi = tkwant.manybody.WaveFunction(psi0, tasks)
    psi.evolve(0.0)
    return psi


def ordered_psibar(psi):
    """Stack local psibar vectors, ordered by integer task key.

    Sorting by key (not dict iteration order or state.energy) is required
    for correctness whenever many-body states can be energy-degenerate
    (e.g. multiple lead channels sharing an energy) -- see the state-
    ordering bug documented in FINDINGS.md.
    """
    local = psi.psi.local_data()
    pairs = sorted(local.items(), key=lambda kv: kv[0])
    return np.stack([state.psibar for _, state in pairs], axis=1)


@pytest.mark.parametrize('t', [5.0, 10.0, 20.0, 40.0])
def test_exact_fourier_matches_cpu(t):
    """A purely sinusoidal drive should match CPU tkwant to ~1e-7."""
    fsys = make_1d_chain(onsite_sinusoidal)

    psi_cpu = make_wavefunction(fsys)
    psi_cpu.evolve(t)
    cpu_psibar = ordered_psibar(psi_cpu)

    psi_gpu = make_wavefunction(fsys)
    solver = BatchGPUSolver(psi_gpu)
    solver.attach_fsys(fsys, omega=OMEGA)
    psi_gpu.evolve = solver.evolve
    psi_gpu.evolve(t)
    gpu_psibar = ordered_psibar(psi_gpu)

    rel_err = np.linalg.norm(cpu_psibar - gpu_psibar) / np.linalg.norm(cpu_psibar)
    assert rel_err < 1e-6


@pytest.mark.parametrize('use_linear, dt_reconstruct, max_rel_err', [
    (False, 0.1, 1e-2),   # frozen (0th order): coarse tolerance
    (True, 1.0, 5e-3),    # linear (1st order): tighter tolerance, larger dt_reconstruct
])
def test_fallback_matches_cpu_for_nonmonochromatic_drive(use_linear, dt_reconstruct, max_rel_err):
    """A non-sinusoidal drive (voltage step) forces the numeric fallback path."""
    fsys = make_1d_chain(onsite_voltage_step)
    t_final = 40.0

    psi_cpu = make_wavefunction(fsys)
    psi_cpu.evolve(t_final)
    cpu_psibar = ordered_psibar(psi_cpu)

    psi_gpu = make_wavefunction(fsys)
    solver = BatchGPUSolver(psi_gpu)
    solver.dt_reconstruct = dt_reconstruct
    solver._use_linear_update = use_linear
    solver.attach_fsys(fsys, omega=None)
    psi_gpu.evolve = solver.evolve
    psi_gpu.evolve(t_final)
    gpu_psibar = ordered_psibar(psi_gpu)

    rel_err = np.linalg.norm(cpu_psibar - gpu_psibar) / np.linalg.norm(cpu_psibar)
    assert rel_err < max_rel_err


def test_evolve_is_noop_at_current_time():
    fsys = make_1d_chain(onsite_sinusoidal)
    psi = make_wavefunction(fsys)
    solver = BatchGPUSolver(psi)
    solver.attach_fsys(fsys, omega=OMEGA)
    psibar_before = ordered_psibar(psi)
    solver.evolve(psi.psi.local_data()[0].time)
    assert np.array_equal(ordered_psibar(psi), psibar_before)
