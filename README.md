# tkwant-gpu

A GPU-batched time-evolution backend for [tkwant](https://tkwant.kwant-project.org/)'s
`manybody.WaveFunction`.

tkwant evolves each one-body scattering state in a many-body wavefunction
sequentially on the CPU. This project stacks all locally held one-body
states into a single matrix and integrates them together with a custom
batched Dormand-Prince (dopri5) solver on GPU (CuPy), replacing `N`
sequential CPU ODE solves with one batched GPU solve.

See [`gpu_solver_note.pdf`](gpu_solver_note.pdf) for the full derivation,
validation tables, and benchmarks. Summary below.

## If you want to use this for a simulation

Benchmarked against tkwant's own MPI-parallel CPU baseline (not
single-core, which is not a fair comparison once `N_states` is more than
a handful).

- **Speedup grows with problem size.** At small batch size
  (`N_states` around 10) the CPU wins outright, by more than an order of
  magnitude, because the GPU pays a fixed per-call dispatch cost that a
  small batch cannot amortize. The crossover is somewhere around
  `N_states` ~100-200 for typical lattices. At production scale
  (`n_dof` = 18,936, `N_states` = 30, long propagation to `t` = 500), the
  GPU wins by 75x against a 24-core CPU baseline. The controlling
  variables are `(n_dof, N_states, t_max)`, not lattice type.
- **A drive confined to a few sites and much faster than the
  reconstruction interval needs `enable_continuous_local_drive()` and a
  bound on `max_step`.** Without `max_step`, both tkwant's own adaptive
  CPU stepper and this GPU solver can silently step over a narrow pulse
  and return a wrong-but-plausible answer with no error raised.
- **Self-consistent (mean-field) runs are validated only at small scale**
  (`n_dof` = 197, `N_states` = 8, `rel_err` = 8e-4 against native CPU).
  There is no speed benchmark yet for `GPUSelfConsistentState` at
  production scale.
- Not yet tested: multi-tone drives, spin-orbit coupling, disorder,
  `manybody.State`'s adaptive interval refinement, multiple GPUs.

## If you want to dig into the implementation

Every one-body task sees the same time-dependent effective Hamiltonian
`H_eff(t)`. Stacking the `N_states` correction vectors as columns of one
matrix turns `N_states` independent ODEs into a single batched matrix
ODE, integrated with a custom GPU dopri5 step so that one batched sparse
matrix-matrix product replaces `N_states` sequential CPU sparse
matrix-vector products per substep. `H_eff` isn't exposed by tkwant
directly, so it's recovered once by column-probing tkwant's own
right-hand-side function. When the drive is purely sinusoidal, `H(t)`
has an exact closed form in `cos(wt)`/`sin(wt)`, eliminating per-step
Hamiltonian reassembly entirely. General drives fall back to periodic
numeric reassembly (frozen or linearly-interpolated), accurate once the
reconstruction interval is set as a fraction of the drive's own
timescale rather than an absolute value.

## Contents

- `gpu_solver_core.py` -- `BatchGPUSolver`, the generic batched solver. Only
  uses tkwant's public per-state API (`state.kernel.rhs()`, `state.psibar`,
  `state.psi_st`) and `fsys.hamiltonian_submatrix()`. No dependency on
  orbital count, lattice symmetry, or dimensionality.
- `gpu_selfconsistent.py` -- `GPUSelfConsistentState`, an extension for
  tkwant's self-consistent (mean-field) solver
  (`tkwant.interaction.SelfConsistentState`).
- `test_gpu_solver_core.py`, `test_gpu_selfconsistent.py` -- validation
  against native CPU tkwant.

## Requirements

- `tkwant`, `kwant`, `kwantspectrum`
- `cupy` + a CUDA-capable GPU

## Usage

```python
from gpu_solver_core import BatchGPUSolver

solver = BatchGPUSolver(psi)              # psi: tkwant.manybody.WaveFunction
solver.attach_fsys(fsys, omega=omega)     # omega=None for non-monochromatic drives
psi.evolve = solver.evolve
psi.evolve(t)
```

## License

BSD 3-Clause, see `LICENSE`.
