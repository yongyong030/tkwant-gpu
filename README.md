# tkwant-gpu

A GPU-batched time-evolution backend for [tkwant](https://tkwant.kwant-project.org/)'s
`manybody.WaveFunction`.

tkwant evolves each one-body scattering state in a many-body wavefunction
sequentially on the CPU. This project stacks all locally held one-body
states into a single matrix and integrates them together with a custom
batched Dormand-Prince (dopri5) solver on GPU (CuPy), replacing `N`
sequential CPU ODE solves with one batched GPU solve.

See `gpu_solver_note.pdf` (shared alongside this repo) for the full
derivation, validation tables, and benchmarks.

## Contents

- `gpu_solver_core.py` -- `BatchGPUSolver`, the generic batched solver. Only
  uses tkwant's public per-state API (`state.kernel.rhs()`, `state.psibar`,
  `state.psi_st`) and `fsys.hamiltonian_submatrix()`; no dependency on
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
