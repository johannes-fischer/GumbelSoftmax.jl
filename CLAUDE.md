# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GumbelSoftmax.jl implements differentiable discrete sampling for Julia:
1. **Gumbel-Softmax** — continuous relaxation of the categorical distribution
2. **Rao-Blackwellized Gumbel-Softmax** — variance-reduced variant with Monte Carlo sampling

Supports both Zygote (reverse-mode) and ForwardDiff (forward-mode) AD, plus GPU via CUDA.

## Running Tests

Do **not** use `Pkg.test()` — it recompiles everything and is too slow. Instead:

```bash
julia --project=. -e 'using GumbelSoftmax; include("test/runtests.jl")'
```

Or from an active Julia REPL (preferred, especially with Revise.jl loaded):
```julia
include("test/runtests.jl")
```

## Architecture

### Tensor Shape Convention

All functions expect `(categorical_dimension, batch_dimensions...)`:
```julia
logits = randn(3, 4, 10)  # 3 classes, 4 distributions, batch of 10
```

### Source Layout

- `src/utils.jl` — `stop_gradient()` bridging both AD systems (ChainRulesCore for Zygote, ForwardDiff.Dual extraction for ForwardDiff). Critical for the straight-through estimator; do not remove or replace.
- `src/gumbel_softmax.jl` — `sample_gumbel_softmax()` and `sample_softmax()`. Hard mode uses argmax + straight-through gradient trick. `sample_gumbel()` dispatches on CuArray vs Array for GPU/CPU.
- `src/rao_gumbel_softmax.jl` — `sample_rao_gumbel_softmax()` uses `slicemap()` for batch processing; a separate `ForwardDiff.Dual` dispatch uses `mapslices()` instead.

### Exported API

- `sample_gumbel_softmax(; probs, logits, tau, hard, epsilon)`
- `sample_rao_gumbel_softmax(; probs, logits, k, tau, I, epsilon)`
- `sample_softmax(; probs, logits, tau, hard, epsilon)`
- `stop_gradient(x)`

### AD System Integration

The package supports Zygote and ForwardDiff simultaneously but they must **not** be composed. Functions detect `ForwardDiff.Dual` types and use alternate code paths (e.g., `mapslices` instead of `slicemap`).

## Key Pitfalls

- **Shape errors**: Always verify `(cat_dim, batch_dims...)` ordering
- **`stop_gradient()`**: Intentional for the straight-through estimator — never remove
- **Numerical stability**: All functions default to `epsilon=1e-10` (Float32) to prevent `-log(0)`
- **Project.toml / Manifest.toml**: Never manually edit; manage via `Pkg` calls
- **Import order for GPU/deep learning**: `using CUDA; using cuDNN; using Flux`
