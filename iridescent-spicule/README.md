# Multi-Agent Cooperative Decentralized Localization

Implementation of sensor network localization using convex optimization with mixed-integer programming for robustness.

## Problem

Solve the non-convex sensor network localization problem:
```
min Σ (||x_i - x_j|| - d_ij)²
```
where distances `d_ij` are noisy and may contain outliers.

## Methods

1. **SDP Relaxation** - Centralized semidefinite programming with Gram matrix lifting
2. **Distributed ADMM** - Decentralized consensus algorithm for scalability
3. **MIQP Outlier Rejection** - Mixed-integer programming with binary trust variables

## Installation

```bash
cd /home/reyshwanth/.gemini/antigravity/playground/iridescent-spicule

# Install Julia dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Optional: Install SCIP for true MIQP (otherwise uses continuous relaxation)
# julia --project=. -e 'using Pkg; Pkg.add("SCIP")'
```

## Usage

```bash
# Default configuration (20 agents, 460 anchors @ 23:1 ratio, 20% outliers)
julia main.jl

# Small network test (5 agents, 115 anchors @ 23:1 ratio)
julia main.jl --test small

# Large network test (50 agents, 1150 anchors @ 23:1 ratio)
julia main.jl --test large

# High outlier ratio test (20 agents, 460 anchors @ 23:1 ratio, 30% outliers)
julia main.jl --test outliers
```

> [!NOTE]
> **High Anchor Ratio**: All configurations use a 23:1 anchor-to-agent ratio for maximum localization accuracy with abundant reference points.

## Files

- `problem_data.jl` - Network generation with noise and outliers
- `sdp_relaxation.jl` - Centralized SDP solver
- `admm_distributed.jl` - Distributed ADMM solver
- `miqp_outlier.jl` - MIQP with outlier detection
- `visualization.jl` - Plotting functions
- `main.jl` - Main orchestration script

## Mathematical Details

See `implementation_plan.md` for detailed problem formulation.

## Output

Results saved to:
- `results_sdp.png` - SDP localization
- `results_admm.png` - ADMM localization
- `results_miqp.png` - MIQP localization
- `results_comparison.png` - Side-by-side comparison
- `admm_convergence.png` - ADMM convergence plot
