# Neural ODE Learning - Execution Guide

This file contains the step-by-step commands to run the Neural ODE learning exercise.

## Prerequisites

- Python 3.12+
- uv package manager installed
- Parent directory PDE solver files available (2_tumor_diffusion_pde_analysis.py)

## Step 1: Install Dependencies

### Option A: Using pnode (recommended, but requires PETSc)

```bash
# Install PETSc and pnode
uv pip install petsc petsc4py
uv pip install git+https://github.com/caidao22/pnode.git
```

**Note**: If PETSc installation fails (common on some systems), use Option B below.

### Option B: Fallback to torchdiffeq (simpler installation)

```bash
# Install torchdiffeq as fallback
uv pip install torchdiffeq
```

Then edit `neural_ode_learning.py` line ~30 to set `USE_PNODE = False`

## Step 2: Create Directory Structure

```bash
# Create output directories
mkdir -p node/out/neural_ode_dataset
mkdir -p node/out/neural_ode_models
mkdir -p node/out/neural_ode_results
mkdir -p node/figs
```

## Step 3: Generate Training Data

Generate 50 trajectories from the PDE model with varying parameters and initial conditions.

```bash
cd node
python neural_ode_learning.py --mode generate_data
```

**Expected output**:
- `out/neural_ode_dataset/train_trajectories.npz` (40 trajectories)
- `out/neural_ode_dataset/val_trajectories.npz` (5 trajectories)
- `out/neural_ode_dataset/test_trajectories.npz` (5 trajectories)
- `out/neural_ode_dataset/metadata.json`

**Runtime**: ~45-50 minutes

## Step 4: Train Small MLP Model

Train the baseline small MLP architecture (2 layers, 32 neurons, ~1.3k parameters).

```bash
python neural_ode_learning.py --mode train --model small_mlp --epochs 500
```

**Expected output**:
- `out/neural_ode_models/model_small_mlp.pt`
- `out/neural_ode_models/training_history_small_mlp.csv`

**Runtime**: ~15-20 minutes

## Step 5: Train Large MLP Model

Train the large MLP architecture with Fourier time encoding (4 layers, 128 neurons, ~33k parameters).

```bash
python neural_ode_learning.py --mode train --model large_mlp --epochs 500
```

**Expected output**:
- `out/neural_ode_models/model_large_mlp.pt`
- `out/neural_ode_models/training_history_large_mlp.csv`

**Runtime**: ~35-45 minutes

## Step 6: Evaluate Models

Evaluate both models on the test set.

```bash
# Evaluate Small MLP
python neural_ode_learning.py --mode evaluate --model small_mlp

# Evaluate Large MLP
python neural_ode_learning.py --mode evaluate --model large_mlp
```

**Expected output**:
- `out/neural_ode_results/metrics_small_mlp.json`
- `out/neural_ode_results/metrics_large_mlp.json`
- Console output with RMSE, MAE metrics

## Step 7: Generate Visualizations

Create all comparison figures and visualizations.

```bash
python neural_ode_learning.py --mode visualize
```

**Expected output**:
- `figs/neural_ode_data_samples.png` - Sample training trajectories
- `figs/neural_ode_training_curves.png` - Training/validation loss curves
- `figs/neural_ode_test_predictions.png` - Predictions vs ground truth (4-panel: S, R, I, C)
- `figs/neural_ode_architecture_comparison.png` - Model comparison bar chart
- `figs/neural_ode_extrapolation.png` - Extrapolation test beyond training window

## Optional: Advanced Usage

### Use Different ODE Solver (pnode only)

```bash
# Train with DOPRI5 solver instead of default RK4
python neural_ode_learning.py --mode train --model small_mlp --ts_type dopri5

# Other available solvers: euler, rk4, dopri5 (if using pnode)
```

### Adjust Training Hyperparameters

```bash
# Train with custom learning rate and batch size
python neural_ode_learning.py --mode train --model large_mlp --epochs 300 --lr 0.0005 --batch_size 16
```

### Run All Steps in Sequence

```bash
# Complete pipeline (data generation + training + evaluation + visualization)
python neural_ode_learning.py --mode generate_data && \
python neural_ode_learning.py --mode train --model small_mlp --epochs 500 && \
python neural_ode_learning.py --mode train --model large_mlp --epochs 500 && \
python neural_ode_learning.py --mode evaluate --model small_mlp && \
python neural_ode_learning.py --mode evaluate --model large_mlp && \
python neural_ode_learning.py --mode visualize
```

## Expected Results

### Performance Targets

**Small MLP**:
- RMSE (Tumor Burden): < 0.15
- Training time: ~20 minutes
- Inference: < 0.1s per trajectory

**Large MLP**:
- RMSE (Tumor Burden): < 0.08
- Training time: ~40 minutes
- Inference: < 0.2s per trajectory

### Key Findings

- Large MLP should capture periodic dosing patterns better (Fourier encoding)
- Both models should be 100x faster than PDE solver for predictions
- Small degradation expected when extrapolating beyond training window (t > 6)

## Troubleshooting

### PETSc Installation Fails

Use torchdiffeq fallback (see Step 1, Option B).

### Out of Memory During Training

Reduce batch size:
```bash
python neural_ode_learning.py --mode train --model large_mlp --batch_size 4
```

### Data Generation Too Slow

Reduce grid resolution by editing `neural_ode_learning.py` line ~200:
```python
N_grid = 48  # instead of 64
```

### CUDA/GPU Errors

Force CPU mode by editing `neural_ode_learning.py` line ~50:
```python
device = torch.device("cpu")
```

## File Structure After Execution

```
node/
├── README.md
├── COMMANDS.md (this file)
├── neural_ode_learning.py
├── out/
│   ├── neural_ode_dataset/
│   │   ├── train_trajectories.npz
│   │   ├── val_trajectories.npz
│   │   ├── test_trajectories.npz
│   │   └── metadata.json
│   ├── neural_ode_models/
│   │   ├── model_small_mlp.pt
│   │   ├── model_large_mlp.pt
│   │   ├── training_history_small_mlp.csv
│   │   └── training_history_large_mlp.csv
│   └── neural_ode_results/
│       ├── metrics_small_mlp.json
│       ├── metrics_large_mlp.json
│       └── metrics_summary.json
└── figs/
    ├── neural_ode_data_samples.png
    ├── neural_ode_training_curves.png
    ├── neural_ode_test_predictions.png
    ├── neural_ode_architecture_comparison.png
    └── neural_ode_extrapolation.png
```

## Questions or Issues?

Refer to:
- pnode repository: https://github.com/caidao22/pnode
- torchdiffeq repository: https://github.com/rtqichen/torchdiffeq
- Original task description: `node/README.md`
