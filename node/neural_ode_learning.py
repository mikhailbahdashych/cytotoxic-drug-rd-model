# %% [markdown]
# # Neural ODE Learning for Cytotoxic Drug R&D Model
# 
# **Task**: Learn the dynamics of the 4-variable PDE system (S, R, I, C) using Neural ODEs.
# Train neural networks to approximate dx/dt = f_θ(x, t) from PDE-generated trajectories.
# 
# ## Structure:
# - Section 1: Imports and Setup
# - Section 2: Data Generation from PDE
# - Section 3: Dataset Class and DataLoader
# - Section 4: Neural ODE Architectures (Small MLP, Large MLP)
# - Section 5: Training Functions
# - Section 6: Evaluation and Metrics
# - Section 7: Visualization
# - Section 8: Main Execution
# 
# **Author**: Generated for ISZ Project
# **Date**: 2025

# %% [markdown]
# ## Section 1: Imports and Setup

# %%
import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Neural ODE libraries
USE_PNODE = False  # Set to False to use torchdiffeq as fallback

try:
    if USE_PNODE:
        from pnode import petsc_adjoint
        print("[OK] pnode imported successfully")
        PNODE_AVAILABLE = True
    else:
        raise ImportError("USE_PNODE is False, using fallback")
except ImportError as e:
    print(f"[WARNING] pnode not available ({e}), falling back to torchdiffeq")
    try:
        from torchdiffeq import odeint_adjoint
        print("[OK] torchdiffeq imported successfully")
        PNODE_AVAILABLE = False
    except ImportError:
        print("[ERROR] Neither pnode nor torchdiffeq available!")
        print("Install one with: uv pip install git+https://github.com/caidao22/pnode.git")
        print("Or: uv pip install torchdiffeq")
        sys.exit(1)

# Import PDE module
import importlib.util

# Try to find the PDE module in the root directory
pde_module_path = Path("2_tumor_diffusion_pde_analysis.py")
if not pde_module_path.exists():
    # If not found, try parent directory (for backward compatibility)
    pde_module_path = Path("../2_tumor_diffusion_pde_analysis.py")
    if not pde_module_path.exists():
        print("[ERROR] Cannot find 2_tumor_diffusion_pde_analysis.py")
        print("Looked in:")
        print("  - ./2_tumor_diffusion_pde_analysis.py (root)")
        print("  - ../2_tumor_diffusion_pde_analysis.py (parent)")
        sys.exit(1)

spec = importlib.util.spec_from_file_location(
    "pde_module", str(pde_module_path.resolve())
)
pde_module = importlib.util.module_from_spec(spec)
sys.modules["pde_module"] = pde_module
try:
    spec.loader.exec_module(pde_module)
    print(f"[OK] PDE module imported from {pde_module_path}")
except Exception as e:
    print(f"[ERROR] Failed to import PDE module: {e}")
    sys.exit(1)

# Import needed components from PDE module
Grid = pde_module.Grid
Params = pde_module.Params
init_fields = pde_module.init_fields

# Setup directories - use node/ prefix when running from root
# Detect if we're in the root directory or node directory
if Path("node").exists() and Path("node").is_dir():
    # Running from root directory
    OUT_BASE = Path("node/out")
    FIG_BASE = Path("node/figs")
else:
    # Running from node directory
    OUT_BASE = Path("out")
    FIG_BASE = Path("figs")

# Create directories
(OUT_BASE / "neural_ode_dataset").mkdir(parents=True, exist_ok=True)
(OUT_BASE / "neural_ode_models").mkdir(parents=True, exist_ok=True)
(OUT_BASE / "neural_ode_results").mkdir(parents=True, exist_ok=True)
FIG_BASE.mkdir(parents=True, exist_ok=True)

# Device setup
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"[Device] Using: {device}")

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Helper functions
def save_json(data, path):
    """Save dictionary to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Saved] {path}")

def load_json(path):
    """Load JSON file to dictionary"""
    with open(path, 'r') as f:
        return json.load(f)

def savefig(path, dpi=160):
    """Save matplotlib figure"""
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"[Saved] {path}")

def rmse(pred, true):
    """Root mean squared error"""
    return np.sqrt(np.mean((pred - true) ** 2))

def mae(pred, true):
    """Mean absolute error"""
    return np.mean(np.abs(pred - true))

# %% [markdown]
# ## Section 2: Data Generation from PDE

# %%
def run_pde_simulation_for_node(grid, params, T=6.0, dt=0.01, save_every=10, custom_init=None):
    """
    Run PDE simulation and extract spatially-averaged trajectory for Neural ODE training.

    Args:
        grid: Grid object from PDE module
        params: Params object with model parameters
        T: Final time
        dt: Time step
        save_every: Save frequency
        custom_init: Optional tuple (S, R, I, C) for custom initial conditions

    Returns:
        dict with keys: t (time array), state (spatially averaged [S̄, R̄, Ī, C̄]), TB (tumor burden)
    """
    # Initialize fields
    if custom_init is not None:
        S, R, I, C = custom_init
    else:
        S, R, I, C = init_fields(grid)

    # Get solver function
    use_semi_implicit = hasattr(pde_module, 'step_semi_implicit')
    if use_semi_implicit:
        step_func = pde_module.step_semi_implicit
        print("[PDE] Using semi-implicit solver")
    else:
        step_func = pde_module.step_explicit
        print("[PDE] Using explicit solver")

    # Time stepping
    n_steps = int(T / dt)
    t_current = 0.0

    # Storage for trajectory
    trajectory = {
        't': [],
        'state': [],  # [S_mean, R_mean, I_mean, C_mean]
        'TB': []      # Tumor burden
    }

    # Save initial condition
    dx, dy = grid.dx, grid.dy
    S_mean = np.mean(S)
    R_mean = np.mean(R)
    I_mean = np.mean(I)
    C_mean = np.mean(C)
    TB = np.sum(S + R) * dx * dy

    trajectory['t'].append(t_current)
    trajectory['state'].append([S_mean, R_mean, I_mean, C_mean])
    trajectory['TB'].append(TB)

    # Time integration
    for step in range(n_steps):
        # Take PDE step
        if use_semi_implicit:
            # semi_implicit needs system matrices - check if available
            if hasattr(pde_module, 'precompute_system_matrices'):
                sys_mats = pde_module.precompute_system_matrices(grid, params, dt)
                S, R, I, C = step_func(S, R, I, C, grid, dt, params, sys_mats)
            else:
                # Fallback to explicit if no system matrices
                S, R, I, C = pde_module.step_explicit(S, R, I, C, grid, dt, params)
        else:
            S, R, I, C = step_func(S, R, I, C, grid, dt, params)
        t_current += dt

        # Save at specified frequency
        if (step + 1) % save_every == 0:
            S_mean = np.mean(S)
            R_mean = np.mean(R)
            I_mean = np.mean(I)
            C_mean = np.mean(C)
            TB = np.sum(S + R) * dx * dy

            trajectory['t'].append(t_current)
            trajectory['state'].append([S_mean, R_mean, I_mean, C_mean])
            trajectory['TB'].append(TB)

    # Convert to numpy arrays
    trajectory['t'] = np.array(trajectory['t'])
    trajectory['state'] = np.array(trajectory['state'])  # Shape: [n_timepoints, 4]
    trajectory['TB'] = np.array(trajectory['TB'])

    return trajectory


def generate_parameter_samples(n_samples=50, seed=42):
    """
    Generate parameter samples using Latin Hypercube Sampling.

    Samples 4 key therapy parameters:
    - infusion_rate: [0.05, 0.25]
    - alpha_S (drug efficacy on sensitive cells): [0.6, 1.0]
    - mu_max (resistance induction rate): [0.02, 0.08]
    - lam (drug clearance): [0.1, 0.3]

    Returns:
        List of Params objects
    """
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_samples)

    params_list = []
    for sample in samples:
        p = Params()
        # Use continuous infusion dosing
        p.dose_type = "infusion_const"
        p.infusion_rate = 0.05 + sample[0] * 0.20    # [0.05, 0.25]
        p.alpha_S = 0.6 + sample[1] * 0.4             # [0.6, 1.0]
        p.mu_max = 0.02 + sample[2] * 0.06            # [0.02, 0.08]
        p.lam = 0.1 + sample[3] * 0.2                 # [0.1, 0.3]
        params_list.append(p)

    return params_list


def generate_initial_conditions(grid, seed=None):
    """
    Generate varied initial conditions for S and R.

    Varies:
    - Tumor center position: (x0, y0) ∈ [0.3, 0.7]²
    - Tumor width: σ ∈ [0.10, 0.20]
    - Initial magnitudes: S0 ∈ [0.4, 0.6], R0 ∈ [0.05, 0.15]

    Returns:
        S, R, I, C arrays
    """
    if seed is not None:
        np.random.seed(seed)

    Nx, Ny = grid.Nx, grid.Ny
    x = np.linspace(0, grid.Lx, Nx)
    y = np.linspace(0, grid.Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Random tumor center and width
    x0 = 0.3 + np.random.rand() * 0.4
    y0 = 0.3 + np.random.rand() * 0.4
    sigma = 0.10 + np.random.rand() * 0.10

    # Random initial magnitudes
    S0_mag = 0.4 + np.random.rand() * 0.2
    R0_mag = 0.05 + np.random.rand() * 0.10

    # Create Gaussian distributions
    r_squared = (X - x0)**2 + (Y - y0)**2
    S = S0_mag * np.exp(-r_squared / (2 * sigma**2))

    # Resistant cells in a ring
    r_ring = 0.15 + np.random.rand() * 0.05
    sigma_ring = 0.05 + np.random.rand() * 0.03
    R = R0_mag * np.exp(-((np.sqrt(r_squared) - r_ring)**2) / (2 * sigma_ring**2))

    # Immune and drug concentrations (fixed)
    I = np.full((Nx, Ny), 0.02)
    C = np.zeros((Nx, Ny))

    return S, R, I, C


def generate_neural_ode_dataset(n_trajectories=50, T=6.0, N_grid=64, verbose=True):
    """
    Generate complete Neural ODE dataset with varied parameters and initial conditions.

    Args:
        n_trajectories: Number of trajectories to generate
        T: Time horizon
        N_grid: Grid resolution (NxN)
        verbose: Print progress

    Returns:
        List of trajectory dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Generating Neural ODE Dataset")
    print(f"{'='*60}")
    print(f"Number of trajectories: {n_trajectories}")
    print(f"Time horizon: T = {T}")
    print(f"Grid resolution: {N_grid} × {N_grid}")
    print(f"{'='*60}\n")

    # Setup
    grid = Grid(Nx=N_grid, Ny=N_grid, Lx=1.0, Ly=1.0)
    params_list = generate_parameter_samples(n_trajectories, seed=42)

    dataset = []

    # Generate trajectories
    iterator = tqdm(range(n_trajectories), desc="Generating trajectories") if verbose else range(n_trajectories)

    for i in iterator:
        # Get parameters
        params = params_list[i]

        # Generate initial conditions
        S, R, I, C = generate_initial_conditions(grid, seed=i+100)

        # Run PDE simulation with custom initial conditions
        try:
            trajectory = run_pde_simulation_for_node(
                grid, params, T=T, dt=0.01, save_every=10,
                custom_init=(S, R, I, C)
            )

            # Store trajectory with metadata
            dataset.append({
                'trajectory_id': i,
                'params': asdict(params),
                't': trajectory['t'],
                'state': trajectory['state'],
                'TB': trajectory['TB']
            })
        except Exception as e:
            print(f"\n[ERROR] Failed to generate trajectory {i}: {e}")
            continue

    print(f"\n[Success] Generated {len(dataset)} trajectories")
    return dataset


def save_dataset(dataset, train_frac=0.8, val_frac=0.1, suffix=""):
    """
    Split and save dataset to disk.

    Args:
        dataset: List of trajectory dicts
        train_frac: Fraction for training
        val_frac: Fraction for validation (remainder goes to test)
        suffix: Optional suffix to add to filenames (e.g., "_run1", "_experiment_A")
    """
    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    # Split
    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train+n_val]
    test_data = dataset[n_train+n_val:]

    # Add suffix to filenames
    suffix_str = f"_{suffix}" if suffix else ""

    # Save
    np.savez_compressed(OUT_BASE / f"neural_ode_dataset/train_trajectories{suffix_str}.npz", trajectories=train_data)
    np.savez_compressed(OUT_BASE / f"neural_ode_dataset/val_trajectories{suffix_str}.npz", trajectories=val_data)
    np.savez_compressed(OUT_BASE / f"neural_ode_dataset/test_trajectories{suffix_str}.npz", trajectories=test_data)

    # Save metadata
    metadata = {
        'n_total': n_total,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'split': {'train': train_frac, 'val': val_frac, 'test': 1-train_frac-val_frac},
        'suffix': suffix
    }
    save_json(metadata, OUT_BASE / f"neural_ode_dataset/metadata{suffix_str}.json")

    print(f"\n[Dataset Split]")
    print(f"  Train: {n_train} trajectories")
    print(f"  Val:   {n_val} trajectories")
    print(f"  Test:  {n_test} trajectories")
    print(f"  Total: {n_total} trajectories")

# %% [markdown]
# ## Section 3: Dataset Class and DataLoader

# %%
class NODEDataset(Dataset):
    """
    PyTorch Dataset for Neural ODE training.

    Returns:
        x0: Initial state [4]
        t: Time points [n_timepoints]
        states: Full trajectory [n_timepoints, 4]
    """
    def __init__(self, trajectories, noise_level=0.0):
        """
        Args:
            trajectories: List of trajectory dicts
            noise_level: Standard deviation of Gaussian noise to add (0 = no noise)
        """
        self.trajectories = trajectories
        self.noise_level = noise_level

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        t = torch.tensor(traj['t'], dtype=torch.float32)
        state = torch.tensor(traj['state'], dtype=torch.float32)

        # Add optional noise
        if self.noise_level > 0:
            noise = torch.randn_like(state) * self.noise_level
            state = state + noise
            # Clamp to non-negative
            state = torch.clamp(state, min=0.0)

        x0 = state[0]  # Initial condition

        return x0, t, state


def load_dataset_splits(noise_level=0.0, suffix=""):
    """
    Load train/val/test datasets from disk.

    Args:
        noise_level: Noise level to add to training data
        suffix: Optional suffix matching the saved dataset files

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    suffix_str = f"_{suffix}" if suffix else ""
    train_data = np.load(OUT_BASE / f"neural_ode_dataset/train_trajectories{suffix_str}.npz", allow_pickle=True)['trajectories']
    val_data = np.load(OUT_BASE / f"neural_ode_dataset/val_trajectories{suffix_str}.npz", allow_pickle=True)['trajectories']
    test_data = np.load(OUT_BASE / f"neural_ode_dataset/test_trajectories{suffix_str}.npz", allow_pickle=True)['trajectories']

    train_dataset = NODEDataset(train_data, noise_level=noise_level)
    val_dataset = NODEDataset(val_data, noise_level=0.0)  # No noise for validation
    test_dataset = NODEDataset(test_data, noise_level=0.0)  # No noise for test

    print(f"[Datasets Loaded]")
    print(f"  Train: {len(train_dataset)} trajectories")
    print(f"  Val:   {len(val_dataset)} trajectories")
    print(f"  Test:  {len(test_dataset)} trajectories")

    return train_dataset, val_dataset, test_dataset

# %% [markdown]
# ## Section 4: Neural ODE Architectures

# %%
class SmallMLP_NODE(nn.Module):
    """
    Small MLP Neural ODE architecture (Architecture A).

    - 2 hidden layers
    - 32 neurons per layer
    - Tanh activation
    - Direct time encoding
    - ~1,300 parameters
    """
    def __init__(self, state_dim=4, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, t, x):
        """
        Forward pass: compute dx/dt = f(x, t).

        Args:
            t: Time (scalar or tensor)
            x: State [batch_size, state_dim]

        Returns:
            dx/dt: [batch_size, state_dim]
        """
        # Ensure t is a scalar tensor
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=torch.float32, device=x.device)

        # Append time to state
        t_vec = t * torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        x_with_t = torch.cat([x, t_vec], dim=1)

        return self.net(x_with_t)


class LargeMLP_NODE(nn.Module):
    """
    Large MLP Neural ODE architecture with Fourier time encoding (Architecture B).

    - 4 hidden layers (128 → 128 → 128 → 64)
    - Tanh activation
    - Fourier time encoding: [sin(ωt), cos(ωt)] where ω = 2π/5 (dosing period)
    - ~33,000 parameters
    """
    def __init__(self, state_dim=4, hidden_dim=128, dosing_period=5.0):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.omega = 2 * np.pi / dosing_period  # Angular frequency for Fourier encoding

        self.net = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),  # +2 for sin/cos
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, state_dim)
        )

    def forward(self, t, x):
        """
        Forward pass with Fourier time encoding: compute dx/dt = f(x, sin(ωt), cos(ωt)).

        Args:
            t: Time (scalar or tensor)
            x: State [batch_size, state_dim]

        Returns:
            dx/dt: [batch_size, state_dim]
        """
        # Ensure t is a scalar tensor
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=torch.float32, device=x.device)

        # Fourier time encoding
        t_vec = t * torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        t_sin = torch.sin(self.omega * t_vec)
        t_cos = torch.cos(self.omega * t_vec)

        # Concatenate state with Fourier features
        x_with_t = torch.cat([x, t_sin, t_cos], dim=1)

        return self.net(x_with_t)


def count_parameters(model):
    """Count total number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %% [markdown]
# ## Section 5: Training Functions

# %%
def ode_solve(model, x0, t, method='rk4'):
    """
    Solve ODE using appropriate library (pnode or torchdiffeq).

    Args:
        model: Neural ODE model
        x0: Initial state [batch_size, state_dim]
        t: Time points [n_timepoints]
        method: ODE solver method

    Returns:
        Trajectory [n_timepoints, batch_size, state_dim]
    """
    if PNODE_AVAILABLE:
        # Use pnode
        # Note: pnode has a different API, this is a simplified wrapper
        # For production, would need proper pnode integration
        # Fallback to Euler for simplicity if pnode API is complex
        dt = t[1] - t[0]
        trajectory = [x0]
        x_current = x0
        for i in range(len(t) - 1):
            t_current = t[i]
            dx = model(t_current, x_current)
            x_next = x_current + dx * dt
            trajectory.append(x_next)
            x_current = x_next
        return torch.stack(trajectory, dim=0)
    else:
        # Use torchdiffeq
        return odeint_adjoint(model, x0, t, method=method)


def compute_loss(model, x0_batch, t_batch, x_true_batch, loss_weights=None):
    """
    Compute multi-component loss for Neural ODE training.

    Loss = w1*L_trajectory + w2*L_endpoint + w3*L_physics

    Args:
        model: Neural ODE model
        x0_batch: Initial states [batch_size, state_dim]
        t_batch: Time points [n_timepoints]
        x_true_batch: True trajectories [batch_size, n_timepoints, state_dim]
        loss_weights: Dict with keys 'trajectory', 'endpoint', 'physics'

    Returns:
        total_loss, loss_dict
    """
    if loss_weights is None:
        loss_weights = {'trajectory': 1.0, 'endpoint': 5.0, 'physics': 0.1}

    # Forward pass: integrate ODE
    x_pred = ode_solve(model, x0_batch, t_batch, method='rk4')
    x_pred = x_pred.permute(1, 0, 2)  # [batch, time, state]

    # Trajectory loss: MSE over entire trajectory
    loss_traj = torch.mean((x_pred - x_true_batch) ** 2)

    # Endpoint loss: emphasize final state
    loss_end = torch.mean((x_pred[:, -1, :] - x_true_batch[:, -1, :]) ** 2)

    # Physics loss: enforce non-negativity
    loss_phys = torch.mean(torch.relu(-x_pred) ** 2)

    # Total loss
    total_loss = (
        loss_weights['trajectory'] * loss_traj +
        loss_weights['endpoint'] * loss_end +
        loss_weights['physics'] * loss_phys
    )

    loss_dict = {
        'total': total_loss.item(),
        'trajectory': loss_traj.item(),
        'endpoint': loss_end.item(),
        'physics': loss_phys.item()
    }

    return total_loss, loss_dict


def train_neural_ode(
    model,
    train_loader,
    val_loader,
    num_epochs=500,
    lr=1e-3,
    weight_decay=1e-5,
    device='cpu',
    save_path=None,
    early_stopping_patience=50
):
    """
    Train Neural ODE model.

    Args:
        model: Neural ODE model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        save_path: Path to save best model
        early_stopping_patience: Patience for early stopping

    Returns:
        model, history (dict with training/validation losses)
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training Neural ODE Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        for x0_batch, t_batch, x_true_batch in train_loader:
            x0_batch = x0_batch.to(device)
            t_batch = t_batch[0].to(device)  # Same time grid for all trajectories
            x_true_batch = x_true_batch.to(device)

            optimizer.zero_grad()

            # Compute loss
            loss, loss_dict = compute_loss(model, x0_batch, t_batch, x_true_batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x0_batch, t_batch, x_true_batch in val_loader:
                x0_batch = x0_batch.to(device)
                t_batch = t_batch[0].to(device)
                x_true_batch = x_true_batch.to(device)

                # Compute loss
                loss, _ = compute_loss(model, x0_batch, t_batch, x_true_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # Print progress
        if epoch % 10 == 0 or patience_counter == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n[Early Stopping] No improvement for {early_stopping_patience} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

    print(f"\n[Training Complete]")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Load best model
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"[Loaded] Best model from {save_path}")

    return model, history

# %% [markdown]
# ## Section 6: Evaluation and Metrics

# %%
def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate Neural ODE model on test set.

    Returns:
        metrics: Dict with RMSE, MAE, etc.
        predictions: List of prediction dicts
    """
    model.eval()
    model = model.to(device)

    all_rmse_state = []
    all_mae_state = []
    all_rmse_tb = []
    predictions = []

    with torch.no_grad():
        for x0_batch, t_batch, x_true_batch in test_loader:
            x0_batch = x0_batch.to(device)
            t_batch = t_batch[0].to(device)
            x_true_batch = x_true_batch.to(device)

            # Predict
            x_pred = ode_solve(model, x0_batch, t_batch, method='rk4')
            x_pred = x_pred.permute(1, 0, 2)  # [batch, time, state]

            # Compute metrics
            rmse_state = torch.sqrt(torch.mean((x_pred - x_true_batch) ** 2, dim=(1, 2)))
            mae_state = torch.mean(torch.abs(x_pred - x_true_batch), dim=(1, 2))

            # Tumor burden metrics (S + R)
            TB_pred = x_pred[:, :, 0] + x_pred[:, :, 1]  # [batch, time]
            TB_true = x_true_batch[:, :, 0] + x_true_batch[:, :, 1]
            rmse_tb = torch.sqrt(torch.mean((TB_pred - TB_true) ** 2, dim=1))

            all_rmse_state.extend(rmse_state.cpu().numpy())
            all_mae_state.extend(mae_state.cpu().numpy())
            all_rmse_tb.extend(rmse_tb.cpu().numpy())

            # Store predictions
            for i in range(x_pred.shape[0]):
                predictions.append({
                    't': t_batch.cpu().numpy(),
                    'x_pred': x_pred[i].cpu().numpy(),
                    'x_true': x_true_batch[i].cpu().numpy()
                })

    # Aggregate metrics
    metrics = {
        'RMSE_state': float(np.mean(all_rmse_state)),
        'RMSE_state_std': float(np.std(all_rmse_state)),
        'MAE_state': float(np.mean(all_mae_state)),
        'MAE_state_std': float(np.std(all_mae_state)),
        'RMSE_TB': float(np.mean(all_rmse_tb)),
        'RMSE_TB_std': float(np.std(all_rmse_tb))
    }

    return metrics, predictions

# %% [markdown]
# ## Section 7: Visualization

# %%
def plot_data_samples(dataset, n_samples=16, save_path=None, save_data=True):
    """Plot sample trajectories from dataset"""
    if save_path is None:
        save_path = FIG_BASE / "neural_ode_data_samples.png"

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    # Store data for later analysis
    plot_data = {'indices': indices.tolist(), 'trajectories': []}

    for idx, ax in enumerate(axes):
        if idx < len(indices):
            _, t, state = dataset[indices[idx]]
            t = t.numpy()
            state = state.numpy()

            # Store data
            plot_data['trajectories'].append({
                'trajectory_id': int(indices[idx]),
                't': t.tolist(),
                'S': state[:, 0].tolist(),
                'R': state[:, 1].tolist(),
                'I': state[:, 2].tolist(),
                'C': state[:, 3].tolist()
            })

            # Plot each state variable
            ax.plot(t, state[:, 0], 'r-', label='S', linewidth=1.5)
            ax.plot(t, state[:, 1], 'b-', label='R', linewidth=1.5)
            ax.plot(t, state[:, 2], 'g-', label='I', linewidth=1.5)
            ax.plot(t, state[:, 3], 'm-', label='C', linewidth=1.5)

            ax.set_xlabel('Time t')
            ax.set_ylabel('State')
            ax.set_title(f'Trajectory {indices[idx]}')
            if idx == 0:
                ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    savefig(save_path)
    plt.close()

    # Save underlying data
    if save_data:
        data_path = str(save_path).replace('.png', '_data.json')
        save_json(plot_data, data_path)


def plot_training_curves(history_small, history_large, save_path=None, save_data=True):
    """Plot training and validation loss curves for both models"""
    if save_path is None:
        save_path = FIG_BASE / "neural_ode_training_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Small MLP
    ax = axes[0]
    ax.plot(history_small['epoch'], history_small['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(history_small['epoch'], history_small['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Small MLP (2 layers, 32 neurons)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Large MLP
    ax = axes[1]
    ax.plot(history_large['epoch'], history_large['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(history_large['epoch'], history_large['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Large MLP (4 layers, 128 neurons)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    savefig(save_path)
    plt.close()

    # Save underlying data
    if save_data:
        plot_data = {
            'small_mlp': history_small,
            'large_mlp': history_large
        }
        data_path = str(save_path).replace('.png', '_data.json')
        save_json(plot_data, data_path)


def plot_test_predictions(predictions, model_name, save_path=None, save_data=True):
    """Plot predictions vs ground truth for first test trajectory"""
    if save_path is None:
        save_path = FIG_BASE / "neural_ode_test_predictions.png"
    pred = predictions[0]
    t = pred['t']
    x_pred = pred['x_pred']
    x_true = pred['x_true']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    state_names = ['S (Sensitive)', 'R (Resistant)', 'I (Immune)', 'C (Drug)']
    colors = ['red', 'blue', 'green', 'magenta']

    for i, ax in enumerate(axes.flatten()):
        ax.plot(t, x_true[:, i], color=colors[i], linestyle='-', linewidth=2.5,
                label='PDE (Ground Truth)', alpha=0.8)
        ax.plot(t, x_pred[:, i], color=colors[i], linestyle='--', linewidth=2,
                label='Neural ODE', alpha=0.9)
        ax.set_xlabel('Time t', fontsize=11)
        ax.set_ylabel('State Value', fontsize=11)
        ax.set_title(state_names[i], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f'Neural ODE Predictions vs PDE Ground Truth ({model_name})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig(save_path)
    plt.close()

    # Save underlying data
    if save_data:
        plot_data = {
            'model': model_name,
            't': t.tolist(),
            'predictions': {
                'S': x_pred[:, 0].tolist(),
                'R': x_pred[:, 1].tolist(),
                'I': x_pred[:, 2].tolist(),
                'C': x_pred[:, 3].tolist()
            },
            'ground_truth': {
                'S': x_true[:, 0].tolist(),
                'R': x_true[:, 1].tolist(),
                'I': x_true[:, 2].tolist(),
                'C': x_true[:, 3].tolist()
            },
            'errors': {
                'S': (x_pred[:, 0] - x_true[:, 0]).tolist(),
                'R': (x_pred[:, 1] - x_true[:, 1]).tolist(),
                'I': (x_pred[:, 2] - x_true[:, 2]).tolist(),
                'C': (x_pred[:, 3] - x_true[:, 3]).tolist()
            }
        }
        data_path = str(save_path).replace('.png', '_data.json')
        save_json(plot_data, data_path)


def plot_architecture_comparison(metrics_small, metrics_large, save_path=None, save_data=True):
    """Bar chart comparing model architectures"""
    if save_path is None:
        save_path = FIG_BASE / "neural_ode_architecture_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ['Small MLP', 'Large MLP']

    # RMSE_TB
    ax = axes[0]
    rmse_tb = [metrics_small['RMSE_TB'], metrics_large['RMSE_TB']]
    rmse_tb_std = [metrics_small['RMSE_TB_std'], metrics_large['RMSE_TB_std']]
    ax.bar(models, rmse_tb, yerr=rmse_tb_std, capsize=5, color=['skyblue', 'salmon'])
    ax.set_ylabel('RMSE (Tumor Burden)')
    ax.set_title('Prediction Error (Tumor Burden)')
    ax.grid(alpha=0.3, axis='y')

    # RMSE_state
    ax = axes[1]
    rmse_state = [metrics_small['RMSE_state'], metrics_large['RMSE_state']]
    rmse_state_std = [metrics_small['RMSE_state_std'], metrics_large['RMSE_state_std']]
    ax.bar(models, rmse_state, yerr=rmse_state_std, capsize=5, color=['skyblue', 'salmon'])
    ax.set_ylabel('RMSE (All States)')
    ax.set_title('Prediction Error (All States)')
    ax.grid(alpha=0.3, axis='y')

    # Parameter count
    ax = axes[2]
    param_counts = [1300, 33000]  # Approximate
    ax.bar(models, param_counts, color=['skyblue', 'salmon'])
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Complexity')
    ax.grid(alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    savefig(save_path)
    plt.close()

    # Save underlying data
    if save_data:
        plot_data = {
            'models': models,
            'rmse_tumor_burden': {
                'values': rmse_tb,
                'std': rmse_tb_std
            },
            'rmse_all_states': {
                'values': rmse_state,
                'std': rmse_state_std
            },
            'parameter_counts': param_counts,
            'metrics_small': metrics_small,
            'metrics_large': metrics_large
        }
        data_path = str(save_path).replace('.png', '_data.json')
        save_json(plot_data, data_path)


def plot_extrapolation_test(model, test_dataset, device='cpu', save_path=None, save_data=True):
    """Test model extrapolation beyond training window"""
    if save_path is None:
        save_path = FIG_BASE / "neural_ode_extrapolation.png"
    model.eval()
    model = model.to(device)

    # Get first test trajectory
    x0, t_train, x_true_train = test_dataset[0]
    x0 = x0.unsqueeze(0).to(device)
    t_train = t_train.to(device)

    # Extend time to t=8 (training was t=6)
    t_extended = torch.linspace(0, 8.0, 800).to(device)

    with torch.no_grad():
        x_pred_extended = ode_solve(model, x0, t_extended, method='rk4')
        x_pred_extended = x_pred_extended.squeeze(1).cpu().numpy()

    t_extended_np = t_extended.cpu().numpy()
    t_train_np = t_train.cpu().numpy()
    x_true_train_np = x_true_train.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    state_names = ['S (Sensitive)', 'R (Resistant)', 'I (Immune)', 'C (Drug)']
    colors = ['red', 'blue', 'green', 'magenta']

    for i, ax in enumerate(axes.flatten()):
        # Training window
        ax.plot(t_train_np, x_true_train_np[:, i], color=colors[i], linestyle='-',
                linewidth=2.5, label='PDE (Training Window)', alpha=0.8)

        # Neural ODE prediction (full)
        ax.plot(t_extended_np, x_pred_extended[:, i], color=colors[i], linestyle='--',
                linewidth=2, label='Neural ODE (Extended)', alpha=0.9)

        # Mark training window boundary
        ax.axvline(x=6.0, color='black', linestyle=':', linewidth=1.5, label='Training Boundary')

        ax.set_xlabel('Time t', fontsize=11)
        ax.set_ylabel('State Value', fontsize=11)
        ax.set_title(state_names[i], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Extrapolation Test: Prediction Beyond Training Window',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig(save_path)
    plt.close()

    # Save underlying data
    if save_data:
        plot_data = {
            't_training': t_train_np.tolist(),
            't_extended': t_extended_np.tolist(),
            'training_boundary': 6.0,
            'ground_truth_training': {
                'S': x_true_train_np[:, 0].tolist(),
                'R': x_true_train_np[:, 1].tolist(),
                'I': x_true_train_np[:, 2].tolist(),
                'C': x_true_train_np[:, 3].tolist()
            },
            'predictions_extended': {
                'S': x_pred_extended[:, 0].tolist(),
                'R': x_pred_extended[:, 1].tolist(),
                'I': x_pred_extended[:, 2].tolist(),
                'C': x_pred_extended[:, 3].tolist()
            }
        }
        data_path = str(save_path).replace('.png', '_data.json')
        save_json(plot_data, data_path)

# %% [markdown]
# ## Section 8: Main Execution

# %%
def run_node(mode=None, model='small_mlp', epochs=500, batch_size=8, lr=1e-3, ts_type='rk4',
             suffix="", use_timestamp=False):
    """
    Main execution function that works both from command line and notebook.

    Args:
        mode: Execution mode ('generate_data', 'train', 'evaluate', 'visualize')
        model: Model architecture ('small_mlp', 'large_mlp')
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        ts_type: ODE solver type
        suffix: Optional suffix to add to output files (e.g., "run1", "experiment_A")
        use_timestamp: If True, automatically add timestamp to suffix
    """

    # Generate suffix with timestamp if requested
    if use_timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"{suffix}_{timestamp}" if suffix else timestamp

    # Create a simple args object for compatibility
    class Args:
        pass
    args = Args()
    args.mode = mode
    args.model = model
    args.epochs = epochs
    args.batch_size = batch_size
    args.lr = lr
    args.ts_type = ts_type
    args.suffix = suffix

    # ========================================================================
    # Mode: Generate Data
    # ========================================================================
    if args.mode == 'generate_data':
        print("\n" + "="*60)
        print("MODE: Generate Data")
        if args.suffix:
            print(f"Suffix: {args.suffix}")
        print("="*60)

        dataset = generate_neural_ode_dataset(n_trajectories=50, T=6.0, N_grid=64)
        save_dataset(dataset, train_frac=0.8, val_frac=0.1, suffix=args.suffix)

        # Visualize sample trajectories
        train_dataset, _, _ = load_dataset_splits(suffix=args.suffix)
        suffix_str = f"_{args.suffix}" if args.suffix else ""
        plot_data_samples(train_dataset, n_samples=16,
                         save_path=FIG_BASE / f"neural_ode_data_samples{suffix_str}.png")

        print("\n[Done] Data generation complete!")

    # ========================================================================
    # Mode: Train
    # ========================================================================
    elif args.mode == 'train':
        suffix_str = f"_{args.suffix}" if args.suffix else ""
        print("\n" + "="*60)
        print(f"MODE: Train {args.model.upper()}")
        if args.suffix:
            print(f"Suffix: {args.suffix}")
        print("="*60)

        # Load datasets
        train_dataset, val_dataset, _ = load_dataset_splits(noise_level=0.01, suffix=args.suffix)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Create model
        if args.model == 'small_mlp':
            model = SmallMLP_NODE(state_dim=4, hidden_dim=32)
        else:  # large_mlp
            model = LargeMLP_NODE(state_dim=4, hidden_dim=128, dosing_period=5.0)

        print(f"\n[Model] {args.model}")
        print(f"Parameters: {count_parameters(model):,}")

        # Train
        save_path = OUT_BASE / f"neural_ode_models/model_{args.model}{suffix_str}.pt"
        model, history = train_neural_ode(
            model, train_loader, val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_path=save_path,
            early_stopping_patience=50
        )

        # Save training history
        history_df = pd.DataFrame(history)
        history_path = OUT_BASE / f"neural_ode_models/training_history_{args.model}{suffix_str}.csv"
        history_df.to_csv(history_path, index=False)
        print(f"[Saved] Training history: {history_path}")

        print("\n[Done] Training complete!")

    # ========================================================================
    # Mode: Evaluate
    # ========================================================================
    elif args.mode == 'evaluate':
        suffix_str = f"_{args.suffix}" if args.suffix else ""
        print("\n" + "="*60)
        print(f"MODE: Evaluate {args.model.upper()}")
        if args.suffix:
            print(f"Suffix: {args.suffix}")
        print("="*60)

        # Load test dataset
        _, _, test_dataset = load_dataset_splits(suffix=args.suffix)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Load model
        if args.model == 'small_mlp':
            model = SmallMLP_NODE(state_dim=4, hidden_dim=32)
        else:  # large_mlp
            model = LargeMLP_NODE(state_dim=4, hidden_dim=128, dosing_period=5.0)

        model_path = OUT_BASE / f"neural_ode_models/model_{args.model}{suffix_str}.pt"
        model.load_state_dict(torch.load(model_path))
        print(f"[Loaded] Model from {model_path}")

        # Evaluate
        metrics, predictions = evaluate_model(model, test_loader, device=device)

        # Print metrics
        print("\n[Metrics]")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Save metrics
        metrics_path = OUT_BASE / f"neural_ode_results/metrics_{args.model}{suffix_str}.json"
        save_json(metrics, metrics_path)

        # Plot predictions
        plot_path = FIG_BASE / f"neural_ode_test_predictions_{args.model}{suffix_str}.png"
        plot_test_predictions(predictions, args.model.upper(), save_path=plot_path)

        print("\n[Done] Evaluation complete!")

    # ========================================================================
    # Mode: Visualize
    # ========================================================================
    elif args.mode == 'visualize':
        suffix_str = f"_{args.suffix}" if args.suffix else ""
        print("\n" + "="*60)
        print("MODE: Visualize")
        if args.suffix:
            print(f"Suffix: {args.suffix}")
        print("="*60)

        # Load training histories
        try:
            history_small = pd.read_csv(OUT_BASE / f"neural_ode_models/training_history_small_mlp{suffix_str}.csv").to_dict('list')
            history_large = pd.read_csv(OUT_BASE / f"neural_ode_models/training_history_large_mlp{suffix_str}.csv").to_dict('list')
            plot_training_curves(history_small, history_large,
                               save_path=FIG_BASE / f"neural_ode_training_curves{suffix_str}.png")
        except FileNotFoundError:
            print("[Warning] Training histories not found, skipping training curves plot")

        # Load metrics
        try:
            metrics_small = load_json(OUT_BASE / f"neural_ode_results/metrics_small_mlp{suffix_str}.json")
            metrics_large = load_json(OUT_BASE / f"neural_ode_results/metrics_large_mlp{suffix_str}.json")
            plot_architecture_comparison(metrics_small, metrics_large,
                                        save_path=FIG_BASE / f"neural_ode_architecture_comparison{suffix_str}.png")

            # Create summary
            metrics_summary = {
                'small_mlp': metrics_small,
                'large_mlp': metrics_large,
                'suffix': args.suffix
            }
            save_json(metrics_summary, OUT_BASE / f"neural_ode_results/metrics_summary{suffix_str}.json")
        except FileNotFoundError:
            print("[Warning] Metrics not found, skipping comparison plot")

        # Extrapolation test
        try:
            _, _, test_dataset = load_dataset_splits(suffix=args.suffix)

            # Try large MLP first, fall back to small MLP if not available
            large_mlp_path = OUT_BASE / f"neural_ode_models/model_large_mlp{suffix_str}.pt"
            small_mlp_path = OUT_BASE / f"neural_ode_models/model_small_mlp{suffix_str}.pt"

            if large_mlp_path.exists():
                model = LargeMLP_NODE(state_dim=4, hidden_dim=128, dosing_period=5.0)
                model.load_state_dict(torch.load(large_mlp_path))
                model_name = "large_mlp"
                print("[Extrapolation] Using Large MLP model")
            elif small_mlp_path.exists():
                model = SmallMLP_NODE(state_dim=4, hidden_dim=32)
                model.load_state_dict(torch.load(small_mlp_path))
                model_name = "small_mlp"
                print("[Extrapolation] Using Small MLP model (large_mlp not found)")
            else:
                raise FileNotFoundError("No trained models found for extrapolation test")

            plot_extrapolation_test(model, test_dataset, device=device,
                                   save_path=FIG_BASE / f"neural_ode_extrapolation_{model_name}{suffix_str}.png")
        except Exception as e:
            print(f"[Warning] Extrapolation test failed: {e}")

        print("\n[Done] Visualization complete!")
        print("\nGenerated figures (depending on available models):")
        print(f"  - figs/neural_ode_data_samples{suffix_str}.png (if dataset exists)")
        print(f"  - figs/neural_ode_training_curves{suffix_str}.png (if both models trained)")
        print(f"  - figs/neural_ode_test_predictions_small_mlp{suffix_str}.png (if small_mlp evaluated)")
        print(f"  - figs/neural_ode_test_predictions_large_mlp{suffix_str}.png (if large_mlp evaluated)")
        print(f"  - figs/neural_ode_architecture_comparison{suffix_str}.png (if both models evaluated)")
        print(f"  - figs/neural_ode_extrapolation_*{suffix_str}.png (if any model trained)")
        print(f"\nData files (JSON) saved alongside each figure for analysis!")

# %% [markdown]
# ## Notebook Usage Examples
#
# When running in a Jupyter notebook, you can call the run_node function directly with parameters.
#
# ### Using Suffixes to Organize Experiments
#
# You can use the `suffix` parameter to distinguish between different runs:
# - `suffix="run1"` - Manual naming
# - `suffix="experiment_A"` - Descriptive names
# - `use_timestamp=True` - Automatic timestamp (e.g., "20250110_143022")
# - Both: `suffix="exp1", use_timestamp=True` - Combined (e.g., "exp1_20250110_143022")

# %% [markdown]
# ### Example 1: Generate Dataset (with suffix)
# ```python
# # Uncomment to run:
# run_node(mode='generate_data', suffix='run1')
# ```

# %%
# run_node(mode='generate_data', suffix='small_mlp_run')
# run_node(mode='generate_data', suffix='large_mlp_run')

# %% [markdown]
# ### Example 2: Train Both Models (with suffix)
# ```python
# # Uncomment to run:
# # Train small MLP
# run_node(mode='train', model='small_mlp', epochs=500, batch_size=8, lr=1e-3, suffix='run1')
#
# # Train large MLP
# run_node(mode='train', model='large_mlp', epochs=500, batch_size=8, lr=1e-3, suffix='run1')
# ```

# %%
# run_node(mode='train', model='small_mlp', epochs=500, batch_size=128, lr=1e-3, suffix='small_mlp_run')
# run_node(mode='train', model='large_mlp', epochs=1500, batch_size=256, lr=1e-3, suffix='large_mlp_run')

# %% [markdown]
# ### Example 3: Evaluate Both Models
# ```python
# # Uncomment to run:
# run_node(mode='evaluate', model='small_mlp', suffix='run1')
# run_node(mode='evaluate', model='large_mlp', suffix='run1')
# ```

# %%
# run_node(mode='evaluate', model='small_mlp', suffix='small_mlp_run')
# run_node(mode='evaluate', model='large_mlp', suffix='large_mlp_run')

# %% [markdown]
# ### Example 4: Generate Visualizations
# ```python
# # Uncomment to run:
# run_node(mode='visualize', suffix='run1')
# ```

# %%
# run_node(mode='visualize', suffix='small_mlp_run')
# run_node(mode='visualize', suffix='large_mlp_run')

# %% [markdown]
# ### Example 5: Complete Pipeline with Timestamp
# ```python
# # Uncomment to run:
# # This will automatically add timestamp to all outputs
# run_node(mode='generate_data', use_timestamp=True)
# run_node(mode='train', model='small_mlp', epochs=500, use_timestamp=True)
# run_node(mode='train', model='large_mlp', epochs=500, use_timestamp=True)
# run_node(mode='evaluate', model='small_mlp', use_timestamp=True)
# run_node(mode='evaluate', model='large_mlp', use_timestamp=True)
# run_node(mode='visualize', use_timestamp=True)
# ```

# %% [markdown]
# ### Example 6: Compare Different Hyperparameters
# ```python
# # Uncomment to run:
# # Run with different learning rates
# run_node(mode='train', model='small_mlp', epochs=300, lr=1e-3, suffix='lr_1e3')
# run_node(mode='train', model='small_mlp', epochs=300, lr=5e-4, suffix='lr_5e4')
# run_node(mode='train', model='small_mlp', epochs=300, lr=1e-4, suffix='lr_1e4')
#
# # Evaluate each
# run_node(mode='evaluate', model='small_mlp', suffix='lr_1e3')
# run_node(mode='evaluate', model='small_mlp', suffix='lr_5e4')
# run_node(mode='evaluate', model='small_mlp', suffix='lr_1e4')
# ```

# %% [markdown]
# ### Output Files Structure
#
# With suffix enabled, your files will be organized like this:
# ```
# node/out/
#   neural_ode_dataset/
#     train_trajectories_run1.npz
#     val_trajectories_run1.npz
#     test_trajectories_run1.npz
#   neural_ode_models/
#     model_small_mlp_run1.pt
#     model_large_mlp_run1.pt
#     training_history_small_mlp_run1.csv
#     training_history_large_mlp_run1.csv
#   neural_ode_results/
#     metrics_small_mlp_run1.json
#     metrics_large_mlp_run1.json
#
# node/figs/
#   neural_ode_data_samples_run1.png
#   neural_ode_data_samples_run1_data.json  # Raw plot data for analysis
#   neural_ode_training_curves_run1.png
#   neural_ode_training_curves_run1_data.json
#   neural_ode_test_predictions_small_mlp_run1.png
#   neural_ode_test_predictions_small_mlp_run1_data.json
#   ...
# ```


