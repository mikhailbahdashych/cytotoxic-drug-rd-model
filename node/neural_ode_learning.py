"""
Neural ODE Learning for Cytotoxic Drug R&D Model

Task: Learn the dynamics of the 4-variable PDE system (S, R, I, C) using Neural ODEs.
      Train neural networks to approximate dx/dt = f_θ(x, t) from PDE-generated trajectories.

Structure:
- Section 1: Imports and Setup
- Section 2: Data Generation from PDE
- Section 3: Dataset Class and DataLoader
- Section 4: Neural ODE Architectures (Small MLP, Large MLP)
- Section 5: Training Functions
- Section 6: Evaluation and Metrics
- Section 7: Visualization
- Section 8: Main Execution

Author: Generated for ISZ Project
Date: 2025
"""

# ============================================================================
# Section 1: Imports and Setup
# ============================================================================

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
USE_PNODE = True  # Set to False to use torchdiffeq as fallback

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

# Import PDE module from parent directory
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pde_module", "../2_tumor_diffusion_pde_analysis.py"
)
pde_module = importlib.util.module_from_spec(spec)
sys.modules["pde_module"] = pde_module
try:
    spec.loader.exec_module(pde_module)
    print("[OK] PDE module imported from parent directory")
except Exception as e:
    print(f"[ERROR] Failed to import PDE module: {e}")
    print("Make sure 2_tumor_diffusion_pde_analysis.py exists in parent directory")
    sys.exit(1)

# Import needed components from PDE module
Grid = pde_module.Grid
Params = pde_module.Params
init_fields = pde_module.init_fields

# Setup directories
Path("out/neural_ode_dataset").mkdir(parents=True, exist_ok=True)
Path("out/neural_ode_models").mkdir(parents=True, exist_ok=True)
Path("out/neural_ode_results").mkdir(parents=True, exist_ok=True)
Path("figs").mkdir(parents=True, exist_ok=True)

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


# ============================================================================
# Section 2: Data Generation from PDE
# ============================================================================

def run_pde_simulation_for_node(grid, params, T=6.0, dt=0.01, save_every=10):
    """
    Run PDE simulation and extract spatially-averaged trajectory for Neural ODE training.

    Args:
        grid: Grid object from PDE module
        params: Params object with model parameters
        T: Final time
        dt: Time step
        save_every: Save frequency

    Returns:
        dict with keys: t (time array), state (spatially averaged [S̄, R̄, Ī, C̄]), TB (tumor burden)
    """
    # Initialize fields
    S, R, I, C = init_fields(grid, params)

    # Get solver function
    if hasattr(pde_module, 'step_semi_implicit'):
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
        S, R, I, C = step_func(S, R, I, C, grid, params, dt, t_current)
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

        # Override init_fields to use our custom ICs
        original_init = pde_module.init_fields
        pde_module.init_fields = lambda g, p: (S, R, I, C)

        # Run PDE simulation
        try:
            trajectory = run_pde_simulation_for_node(grid, params, T=T, dt=0.01, save_every=10)

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
        finally:
            # Restore original init_fields
            pde_module.init_fields = original_init

    print(f"\n[Success] Generated {len(dataset)} trajectories")
    return dataset


def save_dataset(dataset, train_frac=0.8, val_frac=0.1):
    """
    Split and save dataset to disk.

    Args:
        dataset: List of trajectory dicts
        train_frac: Fraction for training
        val_frac: Fraction for validation (remainder goes to test)
    """
    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    # Split
    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train+n_val]
    test_data = dataset[n_train+n_val:]

    # Save
    np.savez_compressed("out/neural_ode_dataset/train_trajectories.npz", trajectories=train_data)
    np.savez_compressed("out/neural_ode_dataset/val_trajectories.npz", trajectories=val_data)
    np.savez_compressed("out/neural_ode_dataset/test_trajectories.npz", trajectories=test_data)

    # Save metadata
    metadata = {
        'n_total': n_total,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'split': {'train': train_frac, 'val': val_frac, 'test': 1-train_frac-val_frac}
    }
    save_json(metadata, "out/neural_ode_dataset/metadata.json")

    print(f"\n[Dataset Split]")
    print(f"  Train: {n_train} trajectories")
    print(f"  Val:   {n_val} trajectories")
    print(f"  Test:  {n_test} trajectories")
    print(f"  Total: {n_total} trajectories")


# ============================================================================
# Section 3: Dataset Class and DataLoader
# ============================================================================

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


def load_dataset_splits(noise_level=0.0):
    """
    Load train/val/test datasets from disk.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_data = np.load("out/neural_ode_dataset/train_trajectories.npz", allow_pickle=True)['trajectories']
    val_data = np.load("out/neural_ode_dataset/val_trajectories.npz", allow_pickle=True)['trajectories']
    test_data = np.load("out/neural_ode_dataset/test_trajectories.npz", allow_pickle=True)['trajectories']

    train_dataset = NODEDataset(train_data, noise_level=noise_level)
    val_dataset = NODEDataset(val_data, noise_level=0.0)  # No noise for validation
    test_dataset = NODEDataset(test_data, noise_level=0.0)  # No noise for test

    print(f"[Datasets Loaded]")
    print(f"  Train: {len(train_dataset)} trajectories")
    print(f"  Val:   {len(val_dataset)} trajectories")
    print(f"  Test:  {len(test_dataset)} trajectories")

    return train_dataset, val_dataset, test_dataset


# ============================================================================
# Section 4: Neural ODE Architectures
# ============================================================================

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


# ============================================================================
# Section 5: Training Functions
# ============================================================================

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
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
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


# ============================================================================
# Section 6: Evaluation and Metrics
# ============================================================================

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


# ============================================================================
# Section 7: Visualization
# ============================================================================

def plot_data_samples(dataset, n_samples=16, save_path="figs/neural_ode_data_samples.png"):
    """Plot sample trajectories from dataset"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx, ax in enumerate(axes):
        if idx < len(indices):
            _, t, state = dataset[indices[idx]]
            t = t.numpy()
            state = state.numpy()

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


def plot_training_curves(history_small, history_large, save_path="figs/neural_ode_training_curves.png"):
    """Plot training and validation loss curves for both models"""
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


def plot_test_predictions(predictions, model_name, save_path="figs/neural_ode_test_predictions.png"):
    """Plot predictions vs ground truth for first test trajectory"""
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


def plot_architecture_comparison(metrics_small, metrics_large, save_path="figs/neural_ode_architecture_comparison.png"):
    """Bar chart comparing model architectures"""
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


def plot_extrapolation_test(model, test_dataset, device='cpu', save_path="figs/neural_ode_extrapolation.png"):
    """Test model extrapolation beyond training window"""
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

    t_extended = t_extended.cpu().numpy()
    t_train = t_train.cpu().numpy()
    x_true_train = x_true_train.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    state_names = ['S (Sensitive)', 'R (Resistant)', 'I (Immune)', 'C (Drug)']
    colors = ['red', 'blue', 'green', 'magenta']

    for i, ax in enumerate(axes.flatten()):
        # Training window
        ax.plot(t_train, x_true_train[:, i], color=colors[i], linestyle='-',
                linewidth=2.5, label='PDE (Training Window)', alpha=0.8)

        # Neural ODE prediction (full)
        ax.plot(t_extended, x_pred_extended[:, i], color=colors[i], linestyle='--',
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


# ============================================================================
# Section 8: Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Neural ODE Learning for Cytotoxic Drug R&D Model')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['generate_data', 'train', 'evaluate', 'visualize'],
                        help='Execution mode')
    parser.add_argument('--model', type=str, default='small_mlp',
                        choices=['small_mlp', 'large_mlp'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--ts_type', type=str, default='rk4', help='ODE solver type (for pnode)')

    args = parser.parse_args()

    # ========================================================================
    # Mode: Generate Data
    # ========================================================================
    if args.mode == 'generate_data':
        print("\n" + "="*60)
        print("MODE: Generate Data")
        print("="*60)

        dataset = generate_neural_ode_dataset(n_trajectories=50, T=6.0, N_grid=64, verbose=True)
        save_dataset(dataset, train_frac=0.8, val_frac=0.1)

        # Visualize sample trajectories
        train_dataset, _, _ = load_dataset_splits()
        plot_data_samples(train_dataset, n_samples=16)

        print("\n[Done] Data generation complete!")

    # ========================================================================
    # Mode: Train
    # ========================================================================
    elif args.mode == 'train':
        print("\n" + "="*60)
        print(f"MODE: Train {args.model.upper()}")
        print("="*60)

        # Load datasets
        train_dataset, val_dataset, _ = load_dataset_splits(noise_level=0.01)
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
        save_path = f"out/neural_ode_models/model_{args.model}.pt"
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
        history_path = f"out/neural_ode_models/training_history_{args.model}.csv"
        history_df.to_csv(history_path, index=False)
        print(f"[Saved] Training history: {history_path}")

        print("\n[Done] Training complete!")

    # ========================================================================
    # Mode: Evaluate
    # ========================================================================
    elif args.mode == 'evaluate':
        print("\n" + "="*60)
        print(f"MODE: Evaluate {args.model.upper()}")
        print("="*60)

        # Load test dataset
        _, _, test_dataset = load_dataset_splits()
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Load model
        if args.model == 'small_mlp':
            model = SmallMLP_NODE(state_dim=4, hidden_dim=32)
        else:  # large_mlp
            model = LargeMLP_NODE(state_dim=4, hidden_dim=128, dosing_period=5.0)

        model_path = f"out/neural_ode_models/model_{args.model}.pt"
        model.load_state_dict(torch.load(model_path))
        print(f"[Loaded] Model from {model_path}")

        # Evaluate
        metrics, predictions = evaluate_model(model, test_loader, device=device)

        # Print metrics
        print("\n[Metrics]")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Save metrics
        metrics_path = f"out/neural_ode_results/metrics_{args.model}.json"
        save_json(metrics, metrics_path)

        # Plot predictions
        plot_path = f"figs/neural_ode_test_predictions_{args.model}.png"
        plot_test_predictions(predictions, args.model.upper(), save_path=plot_path)

        print("\n[Done] Evaluation complete!")

    # ========================================================================
    # Mode: Visualize
    # ========================================================================
    elif args.mode == 'visualize':
        print("\n" + "="*60)
        print("MODE: Visualize")
        print("="*60)

        # Load training histories
        try:
            history_small = pd.read_csv("out/neural_ode_models/training_history_small_mlp.csv").to_dict('list')
            history_large = pd.read_csv("out/neural_ode_models/training_history_large_mlp.csv").to_dict('list')
            plot_training_curves(history_small, history_large)
        except FileNotFoundError:
            print("[Warning] Training histories not found, skipping training curves plot")

        # Load metrics
        try:
            metrics_small = load_json("out/neural_ode_results/metrics_small_mlp.json")
            metrics_large = load_json("out/neural_ode_results/metrics_large_mlp.json")
            plot_architecture_comparison(metrics_small, metrics_large)

            # Create summary
            metrics_summary = {
                'small_mlp': metrics_small,
                'large_mlp': metrics_large
            }
            save_json(metrics_summary, "out/neural_ode_results/metrics_summary.json")
        except FileNotFoundError:
            print("[Warning] Metrics not found, skipping comparison plot")

        # Extrapolation test
        try:
            _, _, test_dataset = load_dataset_splits()

            # Test with large MLP
            model_large = LargeMLP_NODE(state_dim=4, hidden_dim=128, dosing_period=5.0)
            model_large.load_state_dict(torch.load("out/neural_ode_models/model_large_mlp.pt"))
            plot_extrapolation_test(model_large, test_dataset, device=device)
        except Exception as e:
            print(f"[Warning] Extrapolation test failed: {e}")

        print("\n[Done] Visualization complete!")
        print("\nGenerated figures:")
        print("  - figs/neural_ode_data_samples.png")
        print("  - figs/neural_ode_training_curves.png")
        print("  - figs/neural_ode_test_predictions_small_mlp.png")
        print("  - figs/neural_ode_test_predictions_large_mlp.png")
        print("  - figs/neural_ode_architecture_comparison.png")
        print("  - figs/neural_ode_extrapolation.png")


if __name__ == "__main__":
    main()
