# %% [markdown]
# # Zadanie 7 — SuperNet i Supermodel
#
# Cel:
# 1) Stworzyć **Supermodel** — hybrydowy model ODE z neuronową korektą:
#    ẏ(t) = f_ODE(y; θ_calib) + g_φ(y, t)
# 2) Stworzyć **SuperNet** — rozszerzenie PINN na rodzinę parametrów terapii:
#    wejście: (x, y, t, p_dose), wyjście: [Ŝ, R̂, Î, Ĉ]
# 3) Porównać predykcje obu modeli z PDE i ODE.
# 4) Zapisać wykresy do figs/, metryki do out/.

# %%
import os, json, math, time, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import copy

# Katalogi
Path("figs").mkdir(exist_ok=True)
Path("out").mkdir(exist_ok=True)

def savefig_root(name, dpi=160):
    """Zapisuje wykres w katalogu głównym projektu"""
    plt.savefig(name, dpi=dpi, bbox_inches="tight")
    print(f"[Zapisano wykres] {name}")

def savefig_fig(name, dpi=160):
    """Zapisuje wykres w figs/"""
    if not str(name).startswith("figs/"):
        name = f"figs/{name}"
    plt.savefig(name, dpi=dpi, bbox_inches="tight")
    print(f"[Zapisano wykres] {name}")

def save_json(obj, path):
    """Zapisuje JSON do out/"""
    if not str(path).startswith("out/"):
        path = f"out/{path}"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[Zapisano JSON] {path}")

def save_csv(df, path):
    """Zapisuje CSV do out/"""
    if not str(path).startswith("out/"):
        path = f"out/{path}"
    df.to_csv(path, index=False)
    print(f"[Zapisano CSV] {path}")

# Metryki
def rmse(pred, true):
    return np.sqrt(np.mean((pred - true)**2))

def mae(pred, true):
    return np.mean(np.abs(pred - true))

# Reprodukowalność
np.random.seed(123)
random.seed(123)

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Device] Używam: {device}")

# tqdm dla progress bars
try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[INFO] tqdm niedostępny - progress bars wyłączone")

# SciPy dla integracji ODE
try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
    print("[OK] SciPy dostępny")
except Exception as e:
    SCIPY_AVAILABLE = False
    print("Uwaga: SciPy niedostępny:", e)

# %% [markdown]
# ## 1. Importy modułów PDE i ODE
#
# Importujemy istniejące moduły z poprzednich zadań.

# %%
import importlib.util

# --- Import modułu PDE (Zadanie 2) ---
PDE_OK = True
try:
    candidates = ["2_tumor_diffusion_pde_analysis.py", "tumor_diffusion_pde_analysis.py"]
    pdemod = None
    for cand in candidates:
        if Path(cand).exists():
            spec = importlib.util.spec_from_file_location("pdemod", cand)
            pdemod = importlib.util.module_from_spec(spec)
            sys.modules["pdemod"] = pdemod
            spec.loader.exec_module(pdemod)
            break
    if pdemod is None:
        raise FileNotFoundError("Nie znaleziono pliku modułu PDE.")

    # Pobierz obiekty z modułu PDE
    Grid = pdemod.Grid
    Params = pdemod.Params
    run_simulation = pdemod.run_simulation
    init_fields = pdemod.init_fields
    tumor_burden = pdemod.tumor_burden
    p_pde_base = pdemod.p  # bazowe parametry PDE

    print("[OK] Moduł PDE załadowany.")
except Exception as e:
    PDE_OK = False
    print("Uwaga: nie udało się załadować modułu PDE:", e)

# %% [markdown]
# ## 2. Definicja parametrów ODE
#
# Używamy tych samych parametrów co w Zadaniu 3.

# %%
@dataclass
class ODEParams:
    # Wzrost logistyczny
    rho_S: float = 0.04
    rho_R: float = 0.03
    K: float = 1.0

    # Cytotoksyczność
    alpha_S: float = 0.8
    alpha_R: float = 0.12

    # Immunologia
    sigma: float = 0.05
    delta: float = 0.1
    gamma_S: float = 0.02
    gamma_R: float = 0.02

    # Farmakokinetyka leku
    lam: float = 0.2
    beta: float = 0.0

    # Indukcja oporności
    mu_max: float = 0.05
    C50: float = 0.2
    m_hill: int = 3

    # Dawkowanie
    dose_type: str = "infusion_const"  # dla tego zadania używamy infuzji
    dose_A: float = 1.0
    dose_period: float = 5.0
    infusion_rate: float = 0.15

def mu_of_C(C, mu_max, C50, m):
    """Funkcja Hill dla indukcji oporności"""
    C_nonneg = np.maximum(C, 0.0)
    ratio = np.power(C_nonneg / (C50 + 1e-12), m)
    return mu_max * (ratio / (1.0 + ratio))

def dosing_term_exact(t, dt, period, A):
    """Bolus jako krótki impuls"""
    tau = 0.01 * period
    phase = t % period
    return A / tau if phase < tau else 0.0

def ode_rhs(t, y, p: ODEParams, dt_for_dose=None):
    """
    Prawa strona ODE: dy/dt = f(t, y; p)
    y = [S, R, I, C]
    """
    S, R, I, C = y
    N = S + R

    # Wzrost logistyczny
    dS = p.rho_S * S * (1 - N/p.K)
    dR = p.rho_R * R * (1 - N/p.K)

    # Zabijanie lekiem i immunologią
    dS -= p.alpha_S * C * S + p.gamma_S * I * S
    dR -= p.alpha_R * C * R + p.gamma_R * I * R

    # Indukcja oporności
    mu = mu_of_C(C, p.mu_max, p.C50, p.m_hill)
    dS -= mu * S
    dR += mu * S

    # Immunologia
    dI = p.sigma * N - p.delta * I

    # Lek
    dC = -p.lam * C - p.beta * C * N
    if p.dose_type == "infusion_const":
        I_in = p.infusion_rate
    elif p.dose_type == "bolus_periodic":
        I_in = dosing_term_exact(t, dt_for_dose if dt_for_dose is not None else 1e-2,
                                 p.dose_period, p.dose_A)
    else:
        I_in = 0.0
    dC += I_in

    return np.array([dS, dR, dI, dC])

def simulate_ode(p: ODEParams, y0, t_grid):
    """
    Symulacja ODE na siatce czasowej t_grid.
    Zwraca słownik z polami: t, S, R, I, C, TB.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy wymagany do symulacji ODE")

    dt_effective = np.mean(np.diff(t_grid)) if len(t_grid) > 1 else 1e-2

    def fun(t, y):
        return ode_rhs(t, y, p, dt_for_dose=dt_effective)

    sol = solve_ivp(fun, (t_grid[0], t_grid[-1]), y0, t_eval=t_grid,
                   rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(sol.message)

    Y = sol.y.T  # shape: [T, 4]
    S, R, I, C = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
    TB = S + R

    return {"t": t_grid, "S": S, "R": R, "I": I, "C": C, "TB": TB}

# %% [markdown]
# ## 3. Generowanie/wczytanie danych referencyjnych PDE
#
# TODO: Dostosuj do swoich istniejących danych lub użyj symulacji PDE.

# %%
def load_or_generate_pde_reference(T=6.0, N=64, solver="semi_implicit"):
    """
    Wczytuje lub generuje trajektorię TB(t) z PDE.

    TODO: Jeśli masz zapisane dane z wcześniejszych zadań (np. po asymilacji),
    wczytaj je tutaj. W przeciwnym razie uruchom symulację PDE.

    Zwraca: dict {"t": array, "TB": array, "S_final": array, "R_final": array, ...}
    """
    # Próba wczytania z istniejącego pliku
    possible_files = [
        "out/pinn_reference_pde.csv",
        "out/assimilation_pde_trajectory.csv",
        "out/compare_solvers_same_dt.csv",
    ]

    for fname in possible_files:
        if Path(fname).exists():
            df = pd.read_csv(fname)
            # Sprawdź format
            if "t" in df.columns and "tumor_burden" in df.columns:
                print(f"[OK] Wczytano PDE z: {fname}")
                return {"t": df["t"].values, "TB": df["tumor_burden"].values}
            elif "t" in df.columns and "TB_explicit" in df.columns:
                print(f"[OK] Wczytano PDE z: {fname}")
                return {"t": df["t"].values, "TB": df["TB_explicit"].values}

    # Jeśli nie znaleziono, wygeneruj nową symulację
    if PDE_OK:
        print("[INFO] Generuję nową symulację PDE...")
        grid = Grid(Nx=N, Ny=N)
        p = copy.deepcopy(p_pde_base)
        p.dose_type = "infusion_const"
        p.infusion_rate = 0.15

        dt = 0.01

        (S_final, R_final, I_final, C_final), traj, info = run_simulation(solver, grid, p, T=T, dt=dt, save_every=50)

        t_arr = np.array([rec["t"] for rec in traj])
        TB_arr = np.array([rec["tumor_burden"] for rec in traj])

        # Zapisz do pliku
        df_ref = pd.DataFrame({"t": t_arr, "tumor_burden": TB_arr})
        save_csv(df_ref, "supermodel_pde_reference.csv")

        return {
            "t": t_arr,
            "TB": TB_arr,
            "S_final": S_final,
            "R_final": R_final,
            "I_final": I_final,
            "C_final": C_final
        }
    else:
        raise RuntimeError("Nie można wczytać ani wygenerować danych PDE")

pde_ref = load_or_generate_pde_reference(T=6.0, N=64)
t_pde = pde_ref["t"]
TB_pde = pde_ref["TB"]

print(f"[OK] Dane PDE: {len(t_pde)} punktów czasowych, t ∈ [{t_pde[0]:.2f}, {t_pde[-1]:.2f}]")

# %% [markdown]
# ## 4. Symulacja ODE z kalibrowanymi parametrami
#
# Używamy parametrów ODE dopasowanych w Zadaniu 3/5.

# %%
# TODO: Jeśli masz skalibrowane parametry z Zadania 5 (asymilacja),
# wczytaj je tutaj. W przeciwnym razie użyj wartości domyślnych.

# Próba wczytania z JSON
calib_file = "out/assimilation_abc_ode_params.json"
if Path(calib_file).exists():
    with open(calib_file, "r") as f:
        calib_data = json.load(f)
    p_ode = ODEParams(**calib_data)
    print(f"[OK] Wczytano skalibrowane parametry ODE z {calib_file}")
else:
    p_ode = ODEParams()
    print("[INFO] Używam domyślnych parametrów ODE")

# Warunki początkowe ODE
y0_ode = np.array([0.8, 0.2, 0.0, 0.0])  # [S, R, I, C]

# Symulacja ODE
ode_traj = simulate_ode(p_ode, y0_ode, t_pde)
TB_ode = ode_traj["TB"]

print(f"[OK] Symulacja ODE zakończona")
print(f"     RMSE(ODE vs PDE): {rmse(TB_ode, TB_pde):.6f}")
print(f"     MAE(ODE vs PDE):  {mae(TB_ode, TB_pde):.6f}")

# %% [markdown]
# ## 5. SUPERMODEL — Definicja
#
# Supermodel = ODE + neuronowa korekta g_φ(y, t)
#
# ẏ(t) = f_ODE(y; θ_calib) + g_φ(y, t)

# %%
class CorrectionNetwork(nn.Module):
    """
    Mała sieć neuronowa do korekcji ODE.
    Wejście: [S, R, I, C, t] (5-wymiarowy)
    Wyjście: [dS, dR, dI, dC] (korekta dla każdej zmiennej)
    """
    def __init__(self, state_dim=4, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []

        # Wejście: stan + czas
        input_dim = state_dim + 1
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Warstwy ukryte
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Wyjście: korekta dla każdej zmiennej stanu
        layers.append(nn.Linear(hidden_dim, state_dim))

        self.net = nn.Sequential(*layers)

        # Inicjalizacja wagami o małej skali
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, y, t):
        """
        y: [batch, 4] - stan [S, R, I, C]
        t: [batch, 1] - czas
        Returns: [batch, 4] - korekta
        """
        inp = torch.cat([y, t], dim=1)
        return self.net(inp)

# %% [markdown]
# ## 6. SUPERMODEL — Integrator ODE w PyTorch

# %%
class SupermodelIntegrator:
    """
    Integruje ODE z korektą neuronową używając metody Eulera/RK4.
    """
    def __init__(self, ode_params: ODEParams, correction_net: CorrectionNetwork,
                 device="cpu", method="euler"):
        self.p = ode_params
        self.g_net = correction_net
        self.device = device
        self.method = method

    def ode_rhs_torch(self, t_val, y_tensor):
        """
        Prawa strona ODE w PyTorch (bez korekty).
        t_val: skalar (float)
        y_tensor: [batch, 4]
        Returns: [batch, 4]
        """
        S = y_tensor[:, 0:1]
        R = y_tensor[:, 1:2]
        I = y_tensor[:, 2:3]
        C = y_tensor[:, 3:4]

        N = S + R

        # Wzrost
        dS = self.p.rho_S * S * (1 - N / self.p.K)
        dR = self.p.rho_R * R * (1 - N / self.p.K)

        # Cytotoksyczność
        dS = dS - self.p.alpha_S * C * S - self.p.gamma_S * I * S
        dR = dR - self.p.alpha_R * C * R - self.p.gamma_R * I * R

        # Oporność (funkcja Hill)
        C_np = C.detach().cpu().numpy()
        mu_np = mu_of_C(C_np, self.p.mu_max, self.p.C50, self.p.m_hill)
        mu = torch.tensor(mu_np, dtype=torch.float32, device=self.device)

        dS = dS - mu * S
        dR = dR + mu * S

        # Immunologia
        dI = self.p.sigma * N - self.p.delta * I

        # Lek
        dC = -self.p.lam * C - self.p.beta * C * N

        # Dawkowanie
        if self.p.dose_type == "infusion_const":
            dC = dC + self.p.infusion_rate
        elif self.p.dose_type == "bolus_periodic":
            I_in = dosing_term_exact(t_val, 0.01, self.p.dose_period, self.p.dose_A)
            dC = dC + I_in

        return torch.cat([dS, dR, dI, dC], dim=1)

    def rhs_with_correction(self, t_val, y_tensor):
        """
        Pełna prawa strona: ODE + korekta neuronowa.
        """
        f_ode = self.ode_rhs_torch(t_val, y_tensor)

        t_tensor = torch.full((y_tensor.shape[0], 1), t_val,
                             dtype=torch.float32, device=self.device)
        g_corr = self.g_net(y_tensor, t_tensor)

        return f_ode + g_corr

    def integrate(self, y0, t_grid):
        """
        Integruje Supermodel na siatce czasowej t_grid.
        y0: [4] - warunek początkowy
        t_grid: array of time points
        Returns: [len(t_grid), 4]
        """
        y0_tensor = torch.tensor(y0, dtype=torch.float32, device=self.device).unsqueeze(0)

        trajectory = [y0_tensor.detach().cpu().numpy()[0]]
        y_current = y0_tensor

        for i in range(len(t_grid) - 1):
            t = t_grid[i]
            dt = t_grid[i + 1] - t

            if self.method == "euler":
                # Metoda Eulera
                dydt = self.rhs_with_correction(t, y_current)
                y_next = y_current + dt * dydt
            elif self.method == "rk4":
                # Runge-Kutta 4. rzędu
                k1 = self.rhs_with_correction(t, y_current)
                k2 = self.rhs_with_correction(t + 0.5*dt, y_current + 0.5*dt*k1)
                k3 = self.rhs_with_correction(t + 0.5*dt, y_current + 0.5*dt*k2)
                k4 = self.rhs_with_correction(t + dt, y_current + dt*k3)
                y_next = y_current + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Clipowanie do wartości nieujemnych
            y_next = torch.clamp(y_next, min=0.0)

            y_current = y_next
            trajectory.append(y_current.detach().cpu().numpy()[0])

        return np.array(trajectory)

# %% [markdown]
# ## 7. SUPERMODEL — Trenowanie

# %%
def train_supermodel(correction_net, p_ode, y0, t_target, TB_target,
                    epochs=3000, lr=1e-3, reg_weight=0.0, device="cpu"):
    """
    Trenuje sieć korekcyjną g_φ tak, aby TB_supermodel(t) ≈ TB_target(t).

    Parametry:
    - correction_net: instancja CorrectionNetwork
    - p_ode: parametry ODE
    - y0: warunek początkowy [4]
    - t_target: array czasów
    - TB_target: array wartości TB(t) z PDE
    - epochs: liczba epok treningu
    - lr: learning rate
    - reg_weight: waga regularyzacji (kara za duże korekcje)
    - device: "cuda" lub "cpu"

    Zwraca:
    - trained_net: wytrenowana sieć
    - losses: historia strat
    """
    correction_net = correction_net.to(device)
    optimizer = optim.Adam(correction_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

    integrator = SupermodelIntegrator(p_ode, correction_net, device=device, method="rk4")

    TB_target_tensor = torch.tensor(TB_target, dtype=torch.float32, device=device)

    losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 500

    print("[Supermodel] Start treningu...")

    # Progress bar
    epoch_iter = trange(epochs, desc="Supermodel") if TQDM_AVAILABLE else range(epochs)

    for epoch in epoch_iter:
        optimizer.zero_grad()

        # Integracja
        traj = integrator.integrate(y0, t_target)
        TB_pred = traj[:, 0] + traj[:, 1]  # S + R
        TB_pred_tensor = torch.tensor(TB_pred, dtype=torch.float32, device=device)

        # Strata MSE
        loss_mse = torch.mean((TB_pred_tensor - TB_target_tensor)**2)

        # Regularyzacja (opcjonalnie)
        loss_reg = 0.0
        if reg_weight > 0.0:
            for param in correction_net.parameters():
                loss_reg += torch.sum(param**2)
            loss_reg = reg_weight * loss_reg

        loss = loss_mse + loss_reg

        loss.backward()
        optimizer.step()

        # Fix warning: detach loss before passing to scheduler
        scheduler.step(loss.detach())

        loss_val = loss.item()
        losses.append(loss_val)

        # Early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            if TQDM_AVAILABLE:
                epoch_iter.close()
            print(f"[Supermodel] Early stopping at epoch {epoch}")
            break

        # Update progress bar
        if TQDM_AVAILABLE:
            epoch_iter.set_postfix({"loss": f"{loss_val:.6f}", "mse": f"{loss_mse.item():.6f}"})
        elif (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_val:.6f}, "
                  f"MSE: {loss_mse.item():.6f}")

    print(f"[Supermodel] Trening zakończony. Finalna strata: {losses[-1]:.6f}")

    return correction_net, losses

# Inicjalizacja sieci korekcyjnej
correction_net = CorrectionNetwork(state_dim=4, hidden_dim=64, num_layers=2)
print(f"[Supermodel] Sieć korekcyjna: {sum(p.numel() for p in correction_net.parameters())} parametrów")

# Trenowanie
correction_net, train_losses = train_supermodel(
    correction_net, p_ode, y0_ode, t_pde, TB_pde,
    epochs=3000, lr=1e-3, reg_weight=1e-6, device=device
)

# Zapisz model
torch.save(correction_net.state_dict(), "out/supermodel_correction_net.pt")
print("[OK] Model Supermodel zapisany do out/supermodel_correction_net.pt")

# %% [markdown]
# ## 8. SUPERMODEL — Ewaluacja

# %%
# Symulacja Supermodel
integrator = SupermodelIntegrator(p_ode, correction_net, device=device, method="rk4")
traj_supermodel = integrator.integrate(y0_ode, t_pde)
TB_supermodel = traj_supermodel[:, 0] + traj_supermodel[:, 1]

# Metryki
metrics_supermodel = {
    "ODE_vs_PDE": {
        "RMSE": float(rmse(TB_ode, TB_pde)),
        "MAE": float(mae(TB_ode, TB_pde))
    },
    "Supermodel_vs_PDE": {
        "RMSE": float(rmse(TB_supermodel, TB_pde)),
        "MAE": float(mae(TB_supermodel, TB_pde))
    },
    "Supermodel_vs_ODE": {
        "RMSE": float(rmse(TB_supermodel, TB_ode)),
        "MAE": float(mae(TB_supermodel, TB_ode))
    }
}

save_json(metrics_supermodel, "supermodel_metrics.json")

# Zapisz trajektorie
df_traj = pd.DataFrame({
    "t": t_pde,
    "TB_pde": TB_pde,
    "TB_ode": TB_ode,
    "TB_supermodel": TB_supermodel
})
save_csv(df_traj, "supermodel_tb_curves.csv")

# Wykres
plt.figure(figsize=(8, 5))
plt.plot(t_pde, TB_pde, 'k-', lw=2, label="PDE (reference)")
plt.plot(t_pde, TB_ode, 'b--', lw=2, label="ODE (calibrated)")
plt.plot(t_pde, TB_supermodel, 'r-', lw=2, alpha=0.8, label="Supermodel (ODE + NN correction)")
plt.xlabel("Czas t", fontsize=12)
plt.ylabel("Tumor Burden TB(t)", fontsize=12)
plt.title("Porównanie: PDE vs ODE vs Supermodel", fontsize=13)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
savefig_fig("supermodel_tb_compare.png")
plt.show()

print("\n[Supermodel] Metryki:")
for model, metrics in metrics_supermodel.items():
    print(f"  {model}: RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}")

# %% [markdown]
# ## 9. SUPERNET — Definicja
#
# SuperNet rozszerza PINN o dodatkowy parametr terapii p_dose.
# Wejście: (x, y, t, p_dose)
# Wyjście: [Ŝ, R̂, Î, Ĉ]

# %%
class InputNormalizer:
    """Normalizacja wejść do sieci"""
    def __init__(self, x_range=(0, 1), y_range=(0, 1), t_range=(0, 6), p_range=(0.5, 1.5)):
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.p_range = p_range

    def normalize(self, x, y, t, p):
        """Normalizuje wszystkie wejścia do [-1, 1]"""
        x_n = 2 * (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) - 1
        y_n = 2 * (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) - 1
        t_n = 2 * (t - self.t_range[0]) / (self.t_range[1] - self.t_range[0]) - 1
        p_n = 2 * (p - self.p_range[0]) / (self.p_range[1] - self.p_range[0]) - 1
        return x_n, y_n, t_n, p_n

# Skale wyjściowe
OUT_SCALE = {"S": 1.0, "R": 1.0, "I": 0.5, "C": 0.5}

class SuperNet(nn.Module):
    """
    Rozszerzony PINN z dodatkowym parametrem p_dose.
    Wejście: (x, y, t, p_dose)
    Wyjście: [S, R, I, C]
    """
    def __init__(self, in_dim=4, out_dim=4, width=128, depth=6):
        super().__init__()
        self.normalizer = InputNormalizer()

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x, y, t, p):
        """
        x, y, t, p: tensory [N, 1]
        Returns: S, R, I, C - każde [N, 1]
        """
        xn, yn, tn, pn = self.normalizer.normalize(x, y, t, p)
        inp = torch.cat([xn, yn, tn, pn], dim=1)

        raw = self.net(inp)

        # Softplus dla nieujemności
        S = self.softplus(raw[:, 0:1]) * OUT_SCALE["S"]
        R = self.softplus(raw[:, 1:2]) * OUT_SCALE["R"]
        I = self.softplus(raw[:, 2:3]) * OUT_SCALE["I"]
        C = self.softplus(raw[:, 3:4]) * OUT_SCALE["C"]

        return S, R, I, C

# %% [markdown]
# ## 10. SUPERNET — Generowanie danych treningowych
#
# Generujemy dane dla kilku wartości p_dose: {0.5, 1.0, 1.5}
# Każda wartość odpowiada innej intensywności terapii.

# %%
def generate_pde_data_for_dose(p_dose, T=6.0, N=64, base_infusion=0.15):
    """
    Generuje dane PDE dla danej wartości p_dose.
    p_dose - skalar skalujący infusion_rate (0.5 = słaba, 1.0 = nominalna, 1.5 = silna)

    Zwraca: dict {"t": array, "TB": array, "snapshots": list of dicts}
    """
    if not PDE_OK:
        raise RuntimeError("Moduł PDE nie jest dostępny")

    grid = Grid(Nx=N, Ny=N)
    p = copy.deepcopy(p_pde_base)
    p.dose_type = "infusion_const"
    p.infusion_rate = base_infusion * p_dose

    dt = 0.01

    (S_final, R_final, I_final, C_final), traj, info = run_simulation("semi_implicit", grid, p, T=T, dt=dt, save_every=50)

    t_arr = np.array([rec["t"] for rec in traj])
    TB_arr = np.array([rec["tumor_burden"] for rec in traj])

    # Snapshoty w wybranych czasach (np. t=2, 4, 6)
    snapshot_times = [2.0, 4.0, 6.0]
    snapshots = []

    for t_snap in snapshot_times:
        # Znajdź najbliższy punkt czasowy
        idx = np.argmin(np.abs(t_arr - t_snap))

        # Wygeneruj ponownie do tego czasu (lub użyj zapisanych pól)
        # W uproszczeniu użyjemy finalnych pól dla t=6
        if abs(t_arr[idx] - T) < 0.1:
            snapshots.append({
                "t": T,
                "S": S_final.copy(),
                "R": R_final.copy(),
                "I": I_final.copy(),
                "C": C_final.copy()
            })

    return {
        "t": t_arr,
        "TB": TB_arr,
        "snapshots": snapshots,
        "grid": grid
    }

# Generowanie danych dla różnych dawek
dose_scenarios = [0.5, 1.0, 1.5]
pde_data_scenarios = {}

print("\n[SuperNet] Generowanie danych PDE dla różnych scenariuszy...")
for p_dose in dose_scenarios:
    print(f"  Scenariusz p_dose = {p_dose}...")
    pde_data_scenarios[p_dose] = generate_pde_data_for_dose(p_dose, T=6.0, N=64)
    print(f"    -> TB(T={pde_data_scenarios[p_dose]['t'][-1]:.2f}) = "
          f"{pde_data_scenarios[p_dose]['TB'][-1]:.4f}")

# %% [markdown]
# ## 11. SUPERNET — Przygotowanie danych treningowych

# %%
def prepare_supernet_training_data(pde_data_scenarios, n_collocation=2000, n_ic=500, n_bc=500):
    """
    Przygotowuje dane treningowe dla SuperNet.

    Zwraca:
    - collocation_points: punkty (x,y,t,p) do residual loss
    - ic_data: warunki początkowe (x,y,0,p) -> wartości pól
    - bc_data: warunki brzegowe
    - tb_data: trajektorie TB(t,p)
    """
    # Punkty kolokacyjne (równomiernie w (x,y,t,p))
    grid_sample = pde_data_scenarios[1.0]["grid"]  # użyj siatki nominalnej

    collocation = []
    for p_dose in dose_scenarios:
        for _ in range(n_collocation // len(dose_scenarios)):
            x = np.random.uniform(0, grid_sample.Lx)
            y = np.random.uniform(0, grid_sample.Ly)
            t = np.random.uniform(0, 6.0)
            collocation.append([x, y, t, p_dose])

    collocation = np.array(collocation)

    # Warunki początkowe (t=0)
    ic_data = {"points": [], "values": []}
    for p_dose in dose_scenarios:
        grid = pde_data_scenarios[p_dose]["grid"]
        X, Y = grid.X, grid.Y

        # Losowo wybierz n_ic punktów
        idx = np.random.choice(X.size, size=n_ic // len(dose_scenarios), replace=False)
        x_ic = X.ravel()[idx]
        y_ic = Y.ravel()[idx]
        t_ic = np.zeros_like(x_ic)
        p_ic = np.full_like(x_ic, p_dose)

        # TODO: Wartości początkowe - tutaj używamy uproszczonej inicjalizacji
        # W pełnej wersji użyj init_fields z modułu PDE
        S_ic = 0.8 * np.exp(-((x_ic - 0.5)**2 + (y_ic - 0.5)**2) / 0.04)
        R_ic = 0.2 * np.exp(-((x_ic - 0.5)**2 + (y_ic - 0.5)**2) / 0.04)
        I_ic = np.zeros_like(x_ic)
        C_ic = np.zeros_like(x_ic)

        for i in range(len(x_ic)):
            ic_data["points"].append([x_ic[i], y_ic[i], t_ic[i], p_ic[i]])
            ic_data["values"].append([S_ic[i], R_ic[i], I_ic[i], C_ic[i]])

    ic_data["points"] = np.array(ic_data["points"])
    ic_data["values"] = np.array(ic_data["values"])

    # Trajektorie TB(t)
    tb_data = {"t": [], "p": [], "TB": []}
    for p_dose in dose_scenarios:
        t_arr = pde_data_scenarios[p_dose]["t"]
        TB_arr = pde_data_scenarios[p_dose]["TB"]

        for i in range(len(t_arr)):
            tb_data["t"].append(t_arr[i])
            tb_data["p"].append(p_dose)
            tb_data["TB"].append(TB_arr[i])

    tb_data["t"] = np.array(tb_data["t"])
    tb_data["p"] = np.array(tb_data["p"])
    tb_data["TB"] = np.array(tb_data["TB"])

    return {
        "collocation": collocation,
        "ic": ic_data,
        "tb": tb_data
    }

training_data = prepare_supernet_training_data(pde_data_scenarios)
print(f"\n[SuperNet] Dane treningowe przygotowane:")
print(f"  Punkty kolokacyjne: {len(training_data['collocation'])}")
print(f"  Warunki początkowe: {len(training_data['ic']['points'])}")
print(f"  Punkty TB(t): {len(training_data['tb']['t'])}")

# %% [markdown]
# ## 12. SUPERNET — Funkcje straty (PDE residuals)

# %%
def compute_pde_residuals(supernet, x, y, t, p, params):
    """
    Oblicza residua równań PDE.
    Zwraca: res_S, res_R, res_I, res_C (każde [N, 1])
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    S, R, I, C = supernet(x, y, t, p)

    # Pochodne czasowe
    S_t = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S), create_graph=True)[0]
    R_t = torch.autograd.grad(R, t, grad_outputs=torch.ones_like(R), create_graph=True)[0]
    I_t = torch.autograd.grad(I, t, grad_outputs=torch.ones_like(I), create_graph=True)[0]
    C_t = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]

    # Pochodne przestrzenne (laplacian)
    S_x = torch.autograd.grad(S, x, grad_outputs=torch.ones_like(S), create_graph=True)[0]
    S_xx = torch.autograd.grad(S_x, x, grad_outputs=torch.ones_like(S_x), create_graph=True)[0]
    S_y = torch.autograd.grad(S, y, grad_outputs=torch.ones_like(S), create_graph=True)[0]
    S_yy = torch.autograd.grad(S_y, y, grad_outputs=torch.ones_like(S_y), create_graph=True)[0]

    R_x = torch.autograd.grad(R, x, grad_outputs=torch.ones_like(R), create_graph=True)[0]
    R_xx = torch.autograd.grad(R_x, x, grad_outputs=torch.ones_like(R_x), create_graph=True)[0]
    R_y = torch.autograd.grad(R, y, grad_outputs=torch.ones_like(R), create_graph=True)[0]
    R_yy = torch.autograd.grad(R_y, y, grad_outputs=torch.ones_like(R_y), create_graph=True)[0]

    I_x = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I), create_graph=True)[0]
    I_xx = torch.autograd.grad(I_x, x, grad_outputs=torch.ones_like(I_x), create_graph=True)[0]
    I_y = torch.autograd.grad(I, y, grad_outputs=torch.ones_like(I), create_graph=True)[0]
    I_yy = torch.autograd.grad(I_y, y, grad_outputs=torch.ones_like(I_y), create_graph=True)[0]

    C_x = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    C_xx = torch.autograd.grad(C_x, x, grad_outputs=torch.ones_like(C_x), create_graph=True)[0]
    C_y = torch.autograd.grad(C, y, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    C_yy = torch.autograd.grad(C_y, y, grad_outputs=torch.ones_like(C_y), create_graph=True)[0]

    # Parametry (użyj wartości z p_pde_base)
    D_S = params.get("D_S", 0.0)
    D_R = params.get("D_R", 0.0)
    D_I = params.get("D_I", 0.0)
    D_C = params.get("D_C", 0.01)

    rho_S = params.get("rho_S", 0.04)
    rho_R = params.get("rho_R", 0.03)
    K = params.get("K", 1.0)
    alpha_S = params.get("alpha_S", 0.8)
    alpha_R = params.get("alpha_R", 0.1)
    gamma_S = params.get("gamma_S", 0.02)
    gamma_R = params.get("gamma_R", 0.02)
    sigma = params.get("sigma", 0.05)
    delta = params.get("delta", 0.1)
    lam = params.get("lam", 0.2)
    beta = params.get("beta", 0.0)
    mu_max = params.get("mu_max", 0.05)
    C50 = params.get("C50", 0.2)
    m_hill = params.get("m_hill", 3)

    # Reakcje
    N = S + R

    # Funkcja Hill (oporność)
    C_np = C.detach().cpu().numpy()
    mu_np = mu_of_C(C_np, mu_max, C50, m_hill)
    mu = torch.tensor(mu_np, dtype=torch.float32, device=C.device)

    # Równania PDE
    # S: ∂S/∂t = D_S∇²S + rho_S*S*(1-N/K) - alpha_S*C*S - gamma_S*I*S - mu*S
    res_S = S_t - D_S * (S_xx + S_yy) - (
        rho_S * S * (1 - N / K) - alpha_S * C * S - gamma_S * I * S - mu * S
    )

    # R: ∂R/∂t = D_R∇²R + rho_R*R*(1-N/K) - alpha_R*C*R - gamma_R*I*R + mu*S
    res_R = R_t - D_R * (R_xx + R_yy) - (
        rho_R * R * (1 - N / K) - alpha_R * C * R - gamma_R * I * R + mu * S
    )

    # I: ∂I/∂t = D_I∇²I + sigma*N - delta*I
    res_I = I_t - D_I * (I_xx + I_yy) - (sigma * N - delta * I)

    # C: ∂C/∂t = D_C∇²C - lam*C - beta*C*N + I_in(p)
    # I_in zależy od p (p * base_infusion_rate)
    base_infusion = 0.15
    I_in = p * base_infusion

    res_C = C_t - D_C * (C_xx + C_yy) - (-lam * C - beta * C * N + I_in)

    return res_S, res_R, res_I, res_C

# %% [markdown]
# ## 13. SUPERNET — Trenowanie

# %%
def train_supernet(supernet, training_data, pde_params, epochs=5000, lr=1e-3,
                  batch_size=512, device="cpu"):
    """
    Trenuje SuperNet na danych z wielu scenariuszy.

    Straty:
    - PDE residuals (fizyka)
    - Initial conditions
    - TB trajectory matching
    """
    supernet = supernet.to(device)
    optimizer = optim.Adam(supernet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                     patience=300, verbose=False)

    # Dane treningowe
    coll_pts = training_data["collocation"]
    ic_pts = training_data["ic"]["points"]
    ic_vals = training_data["ic"]["values"]
    tb_t = training_data["tb"]["t"]
    tb_p = training_data["tb"]["p"]
    tb_vals = training_data["tb"]["TB"]

    # Przelicz parametry PDE na słownik
    pde_params_dict = asdict(pde_params) if hasattr(pde_params, '__dataclass_fields__') else pde_params

    losses_hist = []

    print("\n[SuperNet] Start treningu...")

    # Progress bar
    epoch_iter = trange(epochs, desc="SuperNet") if TQDM_AVAILABLE else range(epochs)

    for epoch in epoch_iter:
        optimizer.zero_grad()

        # === 1. PDE Residuals ===
        # Losuj batch punktów kolokacyjnych
        idx_coll = np.random.choice(len(coll_pts), size=min(batch_size, len(coll_pts)), replace=False)
        batch_coll = coll_pts[idx_coll]

        x_coll = torch.tensor(batch_coll[:, 0:1], dtype=torch.float32, device=device)
        y_coll = torch.tensor(batch_coll[:, 1:2], dtype=torch.float32, device=device)
        t_coll = torch.tensor(batch_coll[:, 2:3], dtype=torch.float32, device=device)
        p_coll = torch.tensor(batch_coll[:, 3:4], dtype=torch.float32, device=device)

        res_S, res_R, res_I, res_C = compute_pde_residuals(supernet, x_coll, y_coll, t_coll,
                                                           p_coll, pde_params_dict)

        loss_pde = torch.mean(res_S**2 + res_R**2 + res_I**2 + res_C**2)

        # === 2. Initial Conditions ===
        idx_ic = np.random.choice(len(ic_pts), size=min(batch_size//2, len(ic_pts)), replace=False)
        batch_ic_pts = ic_pts[idx_ic]
        batch_ic_vals = ic_vals[idx_ic]

        x_ic = torch.tensor(batch_ic_pts[:, 0:1], dtype=torch.float32, device=device)
        y_ic = torch.tensor(batch_ic_pts[:, 1:2], dtype=torch.float32, device=device)
        t_ic = torch.tensor(batch_ic_pts[:, 2:3], dtype=torch.float32, device=device)
        p_ic = torch.tensor(batch_ic_pts[:, 3:4], dtype=torch.float32, device=device)

        S_ic, R_ic, I_ic, C_ic = supernet(x_ic, y_ic, t_ic, p_ic)

        S_ic_true = torch.tensor(batch_ic_vals[:, 0:1], dtype=torch.float32, device=device)
        R_ic_true = torch.tensor(batch_ic_vals[:, 1:2], dtype=torch.float32, device=device)
        I_ic_true = torch.tensor(batch_ic_vals[:, 2:3], dtype=torch.float32, device=device)
        C_ic_true = torch.tensor(batch_ic_vals[:, 3:4], dtype=torch.float32, device=device)

        loss_ic = torch.mean((S_ic - S_ic_true)**2 + (R_ic - R_ic_true)**2 +
                            (I_ic - I_ic_true)**2 + (C_ic - C_ic_true)**2)

        # === 3. TB Trajectory ===
        # Dla uproszczenia: próbkujemy punkty na siatce przestrzennej i całkujemy TB
        # W pełnej wersji użyj właściwej kwadratury
        idx_tb = np.random.choice(len(tb_t), size=min(batch_size//4, len(tb_t)), replace=False)
        t_tb_batch = tb_t[idx_tb]
        p_tb_batch = tb_p[idx_tb]
        TB_true_batch = tb_vals[idx_tb]

        # Dla każdego punktu (t, p) oblicz TB przez całkowanie po (x,y)
        TB_pred_list = []
        for i in range(len(t_tb_batch)):
            # Próbkuj punkty przestrzenne
            n_spatial = 20
            x_samp = np.linspace(0, 1, n_spatial)
            y_samp = np.linspace(0, 1, n_spatial)
            X_samp, Y_samp = np.meshgrid(x_samp, y_samp)
            x_flat = X_samp.ravel()
            y_flat = Y_samp.ravel()
            t_flat = np.full_like(x_flat, t_tb_batch[i])
            p_flat = np.full_like(x_flat, p_tb_batch[i])

            x_t = torch.tensor(x_flat.reshape(-1, 1), dtype=torch.float32, device=device)
            y_t = torch.tensor(y_flat.reshape(-1, 1), dtype=torch.float32, device=device)
            t_t = torch.tensor(t_flat.reshape(-1, 1), dtype=torch.float32, device=device)
            p_t = torch.tensor(p_flat.reshape(-1, 1), dtype=torch.float32, device=device)

            with torch.no_grad():  # oszczędność pamięci
                S_samp, R_samp, _, _ = supernet(x_t, y_t, t_t, p_t)

            TB_samp = S_samp + R_samp
            TB_integral = torch.mean(TB_samp)  # uproszczone całkowanie
            TB_pred_list.append(TB_integral)

        TB_pred_tensor = torch.stack(TB_pred_list)
        TB_true_tensor = torch.tensor(TB_true_batch, dtype=torch.float32, device=device)

        loss_tb = torch.mean((TB_pred_tensor - TB_true_tensor)**2)

        # === Łączna strata ===
        # Wagi dla różnych komponentów (do dostrojenia)
        w_pde = 1.0
        w_ic = 10.0
        w_tb = 100.0

        loss = w_pde * loss_pde + w_ic * loss_ic + w_tb * loss_tb

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(supernet.parameters(), max_norm=1.0)

        optimizer.step()

        # Fix warning: detach loss before passing to scheduler
        scheduler.step(loss.detach())

        loss_val = loss.item()
        losses_hist.append(loss_val)

        # Update progress bar
        if TQDM_AVAILABLE:
            epoch_iter.set_postfix({
                "loss": f"{loss_val:.4f}",
                "pde": f"{loss_pde.item():.4f}",
                "ic": f"{loss_ic.item():.4f}",
                "tb": f"{loss_tb.item():.4f}"
            })
        elif (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_val:.6f}, "
                  f"PDE: {loss_pde.item():.6f}, IC: {loss_ic.item():.6f}, TB: {loss_tb.item():.6f}")

    print(f"[SuperNet] Trening zakończony. Finalna strata: {losses_hist[-1]:.6f}")

    return supernet, losses_hist

# Inicjalizacja SuperNet
supernet = SuperNet(in_dim=4, out_dim=4, width=128, depth=6)
print(f"\n[SuperNet] Parametry sieci: {sum(p.numel() for p in supernet.parameters())/1e6:.2f}M")

# Trenowanie (uwaga: może być czasochłonne)
# W produkcji użyj większej liczby epok
if PDE_OK:
    supernet, supernet_losses = train_supernet(
        supernet, training_data, p_pde_base,
        epochs=2000, lr=1e-3, batch_size=512, device=device
    )

    # Zapisz model
    torch.save(supernet.state_dict(), "out/supernet_model.pt")
    print("[OK] Model SuperNet zapisany do out/supernet_model.pt")
else:
    print("[UWAGA] SuperNet nie został wytrenowany - brak modułu PDE")

# %% [markdown]
# ## 14. SUPERNET — Ewaluacja i porównanie

# %%
def evaluate_supernet_scenario(supernet, grid, t_eval, p_dose, device="cpu"):
    """
    Ewaluuje SuperNet dla danego scenariusza (p_dose).
    Zwraca TB(t) oraz pola w ostatnim czasie.
    """
    supernet.eval()

    X, Y = grid.X, grid.Y
    TB_over_time = []

    with torch.no_grad():
        for t_val in t_eval:
            x_flat = torch.tensor(X.ravel().reshape(-1, 1), dtype=torch.float32, device=device)
            y_flat = torch.tensor(Y.ravel().reshape(-1, 1), dtype=torch.float32, device=device)
            t_flat = torch.full_like(x_flat, t_val)
            p_flat = torch.full_like(x_flat, p_dose)

            S, R, I, C = supernet(x_flat, y_flat, t_flat, p_flat)

            S_field = S.cpu().numpy().reshape(X.shape)
            R_field = R.cpu().numpy().reshape(X.shape)

            # Całka TB = ∫(S+R) dx dy
            TB = np.sum(S_field + R_field) * grid.dx * grid.dy
            TB_over_time.append(TB)

    return np.array(TB_over_time)

if PDE_OK:
    print("\n[SuperNet] Ewaluacja dla scenariuszy...")

    supernet_metrics = {}
    grid_eval = Grid(Nx=64, Ny=64)

    for p_dose in dose_scenarios:
        print(f"  Scenariusz p_dose = {p_dose}...")

        # PDE reference
        t_ref = pde_data_scenarios[p_dose]["t"]
        TB_pde_ref = pde_data_scenarios[p_dose]["TB"]

        # SuperNet
        TB_supernet = evaluate_supernet_scenario(supernet, grid_eval, t_ref, p_dose, device)

        # ODE (z parametrem p_dose - musimy przeskalować infusion_rate)
        p_ode_scaled = copy.deepcopy(p_ode)
        p_ode_scaled.infusion_rate = 0.15 * p_dose
        ode_traj_scaled = simulate_ode(p_ode_scaled, y0_ode, t_ref)
        TB_ode_scaled = ode_traj_scaled["TB"]

        # Supermodel (używamy tej samej sieci korekcyjnej - może być nieoptymalne)
        integrator_scaled = SupermodelIntegrator(p_ode_scaled, correction_net, device=device, method="rk4")
        traj_sm_scaled = integrator_scaled.integrate(y0_ode, t_ref)
        TB_supermodel_scaled = traj_sm_scaled[:, 0] + traj_sm_scaled[:, 1]

        # Metryki
        metrics = {
            "ODE_vs_PDE": {
                "RMSE": float(rmse(TB_ode_scaled, TB_pde_ref)),
                "MAE": float(mae(TB_ode_scaled, TB_pde_ref))
            },
            "Supermodel_vs_PDE": {
                "RMSE": float(rmse(TB_supermodel_scaled, TB_pde_ref)),
                "MAE": float(mae(TB_supermodel_scaled, TB_pde_ref))
            },
            "SuperNet_vs_PDE": {
                "RMSE": float(rmse(TB_supernet, TB_pde_ref)),
                "MAE": float(mae(TB_supernet, TB_pde_ref))
            }
        }

        supernet_metrics[f"dose_{p_dose}"] = metrics

        # Wykres
        plt.figure(figsize=(8, 5))
        plt.plot(t_ref, TB_pde_ref, 'k-', lw=2.5, label="PDE (reference)")
        plt.plot(t_ref, TB_ode_scaled, 'b--', lw=2, label="ODE")
        plt.plot(t_ref, TB_supermodel_scaled, 'g:', lw=2, label="Supermodel")
        plt.plot(t_ref, TB_supernet, 'r-.', lw=2, label="SuperNet")
        plt.xlabel("Czas t", fontsize=12)
        plt.ylabel("Tumor Burden TB(t)", fontsize=12)
        plt.title(f"Porównanie dla p_dose = {p_dose}", fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)

        fname = f"out/supermodel_supernet_tb_dose{str(p_dose).replace('.', '')}.png"
        savefig_root(fname)
        plt.show()

        print(f"    RMSE(SuperNet vs PDE): {metrics['SuperNet_vs_PDE']['RMSE']:.6f}")

    # Zapisz metryki
    save_json(supernet_metrics, "supernet_metrics.json")

    print("\n[SuperNet] Podsumowanie metryk:")
    for scenario, metrics in supernet_metrics.items():
        print(f"\n  {scenario}:")
        for model, vals in metrics.items():
            print(f"    {model}: RMSE={vals['RMSE']:.6f}, MAE={vals['MAE']:.6f}")

# %% [markdown]
# ## 15. Podsumowanie końcowe

# %%
print("\n" + "="*70)
print("PODSUMOWANIE ZADANIA 7")
print("="*70)

print("\n[SUPERMODEL]")
print("  Model: ODE + neuronowa korekta g_φ(y,t)")
print(f"  Parametry sieci: {sum(p.numel() for p in correction_net.parameters())}")
print("  Zapisane pliki:")
print("    - out/supermodel_correction_net.pt")
print("    - out/supermodel_metrics.json")
print("    - out/supermodel_tb_curves.csv")
print("    - out/supermodel_tb_compare.png")

if PDE_OK:
    print("\n[SUPERNET]")
    print("  Model: PINN rozszerzony o parametr terapii (x,y,t,p)")
    print(f"  Parametry sieci: {sum(p.numel() for p in supernet.parameters())/1e6:.2f}M")
    print(f"  Scenariusze treningowe: {dose_scenarios}")
    print("  Zapisane pliki:")
    print("    - out/supernet_model.pt")
    print("    - out/supernet_metrics.json")
    for p_dose in dose_scenarios:
        fname = f"supermodel_supernet_tb_dose{str(p_dose).replace('.', '')}.png"
        print(f"    - {fname}")

print("\n[METRYKI - Scenariusz nominalny p_dose=1.0]")
if "Supermodel_vs_PDE" in metrics_supermodel:
    print(f"  Supermodel vs PDE: RMSE={metrics_supermodel['Supermodel_vs_PDE']['RMSE']:.6f}")
if PDE_OK and "dose_1.0" in supernet_metrics:
    print(f"  SuperNet vs PDE:   RMSE={supernet_metrics['dose_1.0']['SuperNet_vs_PDE']['RMSE']:.6f}")

print("\n" + "="*70)
print("Zadanie 7 zakończone!")
print("="*70)

# %%
