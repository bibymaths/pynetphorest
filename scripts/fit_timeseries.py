#!/usr/bin/env python3
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
import time


# ------------------------------------------------------------
#  ODE SYSTEM AND SIMULATION
# ------------------------------------------------------------

def ode_rhs(state, t, k_on, k_off, gamma, alpha, C):
    """
    state = [S, x_0, x_1, ..., x_{N-1}]
    dS/dt = -gamma * S
    dx_i/dt = k_on[i] * S * mod * (1 - x_i) - k_off[i] * x_i
    """
    S = state[0]
    x = state[1:]

    # 1. Global activation decay
    dS = -gamma * S

    # 2. Local coupling
    # Clip modulation to >= 0 to prevent negative rates
    coupling = alpha * (C @ x)
    mod = np.maximum(1.0 + coupling, 0.0)

    # 3. Kinetics
    v_on = k_on * S * mod * (1 - x)
    v_off = k_off * x
    dx = v_on - v_off

    return np.concatenate(([dS], dx))


def simulate(k_on, k_off, gamma, alpha, C, t_eval, x0=None):
    N = C.shape[0]
    if x0 is None:
        x_init = np.zeros(N)
    else:
        x_init = x0

    state0 = np.concatenate(([1.0], x_init))

    def rhs(state, t):
        return ode_rhs(state, t, k_on, k_off, gamma, alpha, C)

    sol = odeint(rhs, state0, t_eval, hmax=np.inf)
    return sol[:, 1:].T  # Return only x


# ------------------------------------------------------------
#  FITTING LOGIC
# ------------------------------------------------------------

def get_residuals(params, t_points, y_data, site_idx,
                  current_k_on, current_k_off, fixed_gamma,
                  alpha, C, reg_weight):
    """
    Cost function.
    We optimize parameters for 'site_idx', but we simulate the FULL network
    using 'current_k_on' and 'current_k_off' for the neighbors.
    """
    # Params being optimized for this specific site
    k_on_i, k_off_i = params

    # Construct temporary parameter vectors for simulation
    # We copy the global current state, then overwrite the site we are fitting
    sim_k_on = current_k_on.copy()
    sim_k_off = current_k_off.copy()

    sim_k_on[site_idx] = k_on_i
    sim_k_off[site_idx] = k_off_i

    # Use the fixed global gamma
    gamma_val = fixed_gamma

    # Initial condition (Use data for the target site, zeros/current for others?)
    # Ideally, we use the data-derived IC for the target, and 0 for others
    # (assuming they start at 0 before stimulation).
    x0 = np.zeros(C.shape[0])
    x0[site_idx] = y_data[0]
    # Note: For neighbors, we assume x0=0 fits the biology (starved cells).
    # If neighbors have high basal levels, we would need their data here too.

    # Simulate full network
    X_sim = simulate(sim_k_on, sim_k_off, gamma_val, alpha, C, t_points, x0=x0)

    # Extract prediction for ONLY the site we are fitting
    y_sim = X_sim[site_idx, :]

    # Residuals
    res_data = y_sim - y_data

    # Regularization (L2 penalty on the active params)
    res_reg = np.array(params) * np.sqrt(reg_weight)

    return np.concatenate((res_data, res_reg))


def fit_site_iterative(site_idx, y_data, t_points,
                       global_k_on, global_k_off, global_gamma,
                       alpha, C, reg_weight, n_starts=5):
    """
    Fits site 'site_idx' while keeping all other sites fixed at 'global_k_...' values.
    """

    lower_bounds = [0.0, 0.0]
    upper_bounds = [1000.0, 1000.0]

    best_cost = np.inf
    best_params = None

    # Multi-start
    for i in range(n_starts):
        p0 = np.exp(np.random.uniform(np.log(0.01), np.log(100.0), 2))
        p0 = np.clip(p0, lower_bounds, upper_bounds)

        try:
            res = least_squares(
                get_residuals, p0,
                bounds=(lower_bounds, upper_bounds),
                args=(t_points, y_data, site_idx, global_k_on, global_k_off, global_gamma, alpha, C, reg_weight),
                method='trf', ftol=1e-4
            )
            if res.cost < best_cost:
                best_cost = res.cost
                best_params = res.x
        except:
            continue

    if best_params is None: return global_k_on[site_idx], global_k_off[site_idx]

    # Fine tune
    res_final = least_squares(
        get_residuals, best_params,
        bounds=(lower_bounds, upper_bounds),
        args=(t_points, y_data, site_idx, global_k_on, global_k_off, global_gamma, alpha, C, reg_weight * 0.1),
        method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8
    )

    return res_final.x


# ------------------------------------------------------------
#  PRE-PASS (Find Gamma)
# ------------------------------------------------------------
def fit_gamma_initial(site_idx, y_data, t_points, alpha, C):
    """
    Simple decoupled fit just to estimate Gamma.
    """

    def model(t, k_on, k_off, g):
        # Decoupled simulation (alpha=0 effect)
        # Analytical approximation or simple ODE for 1 site
        # We'll just use the full ODE with others zeroed
        N = C.shape[0]
        kv_on = np.zeros(N);
        kv_on[site_idx] = k_on
        kv_off = np.zeros(N);
        kv_off[site_idx] = k_off
        x0 = np.zeros(N);
        x0[site_idx] = y_data[0]
        X = simulate(kv_on, kv_off, g, 0.0, C, t, x0=x0)  # alpha=0 here effectively
        return X[site_idx, :]

    try:
        popt, _ = from_scipy_optimize_curve_fit(model, t_points, y_data, p0=[1, 0.1, 0.1], bounds=(0, [100, 100, 10]))
        return popt[2]  # gamma
    except:
        return 0.1


from scipy.optimize import curve_fit as from_scipy_optimize_curve_fit


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residues-tsv", required=True)
    parser.add_argument("--C-matrix", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--outdir", default="fit_results")
    parser.add_argument("--rounds", type=int, default=3, help="Number of refinement rounds")
    parser.add_argument("--reg", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load Data ---
    print(f"[*] Loading Data...")
    C_df = pd.read_csv(args.C_matrix, sep="\t", index_col=0)
    df = pd.read_csv(args.residues_tsv, sep="\t")

    TIME_POINTS = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0])

    # Align Data
    gene_name = df.columns[0]
    residues = df["Residue"].tolist()
    ts_sites = [f"{gene_name}_{r}" for r in residues]

    # Filter
    valid_sites = [s for s in ts_sites if s in C_df.index]
    keep_idx = [i for i, s in enumerate(ts_sites) if s in valid_sites]

    df = df.iloc[keep_idx]
    residues = [residues[i] for i in keep_idx]
    ts_sites = [ts_sites[i] for i in keep_idx]

    Y = df.iloc[:, 2:].to_numpy()  # Assuming cols 0,1 are metadata

    C_df = C_df.loc[ts_sites, ts_sites]
    C = C_df.to_numpy()
    N = len(residues)

    print(f"[*] Fitting {N} sites. Alpha={args.alpha}")

    # --- Step 1: Initialize Global Params ---
    global_k_on = np.ones(N) * 0.1
    global_k_off = np.ones(N) * 0.1

    # Estimate global gamma (take median of individual estimates)
    print("[*] Estimating global signal decay (gamma)...")
    gammas = []
    for i in range(N):
        g = fit_gamma_initial(i, Y[i], TIME_POINTS, args.alpha, C)
        gammas.append(g)

    GLOBAL_GAMMA = np.median(gammas)
    print(f"[*] Fixed Global Gamma: {GLOBAL_GAMMA:.4f}")

    # --- Step 2: Iterative Round-Robin Fit ---
    for r in range(args.rounds):
        print(f"\n--- Round {r + 1}/{args.rounds} ---")

        # In each round, iterate through all sites
        for i in range(N):
            resname = residues[i]

            # Fit site i, using current best guesses for everyone else
            new_kon, new_koff = fit_site_iterative(
                i, Y[i], TIME_POINTS,
                global_k_on, global_k_off, GLOBAL_GAMMA,
                args.alpha, C, args.reg, n_starts=5
            )

            # Update global state immediately (Gauss-Seidel style update)
            global_k_on[i] = new_kon
            global_k_off[i] = new_koff

            print(f"    {resname:10s} -> k_on={new_kon:.2f} k_off={new_koff:.2f}")

    # --- Step 3: Final Output & Plotting ---
    print("\n[*] Saving results...")
    results = []

    # Final simulation for plotting
    X_final = simulate(global_k_on, global_k_off, GLOBAL_GAMMA, args.alpha, C, TIME_POINTS, x0=Y[:, 0])

    for i, resname in enumerate(residues):
        results.append({
            "Residue": resname,
            "k_on": global_k_on[i],
            "k_off": global_k_off[i],
            "gamma": GLOBAL_GAMMA
        })

        plt.figure(figsize=(6, 4))
        plt.plot(TIME_POINTS, Y[i], "o", color="black", alpha=0.6, label="Data")
        plt.plot(TIME_POINTS, X_final[i], "-", color="crimson", lw=2, label="Coupled Fit")
        plt.title(f"{resname} Final Fit")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{args.outdir}/{resname}.png")
        plt.close()

    pd.DataFrame(results).to_csv(f"{args.outdir}/fitted_params.tsv", sep="\t", index=False)
    print(f"[*] Done. Parameters saved to {args.outdir}/fitted_params.tsv")


if __name__ == "__main__":
    main()