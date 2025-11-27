#!/usr/bin/env python3
"""
Coupled EGFR phosphosite model with:
- Biologically reasonable input S(t)
- Row-normalised coupling matrices
- Separate global (PTM) and local (distance) coupling with β_global, β_local

Usage (example):

  ./fit_egfr_coupled_globloc.py \
      --data test.tsv \
      --c-global C_ptm.tsv \
      --c-local  C_dist.tsv \
      --outdir fit_egfr_globloc

If you only have one hybrid matrix, you can just pass the same file twice.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares
import argparse
import os


# ------------------------------------------------------------
#  TIMEPOINTS & DATA
# ------------------------------------------------------------

DEFAULT_TIMEPOINTS = np.array(
    [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0,
     30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
)


def load_egfr_data(path, timepoints=DEFAULT_TIMEPOINTS):
    """
    TSV format:
        EGFR, Residue, v1, v2, ..., v14

    Returns
    -------
    residues : list of str
    t        : (T,) float
    Y        : (N, T) float  (FC values)
    """
    df = pd.read_csv(path, sep="\t")
    value_cols = [c for c in df.columns if c.startswith("v")]
    if len(value_cols) != len(timepoints):
        raise ValueError(
            f"Expected {len(timepoints)} value cols, found {len(value_cols)}."
        )
    residues = df["Residue"].astype(str).tolist()
    Y = df[value_cols].values.astype(float)
    return residues, np.array(timepoints, float), Y


def scale_fc_to_unit_interval(Y):
    """
    Map FC per site to internal p_i(t) in [0,1].

    For each row i:
        b_i = y_i(0)
        A_i = max_t y_i(t) - b_i
        p_i(t) = (y_i(t) - b_i) / max(A_i, eps)

    Returns
    -------
    P_scaled : (N, T)
    baselines : (N,)
    amplitudes: (N,)
    """
    N, T = Y.shape
    P = np.zeros_like(Y)
    baselines = np.zeros(N)
    amplitudes = np.zeros(N)
    eps = 1e-6

    for i in range(N):
        y = Y[i]
        b = y[0]
        A = y.max() - b
        if A < eps:
            A = 1.0
        p = (y - b) / A
        p = np.clip(p, 0.0, 1.0)
        P[i] = p
        baselines[i] = b
        amplitudes[i] = A

    return P, baselines, amplitudes


# ------------------------------------------------------------
#  COUPLING MATRICES
# ------------------------------------------------------------

def load_c_matrix(path):
    """
    Load a square TSV matrix with row/col labels.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    if df.shape[0] != df.shape[1] or not all(df.index == df.columns):
        raise ValueError(f"{path}: C matrix must be square with matching row/col labels.")
    return df


def row_normalize(C):
    """
    Row-normalise C so that each row sums to 1 (or stays zero if originally zero).
    """
    C = C.copy()
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return C / row_sums


def align_sites_with_two_C(residues, Cg_df, Cl_df):
    """
    Align residues with both C_global and C_local matrices.

    Strategy:
      - Try exact match of residue in index.
      - Else look for any index that endswith(residue).
      - Same mapping must exist in both Cg_df and Cl_df.

    Returns
    -------
    Cg : (N,N) numpy array
    Cl : (N,N) numpy array
    residues_aligned : list[str]
    """
    idx_g = list(Cg_df.index)
    idx_l = list(Cl_df.index)

    mapping = {}  # residue -> row/col label

    for res in residues:
        # exact in global?
        if res in idx_g and res in idx_l:
            mapping[res] = res
            continue

        # suffix match
        cand_g = [name for name in idx_g if name.endswith(res)]
        cand_l = [name for name in idx_l if name.endswith(res)]

        if len(cand_g) == 0 or len(cand_l) == 0:
            raise ValueError(
                f"Could not map residue '{res}' in both C_global and C_local."
            )

        # heuristic: shortest name
        cand_g.sort(key=len)
        cand_l.sort(key=len)
        lab_g = cand_g[0]
        lab_l = cand_l[0]

        if lab_g != lab_l:
            # It's okay if labels differ between matrices; we track separately
            mapping[res] = (lab_g, lab_l)
        else:
            mapping[res] = lab_g

    ordered_rows_g = []
    ordered_rows_l = []

    for res in residues:
        lab = mapping[res]
        if isinstance(lab, tuple):
            lab_g, lab_l = lab
        else:
            lab_g = lab_l = lab
        ordered_rows_g.append(lab_g)
        ordered_rows_l.append(lab_l)

    Cg = Cg_df.loc[ordered_rows_g, ordered_rows_g].values.astype(float)
    Cl = Cl_df.loc[ordered_rows_l, ordered_rows_l].values.astype(float)

    return Cg, Cl, residues


# ------------------------------------------------------------
#  ODE MODEL
# ------------------------------------------------------------

def egfr_rhs(x, t, params, Cg, Cl):
    """
    RHS for coupled EGFR phosphosite model.

    x = [S, p_0, ..., p_{N-1}]
    params:
        k_act, k_deact, beta_g, beta_l, k_on (N,), k_off (N,)
    Cg : global (PTM) C
    Cl : local (distance) C
    """
    N = Cg.shape[0]
    S = x[0]
    p = x[1:]

    k_act = params["k_act"]
    k_deact = params["k_deact"]
    beta_g = params["beta_g"]
    beta_l = params["beta_l"]
    k_on = params["k_on"]
    k_off = params["k_off"]

    # Input dynamics
    dS = k_act * (1.0 - S) - k_deact * S

    # Coupling
    coup_g = beta_g * (Cg @ p)
    coup_l = beta_l * (Cl @ p)
    total_coup = coup_g + coup_l

    # Site dynamics
    v_on = (k_on * S + total_coup) * (1.0 - p)
    v_off = k_off * p
    dp = v_on - v_off

    dx = np.zeros_like(x)
    dx[0] = dS
    dx[1:] = dp
    return dx


def simulate_p(t, Cg, Cl, p0, theta):
    """
    Simulate p_i(t) for given parameter vector.

    theta =
      [log_k_act, log_k_deact,
       log_beta_g, log_beta_l,
       log_k_on_0..N-1, log_k_off_0..N-1]
    """
    N = Cg.shape[0]
    T = len(t)

    log_k_act = theta[0]
    log_k_deact = theta[1]
    log_beta_g = theta[2]
    log_beta_l = theta[3]
    log_k_on = theta[4:4 + N]
    log_k_off = theta[4 + N:4 + 2 * N]

    params = {
        "k_act": np.exp(log_k_act),
        "k_deact": np.exp(log_k_deact),
        "beta_g": np.exp(log_beta_g),
        "beta_l": np.exp(log_beta_l),
        "k_on": np.exp(log_k_on),
        "k_off": np.exp(log_k_off),
    }

    x0 = np.zeros(N + 1)
    x0[0] = 0.0     # S(0)
    x0[1:] = p0     # p(0) from data

    X = odeint(egfr_rhs, x0, t, args=(params, Cg, Cl))
    P_sim = X[:, 1:].T
    return P_sim


# ------------------------------------------------------------
#  FITTING
# ------------------------------------------------------------

def residuals(theta, t, Cg, Cl, P_data, reg_lambda=1e-3):
    """
    Residuals for least_squares: data + L2 on log-params.
    """
    N, T = P_data.shape
    P_sim = simulate_p(t, Cg, Cl, p0=P_data[:, 0], theta=theta)
    data_res = (P_sim - P_data).ravel()
    reg_res = np.sqrt(reg_lambda) * theta
    return np.concatenate([data_res, reg_res])


def fit_model(t, Cg, Cl, P_data):
    """
    Fit β_global, β_local, k_act, k_deact, k_on, k_off.
    """
    N, T = P_data.shape

    # Initial guesses
    k_act0 = 1.0
    k_deact0 = 0.01
    beta_g0 = 0.05
    beta_l0 = 0.05
    k_on0 = np.full(N, 0.1)
    k_off0 = np.full(N, 0.05)

    theta0 = np.concatenate([
        np.log([k_act0, k_deact0, beta_g0, beta_l0]),
        np.log(k_on0),
        np.log(k_off0),
    ])

    # Bounds (log-space)
    lower = np.log(1e-4) * np.ones_like(theta0)
    upper = np.log(10.0) * np.ones_like(theta0)

    result = least_squares(
        residuals,
        theta0,
        bounds=(lower, upper),
        args=(t, Cg, Cl, P_data),
        verbose=2,
    )
    return result.x, result


# ------------------------------------------------------------
#  PLOTTING
# ------------------------------------------------------------

def plot_fits(outdir, residues, t, Y_data, P_data, baselines, amplitudes, P_sim):
    os.makedirs(outdir, exist_ok=True)
    N, T = Y_data.shape

    Y_sim = np.zeros_like(Y_data)
    for i in range(N):
        Y_sim[i] = baselines[i] + amplitudes[i] * P_sim[i]

    for i, res in enumerate(residues):
        plt.figure(figsize=(6, 4))

        plt.subplot(2, 1, 1)
        plt.plot(t, Y_data[i], "o-", label="Data FC")
        plt.plot(t, Y_sim[i], "-", label="Sim FC")
        plt.ylabel("Fold Change")
        plt.title(f"EGFR {res}")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t, P_data[i], "o-", label="Data p (scaled)")
        plt.plot(t, P_sim[i], "-", label="Sim p")
        plt.xlabel("Time (min)")
        plt.ylabel("p (0-1)")
        plt.legend()

        plt.tight_layout()
        fname = os.path.join(outdir, f"EGFR_{res}.png")
        plt.savefig(fname, dpi=150)
        plt.close()


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit EGFR phosphosite model with global/local coupling."
    )
    parser.add_argument("--data", required=True,
                        help="TSV with EGFR phosphosite FC (e.g. test.tsv).")
    parser.add_argument("--c-global", required=True,
                        help="TSV C matrix for global (PTM) coupling.")
    parser.add_argument("--c-local", required=True,
                        help="TSV C matrix for local (distance) coupling.")
    parser.add_argument("--outdir", default="results_egfr_globloc",
                        help="Output directory for plots and parameters.")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Data
    residues, t, Y = load_egfr_data(args.data)
    print(f"[*] Loaded data for {len(residues)} sites, {Y.shape[1]} timepoints.")

    P_scaled, baselines, amplitudes = scale_fc_to_unit_interval(Y)
    print("[*] Scaled FC to p in [0,1].")

    # 2) Coupling matrices
    Cg_df = load_c_matrix(args.c_global)
    Cl_df = load_c_matrix(args.c_local)

    Cg, Cl, residues_aligned = align_sites_with_two_C(residues, Cg_df, Cl_df)

    # Row-normalise both
    Cg = row_normalize(Cg)
    Cl = row_normalize(Cl)
    print(f"[*] Loaded C_global and C_local, both row-normalised, shape {Cg.shape}.")

    # 3) Fit
    theta_opt, result = fit_model(t, Cg, Cl, P_scaled)
    print("[*] Optimization finished.")
    print(f"    Success: {result.success}, message: {result.message}")
    print(f"    Final cost: {result.cost:.4g}")

    # Decode
    N = Cg.shape[0]
    log_k_act = theta_opt[0]
    log_k_deact = theta_opt[1]
    log_beta_g = theta_opt[2]
    log_beta_l = theta_opt[3]
    log_k_on = theta_opt[4:4 + N]
    log_k_off = theta_opt[4 + N:4 + 2 * N]

    params_decoded = {
        "k_act": float(np.exp(log_k_act)),
        "k_deact": float(np.exp(log_k_deact)),
        "beta_g": float(np.exp(log_beta_g)),
        "beta_l": float(np.exp(log_beta_l)),
        "k_on": np.exp(log_k_on),
        "k_off": np.exp(log_k_off),
        "residues": residues_aligned,
    }

    np.savez(os.path.join(args.outdir, "fitted_params.npz"), **params_decoded)
    print("[*] Saved parameters to", os.path.join(args.outdir, "fitted_params.npz"))

    # 4) Simulate with fitted params
    P_sim = simulate_p(t, Cg, Cl, p0=P_scaled[:, 0], theta=theta_opt)

    # 5) Plots
    plot_fits(args.outdir, residues_aligned, t,
              Y_data=Y,
              P_data=P_scaled,
              baselines=baselines,
              amplitudes=amplitudes,
              P_sim=P_sim)
    print(f"[*] Saved plots to {args.outdir}")


if __name__ == "__main__":
    main()
