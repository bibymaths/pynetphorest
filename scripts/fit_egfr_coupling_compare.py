#!/usr/bin/env python3
"""
Compare coupling models for EGFR phosphosites:

1) FULL:    beta_g, beta_l free
2) NOGLOB:  beta_g = 0, beta_l free
3) NOLOC:   beta_l = 0, beta_g free

Same ODE structure as before, just different parameterisations.
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
    df = pd.read_csv(path, sep="\t", index_col=0)
    if df.shape[0] != df.shape[1] or not all(df.index == df.columns):
        raise ValueError(f"{path}: C matrix must be square with matching labels.")
    return df


def row_normalize(C):
    C = C.copy()
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return C / row_sums


def align_sites_with_two_C(residues, Cg_df, Cl_df):
    idx_g = list(Cg_df.index)
    idx_l = list(Cl_df.index)

    mapping_g = {}
    mapping_l = {}

    for res in residues:
        # exact
        if res in idx_g:
            mapping_g[res] = res
        else:
            cands = [name for name in idx_g if name.endswith(res)]
            if not cands:
                raise ValueError(f"Could not map residue '{res}' in C_global")
            cands.sort(key=len)
            mapping_g[res] = cands[0]

        if res in idx_l:
            mapping_l[res] = res
        else:
            cands = [name for name in idx_l if name.endswith(res)]
            if not cands:
                raise ValueError(f"Could not map residue '{res}' in C_local")
            cands.sort(key=len)
            mapping_l[res] = cands[0]

    rows_g = [mapping_g[res] for res in residues]
    rows_l = [mapping_l[res] for res in residues]

    Cg = Cg_df.loc[rows_g, rows_g].values.astype(float)
    Cl = Cl_df.loc[rows_l, rows_l].values.astype(float)

    return Cg, Cl, residues


# ------------------------------------------------------------
#  ODE MODEL
# ------------------------------------------------------------

def egfr_rhs(x, t, params, Cg, Cl):
    """
    x = [S, p_0, ..., p_{N-1}]
    params:
      k_act, k_deact, beta_g, beta_l, k_on (N,), k_off (N,)
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

    dS = k_act * (1.0 - S) - k_deact * S

    coup_g = beta_g * (Cg @ p)
    coup_l = beta_l * (Cl @ p)
    total_coup = coup_g + coup_l

    v_on = (k_on * S + total_coup) * (1.0 - p)
    v_off = k_off * p
    dp = v_on - v_off

    dx = np.zeros_like(x)
    dx[0] = dS
    dx[1:] = dp
    return dx


def simulate_p(t, Cg, Cl, p0, params):
    """
    params is a dict: k_act, k_deact, beta_g, beta_l, k_on, k_off.
    """
    N = Cg.shape[0]
    x0 = np.zeros(N + 1)
    x0[0] = 0.0   # S(0)
    x0[1:] = p0   # p(0)

    X = odeint(egfr_rhs, x0, t, args=(params, Cg, Cl))
    P_sim = X[:, 1:].T
    return P_sim


# ------------------------------------------------------------
#  RESIDUALS FOR THREE CASES
# ------------------------------------------------------------

def residuals_full(theta, t, Cg, Cl, P_data, reg_lambda=1e-3):
    """
    FULL: beta_g and beta_l free.

    theta = [log_k_act, log_k_deact,
             log_beta_g, log_beta_l,
             log_k_on_0..N-1, log_k_off_0..N-1]
    """
    N, T = P_data.shape
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

    P_sim = simulate_p(t, Cg, Cl, p0=P_data[:, 0], params=params)
    data_res = (P_sim - P_data).ravel()
    reg_res = np.sqrt(reg_lambda) * theta
    return np.concatenate([data_res, reg_res])


def residuals_noglob(theta, t, Cg, Cl, P_data, reg_lambda=1e-3):
    """
    NOGLOB: beta_g = 0, beta_l free.

    theta = [log_k_act, log_k_deact,
             log_beta_l,
             log_k_on_0..N-1, log_k_off_0..N-1]
    """
    N, T = P_data.shape
    log_k_act = theta[0]
    log_k_deact = theta[1]
    log_beta_l = theta[2]
    log_k_on = theta[3:3 + N]
    log_k_off = theta[3 + N:3 + 2 * N]

    params = {
        "k_act": np.exp(log_k_act),
        "k_deact": np.exp(log_k_deact),
        "beta_g": 0.0,  # FIXED
        "beta_l": np.exp(log_beta_l),
        "k_on": np.exp(log_k_on),
        "k_off": np.exp(log_k_off),
    }

    P_sim = simulate_p(t, Cg, Cl, p0=P_data[:, 0], params=params)
    data_res = (P_sim - P_data).ravel()
    reg_res = np.sqrt(reg_lambda) * theta
    return np.concatenate([data_res, reg_res])


def residuals_noloc(theta, t, Cg, Cl, P_data, reg_lambda=1e-3):
    """
    NOLOC: beta_l = 0, beta_g free.

    theta = [log_k_act, log_k_deact,
             log_beta_g,
             log_k_on_0..N-1, log_k_off_0..N-1]
    """
    N, T = P_data.shape
    log_k_act = theta[0]
    log_k_deact = theta[1]
    log_beta_g = theta[2]
    log_k_on = theta[3:3 + N]
    log_k_off = theta[3 + N:3 + 2 * N]

    params = {
        "k_act": np.exp(log_k_act),
        "k_deact": np.exp(log_k_deact),
        "beta_g": np.exp(log_beta_g),
        "beta_l": 0.0,  # FIXED
        "k_on": np.exp(log_k_on),
        "k_off": np.exp(log_k_off),
    }

    P_sim = simulate_p(t, Cg, Cl, p0=P_data[:, 0], params=params)
    data_res = (P_sim - P_data).ravel()
    reg_res = np.sqrt(reg_lambda) * theta
    return np.concatenate([data_res, reg_res])


# ------------------------------------------------------------
#  FIT HELPERS
# ------------------------------------------------------------

def fit_case(case, t, Cg, Cl, P_data):
    """
    case ∈ {"full", "noglob", "noloc"}
    Returns (theta_opt, result, decoded_params)
    """
    N, T = P_data.shape

    # Initial guesses
    k_act0 = 1.0
    k_deact0 = 0.01
    beta_g0 = 0.05
    beta_l0 = 0.05
    k_on0 = np.full(N, 0.1)
    k_off0 = np.full(N, 0.05)

    if case == "full":
        theta0 = np.concatenate([
            np.log([k_act0, k_deact0, beta_g0, beta_l0]),
            np.log(k_on0),
            np.log(k_off0),
        ])
        lower = np.log(1e-4) * np.ones_like(theta0)
        upper = np.log(10.0) * np.ones_like(theta0)
        fun = residuals_full

    elif case == "noglob":
        theta0 = np.concatenate([
            np.log([k_act0, k_deact0, beta_l0]),
            np.log(k_on0),
            np.log(k_off0),
        ])
        lower = np.log(1e-4) * np.ones_like(theta0)
        upper = np.log(10.0) * np.ones_like(theta0)
        fun = residuals_noglob

    elif case == "noloc":
        theta0 = np.concatenate([
            np.log([k_act0, k_deact0, beta_g0]),
            np.log(k_on0),
            np.log(k_off0),
        ])
        lower = np.log(1e-4) * np.ones_like(theta0)
        upper = np.log(10.0) * np.ones_like(theta0)
        fun = residuals_noloc

    else:
        raise ValueError(f"Unknown case: {case}")

    result = least_squares(
        fun,
        theta0,
        bounds=(lower, upper),
        args=(t, Cg, Cl, P_data),
        verbose=2,
    )

    theta_opt = result.x

    # Decode to a params dict for convenience
    if case == "full":
        log_k_act = theta_opt[0]
        log_k_deact = theta_opt[1]
        log_beta_g = theta_opt[2]
        log_beta_l = theta_opt[3]
        log_k_on = theta_opt[4:4 + N]
        log_k_off = theta_opt[4 + N:4 + 2 * N]

        params = {
            "k_act": float(np.exp(log_k_act)),
            "k_deact": float(np.exp(log_k_deact)),
            "beta_g": float(np.exp(log_beta_g)),
            "beta_l": float(np.exp(log_beta_l)),
            "k_on": np.exp(log_k_on),
            "k_off": np.exp(log_k_off),
        }

    elif case == "noglob":
        log_k_act = theta_opt[0]
        log_k_deact = theta_opt[1]
        log_beta_l = theta_opt[2]
        log_k_on = theta_opt[3:3 + N]
        log_k_off = theta_opt[3 + N:3 + 2 * N]

        params = {
            "k_act": float(np.exp(log_k_act)),
            "k_deact": float(np.exp(log_k_deact)),
            "beta_g": 0.0,
            "beta_l": float(np.exp(log_beta_l)),
            "k_on": np.exp(log_k_on),
            "k_off": np.exp(log_k_off),
        }

    else:  # noloc
        log_k_act = theta_opt[0]
        log_k_deact = theta_opt[1]
        log_beta_g = theta_opt[2]
        log_k_on = theta_opt[3:3 + N]
        log_k_off = theta_opt[3 + N:3 + 2 * N]

        params = {
            "k_act": float(np.exp(log_k_act)),
            "k_deact": float(np.exp(log_k_deact)),
            "beta_g": float(np.exp(log_beta_g)),
            "beta_l": 0.0,
            "k_on": np.exp(log_k_on),
            "k_off": np.exp(log_k_off),
        }

    return theta_opt, result, params


# ------------------------------------------------------------
#  MAIN: RUN ALL THREE
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare EGFR coupling models: full vs noglob vs noloc."
    )
    parser.add_argument("--data", required=True,
                        help="EGFR phosphosite FC TSV (e.g. test.tsv)")
    parser.add_argument("--c-global", required=True,
                        help="PTM-based C matrix TSV (C_ptm.tsv)")
    parser.add_argument("--c-local", required=True,
                        help="Distance-based C matrix TSV (C_dist.tsv)")
    parser.add_argument("--outdir", default="fit_egfr_coupling_compare",
                        help="Output dir for params and (optional) plots.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    residues, t, Y = load_egfr_data(args.data)
    P_scaled, baselines, amplitudes = scale_fc_to_unit_interval(Y)

    Cg_df = load_c_matrix(args.c_global)
    Cl_df = load_c_matrix(args.c_local)
    Cg, Cl, residues_aligned = align_sites_with_two_C(residues, Cg_df, Cl_df)

    Cg = row_normalize(Cg)
    Cl = row_normalize(Cl)
    print(f"[*] Using C_global and C_local, row-normalised, shape {Cg.shape}.")

    # FULL
    print("\n=== FIT: FULL (beta_g, beta_l free) ===")
    theta_full, res_full, pars_full = fit_case("full", t, Cg, Cl, P_scaled)
    print(f"[FULL] cost = {res_full.cost:.4g}")
    print(f"[FULL] k_act={pars_full['k_act']:.4g}, k_deact={pars_full['k_deact']:.4g}, "
          f"beta_g={pars_full['beta_g']:.4g}, beta_l={pars_full['beta_l']:.4g}")

    # NOGLOB
    print("\n=== FIT: NOGLOB (beta_g = 0, beta_l free) ===")
    theta_ng, res_ng, pars_ng = fit_case("noglob", t, Cg, Cl, P_scaled)
    print(f"[NOGLOB] cost = {res_ng.cost:.4g}")
    print(f"[NOGLOB] k_act={pars_ng['k_act']:.4g}, k_deact={pars_ng['k_deact']:.4g}, "
          f"beta_g={pars_ng['beta_g']:.4g}, beta_l={pars_ng['beta_l']:.4g}")

    # NOLOC
    print("\n=== FIT: NOLOC (beta_l = 0, beta_g free) ===")
    theta_nl, res_nl, pars_nl = fit_case("noloc", t, Cg, Cl, P_scaled)
    print(f"[NOLOC] cost = {res_nl.cost:.4g}")
    print(f"[NOLOC] k_act={pars_nl['k_act']:.4g}, k_deact={pars_nl['k_deact']:.4g}, "
          f"beta_g={pars_nl['beta_g']:.4g}, beta_l={pars_nl['beta_l']:.4g}")

    # Save params
    np.savez(os.path.join(args.outdir, "params_full.npz"), **pars_full, residues=residues_aligned)
    np.savez(os.path.join(args.outdir, "params_noglob.npz"), **pars_ng, residues=residues_aligned)
    np.savez(os.path.join(args.outdir, "params_noloc.npz"), **pars_nl, residues=residues_aligned)

    print(f"\n[*] Saved parameter sets to {args.outdir}")


if __name__ == "__main__":
    main()
