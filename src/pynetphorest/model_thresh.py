#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NetPhorest Python Implementation
================================

Author : Abhinav Mishra <mishraabhinav36@gmail.com>
Date   : 2025-06-15

Description
-----------
This module provides functions to evaluate a trained crosstalk prediction model. It includes loading evaluation data,
computing metrics (AP, ROC AUC, Brier score), generating plots (PR curve, ROC curve, confusion matrix), summarizing feature importances,
analyzing prediction TSV files, and computing subgroup metrics based on edge metadata.

License
-------
# Copyright (c) 2025, Abhinav Mishra
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def load_model(path: str):
    with open(path, "rb") as f:
        clf = pickle.load(f)
    return clf


def load_eval_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    return X_test, y_test


def load_full_dataset(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    y_all = data["y"]
    return y_all


def load_edge_metadata(path: str) -> List[Dict]:
    meta = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def compute_test_indices(y_all: np.ndarray, test_size: float = 0.2, seed: int = 42) -> np.ndarray:
    """
    Reproduce the same train/test split used during training.
    We stratify by y and use the same random_state.
    """
    indices = np.arange(len(y_all))
    _, idx_test = train_test_split(
        indices,
        test_size=test_size,
        stratify=y_all,
        random_state=seed,
    )
    return idx_test


def residue_group_from_meta(m: Dict) -> str:
    """
    Classify a sample into residue group based on aa1/aa2.

    Groups:
    - S, T, Y if both sites share that residue
    - mixed otherwise
    """
    aa1 = m.get("aa1", "")
    aa2 = m.get("aa2", "")
    if aa1 == aa2 and aa1 in {"S", "T", "Y"}:
        return aa1
    return "mixed"


def build_residue_groups_for_test(meta_all: List[Dict], idx_test: np.ndarray) -> List[str]:
    """
    Build a residue-group label for each test sample, in test set order.
    """
    groups = []
    for idx in idx_test:
        g = residue_group_from_meta(meta_all[idx])
        groups.append(g)
    return groups


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute MCC, guarding against degenerate cases (only one class present).
    """
    if y_true.size == 0:
        return float("nan")
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        # In degenerate cases MCC is not informative; return 0 or NaN.
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def eval_thresholds(
        probs: np.ndarray,
        y_true: np.ndarray,
        residue_groups: List[str],
        thresholds: np.ndarray,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Compute global and per-residue metrics for a set of thresholds.

    Returns:
    - global_rows: list of dicts, one per threshold
    - residue_rows: list of dicts, one per (threshold, residue_group)
    """
    global_rows: List[Dict] = []
    residue_rows: List[Dict] = []

    residue_groups = np.array(residue_groups)

    for th in thresholds:
        y_pred = (probs >= th).astype(int)

        tp, fp, tn, fn = compute_confusion(y_true, y_pred)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = safe_mcc(y_true, y_pred)

        global_rows.append(
            {
                "threshold": th,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": mcc,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "pred_pos": tp + fp,
                "n": int(y_true.size),
            }
        )

        # per-residue groups
        for group_name in ["S", "T", "Y", "mixed"]:
            mask = residue_groups == group_name
            y_g = y_true[mask]
            y_pred_g = y_pred[mask]

            n_g = int(mask.sum())
            if n_g == 0:
                # No samples in this group for this split
                residue_rows.append(
                    {
                        "threshold": th,
                        "group": group_name,
                        "n": 0,
                        "n_pos": 0,
                        "precision": float("nan"),
                        "recall": float("nan"),
                        "f1": float("nan"),
                        "mcc": float("nan"),
                        "tp": 0,
                        "fp": 0,
                        "tn": 0,
                        "fn": 0,
                    }
                )
                continue

            tp_g, fp_g, tn_g, fn_g = compute_confusion(y_g, y_pred_g)

            precision_g = precision_score(y_g, y_pred_g, zero_division=0)
            recall_g = recall_score(y_g, y_pred_g, zero_division=0)
            f1_g = f1_score(y_g, y_pred_g, zero_division=0)
            mcc_g = safe_mcc(y_g, y_pred_g)

            residue_rows.append(
                {
                    "threshold": th,
                    "group": group_name,
                    "n": n_g,
                    "n_pos": int((y_g == 1).sum()),
                    "precision": precision_g,
                    "recall": recall_g,
                    "f1": f1_g,
                    "mcc": mcc_g,
                    "tp": tp_g,
                    "fp": fp_g,
                    "tn": tn_g,
                    "fn": fn_g,
                }
            )

    return global_rows, residue_rows


def print_global_table(rows: List[Dict]) -> None:
    print("# GLOBAL METRICS")
    print("# threshold\tprecision\trecall\tF1\tMCC\tpred_pos\tTP\tFP\tTN\tFN\tN")
    for r in rows:
        print(
            f"{r['threshold']:.2f}\t"
            f"{r['precision']:.3f}\t"
            f"{r['recall']:.3f}\t"
            f"{r['f1']:.3f}\t"
            f"{r['mcc']:.3f}\t"
            f"{r['pred_pos']}\t"
            f"{r['tp']}\t"
            f"{r['fp']}\t"
            f"{r['tn']}\t"
            f"{r['fn']}\t"
            f"{r['n']}"
        )


def print_residue_table(rows: List[Dict]) -> None:
    print("\n# PER-RESIDUE METRICS")
    print("# threshold\tgroup\tn\tn_pos\tprecision\trecall\tF1\tMCC\tTP\tFP\tTN\tFN")
    for r in rows:
        prec = r["precision"]
        rec = r["recall"]
        f1 = r["f1"]
        mcc = r["mcc"]

        def fmt(x):
            if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                return "nan"
            return f"{x:.3f}"

        print(
            f"{r['threshold']:.2f}\t"
            f"{r['group']}\t"
            f"{r['n']}\t"
            f"{r['n_pos']}\t"
            f"{fmt(prec)}\t"
            f"{fmt(rec)}\t"
            f"{fmt(f1)}\t"
            f"{fmt(mcc)}\t"
            f"{r['tp']}\t"
            f"{r['fp']}\t"
            f"{r['tn']}\t"
            f"{r['fn']}"
        )


def write_tsv(path: str, header: List[str], rows: List[Dict]) -> None:
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            fields = [str(r[h]) for h in header]
            f.write("\t".join(fields) + "\n")


def run_sweep_thresh(
        model: str = "crosstalk_model.pkl",
        eval_npz: str = "eval_data.npz",
        full_npz: str = "full_dataset.npz",
        meta_json: str = "edge_metadata.json",
        min_th: float = 0.10,
        max_th: float = 0.90,
        step: float = 0.05,
        out_global: str | None = None,
        out_residues: str | None = None,
):
    """
    Programmatic entry point for threshold sweep.

    Returns (global_rows, residue_rows), where each is a list[dict].
    """
    # Load pieces
    clf = load_model(model)
    X_test, y_test = load_eval_data(eval_npz)
    y_all = load_full_dataset(full_npz)
    meta_all = load_edge_metadata(meta_json)

    # Sanity checks
    if len(y_all) != len(meta_all):
        raise RuntimeError(
            f"Mismatch: y_all has {len(y_all)} samples but edge_metadata.json "
            f"has {len(meta_all)} lines."
        )

    # Reconstruct test indices and residue groups for test samples
    idx_test = compute_test_indices(y_all, test_size=0.2, seed=42)
    if len(idx_test) != len(y_test):
        raise RuntimeError(
            f"Test index length {len(idx_test)} != y_test length {len(y_test)}. "
            "Train/test split in this script must match training."
        )

    residue_groups = build_residue_groups_for_test(meta_all, idx_test)

    # Get probabilities once
    probs = clf.predict_proba(X_test)[:, 1]

    thresholds = np.arange(min_th, max_th + 1e-9, step)

    global_rows, residue_rows = eval_thresholds(
        probs=probs,
        y_true=y_test,
        residue_groups=residue_groups,
        thresholds=thresholds,
    )

    # Optional TSV outputs
    if out_global:
        header_g = [
            "threshold",
            "precision",
            "recall",
            "f1",
            "mcc",
            "tp",
            "fp",
            "tn",
            "fn",
            "pred_pos",
            "n",
        ]
        write_tsv(out_global, header_g, global_rows)

    if out_residues:
        header_r = [
            "threshold",
            "group",
            "n",
            "n_pos",
            "precision",
            "recall",
            "f1",
            "mcc",
            "tp",
            "fp",
            "tn",
            "fn",
        ]
        write_tsv(out_residues, header_r, residue_rows)

    return global_rows, residue_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep thresholds and compute global + per-residue metrics "
            "including MCC."
        )
    )
    parser.add_argument(
        "--model",
        default="crosstalk_model.pkl"
    )
    parser.add_argument(
        "--eval",
        default="eval_data.npz"
    )
    parser.add_argument(
        "--full",
        default="full_dataset.npz"
    )
    parser.add_argument(
        "--meta",
        default="edge_metadata.json"
    )
    parser.add_argument(
        "--min-th",
        type=float,
        default=0.10
    )
    parser.add_argument(
        "--max-th",
        type=float,
        default=0.90
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--out-global",
        default=None
    )
    parser.add_argument(
        "--out-residues",
        default=None
    )
    args = parser.parse_args()

    global_rows, residue_rows = run_sweep_thresh(
        model=args.model,
        eval_npz=args.eval,
        full_npz=args.full,
        meta_json=args.meta,
        min_th=args.min_th,
        max_th=args.max_th,
        step=args.step,
        out_global=args.out_global,
        out_residues=args.out_residues,
    )

    if not args.out_global:
        print_global_table(global_rows)
    if not args.out_residues:
        print_residue_table(residue_rows)


if __name__ == "__main__":
    main()
