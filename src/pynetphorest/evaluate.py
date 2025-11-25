#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)


def load_eval_data(npz_path: Path):
    """
    Expect an .npz file with at least:
      - X_test
      - y_test
    Optional:
      - w_test  (sample weights)
    """

    data = np.load(npz_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]

    w_test = None
    if "w_test" in data.files:
        w_raw = data["w_test"]
        if isinstance(w_raw, np.ndarray) and w_raw.dtype == object and w_raw.shape == ():
            w_test = None
        elif w_raw is None:
            w_test = None
        else:
            w_test = w_raw

    return X_test, y_test, w_test


def load_edge_metadata(meta_path: Path):
    """
    Expect a JSON-lines file (one dict per line) with keys like:
      type, rrcs1, rrcs2, ...
    """
    meta = []
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def metrics_and_curves(clf, X_test, y_test, w_test, out_prefix: Path):
    """
    Compute metrics and generate PR/ROC/confusion plots.
    """

    print("\n=== EVALUATION METRICS ===")
    probs = clf.predict_proba(X_test)[:, 1]

    # sanitize w_test again, just in case
    if isinstance(w_test, np.ndarray) and w_test.dtype == object and w_test.shape == ():
        w_test = None

    # --- Average precision ---
    if w_test is None:
        ap = average_precision_score(y_test, probs)
    else:
        ap = average_precision_score(y_test, probs, sample_weight=w_test)
    print(f"AP Score:   {ap:.4f}")

    # --- ROC AUC ---
    try:
        if w_test is None:
            roc = roc_auc_score(y_test, probs)
        else:
            roc = roc_auc_score(y_test, probs, sample_weight=w_test)
        print(f"ROC AUC:    {roc:.4f}")
    except ValueError:
        print("ROC AUC:    cannot compute (only one class present in y_test)")

    # --- Brier score ---
    if w_test is None:
        brier = brier_score_loss(y_test, probs)
    else:
        brier = brier_score_loss(y_test, probs, sample_weight=w_test)
    print(f"Brier score:{brier:.4f}")

    # Threshold @0.8
    y_pred = (probs >= 0.8).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix @0.8:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    print(f"Precision@0.8: {prec:.3f}")
    print(f"Recall@0.8:    {rec:.3f}")

    # PR curve
    if w_test is None:
        precs, recs, _ = precision_recall_curve(y_test, probs)
    else:
        precs, recs, _ = precision_recall_curve(y_test, probs, sample_weight=w_test)

    plt.figure()
    plt.plot(recs, precs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve")
    pr_path = out_prefix.with_suffix(".pr.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    print(f"Saved PR curve to {pr_path}")

    # ROC curve plot (if valid)
    try:
        fpr, tpr, _ = roc_curve(y_test, probs, sample_weight=w_test)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        roc_path = out_prefix.with_suffix(".roc.png")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()
        print(f"Saved ROC curve to {roc_path}")
    except ValueError:
        print("Skipped ROC curve plot (only one class present).")

    # Probability histogram
    plt.figure()
    plt.hist(probs, bins=50)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability distribution")
    prob_hist_path = out_prefix.with_suffix(".probs.png")
    plt.savefig(prob_hist_path, bbox_inches="tight")
    plt.close()
    print(f"Saved probability histogram to {prob_hist_path}")

    # Confusion-matrix heatmap
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Confusion matrix @0.8")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    cm_path = out_prefix.with_suffix(".cm.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_path}")

    return probs


def feature_group_importance(clf, out_prefix: Path):
    """
    Summarize and plot feature-group importance assuming:
      - index 0: avg posterior
      - index 1–5: top5 posteriors
      - index 6+: peptide encoding
    """
    if not hasattr(clf, "feature_importances_"):
        print("\nNo feature_importances_ on model; skipping importance plot.")
        return

    fi = clf.feature_importances_
    print("\n=== FEATURE IMPORTANCES ===")
    print("Top 10 raw important feature indices:",
          np.argsort(fi)[::-1][:10].tolist())

    avg_imp = fi[0]
    top5_imp = fi[1:6].sum() if fi.shape[0] > 1 else 0.0
    enc_imp = fi[6:].sum() if fi.shape[0] > 6 else 0.0

    print(f"Importance – avg posterior:    {avg_imp:.4f}")
    print(f"Importance – top5 posteriors:  {top5_imp:.4f}")
    print(f"Importance – peptide encoding: {enc_imp:.4f}")

    # Bar plot of the three groups
    labels = ["avg_posterior", "top5_posteriors", "encoding"]
    values = [avg_imp, top5_imp, enc_imp]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Importance")
    plt.title("Feature-group importance")
    imp_path = out_prefix.with_suffix(".feat_importance.png")
    plt.savefig(imp_path, bbox_inches="tight")
    plt.close()
    print(f"Saved feature-group importance plot to {imp_path}")


def summarize_posteriors_from_dataset(npz_path: Path):
    """
    If you saved the full dataset as npz with X and y, you can
    inspect distribution of the first feature (avg posterior).
    """
    print("\n=== DATASET FEATURE SUMMARY (avg posterior) ===")
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    avg_post_pos = X[y == 1, 0]
    avg_post_neg = X[y == 0, 0]

    print("Avg posterior (pos mean):", float(avg_post_pos.mean()))
    print("Avg posterior (neg mean):", float(avg_post_neg.mean()))
    print("Avg posterior (pos std): ", float(avg_post_pos.std()))
    print("Avg posterior (neg std): ", float(avg_post_neg.std()))


def summarize_rrcs_from_edge_meta(edge_meta_path: Path):
    """
    Use edge_metadata to inspect true rRCS distribution.
    """
    print("\n=== TRUE rRCS SUMMARY (from edge metadata) ===")
    meta = load_edge_metadata(edge_meta_path)

    r_pos = []
    r_neg = []
    for m in meta:
        mean_rrcs = 0.8 * (m.get("rrcs1", 0.0) + m.get("rrcs2", 0.0))
        if m.get("type") == "pos":
            r_pos.append(mean_rrcs)
        else:
            r_neg.append(mean_rrcs)

    if r_pos:
        r_pos = np.array(r_pos)
        print("rRCS pos mean:", float(r_pos.mean()))
        print("rRCS pos std: ", float(r_pos.std()))
    if r_neg:
        r_neg = np.array(r_neg)
        print("rRCS neg mean:", float(r_neg.mean()))
        print("rRCS neg std: ", float(r_neg.std()))


def analyze_predictions_tsv(pred_path: Path):
    """
    Analyze a crosstalk_predictions.tsv-style file:
      Protein  Site1  Site2  Prob
    """
    print("\n=== PREDICTION TSV SUMMARY ===")
    edge_counts = {}
    max_prob = {}
    total_prob = {}
    all_probs = []

    with open(pred_path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            prot, _, _, prob_str = parts
            prob = float(prob_str)
            all_probs.append(prob)
            edge_counts[prot] = edge_counts.get(prot, 0) + 1
            total_prob[prot] = total_prob.get(prot, 0.0) + prob
            max_prob[prot] = max(max_prob.get(prot, 0.0), prob)

    if not all_probs:
        print("No edges in predictions file.")
        return

    all_probs = np.array(all_probs)
    print("Total edges:", all_probs.size)
    for q in [0.8, 0.9, 0.95, 0.99]:
        print(f"prob p{int(q*100)}: {np.quantile(all_probs, q):.3f}")
    print("mean prob:", float(all_probs.mean()), "std:", float(all_probs.std()))

    print("\nTop 10 proteins by number of edges:")
    top_prots = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for prot, cnt in top_prots:
        mean_p = total_prob[prot] / cnt
        print(f"{prot}: edges={cnt}, max_prob={max_prob[prot]:.3f}, "
              f"mean_prob={mean_p:.3f}")

    # Probability histogram for predictions file
    plt.figure()
    plt.hist(all_probs, bins=50)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability distribution (predictions TSV)")
    hist_path = pred_path.with_suffix(".probs.png")
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction probability histogram to {hist_path}")

def _compute_group_metrics(label, idxs, y, probs, out_rows):
    """
    Compute AP / ROC / Brier for a masked subset and append to out_rows.
    """
    if not idxs:
        print(f"{label}: 0 samples, skipping.")
        return

    mask = np.asarray(idxs, dtype=int)
    y_g = y[mask]
    p_g = probs[mask]

    ap = average_precision_score(y_g, p_g)
    try:
        roc = roc_auc_score(y_g, p_g)
    except ValueError:
        roc = float("nan")
    brier = brier_score_loss(y_g, p_g)

    roc_str = "nan" if np.isnan(roc) else f"{roc:.3f}"

    print(
        f"{label}: n={len(y_g)}, AP={ap:.3f}, ROC={roc_str}, Brier={brier:.3f}"
    )

    out_rows.append({
        "group": label,
        "n": int(len(y_g)),
        "ap": float(ap),
        "roc_auc": None if np.isnan(roc) else float(roc),
        "brier": float(brier),
    })

def subgroup_metrics_full_dataset(
    clf,
    dataset_npz_path: Path,
    edge_metadata_path: Path,
    out_prefix: Path,
):
    """
    Per-subgroup metrics using the FULL dataset (X, y) + edge_metadata.

    Assumes:
      - dataset_npz contains X, y
      - edge_metadata.jsonl has one JSON dict per sample, aligned with X/y
      - each metadata dict has:
          type: "pos"/"neg"
          structure: "within"/"between"
          aa1, aa2: residue letters ("S","T","Y",...)
    """
    print("\n=== PER-SUBGROUP METRICS (FULL DATASET) ===")

    data = np.load(dataset_npz_path, allow_pickle=False)
    X = data["X"]
    y = data["y"]

    meta = load_edge_metadata(edge_metadata_path)

    if len(meta) != len(y):
        print(
            f"WARNING: len(metadata)={len(meta)} != len(y)={len(y)}; "
            "cannot do subgroup metrics safely. Skipping."
        )
        return

    probs = clf.predict_proba(X)[:, 1]

    # ----- group indices by residue (aa1/aa2) -----
    # here we only take edges where BOTH sites are S/T/Y respectively;
    # everything else is treated as "mixed"
    res_groups = {"S": [], "T": [], "Y": [], "mixed": []}
    for i, m in enumerate(meta):
        aa1 = m.get("aa1")
        aa2 = m.get("aa2")
        if aa1 == aa2 and aa1 in ("S", "T", "Y"):
            res_groups[aa1].append(i)
        else:
            res_groups["mixed"].append(i)

    # ----- group indices by edge type: within / between -----
    type_groups = {"within": [], "between": []}
    for i, m in enumerate(meta):
        struct = (m.get("structure") or "").lower()
        if struct in type_groups:
            type_groups[struct].append(i)

    # Collect metrics into a table
    rows = []

    print("\n--- By residue group (aa1/aa2) ---")
    for label, idxs in res_groups.items():
        _compute_group_metrics(f"residue:{label}", idxs, y, probs, rows)

    print("\n--- By edge structure (within/between) ---")
    for label, idxs in type_groups.items():
        _compute_group_metrics(f"structure:{label}", idxs, y, probs, rows)

    # Save TSV summary
    import csv
    tsv_path = out_prefix.with_suffix(".subgroups.tsv")
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "n", "ap", "roc_auc", "brier"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved subgroup metrics table to {tsv_path}")

    # Simple bar plots for AP by residue and by structure
    # 1) residue APs
    res_labels = []
    res_ap = []
    for r in ("S", "T", "Y", "mixed"):
        label = f"residue:{r}"
        for row in rows:
            if row["group"] == label:
                res_labels.append(r)
                res_ap.append(row["ap"])
                break

    if res_labels:
        plt.figure()
        plt.bar(res_labels, res_ap)
        plt.ylabel("AP")
        plt.title("Average Precision by residue group")
        res_plot = out_prefix.with_suffix(".subgroups_residues.png")
        plt.savefig(res_plot, bbox_inches="tight")
        plt.close()
        print(f"Saved residue subgroup AP plot to {res_plot}")

    # 2) structure APs
    struct_labels = []
    struct_ap = []
    for t in ("within", "between"):
        label = f"structure:{t}"
        for row in rows:
            if row["group"] == label:
                struct_labels.append(t)
                struct_ap.append(row["ap"])
                break

    if struct_labels:
        plt.figure()
        plt.bar(struct_labels, struct_ap)
        plt.ylabel("AP")
        plt.title("Average Precision by edge type")
        struct_plot = out_prefix.with_suffix(".subgroups_edge_type.png")
        plt.savefig(struct_plot, bbox_inches="tight")
        plt.close()
        print(f"Saved edge-type subgroup AP plot to {struct_plot}")

def run_evaluation(
    model_path: str,
    eval_npz_path: str | None = None,
    dataset_npz_path: str | None = None,
    edge_metadata_path: str | None = None,
    predictions_tsv_path: str | None = None,
    out_prefix: str | None = None,
):
    """
    Programmatic entry point used by the Typer CLI.

    Parameters mirror the CLI flags, but are plain strings.
    """
    os.makedirs(out_prefix, exist_ok=True)
    model_path = Path(model_path)

    if out_prefix is None:
        out_prefix_path = model_path.with_suffix("")  # e.g. crosstalk_model
    else:
        out_prefix_path = Path(out_prefix)

    # Load model
    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # 1) Evaluation metrics and plots if eval data is provided
    if eval_npz_path:
        X_test, y_test, w_test = load_eval_data(Path(eval_npz_path))
        _ = metrics_and_curves(clf, X_test, y_test, w_test, out_prefix_path)
        feature_group_importance(clf, out_prefix_path)
    else:
        print("\nNo eval_npz_path given; skipping test-set metrics and plots.")

    # 2) Dataset-level avg posterior summary
    if dataset_npz_path:
        summarize_posteriors_from_dataset(Path(dataset_npz_path))

    # 3) True rRCS summary from edge metadata
    if edge_metadata_path:
        summarize_rrcs_from_edge_meta(Path(edge_metadata_path))

    # 4) Prediction TSV summary
    if predictions_tsv_path:
        analyze_predictions_tsv(Path(predictions_tsv_path))

    # 5) Subgroup metrics from full dataset + edge metadata
    if dataset_npz_path and edge_metadata_path:
        subgroup_metrics_full_dataset(
            clf,
            Path(dataset_npz_path),
            Path(edge_metadata_path),
            out_prefix_path,
        )

def main():
    parser = argparse.ArgumentParser(
        description="Offline analysis of crosstalk model (.pkl) and outputs."
    )
    parser.add_argument("--model", required=True, help="Path to trained model .pkl")
    parser.add_argument("--eval-npz", help="Path to npz with X_test, y_test, optional w_test")
    parser.add_argument(
        "--dataset-npz",
        help="Optional: npz with full dataset X, y for avg posterior summary",
    )
    parser.add_argument(
        "--edge-metadata",
        help="Optional: JSON-lines file with edge metadata for rRCS summary",
    )
    parser.add_argument(
        "--predictions-tsv",
        help="Optional: crosstalk_predictions.tsv to summarize predictions",
    )
    parser.add_argument(
        "--out-prefix",
        help="Prefix for plots/metrics files (default: model path without extension)",
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        eval_npz_path=args.eval_npz,
        dataset_npz_path=args.dataset_npz,
        edge_metadata_path=args.edge_metadata,
        predictions_tsv_path=args.predictions_tsv,
        out_prefix=args.out_prefix,
    )
