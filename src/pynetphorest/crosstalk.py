#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NetPhorest Python Implementation
================================

Author : Abhinav Mishra <mishraabhinav36@gmail.com>
Date   : 2025-06-15

Description
-----------
This module provides functionality to train a machine learning model to predict
post-translational modification (PTM) crosstalk between phosphorylation sites
based on features derived from the NetPhorest kinase prediction platform.

It includes functions to load protein sequences, extract features for phosphorylation sites,
parse PTMcode data, train a classifier, and predict crosstalk on new protein sequences. The
model uses a Gradient Boosting Classifier and incorporates rRCS-based weighting for positive
samples during training.

Reference
---------
Horn, H. et al. (2014). KinomeXplorer: an integrated platform for
kinome biology studies. Nature Methods, 11(6), 603–604.
https://doi.org/10.1038/nmeth.2968

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

import gzip
import json
import pathlib
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from joblib import delayed, Parallel
from . import core

WINDOW_SIZE = 9
NEGATIVE_RATIO = 3
BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_ATLAS_PATH = core.get_default_atlas_path()


def load_sequences(fasta_path):
    """
    Simple FASTA parser returning dict {header: sequence}.
    Handles multi-line sequences.

    Args:
        fasta_path (str): Path to the FASTA file.
    Returns:
        dict: Mapping of sequence names to sequences.
    """
    seqs = {}
    name = None
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                name = line.strip().split()[0][1:]
                seqs[name] = []
            elif name:
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def extract_site_features(seq, pos, aa, models, rrcs=0.0):
    """
    Features: [Avg_Post, Top5_Post..., Peptide_Encoding...]
    NetPhorest-only; rRCS is NOT used as an input feature.

    Args:
        seq (str): Protein sequence.
        pos (int): Position of the residue in the sequence (0-based).
        aa (str): Amino acid at the position (should be 'S', 'T', or 'Y').
        models (list): List of NetPhorest models to use for scoring.
        rrcs (float): Relative Residue Conservation Score (not used here).
    Returns:
        list or None: Feature vector or None if invalid position/AA.
    """
    if pos < 0 or pos >= len(seq) or seq[pos] != aa:
        return None

        # 1. Physical Context (Peptide)
    peptide = core.get_window(seq, pos, WINDOW_SIZE)
    encoded = core.encode_peptide(peptide)

    # 2. NetPhorest Posterior Scores
    scores = []
    for model in models:
        # Calculate posterior using core logic
        score = core.get_model_posterior(seq, pos, model)
        scores.append(score)

    scores = sorted(scores, reverse=True)
    top_scores = scores[:5]  # Keep top 5 kinase probabilities
    # Pad if fewer than 5 models
    while len(top_scores) < 5:
        top_scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    # NOTE: rrcs intentionally NOT included in the feature vector
    return [avg_score] + top_scores + encoded


def parse_ptmcode_line(line, structure="within"):
    """
    Parse a line from PTMcode data.

    Args:
        line (str): A line from the PTMcode file.
        structure (str): "within" or "between" indicating the type of edge.
    Returns:
        dict or None: Parsed data or None if invalid.
    """
    parts = line.strip().split('\t')

    def parse_res(res_str):
        """
        Parse residue string like 'S588' into (aa, position).
        Position is converted to 0-based index.

        Args:
            res_str (str): Residue string (e.g., 'S588').
        Returns:
            tuple: (amino_acid (str), position (int))
        """
        # 'S588' -> ('S', 587)
        return res_str[0], int(res_str[1:]) - 1

    try:
        if structure == "within":
            # Protein, Species, PTM1, Res1, rRCS1, Prop1, PTM2...
            p1 = parts[0]
            p2 = parts[0]
            ptm1_type, res1_raw, rrcs1 = parts[2], parts[3], float(parts[4])
            ptm2_type, res2_raw, rrcs2 = parts[6], parts[7], float(parts[8])
        else:  # between
            p1, p2 = parts[0], parts[1]
            ptm1_type, res1_raw, rrcs1 = parts[3], parts[4], float(parts[5])
            ptm2_type, res2_raw, rrcs2 = parts[7], parts[8], float(parts[9])

        if "phosphorylation" not in ptm1_type or "phosphorylation" not in ptm2_type:
            return None

        aa1, pos1 = parse_res(res1_raw)
        aa2, pos2 = parse_res(res2_raw)

        return {
            "p1": p1, "p2": p2,
            "aa1": aa1, "pos1": pos1, "rrcs1": rrcs1,
            "aa2": aa2, "pos2": pos2, "rrcs2": rrcs2,
            "label": 1
        }
    except (ValueError, IndexError):
        return None


def train_model(fasta, within_file, between_file, atlas_path=pathlib.Path | None, output_model="crosstalk_model.pkl"):
    """
    Train a crosstalk prediction model using PTMcode data.

    Args
        fasta (str): Path to the FASTA file with protein sequences.
        within_file (str): Path to the PTMcode within-protein edges file.
        between_file (str): Path to the PTMcode between-protein edges file.
        atlas_path (str or None): Path to the NetPhorest atlas file.
        output_model (str): Path to save the trained model.
    Returns:
        None
    """
    if atlas_path is None:
        atlas_path = DEFAULT_ATLAS_PATH
    print(f"Loading Atlas from {atlas_path}...")
    atlas = core.load_atlas(str(atlas_path))
    models = atlas.get("models", []) if isinstance(atlas, dict) else atlas

    models_by_res = {"S": [], "T": [], "Y": []}
    for m in models:
        for r in m.get("residues", []):
            if r in models_by_res:
                models_by_res[r].append(m)

    print("Loading Sequences...")
    sequences = load_sequences(fasta)
    dataset = []
    edge_metadata = []

    files = [(within_file, "within"), (between_file, "between")]
    valid_edges = 0

    site_cache = {}

    def get_site_features(prot, pos, aa, rrcs):
        """
        Cached site features.
        Compute heavy part once per (protein, pos), then just inject rRCS.

        Args:
            prot (str): Protein ID.
            pos (int): Position in the protein (0-based).
            aa (str): Amino acid at the position.
            rrcs (float): rRCS value for this site.
        Returns:
            list or None: Feature vector or None if invalid.
        """
        key = (prot, pos)
        base = site_cache.get(key)
        if base is None:
            seq = sequences.get(prot)
            if seq is None:
                return None
            used_models = models_by_res.get(aa, models)  # if you use models_by_res
            base = extract_site_features(seq, pos, aa, used_models, rrcs=0.0)
            if base is None:
                return None
            site_cache[key] = base

        feat = list(base)
        # feat[0] = rrcs  # overwrite rRCS for this edge
        # NetPhorest-only features; rRCS not injected here
        return feat

    print("Processing PTMcode edges...")
    seq_ids = set(sequences.keys())
    for fpath, ftype in files:
        if not fpath:
            continue

        open_func = gzip.open if fpath.endswith(".gz") else open

        with open_func(fpath, "rt") as f:
            for line in tqdm(f, desc=f"Parsing {ftype}"):
                if not line or line[0] == "#":
                    continue
                edge = parse_ptmcode_line(line, ftype)
                if not edge:
                    continue

                # fast reject by protein ID
                if edge["p1"] not in seq_ids or edge["p2"] not in seq_ids:
                    continue

                feat1 = get_site_features(edge["p1"], edge["pos1"],
                                          edge["aa1"], edge["rrcs1"])
                feat2 = get_site_features(edge["p2"], edge["pos2"],
                                          edge["aa2"], edge["rrcs2"])

                if feat1 and feat2:
                    combined = feat1 + feat2 + [abs(a - b) for a, b in zip(feat1, feat2)]
                    dataset.append(combined + [1])
                    edge_metadata.append({
                        "type": "pos",
                        "p1": edge["p1"],
                        "p2": edge["p2"],
                        "rrcs1": edge["rrcs1"],
                        "rrcs2": edge["rrcs2"],
                        "aa1": edge["aa1"],
                        "aa2": edge["aa2"],
                        "pos1": edge["pos1"],
                        "pos2": edge["pos2"]
                    })
                    valid_edges += 1

    print(f"Generated {valid_edges} positive samples. Generating negatives...")
    neg_count = 0
    target_neg = valid_edges * NEGATIVE_RATIO

    # Precompute STY positions for each protein ONCE
    sty_sites = {}
    for prot, seq in sequences.items():
        sites = [i for i, c in enumerate(seq) if c in "STY"]
        if len(sites) >= 2:
            sty_sites[prot] = sites

    # Optional: shuffle protein order for randomness
    prot_list = list(sty_sites.keys())
    random.shuffle(prot_list)

    with tqdm(total=target_neg, desc="Negatives") as pbar:
        for prot in prot_list:
            seq = sequences[prot]
            sites = sty_sites[prot]

            # generate all unordered pairs of STY sites in this protein
            site_pairs = list(combinations(sites, 2))
            random.shuffle(site_pairs)  # randomize order within protein

            for i1, i2 in site_pairs:
                aa1 = seq[i1]
                aa2 = seq[i2]

                feat1 = get_site_features(prot, i1, aa1, rrcs=0.0)
                feat2 = get_site_features(prot, i2, aa2, rrcs=0.0)

                if feat1 and feat2:
                    combined = feat1 + feat2 + [abs(a - b) for a, b in zip(feat1, feat2)]
                    dataset.append(combined + [0])
                    edge_metadata.append({
                        "type": "neg",
                        "p1": prot,
                        "p2": prot,
                        "rrcs1": 0.0,
                        "rrcs2": 0.0,
                        "aa1": aa1,
                        "aa2": aa2,
                        "pos1": i1,
                        "pos2": i2
                    })
                    neg_count += 1
                    pbar.update(1)

                    if neg_count >= target_neg:
                        break

            if neg_count >= target_neg:
                break

    #### Debugging / dataset summary (optional)

    # print("\n=== DATASET SUMMARY ===")
    #
    # # Total rows
    # print("Total samples:", len(dataset))
    #
    # # Count positives & negatives
    # pos = sum(1 for row in dataset if row[-1] == 1)
    # neg = len(dataset) - pos
    # print("Positives:", pos)
    # print("Negatives:", neg)
    #
    # # Dedup check
    # df_temp = pd.DataFrame(dataset)
    # unique_rows = len(df_temp.drop_duplicates())
    # print("Unique samples:", unique_rows)
    #
    # # Duplication ratio
    # dup_ratio = 1 - (unique_rows / len(df_temp))
    # print(f"Duplicate ratio: {dup_ratio:.3f}")
    #
    # # Feature dimensionality
    # if len(dataset) > 0:
    #     print("Feature vector length:", len(dataset[0]) - 1)
    #
    # rrcs_pos = [row[0] for row in dataset if row[-1] == 1]
    # rrcs_neg = [row[0] for row in dataset if row[-1] == 0]
    #
    # print("RRCS positive mean:", np.mean(rrcs_pos))
    # print("RRCS negative mean:", np.mean(rrcs_neg))
    # print("RRCS positive unique values:", sorted(set(rrcs_pos))[:10])
    # print("RRCS negative unique values:", sorted(set(rrcs_neg))[:10])

    df = pd.DataFrame(dataset)
    df.to_csv("full_dataset.csv", index=False)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    weights = []
    for row_meta in edge_metadata:
        if row_meta["type"] == "pos":
            # rRCS-based weighting (scale to ~0–1)
            w = (row_meta["rrcs1"] + row_meta["rrcs2"]) / 200.0
            w = max(w, 0.1)  # avoid zero-weights
            weights.append(w)
        else:
            weights.append(1.0)

    with open("edge_metadata.json", "w") as f:
        for m in edge_metadata:
            f.write(json.dumps(m) + "\n")

    print("Saved edge_metadata.json")

    df = pd.DataFrame(dataset, dtype="float32")
    X = df.iloc[:, :-1].to_numpy(dtype="float32")
    y = df.iloc[:, -1].to_numpy(dtype="int8")
    weights = np.array(weights, dtype="float32")

    np.savez("full_dataset.npz",
             X=X,
             y=y)
    print("Saved full_dataset.npz with X/y for feature summaries")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Save evaluation split
    np.savez("eval_data.npz",
             X_test=X_test,
             y_test=y_test,
             w_test=w_test if 'sample_weight' in locals() else None)
    print("Saved eval_data.npz with X_test/y_test/w_test")

    print("Training Gradient Boosting Classifier...")

    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        scoring="average_precision",
        random_state=42
    )
    clf.fit(X_train, y_train, sample_weight=w_train)

    print("Evaluation:")
    if len(X_test) > 0:
        probs = clf.predict_proba(X_test)[:, 1]
        print(f"AP Score: {average_precision_score(y_test, probs):.4f}")
        print(f"ROC AUC:    {roc_auc_score(y_test, probs):.4f}")

    with open(output_model, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {output_model}")


def _predict_for_protein(name, seq, models_by_res, models, clf, threshold):
    """
    Predict crosstalk for all STY site pairs in a single protein.

    Args:
        name (str): Protein name/ID.
        seq (str): Protein sequence.
        models_by_res (dict): Residue-specific NetPhorest models.
        models (list): All NetPhorest models.
        clf: Trained classifier.
        threshold (float): Probability threshold for reporting crosstalk.
    Returns:
        list: List of TSV lines with predictions.
    """
    sites = [i for i, c in enumerate(seq) if c in "STY"]
    if len(sites) < 2:
        return []

    # ----- 1) Extract per-site feature vectors -----
    vectors = {}
    for site in sites:
        aa = seq[site]
        used_models = models_by_res.get(aa, models)
        vec = extract_site_features(seq, site, aa, used_models, rrcs=0.0)
        if vec:
            vectors[site] = vec

    valid_sites = list(vectors.keys())
    if len(valid_sites) < 2:
        return []

    # ----- 2) Build all pair feature vectors at once -----
    pair_features = []
    pair_meta = []  # (s1, s2)

    for i in range(len(valid_sites)):
        for j in range(i + 1, len(valid_sites)):
            s1, s2 = valid_sites[i], valid_sites[j]
            v1, v2 = vectors[s1], vectors[s2]

            # Combine: v1 + v2 + abs-diff
            combined = v1 + v2 + [abs(a - b) for a, b in zip(v1, v2)]

            pair_features.append(combined)
            pair_meta.append((s1, s2))

    if not pair_features:
        return []

    # Convert to array
    X_pairs = np.array(pair_features, dtype=np.float32)

    # ----- 3) Vectorized prediction -----
    # If extremely large, chunk to avoid memory blow-up.
    # Chunk size can be increased if your machine handles more.
    CHUNK = 20000
    probs = []

    for k in range(0, len(X_pairs), CHUNK):
        p = clf.predict_proba(X_pairs[k:k + CHUNK])[:, 1]
        probs.append(p)

    probs = np.concatenate(probs)

    # ----- 4) Convert to TSV lines -----
    lines = []
    for (s1, s2), prob in zip(pair_meta, probs):
        aa1 = seq[s1]
        aa2 = seq[s2]

        if aa1 == aa2 == "S":
            th = 0.25
        elif aa1 == aa2 == "Y":
            th = 0.35
        else:
            th = 0.30
        # if prob >= threshold:
        if prob >= th:
            res1 = f"{seq[s1]}{s1 + 1}"
            res2 = f"{seq[s2]}{s2 + 1}"
            lines.append(f"{name}\t{res1}\t{res2}\t{prob:.4f}\n")

    return lines


def predict(
        fasta,
        atlas_path=pathlib.Path | None,
        model_path=pathlib.Path | None,
        out=pathlib.Path | None,
        threshold=0.8,
        n_jobs=-1,
):
    """
    Predict crosstalk for all proteins in a FASTA file.

    Args:
        fasta (str): Path to the FASTA file with protein sequences.
        atlas_path (str or None): Path to the NetPhorest atlas file.
        model_path (str or None): Path to the trained model file.
        out (str or None): Path to the output TSV file.
        threshold (float): Probability threshold for reporting crosstalk.
        n_jobs (int): Number of parallel jobs (-1 for all cores).
    Returns:
        None
    """
    if atlas_path is None:
        atlas_path = DEFAULT_ATLAS_PATH

    print("Loading Atlas and Model...")
    atlas = core.load_atlas(atlas_path)

    # Handle JSON or SQLite atlas
    models = atlas["models"] if isinstance(atlas, dict) else atlas

    # Build residue-specific model lists, same as in train_model
    models_by_res = {"S": [], "T": [], "Y": []}
    for m in models:
        for r in m.get("residues", []):
            if r in models_by_res:
                models_by_res[r].append(m)

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    print(f"Predicting crosstalk for {fasta}...")
    sequences = load_sequences(fasta)
    items = list(sequences.items())

    # Run per-protein prediction in parallel
    # Use backend="loky" for true multi-core (separate processes).
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_predict_for_protein)(name, seq, models_by_res, models, clf, threshold)
        for name, seq in tqdm(items, desc="Proteins", total=len(items), smoothing=0.1)
    )

    # Write output once
    with open(out, "w") as f_out:
        f_out.write("Protein\tSite1\tSite2\tCrosstalk_Prob\n")
        for lines in results:
            for line in lines:
                f_out.write(line)

    print(f"Results written to {out}")
