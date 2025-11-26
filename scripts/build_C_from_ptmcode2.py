#!/usr/bin/env python3
import gzip
import pandas as pd
import numpy as np
import argparse


def load_within(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split("\t")
            if len(parts) < 13:
                continue

            protein = parts[0]
            species = parts[1]
            if species != "Homo sapiens":
                continue
            if parts[2] != "phosphorylation" or parts[6] != "phosphorylation":
                continue

            res1 = parts[3]  # e.g. Y535
            r1 = float(parts[4])
            res2 = parts[7]
            r2 = float(parts[8])

            rows.append((protein, res1, r1, protein, res2, r2))
    return rows


def load_between(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split("\t")
            if len(parts) < 13:
                continue

            p1, p2 = parts[0], parts[1]
            species = parts[2]
            if species != "Homo sapiens":
                continue
            if parts[3] != "phosphorylation" or parts[7] != "phosphorylation":
                continue

            res1 = parts[4]   # e.g. S588
            r1 = float(parts[5])
            res2 = parts[8]   # e.g. S380
            r2 = float(parts[9])

            rows.append((p1, res1, r1, p2, res2, r2))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", required=True)
    parser.add_argument("--within", required=True)
    parser.add_argument("--between", required=True)
    parser.add_argument("--out", default="C_matrix.tsv")
    parser.add_argument("--list-out", default="sites.tsv")
    args = parser.parse_args()

    print("[*] Loading PTMcode2 within...")
    W = load_within(args.within)

    print("[*] Loading PTMcode2 between...")
    B = load_between(args.between)

    rows = W + B
    print(f"[*] Total phosphorylation pairs loaded: {len(rows)}")

    # Collect all unique sites
    site_set = set()
    for p1, r1, _, p2, r2, _ in rows:
        site_set.add(f"{p1}_{r1}")
        site_set.add(f"{p2}_{r2}")

    site_list = sorted(site_set)
    N = len(site_list)
    print(f"[*] Total unique sites: {N}")

    prot = args.protein

    # ---- FILTER: keep ONLY EGFR–EGFR pairs (intra-protein, Option A) ----
    rows = [(p1, res1, r1, p2, res2, r2)
            for (p1, res1, r1, p2, res2, r2) in rows
            if (p1 == prot and p2 == prot)]

    print(f"[*] Pairs after filtering for {prot}-{prot}: {len(rows)}")

    if not rows:
        raise ValueError(f"No PTM pairs found for protein {prot}")

    site_set = set()
    for p1, res1, r1, p2, res2, r2 in rows:
        site_set.add(f"{p1}_{res1}")
        site_set.add(f"{p2}_{res2}")

    site_list = sorted(site_set)
    N = len(site_list)
    print(f"[*] Total unique sites: {N}")

    idx = {s: i for i, s in enumerate(site_list)}

    C = np.zeros((N, N), dtype=float)
    for p1, res1, r1, p2, res2, r2 in rows:
        s1 = f"{p1}_{res1}"
        s2 = f"{p2}_{res2}"
        i, j = idx[s1], idx[s2]

        score = 0.8 * (r1 + r2) / 200.0
        C[i, j] = score
        C[j, i] = score

    # Save matrix and site list
    pd.DataFrame(C, index=site_list, columns=site_list).to_csv(args.out, sep="\t")
    pd.DataFrame({"site": site_list}).to_csv(args.list_out, sep="\t", index=False)

    print(f"[*] Saved C matrix to {args.out}")
    print(f"[*] Saved site list to {args.list_out}")


if __name__ == "__main__":
    main()
