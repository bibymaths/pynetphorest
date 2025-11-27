#!/usr/bin/env python3
import gzip
import pandas as pd
import numpy as np
import argparse
import re


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


def extract_pos(res_str):
    """
    Extract integer position from residue string like 'Y535' or 'S1173'.
    Returns None if it fails.
    """
    m = re.match(r"[A-Z]([0-9]+)", res_str)
    if not m:
        return None
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", required=True)
    parser.add_argument("--within", required=True)
    parser.add_argument("--between", required=True)
    parser.add_argument("--out-global", default="C_ptm.tsv",
                        help="Output PTM-based C matrix (global).")
    parser.add_argument("--out-local", default="C_dist.tsv",
                        help="Output distance-based C matrix (local).")
    parser.add_argument("--list-out", default="sites.tsv")
    parser.add_argument("--length-scale", type=float, default=50.0,
                        help="Length scale L (aa) for distance kernel exp(-|i-j|/L).")
    args = parser.parse_args()

    print("[*] Loading PTMcode2 within...")
    W = load_within(args.within)

    print("[*] Loading PTMcode2 between...")
    B = load_between(args.between)

    rows = W + B
    print(f"[*] Total phosphorylation pairs loaded: {len(rows)}")

    prot = args.protein

    # ---- FILTER: keep ONLY EGFR–EGFR pairs (intra-protein) ----
    rows = [(p1, res1, r1, p2, res2, r2)
            for (p1, res1, r1, p2, res2, r2) in rows
            if (p1 == prot and p2 == prot)]

    print(f"[*] Pairs after filtering for {prot}-{prot}: {len(rows)}")
    if not rows:
        raise ValueError(f"No PTM pairs found for protein {prot}")

    # Collect EGFR sites
    site_set = set()
    for p1, res1, r1, p2, res2, r2 in rows:
        site_set.add(f"{p1}_{res1}")
        site_set.add(f"{p2}_{res2}")

    site_list = sorted(site_set)
    N = len(site_list)
    print(f"[*] Total unique EGFR sites: {N}")

    idx = {s: i for i, s in enumerate(site_list)}

    # ------------------------------------------------------------------
    # 1) PTM-based C (GLOBAL): exactly what you already had
    # ------------------------------------------------------------------
    C_ptm = np.zeros((N, N), dtype=float)
    for p1, res1, r1, p2, res2, r2 in rows:
        s1 = f"{p1}_{res1}"
        s2 = f"{p2}_{res2}"
        i, j = idx[s1], idx[s2]

        score = 0.8 * (r1 + r2) / 200.0   # your original scoring
        C_ptm[i, j] = max(C_ptm[i, j], score)
        C_ptm[j, i] = max(C_ptm[j, i], score)

    # ------------------------------------------------------------------
    # 2) Distance-based C (LOCAL): from residue positions only
    # ------------------------------------------------------------------
    # extract positions from strings like 'EGFR_Y1173'
    positions = []
    for s in site_list:
        _, res = s.split("_", 1)  # 'EGFR', 'Y1173'
        pos = extract_pos(res)
        if pos is None:
            raise ValueError(f"Could not parse position from residue '{res}'")
        positions.append(pos)
    positions = np.array(positions, dtype=float)

    C_dist = np.zeros((N, N), dtype=float)
    L = float(args.length_scale)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = abs(positions[i] - positions[j])  # aa distance
            # Simple exponential kernel: closer sites → stronger coupling
            C_dist[i, j] = np.exp(-d / L)

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    pd.DataFrame(C_ptm, index=site_list, columns=site_list).to_csv(
        args.out_global, sep="\t"
    )
    pd.DataFrame(C_dist, index=site_list, columns=site_list).to_csv(
        args.out_local, sep="\t"
    )
    pd.DataFrame({"site": site_list, "position": positions}).to_csv(
        args.list_out, sep="\t", index=False
    )

    print(f"[*] Saved PTM-based C_global to {args.out_global}")
    print(f"[*] Saved distance-based C_local to {args.out_local}")
    print(f"[*] Saved site list to {args.list_out}")


if __name__ == "__main__":
    main()
