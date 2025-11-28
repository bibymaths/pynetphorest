#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NetPhorest Python Implementation
================================

Author : Abhinav Mishra <mishraabhinav36@gmail.com>
Date   : 2025-06-15

Description
-----------
This script provides a Python implementation of the NetPhorest kinase
substrate prediction algorithm. It reads kinase models from a JSON or
SQLite atlas file and predicts phosphorylation sites in input protein
sequences provided in FASTA format. The output is a tab-separated values
file containing predicted sites along with their scores and metadata.


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

from joblib import Parallel, delayed
import argparse, math, sys, pathlib, os
from tqdm import tqdm
from . import core


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(
        prog="netphorest-py",
        description=(
            "NetPhorest Python Predictor: scans protein sequences for S/T/Y sites and "
            "scores them against kinase models from a NetPhorest atlas."
        ),
        epilog=(
            "Examples:\n"
            "  netphorest-py input.fasta > predictions.tsv\n"
            "  netphorest-py input.fasta --out preds.tsv --atlas netphorest.db\n"
            "  netphorest-py input.fasta --atlas atlas.json\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "fasta",
        type=pathlib.Path,
        metavar="FASTA",
        help=(
            "Input FASTA file with protein sequences.\n"
            "- Must be valid FASTA.\n"
            "- Non-standard residues (e.g., B, Z, J, X) are skipped during scoring.\n"
            "- Use '-' to read FASTA from stdin."
        )
    )

    parser.add_argument(
        "--out",
        type=pathlib.Path,
        metavar="TXT",
        default=None,
        help=(
            "Output TSV file path.\n"
            "- If not provided, results are written to stdout.\n"
            "- If a path is given, parent directories must exist.\n"
            "- Use '-' to force stdout."
        )
    )

    parser.add_argument(
        "--atlas",
        type=pathlib.Path,
        metavar="ATLAS",
        default=None,
        help=(
            "Path to NetPhorest atlas (.db/.sqlite/.json).\n"
            "If omitted, uses the atlas bundled with the package."
        )
    )

    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable Writer->Reader causal linking (Kinase recruits Binder)."
    )

    args = parser.parse_args()

    # FASTA input: allow stdin via "-"
    if str(args.fasta) == "-":
        fasta_handle = sys.stdin
    else:
        if not args.fasta.exists():
            parser.error(f"FASTA file not found: {args.fasta}")
        fasta_handle = open(args.fasta, "r")

    # Resolve atlas path (use package-bundled atlas if not provided)
    if args.atlas is None:
        atlas_path = core.get_default_atlas_path()
    else:
        atlas_path = args.atlas

    if not atlas_path.exists():
        parser.error(f"Atlas file not found: {atlas_path}")

    # Load Models
    models = core.load_atlas(args.atlas)
    sequences = core.parse_fasta(args.fasta)

    writers, readers = [], []
    if args.causal:
        for m in models:
            if core.classify_model_role(m) == 'READER':
                readers.append(m)
            else:
                writers.append(m)
    else:
        # In standard mode, we treat everything as a flat list of models
        writers = models
        readers = []

    # Setup Output
    if args.out:
        out_handle = open(args.out, 'w')
    else:
        out_handle = sys.stdout

    # Decide which models to use based on mode
    models_to_pass = writers if args.causal else models

    def score_position(name, seq_upper, i, res, models, readers=None, causal_mode=False):
        """
        Score a single S/T/Y position of a protein against all models.
        Returns list of output lines (can be empty).
        """
        out_lines = []

        # ------------------------------------------------------------
        # BRANCH 1: STANDARD PREDICTION (Your Exact Original Code)
        # ------------------------------------------------------------
        if not causal_mode:
            for model in models:
                if res not in model["residues"]:
                    continue

                raw_score = 0.0

                # --- Explicit Inline Logic (Preserved) ---
                if model["type"] == "PSSM":
                    peptide = core.get_window(seq_upper, i, model["window"])
                    indices = core.encode_peptide(peptide)
                    # skip if any index is out of amino-acid range
                    if any(idx > 20 for idx in indices):
                        continue
                    raw_score = core.score_pssm(
                        indices,
                        model["weights"],
                        model["divisor"],
                    )

                elif model["type"] == "NN":
                    total_nn_score = 0.0
                    valid_ensemble = True

                    for net in model["networks"]:
                        peptide = core.get_window(seq_upper, i, net["window"])
                        indices = core.encode_peptide(peptide)
                        if any(idx > 20 for idx in indices):
                            valid_ensemble = False
                            break
                        total_nn_score += core.score_feed_forward(
                            indices,
                            net["weights"],
                            net["window"],
                            net["hidden"],
                        )

                    if not valid_ensemble:
                        continue

                    raw_score = total_nn_score / model["divisor"]
                # -----------------------------------------

                if raw_score <= 0.0:
                    continue

                # transform score
                if model["type"] == "PSSM":
                    log_score = math.log(raw_score)
                else:
                    log_score = raw_score

                sig = model["sigmoid"]
                term = sig["slope"] * (sig["inflection"] - log_score)
                if term > 50.0:
                    term = 50.0
                elif term < -50.0:
                    term = -50.0

                posterior = sig["min"] + (sig["max"] - sig["min"]) / (1.0 + math.exp(term))

                if posterior > 0.0:
                    visual = core.get_display_window(seq_upper, i)
                    meta = model["meta"]
                    line = (
                        f"{name}\t{i + 1}\t{res}\t{visual}\t"
                        f"{meta['method']}\t{meta['tree']}\t{meta['classifier']}\t"
                        f"{meta['kinase']}\t{posterior:.6f}\t{meta['prior']:.6f}"
                    )
                    out_lines.append(line)

            return out_lines

        # ------------------------------------------------------------
        # BRANCH 2: CAUSAL PREDICTION (New Extension)
        # ------------------------------------------------------------
        else:
            # Note: We use core.get_model_posterior here to keep the new logic clean
            # and avoid copying the massive block above a second time.

            # A. Find the best Writer (Kinase)
            best_kinase_name = "-"
            best_kinase_prob = 0.0

            for model in models:
                if res not in model["residues"]: continue

                # Using the helper for the new branch only
                prob = core.get_model_posterior(seq_upper, i, model)

                if prob > best_kinase_prob:
                    best_kinase_prob = prob
                    best_kinase_name = model['meta']['kinase']

            if best_kinase_prob < 0.1:
                return []

            # B. Check Readers (Binders)
            has_binder = False
            visual = core.get_display_window(seq_upper, i)

            if readers:
                for reader in readers:
                    if res not in reader["residues"]: continue

                    bind_prob = core.get_model_posterior(seq_upper, i, reader)

                    if bind_prob > 0.5:
                        has_binder = True
                        line = (
                            f"{name}\t{i + 1}\t{res}\t{visual}\t"
                            f"{best_kinase_name}\t{best_kinase_prob:.4f}\t"
                            f"{reader['meta']['kinase']}\t{bind_prob:.4f}\t{reader['meta']['classifier']}"
                        )
                        out_lines.append(line)

            if not has_binder and best_kinase_prob > 0.5:
                line = (
                    f"{name}\t{i + 1}\t{res}\t{visual}\t"
                    f"{best_kinase_name}\t{best_kinase_prob:.4f}\t"
                    f"-\t-\t-"
                )
                out_lines.append(line)

            return out_lines

    def process_one_protein(name, seq, models, readers=None, causal_mode=False, n_inner=None):
        """
        Process one protein:
          - collect all S/T/Y sites
          - run per-site scoring in parallel (threads) inside this process
        """
        seq_upper = seq.upper()
        positions = [(i, res) for i, res in enumerate(seq_upper) if res in ("S", "T", "Y")]

        if not positions:
            return []

        # choose inner parallelism (threads)
        if n_inner is None:
            # heuristic: a few threads per process, but not crazy
            n_inner = max(1, min(4, os.cpu_count() or 1))

        # inner parallel: per-site, thread-based
        # We now pass 'readers' and 'causal_mode' into the worker function
        site_results = Parallel(
            n_jobs=n_inner,
            backend="threading",
        )(
            delayed(score_position)(name, seq_upper, i, res, models, readers, causal_mode)
            for (i, res) in tqdm(positions, desc=f"{name} sites", leave=False)
        )

        # flatten
        out_lines = []
        for lines in site_results:
            if lines:
                out_lines.extend(lines)
        return out_lines

    # ------------------------------------------------------------
    # Outer-level parallelism: per protein (process-based)
    # ------------------------------------------------------------
    seq_items = list(sequences.items())

    if len(seq_items) < 2:
        # small input: no outer processes, but keep inner threading
        results = [
            process_one_protein(
                name, seq,
                models=models_to_pass,
                readers=readers,
                causal_mode=args.causal,
                n_inner=max(1, (os.cpu_count() or 1) // 2)
            )
            for name, seq in seq_items
        ]
    else:
        # outer processes, inner threads
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(process_one_protein)(
                name, seq,
                models=models_to_pass,
                readers=readers,
                causal_mode=args.causal
            )
            for name, seq in tqdm(seq_items, desc="Proteins")
        )

    # write results
    # Write Header
    if args.causal:
        header = "# Substrate\tPos\tRes\tPeptide\tTop_Kinase\tKin_Prob\tRecruited_Binder\tBind_Prob\tBinder_Type"
    else:
        header = "# Name\tPosition\tResidue\tPeptide\tMethod\tTree\tClassifier\tPosterior\tPrior"

    out_handle.write(header + "\n")

    for lines in results:
        for line in lines:
            out_handle.write(line + "\n")

    if out_handle is not sys.stdout:
        out_handle.close()


if __name__ == "__main__":
    main()
