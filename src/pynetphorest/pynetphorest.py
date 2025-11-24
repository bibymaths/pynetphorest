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

import argparse, math, sys, pathlib
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
        default=pathlib.Path("netphorest.db"),
        help=(
            "Path to NetPhorest atlas containing kinase models.\n"
            "Supported formats (auto-detected by extension):\n"
            "- .db / .sqlite : SQLite atlas (recommended for speed)\n"
            "- .json         : JSON atlas\n"
            "Default: netphorest.db"
        )
    )

    args = parser.parse_args()

    # FASTA input: allow stdin via "-"
    if str(args.fasta) == "-":
        fasta_handle = sys.stdin
    else:
        if not args.fasta.exists():
            parser.error(f"FASTA file not found: {args.fasta}")
        fasta_handle = open(args.fasta, "r")

    # ATLAS must exist
    if not args.atlas.exists():
        parser.error(f"Atlas file not found: {args.atlas}")

    # Load Models
    models = core.load_atlas(args.atlas)
    sequences = core.parse_fasta(args.fasta)

    # Setup Output
    if args.out:
        out_handle = open(args.out, 'w')
    else:
        out_handle = sys.stdout

    # Write Header
    header = "# Name\tPosition\tResidue\tPeptide\tMethod\tTree\tClassifier\tPosterior\tPrior"
    out_handle.write(header + "\n")

    # Prediction Loop
    for name, seq in sequences.items():
        seq_upper = seq.upper()

        for i, res in enumerate(seq_upper):
            # NetPhorest only targets S, T, Y
            if res not in ['S', 'T', 'Y']:
                continue

            for model in models:
                # 1. Residue Filter (Context Awareness)
                if res not in model['residues']:
                    continue

                raw_score = 0.0

                # 2. Calculate Raw Score
                if model['type'] == 'PSSM':
                    peptide = core.get_window(seq_upper, i, model['window'])
                    indices = core.encode_peptide(peptide)

                    # Reject unsupported chars
                    if any(idx > 20 for idx in indices):
                        continue

                    raw_score = core.score_pssm(indices, model['weights'], model['divisor'])

                elif model['type'] == 'NN':
                    total_nn_score = 0.0
                    valid_ensemble = True

                    for net in model['networks']:
                        peptide = core.get_window(seq_upper, i, net['window'])
                        indices = core.encode_peptide(peptide)

                        # Reject unsupported chars
                        if any(idx > 20 for idx in indices):
                            valid_ensemble = False
                            break

                        # Feed-forward through the network
                        total_nn_score += core.score_feed_forward(
                            indices, net['weights'], net['window'], net['hidden']
                        )

                    if not valid_ensemble:
                        continue

                    # Average the ensemble score
                    raw_score = total_nn_score / model['divisor']

                # 3. Post-Processing
                if raw_score <= 0:
                    continue

                # Log Transformation for PSSM models only
                if model['type'] == 'PSSM':
                    log_score = math.log(raw_score)
                else:
                    log_score = raw_score

                # Sigmoid Scaling
                sig = model['sigmoid']
                term = sig['slope'] * (sig['inflection'] - log_score)

                if term > 50: term = 50.0
                if term < -50: term = -50.0

                # Calculate Posterior Probability
                posterior = sig['min'] + (sig['max'] - sig['min']) / (1.0 + math.exp(term))

                # 4. Output
                if posterior > 0.0:
                    visual = core.get_display_window(seq_upper, i)
                    meta = model['meta']

                    line = (f"{name}\t{i + 1}\t{res}\t{visual}\t"
                            f"{meta['method']}\t{meta['tree']}\t{meta['classifier']}\t"
                            f"{meta['kinase']}\t{posterior:.6f}\t{meta['prior']:.6f}")
                    out_handle.write(line + "\n")

    if args.out:
        out_handle.close()


if __name__ == "__main__":
    main()