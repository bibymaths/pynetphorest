import argparse
import json
import math
import sys
from pathlib import Path
import netphorest_core as core


def load_atlas(path):
    if not Path(path).exists():
        print(f"Error: Atlas file '{path}' not found.")
        print("Please run builder.py first to generate the JSON atlas.")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)['models']


def parse_fasta(path):
    seqs = {}
    name = None
    parts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if name: seqs[name] = "".join(parts)
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if name: seqs[name] = "".join(parts)
    return seqs


def main():
    parser = argparse.ArgumentParser(description="NetPhorest Python Predictor")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out", help="Output file (tsv)", default=None)
    parser.add_argument("--atlas", help="Path to JSON atlas", default="netphorest_atlas.json")
    args = parser.parse_args()

    # Load Models
    models = load_atlas(args.atlas)
    sequences = parse_fasta(args.fasta)

    # Setup Output
    if args.out:
        out_handle = open(args.out, 'w')
    else:
        out_handle = sys.stdout

    # Match C Header Exactly (Missing Kinase label is intentional to match C output)
    header = "# Name\tPosition\tResidue\tPeptide\tMethod\tTree\tClassifier\tPosterior\tPrior"
    out_handle.write(header + "\n")

    # Prediction Loop
    for name, seq in sequences.items():
        seq_upper = seq.upper()

        for i, res in enumerate(seq_upper):
            # Optimization: NetPhorest only targets S, T, Y
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

                    # C-Code Check: Reject unsupported chars
                    if any(idx > 20 for idx in indices):
                        continue

                    raw_score = core.score_pssm(indices, model['weights'], model['divisor'])

                elif model['type'] == 'NN':
                    total_nn_score = 0.0
                    valid_ensemble = True

                    for net in model['networks']:
                        peptide = core.get_window(seq_upper, i, net['window'])
                        indices = core.encode_peptide(peptide)

                        # C-Code Check: Reject unsupported chars
                        if any(idx > 20 for idx in indices):
                            valid_ensemble = False
                            break

                        total_nn_score += core.score_feed_forward(
                            indices, net['weights'], net['window'], net['hidden']
                        )

                    if not valid_ensemble:
                        continue

                    raw_score = total_nn_score / model['divisor']

                # 3. Post-Processing
                if raw_score <= 0:
                    continue

                if model['type'] == 'PSSM':
                    log_score = math.log(raw_score)
                else:
                    log_score = raw_score

                # Sigmoid Scaling
                sig = model['sigmoid']
                term = sig['slope'] * (sig['inflection'] - log_score)

                if term > 50: term = 50.0
                if term < -50: term = -50.0

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