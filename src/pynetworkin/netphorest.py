import argparse
import json
import math
import sys
from pathlib import Path
import core as core

# Preferred: native python atlas
try:
    from netphorest_atlas import MODELS as DEFAULT_MODELS
except ImportError:
    DEFAULT_MODELS = None


def load_atlas_json(path):
    if not Path(path).exists():
        print(f"Error: Atlas file '{path}' not found.")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)["models"]


def parse_fasta(path):
    seqs = {}
    name = None
    parts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(parts)
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if name:
            seqs[name] = "".join(parts)
    return seqs


def main():
    parser = argparse.ArgumentParser(description="NetPhorest Python Predictor")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out", help="Output file (tsv)", default=None)
    parser.add_argument(
        "--atlas",
        help="Optional JSON atlas path. If omitted, uses netphorest_atlas package.",
        default=None
    )
    args = parser.parse_args()

    # Load Models
    if args.atlas is not None:
        models = load_atlas_json(args.atlas)
    else:
        if DEFAULT_MODELS is None:
            print("Error: netphorest_atlas package not found and --atlas not provided.")
            sys.exit(1)
        models = DEFAULT_MODELS

    sequences = parse_fasta(args.fasta)

    # Setup Output
    if args.out:
        out_handle = open(args.out, 'w')
    else:
        out_handle = sys.stdout

    header = "# Name\tPosition\tResidue\tPeptide\tMethod\tTree\tClassifier\tPosterior\tPrior"
    out_handle.write(header + "\n")

    for name, seq in sequences.items():
        seq_upper = seq.upper()

        for i, res in enumerate(seq_upper):
            if res not in ['S', 'T', 'Y']:
                continue

            for model in models:
                if res not in model['residues']:
                    continue

                raw_score = 0.0

                if model['type'] == 'PSSM':
                    peptide = core.get_window(seq_upper, i, model['window'])
                    indices = core.encode_peptide(peptide)
                    if any(idx > 20 for idx in indices):
                        continue
                    raw_score = core.score_pssm(indices, model['weights'], model['divisor'])

                elif model['type'] == 'NN':
                    total_nn_score = 0.0
                    valid_ensemble = True
                    for net in model['networks']:
                        peptide = core.get_window(seq_upper, i, net['window'])
                        indices = core.encode_peptide(peptide)
                        if any(idx > 20 for idx in indices):
                            valid_ensemble = False
                            break
                        total_nn_score += core.score_feed_forward(
                            indices, net['weights'], net['window'], net['hidden']
                        )
                    if not valid_ensemble:
                        continue
                    raw_score = total_nn_score / model['divisor']

                if raw_score <= 0:
                    continue

                log_score = math.log(raw_score) if model['type'] == 'PSSM' else raw_score

                sig = model['sigmoid']
                term = sig['slope'] * (sig['inflection'] - log_score)
                if term > 50: term = 50.0
                if term < -50: term = -50.0

                posterior = sig['min'] + (sig['max'] - sig['min']) / (1.0 + math.exp(term))

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
