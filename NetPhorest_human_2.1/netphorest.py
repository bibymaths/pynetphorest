import json
import math
import argparse
import sys
from pathlib import Path

# Alphabet Map (Index 0-20)
ALPHABET = "FIVWMLCHYAGNRTPDEQSK"
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


class NetPhorest:
    def __init__(self, atlas_path="netphorest_atlas.json"):
        if not Path(atlas_path).exists():
            print(f"Error: {atlas_path} not found. Please run builder.py first.")
            sys.exit(1)
        with open(atlas_path, "r") as f:
            self.models = json.load(f)

    def _get_window(self, sequence, center_idx, window_size):
        """Extracts sequence context for Scoring"""
        half = window_size // 2
        start = center_idx - half
        end = center_idx + half + 1

        prefix = ""
        suffix = ""

        if start < 0:
            prefix = "-" * abs(start)
            real_start = 0
        else:
            real_start = start

        if end > len(sequence):
            suffix = "-" * (end - len(sequence))
            real_end = len(sequence)
        else:
            real_end = end

        seq_part = sequence[real_start:real_end]
        return prefix + seq_part + suffix

    def _get_display_window(self, sequence, center_idx):
        """Mimics print_peptide() in C code (Fixed 11-mer)"""
        start = center_idx - 5
        end = center_idx + 6
        out = []
        for k in range(start, end):
            if k < 0 or k >= len(sequence):
                out.append('-')
            else:
                out.append(sequence[k])
        out[5] = out[5].lower()
        return "".join(out)

    def _encode(self, peptide):
        return [CHAR_TO_IDX.get(aa, 21 if aa != '-' else 20) for aa in peptide]

    def _sigmoid(self, x):
        if x > 16: return 1.0
        if x < -16: return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def predict(self, fasta_file, output_file=None):
        sequences = self._parse_fasta(fasta_file)

        if output_file:
            out_handle = open(output_file, 'w')
        else:
            out_handle = sys.stdout

        header = "Name\tPosition\tResidue\tPeptide\tMethod\tTree\tClassifier\tKinase\tPosterior\tPrior"
        out_handle.write(header + "\n")

        for name, seq in sequences.items():
            seq_upper = seq.upper()

            for i, res in enumerate(seq_upper):
                if res not in ['S', 'T', 'Y']: continue

                for model in self.models:
                    # --- CRITICAL FIX: RESIDUE FILTER ---
                    # Only run if residue matches model type (S, T, or Y)
                    if res not in model['residues']:
                        continue

                    raw_score = 0.0

                    if model['type'] == 'PSSM':
                        peptide = self._get_window(seq_upper, i, model['window'])
                        indices = self._encode(peptide)
                        raw_score = self._score_pssm(indices, model)

                    elif model['type'] == 'NN':
                        total_nn_score = 0.0
                        valid_ensemble = True
                        for net in model['networks']:
                            peptide = self._get_window(seq_upper, i, net['window'])
                            indices = self._encode(peptide)
                            if any(idx > 20 for idx in indices):
                                valid_ensemble = False;
                                break
                            total_nn_score += self._score_feed_forward(indices, net)

                        if not valid_ensemble: continue
                        raw_score = total_nn_score / model['divisor']

                    if raw_score <= 0: continue

                    log_score = math.log(raw_score)
                    sig = model['sigmoid']
                    term = sig['slope'] * (sig['inflection'] - log_score)
                    if term > 50: term = 50.0
                    if term < -50: term = -50.0
                    posterior = sig['min'] + (sig['max'] - sig['min']) / (1.0 + math.exp(term))

                    if posterior > 0.0:
                        visual = self._get_display_window(seq_upper, i)
                        meta = model['meta']
                        line = (f"{name}\t{i + 1}\t{res}\t{visual}\t"
                                f"{meta['method']}\t{meta['tree']}\t{meta['classifier']}\t"
                                f"{meta['kinase']}\t{posterior:.6f}\t{meta['prior']:.6f}")
                        out_handle.write(line + "\n")

        if output_file: out_handle.close()

    def _score_pssm(self, indices, model):
        weights = model['weights']
        stride = 21
        raw = 1.0
        for pos, aa_idx in enumerate(indices):
            safe_idx = 20 if aa_idx > 20 else aa_idx
            flat_idx = (pos * stride) + safe_idx
            if flat_idx < len(weights):
                raw *= weights[flat_idx]
        return raw / model['divisor']

    def _score_feed_forward(self, indices, network):
        weights = network['weights']
        nw = network['window']
        nh = network['hidden']
        na = 21
        o = [0.0, 0.0]
        stride_input = (na * nw) + 1

        if nh > 0:
            h_vals = []
            for i in range(nh):
                bias_idx = stride_input * (i + 1) - 1
                x = weights[bias_idx]
                block_start = stride_input * i
                for j in range(nw):
                    w_idx = block_start + (na * j) + indices[j]
                    x += weights[w_idx]
                h_vals.append(self._sigmoid(x))

            output_start_offset = stride_input * nh
            stride_output = nh + 1
            for i in range(2):
                bias_idx = output_start_offset + (stride_output * (i + 1)) - 1
                x = weights[bias_idx]
                block_start = output_start_offset + (stride_output * i)
                for j in range(nh):
                    w_idx = block_start + j
                    x += weights[w_idx] * h_vals[j]
                o[i] = self._sigmoid(x)
        else:
            for i in range(2):
                bias_idx = stride_input * (i + 1) - 1
                x = weights[bias_idx]
                block_start = stride_input * i
                for j in range(nw):
                    w_idx = block_start + (na * j) + indices[j]
                    x += weights[w_idx]
                o[i] = self._sigmoid(x)

        return (o[0] + 1.0 - o[1]) / 2.0

    def _parse_fasta(self, path):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    NetPhorest().predict(args.fasta, args.out)