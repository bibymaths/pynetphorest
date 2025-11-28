#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NetPhorest Python Implementation
================================

Author : Abhinav Mishra <mishraabhinav36@gmail.com>
Date   : 2025-06-15

Description
-----------
Core functions for NetPhorest-like neural network scoring.
Pure Python implementations of key algorithms from netphorest.c
for use in PyNetworkin.

This script predicts phosphorylation networks using the atlas and logic
derived from the original NetPhorest/KinomeXplorer methodology.

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

from typing import List, Iterable, Tuple, Iterator, Optional
import json, sys, sqlite3, math
from pathlib import Path
from importlib.resources import files

# Constants
NA = 21
# Maximum window size for sequence context
MAXWINDOW = 99
# Center position in the window
CENTER = 49
# Maximum hidden layer size
MAXHIDDEN = 20
# Amino Acid Alphabet used in NetPhorest
ALPHABET = "FIVWMLCHYAGNRTPDEQSK"
# Mapping from Amino Acid to Index
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
# Common phosphorylation-dependent binding domains (Readers)
READER_DOMAINS = {
    'SH2', 'PTB', 'C2', 'WW', '14-3-3', 'FHA', 'BRCT',
    'Polo-box', 'WD40', 'Broman', 'Chromodomain'
}
# Precomputed Sigmoid Lookup Table for fast approximation
SIGMOID_DATA: List[float] = [
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001, 0.000001,
    0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000002, 0.000002, 0.000002,
    0.000002, 0.000003, 0.000003, 0.000003, 0.000004, 0.000004, 0.000005, 0.000005,
    0.000006, 0.000007, 0.000008, 0.000009, 0.000010, 0.000011, 0.000013, 0.000015,
    0.000017, 0.000019, 0.000021, 0.000024, 0.000028, 0.000031, 0.000035, 0.000040,
    0.000045, 0.000051, 0.000058, 0.000066, 0.000075, 0.000085, 0.000096, 0.000109,
    0.000123, 0.000140, 0.000158, 0.000180, 0.000203, 0.000231, 0.000261, 0.000296,
    0.000335, 0.000380, 0.000431, 0.000488, 0.000553, 0.000626, 0.000710, 0.000804,
    0.000911, 0.001032, 0.001170, 0.001325, 0.001501, 0.001701, 0.001927, 0.002183,
    0.002473, 0.002801, 0.003173, 0.003594, 0.004070, 0.004610, 0.005220, 0.005911,
    0.006693, 0.007577, 0.008577, 0.009708, 0.010987, 0.012432, 0.014064, 0.015906,
    0.017986, 0.020332, 0.022977, 0.025957, 0.029312, 0.033086, 0.037327, 0.042088,
    0.047426, 0.053403, 0.060087, 0.067547, 0.075858, 0.085099, 0.095349, 0.106691,
    0.119203, 0.132964, 0.148047, 0.164516, 0.182426, 0.201813, 0.222700, 0.245085,
    0.268941, 0.294215, 0.320821, 0.348645, 0.377541, 0.407333, 0.437823, 0.468791,
    0.500000, 0.531209, 0.562177, 0.592667, 0.622459, 0.651355, 0.679179, 0.705785,
    0.731059, 0.754915, 0.777300, 0.798187, 0.817574, 0.835484, 0.851953, 0.867036,
    0.880797, 0.893309, 0.904651, 0.914901, 0.924142, 0.932453, 0.939913, 0.946597,
    0.952574, 0.957912, 0.962673, 0.966914, 0.970688, 0.974043, 0.977023, 0.979668,
    0.982014, 0.984094, 0.985936, 0.987568, 0.989013, 0.990292, 0.991423, 0.992423,
    0.993307, 0.994089, 0.994780, 0.995390, 0.995930, 0.996406, 0.996827, 0.997199,
    0.997527, 0.997817, 0.998073, 0.998299, 0.998499, 0.998675, 0.998830, 0.998968,
    0.999089, 0.999196, 0.999290, 0.999374, 0.999447, 0.999512, 0.999569, 0.999620,
    0.999665, 0.999704, 0.999739, 0.999769, 0.999797, 0.999820, 0.999842, 0.999860,
    0.999877, 0.999891, 0.999904, 0.999915, 0.999925, 0.999934, 0.999942, 0.999949,
    0.999955, 0.999960, 0.999965, 0.999969, 0.999972, 0.999976, 0.999979, 0.999981,
    0.999983, 0.999985, 0.999987, 0.999989, 0.999990, 0.999991, 0.999992, 0.999993,
    0.999994, 0.999995, 0.999995, 0.999996, 0.999996, 0.999997, 0.999997, 0.999997,
    0.999998, 0.999998, 0.999998, 0.999998, 0.999999, 0.999999, 0.999999, 0.999999,
    0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000
]


def sigmoid(x: float) -> float:
    """
    Fast Sigmoid Approximation using Lookup Table
    Input x is expected to be in range [-16.0, 16.0].

    Parameters
    ----------
    x : float
        The input value for the sigmoid function.
    Returns
    -------
    float
        The sigmoid of the input value.
    """
    if x <= -16.0:
        return 0.0
    if x >= 16.0:
        return 1.0
    # index = int(8*x + 128)
    return SIGMOID_DATA[int(8.0 * x + 128.0)]


def encode_peptide(peptide):
    """
    Maps peptide string to indices.
    Returns 0-19 for valid AAs.
    Returns 20 for Gap ('-').
    Returns 21 for Unsupported (U, X, Z, B, J, O).

    Parameters
    ----------
    peptide : str
        The peptide sequence.
    Returns
    -------
    List[int]
        List of indices corresponding to the peptide sequence.
    """
    return [CHAR_TO_IDX.get(aa, 21 if aa != '-' else 20) for aa in peptide]


def get_window(sequence, center_idx, window_size):
    """
    Centered window extraction with padding.
    Pads with '-' if window exceeds sequence bounds.

    Parameters
    ----------
    sequence : str
        The full sequence.
    center_idx : int
        The center index for the window.
    window_size : int
        The size of the window to extract.
    Returns
    -------
    str
        The extracted window with padding if necessary.
    """
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


def get_display_window(sequence, center_idx):
    """
    Extracts an 11-character window centered at center_idx.
    Pads with '-' if window exceeds sequence bounds.
    The center character is converted to lowercase.

    Parameters
    ----------
    sequence : str
        The full sequence.
    center_idx : int
        The center index for the window.
    Returns
    -------
    str
        The 11-character display window with center in lowercase.
    """
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


def score_pssm(indices, weights, divisor):
    """
    PSSM Scoring Function:

    - Each position in the peptide window contributes a weight based on the amino acid.
    - The weights are stored in a flat array where each position has 21 weights (20 AAs + 1 Gap).
    - The final score is the product of all position weights divided by the divisor.

    Parameters
    ----------
    indices : List[int]
        List of amino acid indices for the peptide window.
    weights : List[float]
        Flat list of weights for the PSSM.
    divisor : float
        The divisor to normalize the final score.
    Returns
    -------
    float
        The computed raw score for the peptide window.
    """
    stride = 21
    raw = 1.0
    for pos, aa_idx in enumerate(indices):
        safe_idx = 20 if aa_idx > 20 else aa_idx
        flat_idx = (pos * stride) + safe_idx
        if flat_idx < len(weights):
            raw *= weights[flat_idx]
    return raw / divisor


def slide_windows(seq: str) -> Iterator[Tuple[int, List[int]]]:
    """
    Sliding Window Generator:

    Yields (position, window) tuples for each position in the sequence.
    Windows are of size MAXWINDOW, centered around the current position.
    Pads with NA-1 for positions outside the sequence bounds.

    Parameters
    ----------
    seq : str
        The input amino acid sequence.
    Yields
    -------
    Iterator[Tuple[int, List[int]]]
        Yields tuples of (position, window) where window is a list of indices.
    """
    window = [NA - 1] * MAXWINDOW

    def aa_to_idx(c: str) -> Optional[int]:
        """
        Maps amino acid character to index.
        Returns None for unsupported characters.

        Parameters
        ----------
        c : str
            The amino acid character.
        Returns
        -------
        Optional[int]
            The index of the amino acid or None if unsupported.
        """
        try:
            return ALPHABET.index(c)
        except ValueError:
            return None

    n = 0

    for pos, c in enumerate(seq, start=1):
        if not c.isalpha():
            continue
        idx = aa_to_idx(c.upper())
        if idx is None:
            continue

        for i in range(1, MAXWINDOW):
            window[MAXWINDOW - i] = window[MAXWINDOW - i - 1]

        window[0] = idx
        n += 1

        yield n, window

    # tail
    for _ in range((MAXWINDOW - 1) // 2):

        for i in range(1, MAXWINDOW):
            window[MAXWINDOW - i] = window[MAXWINDOW - i - 1]

        window[0] = NA - 1
        n += 1

        yield n, window


def score_feed_forward(indices, weights, window, hidden):
    """
    Feed-Forward Neural Network Scoring Function:

    - The function supports both single-layer perceptrons (no hidden layer)
      and networks with one hidden layer.
    - Inputs are weighted and summed, biases are added, and the sigmoid
      activation function is applied at each neuron.
    - The final output is a combination of the two output neurons.

    Parameters
    ----------
    indices : List[int]
        List of amino acid indices for the peptide window.
    weights : List[float]
        Flat list of weights for the neural network.
    window : int
        The size of the input window.
    hidden : int
        The number of hidden layer neurons (0 for perceptron).
    Returns
    -------
    float
        The computed score from the neural network.
    """

    # Reverse indices to align with C-code's C-to-N weight layout
    rev_indices = indices[::-1]

    nw = window
    nh = hidden
    na = 21

    o = [0.0, 0.0]
    stride_input = (na * nw) + 1

    if nh > 0:
        # --- HIDDEN LAYER ---
        h_vals = []
        for i in range(nh):

            bias_idx = stride_input * (i + 1) - 1
            x = weights[bias_idx]

            # Sum Inputs
            block_start = stride_input * i
            for j in range(nw):
                w_idx = block_start + (na * j) + rev_indices[j]
                x += weights[w_idx]

            h_vals.append(sigmoid(x))

        # --- OUTPUT LAYER (2 Neurons) ---
        output_start_offset = stride_input * nh
        stride_output = nh + 1
        for i in range(2):
            bias_idx = output_start_offset + (stride_output * (i + 1)) - 1
            x = weights[bias_idx]

            block_start = output_start_offset + (stride_output * i)
            for j in range(nh):
                w_idx = block_start + j
                x += weights[w_idx] * h_vals[j]
            o[i] = sigmoid(x)

    else:
        # --- PERCEPTRON (No Hidden Layer) ---
        for i in range(2):
            bias_idx = stride_input * (i + 1) - 1
            x = weights[bias_idx]

            block_start = stride_input * i
            for j in range(nw):
                w_idx = block_start + (na * j) + rev_indices[j]
                x += weights[w_idx]
            o[i] = sigmoid(x)

    # --- OUTPUTS ---
    return (o[0] + 1.0 - o[1]) / 2.0


def fasta_iter(lines: Iterable[str]) -> Iterator[Tuple[str, str]]:
    """
    Simple FASTA parser yielding (name, sequence) tuples.

    Parameters
    ----------
    lines : Iterable[str]
        Lines from a FASTA file.
    Yields
    -------
    Iterator[Tuple[str, str]]
        Yields tuples of (name, sequence).
    """
    name = None
    seq_chunks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(seq_chunks)
            name = line[1:].split()[0]
            seq_chunks = []
        else:
            seq_chunks.append(line)
    if name is not None:
        yield name, "".join(seq_chunks)

def get_default_atlas_path() -> Path:
    """
    Return the path to the bundled NetPhorest atlas inside the package.

    Preference order:
    - netphorest.db
    - netphorest.json
    """
    try:
        base = files("pynetphorest")
        base_path = Path(base)
    except Exception:
        # Fallback for editable installs / local runs
        base_path = Path(__file__).resolve().parent

    for fname in ("netphorest.db", "netphorest.json"):
        cand = base_path /  'models' / fname
        if cand.exists():
            return cand

    # Last resort – will still error in load_atlas if missing
    return base_path / 'models' / "netphorest.db"

def load_atlas(path):
    """
    Load kinase models from a JSON or SQLite atlas file.
    Supports both legacy JSON format and modern SQLite database format.

    Args:
        path (str): Path to the atlas file (JSON or SQLite .db).
    Returns:
        list: List of kinase model dictionaries.
    """
    if path is None:
        p = get_default_atlas_path()
    else:
        p = Path(path)

    if not p.exists():
        print(f"Error: Atlas file '{p}' not found.")
        sys.exit(1)

    # Case 1: Load from SQLite Database
    if p.suffix == '.db':
        conn = sqlite3.connect(p)
        conn.row_factory = sqlite3.Row  # Allows accessing columns by name
        cursor = conn.cursor()

        models = []

        # Fetch all models
        cursor.execute("SELECT * FROM models")
        rows = cursor.fetchall()

        for row in rows:

            # Build base model structure
            model = {
                'id': row['id'],
                'type': row['type'],
                'residues': row['residues'].split(',') if row['residues'] else [],
                'divisor': row['divisor'],
                'sigmoid': {
                    'slope': row['sig_slope'],
                    'inflection': row['sig_inflection'],
                    'min': row['sig_min'],
                    'max': row['sig_max']
                },

                # Metadata for output
                'meta': {
                    'method': row['method'],
                    'tree': row['organism'],
                    'classifier': row['classifier'],
                    'kinase': row['kinase'],
                    'prior': row['prior']
                }
            }

            # Fetch the weights (components) for this model
            cursor.execute("""
                           SELECT window_size, hidden_units, weights
                           FROM model_components
                           WHERE model_id = ?
                           ORDER BY component_index
                           """, (row['id'],))

            components = cursor.fetchall()

            # Handle Neural Network Structure
            if model['type'] == 'NN':
                model['networks'] = []
                for comp in components:
                    model['networks'].append({
                        'window': comp['window_size'],
                        'hidden': comp['hidden_units'],
                        'weights': json.loads(comp['weights'])
                    })

            # Handle PSSM Structure
            elif model['type'] == 'PSSM':
                if components:
                    comp = components[0]
                    model['window'] = comp['window_size']
                    model['weights'] = json.loads(comp['weights'])

            models.append(model)

        conn.close()
        return models

    # Case 2: Load from JSON (Legacy support)
    else:
        with open(p, "r") as f:
            data = json.load(f)
            # Handle both raw list and dictionary wrapper format
            return data['models'] if isinstance(data, dict) and 'models' in data else data


def parse_fasta(path):
    """
    Parse a FASTA file and return a dictionary of sequences.

    Args:
        path (str): Path to the FASTA file.
    Returns:
        dict: Dictionary mapping sequence names to sequences.
    """
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


def classify_model_role(model):
    """
    Determines if a model is a 'Writer' (Kinase) or a 'Reader' (Binding Domain).
    Returns 'WRITER' or 'READER'.
    """
    # Check the classifier string from the atlas metadata
    classifier = model['meta'].get('classifier', '').upper()
    group = model['meta'].get('kinase', '').upper()

    # Check against known reader domains
    if any(domain in classifier for domain in READER_DOMAINS):
        return 'READER'
    if any(domain in group for domain in READER_DOMAINS):
        return 'READER'

    # Default to WRITER (Kinase)
    return 'WRITER'


def get_model_posterior(seq_upper, i, model):
    """
    Calculates the posterior probability for a specific site and model.
    Encapsulates PSSM/NN selection and Sigmoid transformation.
    Returns: float (0.0 to 1.0)
    """
    if model["type"] == "PSSM":
        peptide = get_window(seq_upper, i, model["window"])
        indices = encode_peptide(peptide)
        if any(idx > 20 for idx in indices): return 0.0

        raw_score = score_pssm(indices, model["weights"], model["divisor"])

    elif model["type"] == "NN":
        total_nn_score = 0.0
        valid_ensemble = True
        for net in model["networks"]:
            peptide = get_window(seq_upper, i, net["window"])
            indices = encode_peptide(peptide)
            if any(idx > 20 for idx in indices):
                valid_ensemble = False
                break
            total_nn_score += score_feed_forward(indices, net["weights"], net["window"], net["hidden"])

        if not valid_ensemble: return 0.0
        raw_score = total_nn_score / model["divisor"]
    else:
        return 0.0

    if raw_score <= 0.0: return 0.0

    # Log transformation for PSSM
    log_score = math.log(raw_score) if model["type"] == "PSSM" else raw_score

    # Sigmoid Transformation
    sig = model["sigmoid"]
    term = sig["slope"] * (sig["inflection"] - log_score)

    # Clamp for numerical stability
    if term > 50.0:
        term = 50.0
    elif term < -50.0:
        term = -50.0

    posterior = sig["min"] + (sig["max"] - sig["min"]) / (1.0 + math.exp(term))
    return posterior
