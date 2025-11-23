import math
from typing import List, Iterable, Tuple, Iterator, Optional

NA = 21
MAXWINDOW = 99
CENTER = 49
MAXHIDDEN = 20
# Alphabet Map (Index 0-20)
# Based on netphorest.c: char alphabet[] = "FIVWMLCHYAGNRTPDEQSK-UXZJBO";
# F=0 ... K=19, -=20.
ALPHABET = "FIVWMLCHYAGNRTPDEQSK"
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}

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
    """Lookup-table sigmoid exactly like C."""
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
    """
    return [CHAR_TO_IDX.get(aa, 21 if aa != '-' else 20) for aa in peptide]


def get_window(sequence, center_idx, window_size):
    """Extracts sequence context for Scoring (variable size)"""
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
    Mimics print_peptide() in C code.
    Always returns 11 residues: Center +/- 5.
    """
    start = center_idx - 5
    end = center_idx + 6

    out = []
    for k in range(start, end):
        if k < 0 or k >= len(sequence):
            out.append('-')
        else:
            out.append(sequence[k])

    # Center to lowercase
    out[5] = out[5].lower()
    return "".join(out)


def score_pssm(indices, weights, divisor):
    stride = 21
    raw = 1.0
    for pos, aa_idx in enumerate(indices):
        # Safe idx handles Gap (20) correctly. Unsupported (>20) handled by caller.
        safe_idx = 20 if aa_idx > 20 else aa_idx
        flat_idx = (pos * stride) + safe_idx
        if flat_idx < len(weights):
            raw *= weights[flat_idx]
    return raw / divisor

def slide_windows(seq: str) -> Iterator[Tuple[int, List[int]]]:
    """
    Reproduces main() rolling window behavior:
    - window filled with NA-1 initially
    - shift right each char, insert new at s[0]
    - then flush tail with NA-1
    """
    window = [NA - 1] * MAXWINDOW

    def aa_to_idx(c: str) -> Optional[int]:
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
        # shift
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
    Exact Python Port of netphorest.c:feed_forward()

    CRITICAL FIX: The C code processes the window in 's' which is
    reversed (C-term at index 0). The NN weights are therefore stored
    C-term -> N-term.

    Our 'indices' input is N-term -> C-term.
    We must REVERSE 'indices' to match the C-code weight order.
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
            # Bias is at the END of the block in C code
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

    # --- COMBINE OUTPUTS ---
    return (o[0] + 1.0 - o[1]) / 2.0

def fasta_iter(lines: Iterable[str]) -> Iterator[Tuple[str, str]]:
    """Simple FASTA parser yielding (name, seq)."""
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