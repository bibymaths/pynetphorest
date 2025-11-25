# --------------------------------------------------------------------------------
#
#                                 Likelihood module
# Original Date: 2013/06/05
# Updated: 2025 (Python 3 Support & Optimization)
#
# Description:
#   Handles Bayesian likelihood conversion table generation, smoothing,
#   and scoring for NetworKIN/NetPhorest.
#
# --------------------------------------------------------------------------------

import math
import sys


class CConvEntry:
    """
    Data structure representing a row in the conversion table.
    """

    def __init__(self):
        self.predictions_bin = []
        self.score = 0.0
        self.score_lower_bound = 0.0
        self.score_upper_bound = 0.0
        self.TPR = 0.0  # True Positive Rate
        self.FPR = 0.0  # False Positive Rate
        self.PPV = 0.0  # Positive Predictive Value
        self.FDR = 0.0  # False Discovery Rate
        self.L = 0.0  # Likelihood
        self.num_pos = 0
        self.num_neg = 0


def Count(FCondi, data_list):
    """Counts elements in data_list that satisfy condition FCondi."""
    return sum(1 for x in data_list if FCondi(x))


def mean(data_list):
    """Calculates arithmetic mean."""
    if not data_list:
        return 0.0
    return float(sum(data_list)) / len(data_list)


def ExtendBin(predictions, func_score, lower_bound, upper_bound):
    """
    Selects predictions falling strictly within the score bounds.
    Optimized to list comprehension.
    """
    return [p for p in predictions if lower_bound <= func_score(p) <= upper_bound]


def IsUpPeak(conv_tbl, i):
    """Checks if index i is a local maximum (peak)."""
    if conv_tbl[i - 1].L < conv_tbl[i].L and conv_tbl[i].L >= conv_tbl[i + 1].L:
        return True
    return False


def IsDownPeak(conv_tbl, i):
    """Checks if index i is a local minimum (valley)."""
    if conv_tbl[i - 1].L >= conv_tbl[i].L and conv_tbl[i].L < conv_tbl[i + 1].L:
        return True
    return False


def RecalculateEntryStats(conv_entry, predictions_bin, num_pos, num_neg, func_score, VAZD):
    """
    Helper to recalculate statistics for a merged bin.
    """
    num_predictions = num_pos + num_neg
    num_pos_bin = Count(lambda x: x.kz == 'k', predictions_bin)
    num_neg_bin = Count(lambda x: x.kz == 'z', predictions_bin)
    len_bin = len(predictions_bin)

    conv_entry.predictions_bin = predictions_bin
    conv_entry.score = mean([func_score(x) for x in predictions_bin])
    conv_entry.score_lower_bound = func_score(predictions_bin[-1])
    conv_entry.score_upper_bound = func_score(predictions_bin[0])

    conv_entry.TPR = float(num_pos_bin) / num_pos if num_pos else 0
    conv_entry.FPR = float(num_neg_bin) / num_neg if num_neg else 0
    conv_entry.PPV = float(num_pos_bin) / len_bin if len_bin else 0
    conv_entry.FDR = float(num_neg_bin) / len_bin if len_bin else 0

    # Bayesian Likelihood Formula with VAZD smoothing
    numerator = float(num_pos_bin) / num_pos if num_pos else 0

    denom_term1 = float(num_neg_bin + VAZD * (float(len_bin) / num_predictions))
    denom_term2 = num_neg + VAZD
    denominator = denom_term1 / denom_term2 if denom_term2 else 1

    conv_entry.L = numerator / denominator if denominator else 0
    conv_entry.num_pos = num_pos_bin
    conv_entry.num_neg = num_neg_bin
    return conv_entry


def Smooth3(conv_tbl, i, num_pos, num_neg, func_score, VAZD):
    """
    Merges 3 adjacent bins (i, i+1, i+2) into one to smooth a peak/valley.
    """
    # Combine predictions from 3 bins
    predictions_bin = list(set(conv_tbl[i].predictions_bin) |
                           set(conv_tbl[i + 1].predictions_bin) |
                           set(conv_tbl[i + 2].predictions_bin))

    # Sort: First by class (kz), then by score descending
    # Note: In Py3, sort is stable. 'k'/'z' strings sort deterministically.
    predictions_bin.sort(key=lambda x: x.kz)
    predictions_bin.sort(key=func_score, reverse=True)

    conv_entry = CConvEntry()
    RecalculateEntryStats(conv_entry, predictions_bin, num_pos, num_neg, func_score, VAZD)

    # Replace 3 entries with 1
    conv_tbl.pop(i + 2)
    conv_tbl.pop(i + 1)
    conv_tbl.pop(i)
    conv_tbl.insert(i, conv_entry)


def MergePoints(conv_tbl, i, num_pos, num_neg, func_score, VAZD):
    """
    Merges 2 adjacent bins (i, i+1).
    """
    predictions_bin = list(set(conv_tbl[i].predictions_bin) |
                           set(conv_tbl[i + 1].predictions_bin))

    predictions_bin.sort(key=lambda x: x.kz)
    predictions_bin.sort(key=func_score, reverse=True)

    conv_entry = CConvEntry()
    RecalculateEntryStats(conv_entry, predictions_bin, num_pos, num_neg, func_score, VAZD)

    conv_tbl.pop(i)
    conv_tbl.pop(i)
    conv_tbl.insert(i, conv_entry)


def SmoothUpPeak(conv_tbl, num_pos, num_neg, func_score, VAZD):
    """Smooths local maxima in the likelihood curve."""
    if len(conv_tbl) < 2:
        return False

    has_up_peak = False

    # Check tail
    if conv_tbl[-2].L < conv_tbl[-1].L:
        has_up_peak = True
        MergePoints(conv_tbl, len(conv_tbl) - 2, num_pos, num_neg, func_score, VAZD)
        i = len(conv_tbl) - 5
    else:
        i = len(conv_tbl) - 3

    if i < 0:
        return has_up_peak

    # Scan backwards
    if len(conv_tbl) >= 3:
        while True:
            if IsUpPeak(conv_tbl, i + 1):
                has_up_peak = True
                Smooth3(conv_tbl, i, num_pos, num_neg, func_score, VAZD)
                i -= 3
            else:
                i -= 1
            if i < 0:
                break

    return has_up_peak


def SmoothDownPeak(conv_tbl, num_pos, num_neg, func_score, VAZD):
    """Smooths local minima in the likelihood curve."""
    if len(conv_tbl) < 2:
        return False

    has_down_peak = False

    if len(conv_tbl) >= 3:
        i = len(conv_tbl) - 3
        while True:
            if IsDownPeak(conv_tbl, i + 1):
                has_down_peak = True
                Smooth3(conv_tbl, i, num_pos, num_neg, func_score, VAZD)
                i -= 3
            else:
                i -= 1
            if i < 0:
                break

    # Check head
    if len(conv_tbl) >= 2:
        if conv_tbl[1].L > conv_tbl[0].L:
            has_down_peak = True
            MergePoints(conv_tbl, 0, num_pos, num_neg, func_score, VAZD)

    return has_down_peak


def RemoveRedundancy(conv_tbl, num_pos, num_neg, func_score, VAZD):
    """Merges adjacent bins if they have identical scores."""
    if len(conv_tbl) < 2:
        return False

    has_same_points = False

    i = len(conv_tbl) - 2
    while True:
        if conv_tbl[i].score == conv_tbl[i + 1].score:  # Fixed comparison logic
            has_same_points = True
            MergePoints(conv_tbl, i, num_pos, num_neg, func_score, VAZD)
            i -= 2
        else:
            i -= 1
        if i < 0:
            break

    return has_same_points


def LocalSmooth(conv_tbl, num_pos, num_neg, func_score, VAZD):
    """Iteratively smooths the conversion table until monotonic."""
    has_same_points = True

    while has_same_points:
        has_same_points = RemoveRedundancy(conv_tbl, num_pos, num_neg, func_score, VAZD)

    has_up_peak = True
    has_down_peak = True

    conv_tbl.sort(key=lambda x: x.score, reverse=True)

    while True:
        has_up_peak = SmoothUpPeak(conv_tbl, num_pos, num_neg, func_score, VAZD)
        if not has_up_peak and not has_down_peak:
            break
        has_down_peak = SmoothDownPeak(conv_tbl, num_pos, num_neg, func_score, VAZD)
        if not has_up_peak and not has_down_peak:
            break


def GenerateLikelihoodConversionTbl(predictions, num_pos, num_neg, func_score, VAZD):
    """
    Generates the initial raw conversion table using sliding window binning.
    """
    # Calculate bin size using square root rule
    bin_size = int(len(predictions) / math.sqrt(num_pos)) if num_pos > 0 else 1
    if bin_size < 1: bin_size = 1

    conv_tbl = []

    # Sort predictions
    predictions.sort(key=lambda x: x.kz)
    predictions.sort(key=func_score, reverse=True)

    # Sliding window
    for i in range(0, len(predictions) - bin_size + 1):
        # ExtendBin captures all items with scores equal to the boundary items
        # to ensure boundaries aren't split arbitrarily.
        predictions_bin = ExtendBin(predictions, func_score,
                                    func_score(predictions[i + bin_size - 1]) - 0.0001,
                                    func_score(predictions[i]) + 0.0001)

        conv_entry = CConvEntry()
        RecalculateEntryStats(conv_entry, predictions_bin, num_pos, num_neg, func_score, VAZD)
        conv_tbl.append(conv_entry)

    return conv_tbl


def WriteConversionTableBin(path_conversion_table, conv_tbl):
    """Writes the conversion table to a TSV file."""
    with open(path_conversion_table, 'w') as f:
        f.write("Score\tLower bound\tUpper bound\tLikelihood\tTPR\tFPR\tPPV\tFDR\tNo. positives\tNo. negatives\n")
        for conv_entry in conv_tbl:
            f.write("%.5f\t%.5f\t%.5f\t%.5f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\n" % \
                    (conv_entry.score, conv_entry.score_lower_bound, conv_entry.score_upper_bound,
                     conv_entry.L, conv_entry.TPR, conv_entry.FPR, conv_entry.PPV,
                     conv_entry.FDR, conv_entry.num_pos, conv_entry.num_neg))


def ReadConversionTableBin(path_conversion_table):
    """Reads the conversion table from a TSV file."""
    conv_tbl = []

    try:
        with open(path_conversion_table, 'r') as f:
            lines = f.readlines()

        # Skip header
        for line in lines[1:]:
            tokens = line.split()
            if not tokens: continue

            conv_entry = CConvEntry()
            # File format: Score(0), Lower(1), Upper(2), Likelihood(3)
            conv_entry.score = float(tokens[0])
            conv_entry.score_lower_bound = float(tokens[1])
            conv_entry.score_upper_bound = float(tokens[2])
            conv_entry.L = float(tokens[3])
            conv_tbl.append(conv_entry)

    except Exception as e:
        sys.stderr.write("Error reading conversion table %s: %s\n" % (path_conversion_table, e))

    # Ensure it is sorted descending by score for the binary search/scan in ConvertScore2L
    conv_tbl.sort(key=lambda x: x.score, reverse=True)

    return conv_tbl


def WriteConversionTableFDR(path_conversion_table, conv_tbl):
    """Writes just Score and FDR columns."""
    with open(path_conversion_table, 'w') as f:
        f.write("Score\tFDR\n")
        for conv_entry in conv_tbl:
            f.write("%.5f\t%.3f\n" % (conv_entry.score, conv_entry.FDR))


def ConvertScore2L(score, conv_tbl):
    """
    Interpolates the Likelihood (L) for a given score using the conversion table.
    """
    lower_limit = 0.0001

    # Sort just in case, though usually this is done on read.
    # Python Timsort is O(N) on sorted lists, so this is cheap.
    conv_tbl.sort(key=lambda x: x.score, reverse=True)

    if not conv_tbl:
        return lower_limit

    # Case 1: Score is higher than highest table entry (Extrapolation High)
    if score >= conv_tbl[0].score:
        if conv_tbl[0].L < lower_limit:
            return lower_limit
        return conv_tbl[0].L

    # Case 2: Score is lower than lowest table entry (Extrapolation Low)
    if score <= conv_tbl[-1].score:
        if conv_tbl[-1].L < lower_limit:
            return lower_limit
        return conv_tbl[-1].L

    # Case 3: Interpolation
    for i in range(len(conv_tbl) - 1):
        if conv_tbl[i].score >= score and score >= conv_tbl[i + 1].score:

            # Exact match range or flat region
            if conv_tbl[i].score == conv_tbl[i + 1].score:
                if conv_tbl[i].L < lower_limit:
                    return lower_limit
                return conv_tbl[i].L
            else:
                # Linear Interpolation
                fraction = (score - conv_tbl[i + 1].score) / (conv_tbl[i].score - conv_tbl[i + 1].score)
                L = conv_tbl[i + 1].L + (conv_tbl[i].L - conv_tbl[i + 1].L) * fraction

                if L < lower_limit:
                    return lower_limit
                return L

    return lower_limit