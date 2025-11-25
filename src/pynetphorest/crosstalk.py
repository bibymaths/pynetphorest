import gzip
import pathlib
import random
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from . import core

WINDOW_SIZE = 9
NEGATIVE_RATIO = 3
BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_ATLAS_PATH = BASE_DIR / "netphorest.db"

def load_sequences(fasta_path):
    """Simple FASTA parser returning dict {header: sequence}."""
    seqs = {}
    name = None
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                name = line.strip().split()[0][1:]
                seqs[name] = []
            elif name:
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def extract_site_features(seq, pos, aa, models, rrcs=0.0):
    """
    Generates a feature vector for a single site using NetPhorest logic.
    Features: [rRCS, Avg_Post, Top5_Post..., Peptide_Encoding...]
    """
    if pos < 0 or pos >= len(seq) or seq[pos] != aa:
        return None

        # 1. Physical Context (Peptide)
    peptide = core.get_window(seq, pos, WINDOW_SIZE)
    encoded = core.encode_peptide(peptide)

    # 2. NetPhorest Posterior Scores
    scores = []
    for model in models:
        # Calculate posterior using core logic
        score = core.get_model_posterior(seq, pos, model)
        scores.append(score)

    scores = sorted(scores, reverse=True)
    top_scores = scores[:5]  # Keep top 5 kinase probabilities
    # Pad if fewer than 5 models
    while len(top_scores) < 5:
        top_scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return [rrcs, avg_score] + top_scores + encoded


def parse_ptmcode_line(line, structure="within"):
    parts = line.strip().split('\t')

    def parse_res(res_str):
        # 'S588' -> ('S', 587)
        return res_str[0], int(res_str[1:]) - 1

    try:
        if structure == "within":
            # Protein, Species, PTM1, Res1, rRCS1, Prop1, PTM2...
            p1 = parts[0]
            p2 = parts[0]
            ptm1_type, res1_raw, rrcs1 = parts[2], parts[3], float(parts[4])
            ptm2_type, res2_raw, rrcs2 = parts[6], parts[7], float(parts[8])
        else:  # between
            p1, p2 = parts[0], parts[1]
            ptm1_type, res1_raw, rrcs1 = parts[3], parts[4], float(parts[5])
            ptm2_type, res2_raw, rrcs2 = parts[7], parts[8], float(parts[9])

        if "phosphorylation" not in ptm1_type or "phosphorylation" not in ptm2_type:
            return None

        aa1, pos1 = parse_res(res1_raw)
        aa2, pos2 = parse_res(res2_raw)

        return {
            "p1": p1, "p2": p2,
            "aa1": aa1, "pos1": pos1, "rrcs1": rrcs1,
            "aa2": aa2, "pos2": pos2, "rrcs2": rrcs2,
            "label": 1
        }
    except (ValueError, IndexError):
        return None


def train_model(fasta, within_file, between_file, atlas_path=pathlib.Path | None, output_model="crosstalk_model.pkl"):
    if atlas_path is None:
        atlas_path = DEFAULT_ATLAS_PATH
    print(f"Loading Atlas from {atlas_path}...")
    atlas = core.load_atlas(str(atlas_path))
    models = atlas.get("models", []) if isinstance(atlas, dict) else atlas

    print("Loading Sequences...")
    sequences = load_sequences(fasta)
    dataset = []

    files = [(within_file, "within"), (between_file, "between")]
    valid_edges = 0

    print("Processing PTMcode edges...")
    for fpath, ftype in files:
        if not fpath: continue

        # Handle both raw strings and potential file objects if integrated differently later
        open_func = gzip.open if fpath.endswith(".gz") else open

        with open_func(fpath, 'rt') as f:
            for line in tqdm(f, desc=f"Parsing {ftype}"):
                if line.startswith("##"): continue
                edge = parse_ptmcode_line(line, ftype)
                if not edge: continue

                if edge['p1'] not in sequences or edge['p2'] not in sequences:
                    continue

                feat1 = extract_site_features(sequences[edge['p1']], edge['pos1'], edge['aa1'], models, edge['rrcs1'])
                feat2 = extract_site_features(sequences[edge['p2']], edge['pos2'], edge['aa2'], models, edge['rrcs2'])

                if feat1 and feat2:
                    combined = feat1 + feat2 + [abs(a - b) for a, b in zip(feat1, feat2)]
                    dataset.append(combined + [1])
                    valid_edges += 1

    print(f"Generated {valid_edges} positive samples. Generating negatives...")
    neg_count = 0
    target_neg = valid_edges * NEGATIVE_RATIO
    keys = list(sequences.keys())

    with tqdm(total=target_neg) as pbar:
        while neg_count < target_neg:
            prot = random.choice(keys)
            seq = sequences[prot]
            sty = [i for i, c in enumerate(seq) if c in "STY"]
            if len(sty) < 2: continue

            i1, i2 = random.sample(sty, 2)
            feat1 = extract_site_features(seq, i1, seq[i1], models, rrcs=0.0)
            feat2 = extract_site_features(seq, i2, seq[i2], models, rrcs=0.0)

            if feat1 and feat2:
                combined = feat1 + feat2 + [abs(a - b) for a, b in zip(feat1, feat2)]
                dataset.append(combined + [0])
                neg_count += 1
                pbar.update(1)

    df = pd.DataFrame(dataset)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Training Gradient Boosting Classifier...")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    clf.fit(X_train, y_train)

    print("Evaluation:")
    if len(X_test) > 0:
        probs = clf.predict_proba(X_test)[:, 1]
        print(f"AP Score: {average_precision_score(y_test, probs):.4f}")

    with open(output_model, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {output_model}")


def predict(fasta, atlas_path=pathlib.Path | None, model_path="crosstalk_model.pkl", out="crosstalk_predictions.tsv",
            threshold=0.5):
    if atlas_path is None:
        atlas_path = DEFAULT_ATLAS_PATH
    print("Loading Atlas and Model...")
    atlas = core.load_atlas(atlas_path)
    models = atlas.get("models", [])

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    print(f"Predicting crosstalk for {fasta}...")
    with open(out, "w") as f_out:
        f_out.write("Protein\tSite1\tSite2\tCrosstalk_Prob\n")

        sequences = load_sequences(fasta)
        for name, seq in tqdm(sequences.items()):
            sites = [i for i, c in enumerate(seq) if c in "STY"]
            if len(sites) < 2: continue

            vectors = {}
            for site in sites:
                vec = extract_site_features(seq, site, seq[site], models, rrcs=0.0)
                if vec:
                    vectors[site] = vec

            valid_sites = list(vectors.keys())
            for i in range(len(valid_sites)):
                for j in range(i + 1, len(valid_sites)):
                    s1, s2 = valid_sites[i], valid_sites[j]
                    v1, v2 = vectors[s1], vectors[s2]

                    combined = v1 + v2 + [abs(a - b) for a, b in zip(v1, v2)]
                    prob = clf.predict_proba([combined])[0][1]

                    if prob > threshold:
                        res1 = f"{seq[s1]}{s1 + 1}"
                        res2 = f"{seq[s2]}{s2 + 1}"
                        f_out.write(f"{name}\t{res1}\t{res2}\t{prob:.4f}\n")
    print(f"Results written to {out}")