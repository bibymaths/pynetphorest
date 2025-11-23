import re
import json
import os


def parse_float_arrays(filenames):
    """Parses C float arrays from a list of files."""
    arrays = {}
    pattern = re.compile(r'float const (\w+)\[\] = \{(.*?)\};', re.DOTALL)

    for fname in filenames:
        if not os.path.exists(fname): continue
        print(f"Parsing data from {fname}...")
        with open(fname, 'r') as f:
            content = f.read()
        for match in pattern.finditer(content):
            name = match.group(1)
            values_str = match.group(2).replace('\n', '').replace(' ', '')
            values = [float(x) for x in values_str.split(',') if x]
            arrays[name] = values
    return arrays


def extract_max_val(diff_str):
    parts = re.split(r'(?<![eE])-', diff_str)
    parts = [p for p in parts if p]
    if not parts: return 0.0
    first = parts[0]
    if diff_str.strip().startswith("-") and not diff_str.strip().startswith("-e"):
        return -float(first)
    return float(first)


def parse_residues_from_header(header):
    """Extracts ['S', 'T'] from "if (c == 'S' || c == 'T')" """
    residues = []
    if "c == 'S'" in header: residues.append('S')
    if "c == 'T'" in header: residues.append('T')
    if "c == 'Y'" in header: residues.append('Y')
    return residues if residues else ['S', 'T', 'Y']


def parse_code_blocks(filenames):
    definitions = []

    # Regex to break file into "Context Blocks" (if c == X)
    # We split by the 'if (c ==' line to get chunks guarded by residue checks
    context_split_pattern = re.compile(r'(if \(c == [^\)]+\) \{)')

    for fname in filenames:
        if not os.path.exists(fname): continue
        print(f"Parsing logic from {fname}...")
        with open(fname, 'r') as f:
            content = f.read()

        # Split file into chunks based on "if (c =="
        chunks = context_split_pattern.split(content)

        # Default context if no 'if' found (usually for pssm_code.h preamble)
        current_residues = ['S', 'T', 'Y']

        for chunk in chunks:
            # Check if this chunk IS a header "if (c == ... {"
            if chunk.startswith("if (c =="):
                current_residues = parse_residues_from_header(chunk)
                continue  # The header itself contains no models usually

            # This chunk contains models under the `current_residues` context
            # We now split this chunk into individual models.
            # Models always start with setting o=...

            # --- PARSE PSSM MODELS ---
            # Split by "o = pssm("
            pssm_parts = chunk.split("o = pssm(")
            for part in pssm_parts[1:]:  # Skip text before first model
                try:
                    # Parse Definition
                    def_match = re.match(r's, (\w+), (\d+)\)/([0-9\.eE\+\-]+);', part)
                    if not def_match: continue

                    # Parse Sigmoid (Looking ahead in the part)
                    sig_match = re.search(
                        r'o = ([0-9\.eE\+\-]+)\+\((.*?)\)/\(\d+\+exp\(([0-9\.eE\+\-]+)\*\(([0-9\.eE\+\-]+)-o\)\)\);',
                        part)
                    if not sig_match: continue

                    # Parse Metadata
                    meta_match = re.search(
                        r'printf\(".*?\\t(.*?)\\t(.*?)\\t(.*?)\\t(.*?)\\t%.6f\\t%.6f\\n", o, ([0-9\.eE\+\-]+)\);', part)
                    if not meta_match: continue

                    definitions.append({
                        "type": "PSSM",
                        "residues": current_residues,
                        "weights_id": def_match.group(1),
                        "window": int(def_match.group(2)),
                        "divisor": float(def_match.group(3)),
                        "sigmoid": {
                            "min": float(sig_match.group(1)),
                            "max": extract_max_val(sig_match.group(2)),
                            "slope": float(sig_match.group(3)),
                            "inflection": float(sig_match.group(4))
                        },
                        "meta": {
                            "method": meta_match.group(1), "tree": meta_match.group(2),
                            "classifier": meta_match.group(3), "kinase": meta_match.group(4),
                            "prior": float(meta_match.group(5))
                        }
                    })
                except Exception:
                    continue

            # --- PARSE NN MODELS ---
            # Split by "o = 0;" which resets score for a new NN ensemble
            nn_parts = chunk.split("o = 0;")
            for part in nn_parts[1:]:
                try:
                    # 1. Extract all feed_forwards in this specific model part
                    ff_pattern = re.compile(r'o \+= feed_forward\(s, (\w+), (\d+), (\d+)\);')
                    networks = []
                    for ff in ff_pattern.finditer(part):
                        networks.append({
                            "weights_id": ff.group(1),
                            "window": int(ff.group(2)),
                            "hidden": int(ff.group(3))
                        })

                    if not networks: continue

                    # 2. Divisor (local to this part)
                    div_match = re.search(r'o /= (\d+);', part)
                    divisor = float(div_match.group(1)) if div_match else 1.0

                    # 3. Sigmoid (local)
                    sig_match = re.search(
                        r'o = ([0-9\.eE\+\-]+)\+\((.*?)\)/\(\d+\+exp\(([0-9\.eE\+\-]+)\*\(([0-9\.eE\+\-]+)-o\)\)\);',
                        part)
                    if not sig_match: continue

                    # 4. Metadata (local)
                    meta_match = re.search(
                        r'printf\(".*?\\t(.*?)\\t(.*?)\\t(.*?)\\t(.*?)\\t%.6f\\t%.6f\\n", o, ([0-9\.eE\+\-]+)\);', part)
                    if not meta_match: continue

                    definitions.append({
                        "type": "NN",
                        "residues": current_residues,
                        "networks": networks,
                        "divisor": divisor,
                        "sigmoid": {
                            "min": float(sig_match.group(1)),
                            "max": extract_max_val(sig_match.group(2)),
                            "slope": float(sig_match.group(3)),
                            "inflection": float(sig_match.group(4))
                        },
                        "meta": {
                            "method": meta_match.group(1), "tree": meta_match.group(2),
                            "classifier": meta_match.group(3), "kinase": meta_match.group(4),
                            "prior": float(meta_match.group(5))
                        }
                    })
                except Exception:
                    continue

    return definitions


def build():
    # 1. Load Data
    data_files = ["pssm_data.h", "insr_data.h", "nn_data.h"]
    arrays = parse_float_arrays(data_files)

    # 2. Parse Logic with Context
    logic_files = ["pssm_code.h", "insr_code.h", "nn_code.h"]
    all_models = parse_code_blocks(logic_files)

    atlas = []
    # 3. Link Weights
    for model in all_models:
        if model['type'] == 'PSSM':
            if model['weights_id'] in arrays:
                model['weights'] = arrays[model['weights_id']]
                atlas.append(model)
        elif model['type'] == 'NN':
            valid = True
            for net in model['networks']:
                if net['weights_id'] in arrays:
                    net['weights'] = arrays[net['weights_id']]
                else:
                    valid = False
            if valid:
                atlas.append(model)

    with open("netphorest_atlas.json", "w") as f:
        json.dump(atlas, f, indent=2)
    print(f"Done! Built atlas with {len(atlas)} models.")


if __name__ == "__main__":
    build()