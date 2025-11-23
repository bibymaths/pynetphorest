import re
import json
import os


def parse_float_arrays(filenames):
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
    residues = []
    if "c == 'S'" in header: residues.append('S')
    if "c == 'T'" in header: residues.append('T')
    if "c == 'Y'" in header: residues.append('Y')
    return residues if residues else ['S', 'T', 'Y']


def parse_code_blocks(filenames):
    definitions = []

    # Regex for specific patterns
    # PSSM: o = pssm(s, NAME, WIN)/DIV;
    pssm_re = re.compile(r'o = pssm\(s, (\w+), (\d+)\)/([0-9\.eE\+\-]+);')

    # NN Feed Forward: o += feed_forward(s, NAME, WIN, HIDDEN);
    ff_re = re.compile(r'o \+= feed_forward\(s, (\w+), (\d+), (\d+)\);')

    # NN Divisor: o /= DIV;
    div_re = re.compile(r'o /= (\d+);')

    # Sigmoid: o = MIN+(MAX-MIN)/(1+exp(SLOPE*(INF-o)));
    # We use specific float patterns to be safe
    sig_re = re.compile(r'o = ([0-9\.eE\+\-]+)\+\((.*?)\)/\(\d+\+exp\(([0-9\.eE\+\-]+)\*\(([0-9\.eE\+\-]+)-o\)\)\);')

    # Metadata: printf(..., "METHOD", "TREE", "CLASS", "KINASE", POSTERIOR, PRIOR)
    meta_re = re.compile(r'printf\(".*?\\t(.*?)\\t(.*?)\\t(.*?)\\t(.*?)\\t%.6f\\t%.6f\\n", o, ([0-9\.eE\+\-]+)\);')

    # Context split (same as before, but we process chunks better)
    context_split_pattern = re.compile(r'(if \(c == [^\)]+\) \{)')

    for fname in filenames:
        if not os.path.exists(fname): continue
        print(f"Parsing logic from {fname}...")
        with open(fname, 'r') as f:
            content = f.read()

        chunks = context_split_pattern.split(content)
        current_residues = ['S', 'T', 'Y']

        for chunk in chunks:
            if chunk.startswith("if (c =="):
                current_residues = parse_residues_from_header(chunk)
                continue

                # The chunk contains multiple models.
            # Strategy: Identify every 'printf' (which marks the end of a model).
            # Then look backwards or parse the block leading up to it.

            # We will split by 'printf' to separate models, but keep the printf content attached to the preceding block
            # Actually, splitting by "if (gbHas_unsupported_amino_acid == 0)" is safer for NN blocks in nn_code.h
            # or just splitting by the sigmoid calculation line.

            # BETTER STRATEGY: Regex Find Iter over the whole chunk for Metadata
            # The metadata is the anchor. Once we find a metadata block, we look at the text *immediately preceding it* to find the Sigmoid and the Model Def.

            # Let's find all metadata locations
            meta_matches = list(meta_re.finditer(chunk))

            prev_end = 0
            for i, meta_match in enumerate(meta_matches):
                # Define the text block for this model: from end of previous model to start of this metadata
                # (We add some buffer from the match itself to capture the printf line if needed, but mainly we want BEFORE)
                block_end = meta_match.start()
                block_content = chunk[prev_end:block_end]

                # 1. Find Sigmoid (Last one in the block)
                sig_matches = list(sig_re.finditer(block_content))
                if not sig_matches:
                    # Fallback: maybe the sigmoid is essentially 0-1 linear? (Unlikely in NetPhorest)
                    continue
                sig_match = sig_matches[-1]

                # 2. Identify Type (PSSM or NN)
                # Look for pssm() call
                pssm_match = pssm_re.search(block_content)

                if pssm_match:
                    # It's a PSSM
                    definitions.append({
                        "type": "PSSM",
                        "residues": current_residues,
                        "weights_id": pssm_match.group(1),
                        "window": int(pssm_match.group(2)),
                        "divisor": float(pssm_match.group(3)),
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
                else:
                    # It's likely NN
                    # Find all feed_forward calls in this block
                    networks = []
                    for ff in ff_re.finditer(block_content):
                        networks.append({
                            "weights_id": ff.group(1),
                            "window": int(ff.group(2)),
                            "hidden": int(ff.group(3))
                        })

                    if networks:
                        # Find Divisor (o /= X)
                        div_match = div_re.search(block_content)
                        divisor = float(div_match.group(1)) if div_match else 1.0

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

                # Update pointer
                prev_end = meta_match.end()

    return definitions


def build():
    data_files = ["insr_data.h", "nn_data.h", "pssm_data.h"]
    logic_files = ["insr_code.h", "nn_code.h", "pssm_code.h"]
    arrays = parse_float_arrays(data_files)
    all_models = parse_code_blocks(logic_files)

    atlas = []
    print(f"Found {len(all_models)} definitions.")

    linked_count = 0
    for model in all_models:
        if model['type'] == 'PSSM':
            if model['weights_id'] in arrays:
                model['weights'] = arrays[model['weights_id']]
                atlas.append(model)
                linked_count += 1
        elif model['type'] == 'NN':
            valid = True
            for net in model['networks']:
                if net['weights_id'] in arrays:
                    net['weights'] = arrays[net['weights_id']]
                else:
                    valid = False
            if valid:
                atlas.append(model)
                linked_count += 1

    print(f"Successfully linked {linked_count} models to their weights.")

    with open("netphorest_atlas.json", "w") as f:
        json.dump(atlas, f, indent=2)
    print(f"Done! Saved netphorest_atlas.json")


if __name__ == "__main__":
    build()