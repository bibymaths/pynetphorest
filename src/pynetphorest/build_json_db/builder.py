import re, json, os


def parse_float_arrays(filenames):
    """

    """
    arrays = {}
    pattern = re.compile(r'float const (\w+)\[\] = \{(.*?)\};', re.DOTALL)

    for fname in filenames:
        if not os.path.exists(fname):
            continue
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
    if not parts:
        return 0.0
    first = parts[0]
    if diff_str.strip().startswith("-") and not diff_str.strip().startswith("-e"):
        return -float(first)
    return float(first)


def parse_residues_from_header(header):
    residues = []
    if "c == 'S'" in header:
        residues.append('S')
    if "c == 'T'" in header:
        residues.append('T')
    if "c == 'Y'" in header:
        residues.append('Y')
    return residues if residues else ['S', 'T', 'Y']


def parse_code_blocks(filenames):
    definitions = []

    pssm_re = re.compile(r'o = pssm\(s, (\w+), (\d+)\)/([0-9\.eE\+\-]+);')
    ff_re   = re.compile(r'o \+= feed_forward\(s, (\w+), (\d+), (\d+)\);')
    div_re  = re.compile(r'o /= (\d+);')
    sig_re  = re.compile(
        r'o = ([0-9\.eE\+\-]+)\+\((.*?)\)/\(\d+\+exp\(([0-9\.eE\+\-]+)\*\(([0-9\.eE\+\-]+)-o\)\)\);'
    )
    meta_re = re.compile(
        r'printf\(".*?\\t(.*?)\\t(.*?)\\t(.*?)\\t(.*?)\\t%.6f\\t%.6f\\n", o, ([0-9\.eE\+\-]+)\);'
    )

    context_split_pattern = re.compile(r'(if \(c == [^\)]+\) \{)')

    for fname in filenames:
        if not os.path.exists(fname):
            continue
        print(f"Parsing logic from {fname}...")
        with open(fname, 'r') as f:
            content = f.read()

        chunks = context_split_pattern.split(content)
        current_residues = ['S', 'T', 'Y']

        for chunk in chunks:
            if chunk.startswith("if (c =="):
                current_residues = parse_residues_from_header(chunk)
                continue

            meta_matches = list(meta_re.finditer(chunk))
            prev_end = 0

            for meta_match in meta_matches:
                block_end = meta_match.start()
                block_content = chunk[prev_end:block_end]

                sig_matches = list(sig_re.finditer(block_content))
                if not sig_matches:
                    prev_end = meta_match.end()
                    continue
                sig_match = sig_matches[-1]

                pssm_match = pssm_re.search(block_content)

                if pssm_match:
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
                            "method": meta_match.group(1),
                            "tree": meta_match.group(2),
                            "classifier": meta_match.group(3),
                            "kinase": meta_match.group(4),
                            "prior": float(meta_match.group(5))
                        }
                    })
                else:
                    networks = []
                    for ff in ff_re.finditer(block_content):
                        networks.append({
                            "weights_id": ff.group(1),
                            "window": int(ff.group(2)),
                            "hidden": int(ff.group(3))
                        })

                    if networks:
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
                                "method": meta_match.group(1),
                                "tree": meta_match.group(2),
                                "classifier": meta_match.group(3),
                                "kinase": meta_match.group(4),
                                "prior": float(meta_match.group(5))
                            }
                        })

                prev_end = meta_match.end()

    return definitions


def build():
    # C-order preserved here:
    data_files  = ["insr_data.h", "nn_data.h", "pssm_data.h"]
    logic_files = ["insr_code.h", "nn_code.h", "pssm_code.h"]

    arrays = parse_float_arrays(data_files)
    all_models = parse_code_blocks(logic_files)

    print(f"Found {len(all_models)} definitions.")

    models_out = []
    linked_count = 0

    # stable counters for IDs
    type_counts = {"INSR": 0, "NN": 0, "PSSM": 0}

    for model in all_models:
        mtype = model["type"]
        type_counts[mtype] = type_counts.get(mtype, 0) + 1
        model_id = f"{mtype}_{type_counts[mtype]:03d}"

        if mtype == "PSSM":
            wid = model.get("weights_id")
            if wid in arrays:
                model["weights"] = arrays[wid]
                model["id"] = model_id
                models_out.append(model)
                linked_count += 1

        elif mtype == "NN":
            valid = True
            for net in model["networks"]:
                wid = net.get("weights_id")
                if wid in arrays:
                    net["weights"] = arrays[wid]
                else:
                    valid = False
                    break
            if valid:
                model["id"] = model_id
                models_out.append(model)
                linked_count += 1

        else:
            # INSR models, if any, should behave like PSSM/N
            # If they have weights_id, link them the same way.
            wid = model.get("weights_id")
            if wid and wid in arrays:
                model["weights"] = arrays[wid]
            model["id"] = model_id
            models_out.append(model)
            linked_count += 1

    print(f"Successfully linked {linked_count} models to their weights.")

    atlas = {
        "model_order": ["INSR", "NN", "PSSM"],
        "models": models_out
    }

    with open("netphorest_atlas.json", "w") as f:
        json.dump(atlas, f, indent=2)

    print("Done! Saved netphorest_atlas.json")


if __name__ == "__main__":
    build()
