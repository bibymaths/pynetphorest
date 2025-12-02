# pynetphorest 

![CLI Tool](https://img.shields.io/badge/tool-CLI-blue)
[![PyPI version](https://img.shields.io/pypi/v/pynetphorest.svg)](https://pypi.org/project/pynetphorest/)
[![pip install](https://img.shields.io/badge/pip%20install-pynetphorest-brightgreen)](https://pypi.org/project/pynetphorest/)

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
---  

<p align="center">
  <img src="images/pynetphorest_schematic.svg" width="700">
</p>

`pynetphorest` is a **modern Python re-implementation and extension** of the NetPhorest scoring engine.

---

### **Understanding the biological problem**

Kinase signalling networks control almost every decision a cell makes — growth, stress response, DNA repair, apoptosis,
migration.
These decisions are encoded in **phosphorylation events**, and each event depends on:

* which kinase recognizes a motif,
* the structural context of the site,
* and dynamic interactions between proteins (crosstalk).

Despite two decades of work, most phosphosites still lack an assigned kinase, and **crosstalk between phosphorylation
events remains even more poorly mapped**.
Experimental methods cannot scale to the millions of possible site–kinase combinations.
Bioinformatics tools filled that gap — but many legacy implementations are slow, rigid, unmaintained, and difficult to
extend to modern data.

### **Why this needed to be solved**

Researchers today work with:

* full human proteome FASTAs
* PTMcode2 co-modification networks
* deep phosphoproteomics datasets
* ML workflows and reproducible pipelines

Existing tools could not handle this scale or integrate modern ML approaches.
A modern, fast, clean, and extensible implementation was needed — something that could combine **classic motif-scoring (
NetPhorest)** with **machine-learning models for kinase–kinase crosstalk**, and run end-to-end on real datasets without
legacy constraints.

### **How pynetphorest solves it**

`pynetphorest` is a complete re-implementation of the NetPhorest/NetworKIN logic in modern Python, redesigned to be
transparent, scalable, and extendable:

* **Fast motif-scoring of S/T/Y sites** using PSSMs and NN models
* **Causal “writer→reader” mode** for binder-mediated interactions
* **ML-based crosstalk prediction** (HistGradientBoosting) trained on PTMcode2
* **Unified CLI (`app`)** for scoring, training, predicting, and evaluation
* **Snakemake pipelines** for reproducible workflows
* **Full evaluation suite**: PR/ROC, Brier, MCC, per-residue metrics, subgroup analysis
* **Threshold sweeps** for downstream filtering and biological interpretability

Everything runs on standard Python 3.10+, with no external C dependencies, and can be integrated into any proteomics or
systems-biology pipeline.

---

## Why this matters

Protein phosphorylation is one of the most information-dense regulatory systems in biology. Every signaling
decision—growth, differentiation, DNA damage response, immune activation—depends on **who phosphorylates whom**, and
what downstream binding events are enabled.
Yet, reconstructing these networks experimentally is slow, expensive, and incomplete. Tools like NetPhorest were
groundbreaking at the time, but:

* relied on **legacy C implementations**,
* were difficult to extend or integrate,
* lacked **modern ML evaluation**,
* had limited support for **crosstalk logic** (e.g., kinase → binder causal chains),
* and were not scalable for modern proteome-wide analyses.

For real biological problems—like inferring context-specific signaling rewiring, or integrating phosphoproteomics with
structural knowledge—we need a framework that is fast, transparent, scriptable, and extensible.

---

## What this project contributes

It maintains full compatibility with the original algorithmic logic , while rebuilding the neural-network and PSSM
scoring stack in pure Python for clarity and reproducibility .

It adds three key capabilities:

### **1. A clean, modular scoring engine**

* Pure-Python implementation of all NetPhorest neural networks and PSSMs
* SQLite-based atlas format (fast, portable, inspectable)
* Support for both classic and causal (writer→reader) predictions
* Thread/process-parallel execution for whole-proteome scans

### **2. A full ML pipeline for phosphorylation crosstalk**

The `crosstalk` module trains a machine-learning model on PTMcode2 co-occurrence data to predict functional PTM-PTM
edges.
It includes:

* feature construction,
* dataset assembly,
* model training,
* threshold sweeping,
* full evaluation with PR/ROC/Brier/MCC,
* per-residue and per-structure subgroup analysis.

### **3. A reproducible workflow**

A ready-to-run **Snakemake pipeline** wraps:

* classic NetPhorest scoring,
* causal extension mode,
* ML-based crosstalk training and prediction,
* evaluation and summary statistics.

All outputs are stored in a consistent, versioned directory structure.

---

## How the system works (logic flow)

1. **Load kinase models** from an SQLite atlas or JSON.
   Each model contains:

    * window size,
    * NN/PSSM architecture,
    * sigmoid calibration parameters,
    * kinase metadata.

2. **Scan sequences** for S/T/Y sites.
   For each site, the engine extracts the correct sequence window and computes raw scores → sigmoid posteriors using the
   exact mathematical logic from the original algorithms.

3. **Optional causal mode**:

    * Identify the strongest kinase ("Writer") for a site,
    * Evaluate phospho-binding domains ("Readers"),
    * Emit kinase→binder causal edges only when biological logic allows.

4. **For crosstalk**:

    * Transform PTMcode2 edges into supervised learning data,
    * Train a probabilistic classifier,
    * Export a TSV of predicted crosstalk edges,
    * Evaluate global and subgroup metrics.

The entire workflow is reproducible and analysis-ready for downstream interpretation.

---

## Who is this for?

This toolkit is written for scientists across domains:

* computational biologists needing reproducible kinase scoring
* phosphoproteomics researchers integrating multi-omic datasets
* cancer biologists examining pathway rewiring
* ML researchers building graph-based signaling models
* structural biologists studying phospho-binding domain specificity
* any researcher wanting a **transparent, modifiable, modern** NetPhorest engine

---

## Install the package

```
pip install pynetphorest
```

## Command-line interface

After installation, all functionality is available via the single entrypoint:

```bash
pynetphorest [COMMAND] [SUBCOMMAND] [OPTIONS]
```

Top-level commands:

* `netphorest` – kinase–substrate site prediction
* `crosstalk` – PTM–PTM crosstalk modeling and evaluation

You can always inspect help with:

```bash
pynetphorest --help
pynetphorest netphorest --help
pynetphorest crosstalk --help
```

---

### 1. NetPhorest: kinase–substrate prediction

NetPhorest commands are grouped under:

```bash
pynetphorest netphorest [SUBCOMMAND] ...
```

Currently there is one subcommand: `fasta`.

#### 1.1 `netphorest fasta`

Run NetPhorest on a protein FASTA:

```bash
pynetphorest netphorest fasta FASTA \
    [--atlas ATLAS] \
    [--out TSV] \
    [--causal] \
    [--min-posterior P] \
    [--sigmoid-clamp X]
```

**Positional argument**

* `FASTA`
  Input FASTA file with protein sequences (or `-` for stdin).

**Options**

* `--atlas ATLAS`
  Path to a NetPhorest atlas (`.db`, `.sqlite`, `.json`).
  If omitted, the bundled atlas is used (first `netphorest.db`, then `netphorest.json` inside the package).

* `--out TSV`
  Output TSV file. If not given, results are written to stdout.

* `--causal`
  Enable writer→reader “causal” linking mode (kinase recruits a binding-domain reader).

* `--min-posterior P`
  Only report sites with posterior probability ≥ `P`.
  Use this to filter out low-confidence hits.

* `--sigmoid-clamp X`
  Absolute clamp on the logistic scoring term for numerical stability
  (default `50.0`; set to `0` to disable clamping).

---

### 2. Crosstalk: PTM–PTM interaction modeling

Crosstalk commands are grouped under:

```bash
pynetphorest crosstalk [SUBCOMMAND] ...
```

Available subcommands:

* `train` – train a crosstalk classifier from PTMcode2 + NetPhorest features
* `predict` – predict crosstalk on new FASTA sequences
* `eval` – offline evaluation and plotting for a trained model
* `model-thresh` – threshold sweep and metrics tables

You can inspect them via:

```bash
pynetphorest crosstalk train --help
pynetphorest crosstalk predict --help
pynetphorest crosstalk eval --help
pynetphorest crosstalk model-thresh --help
```

#### 2.1 `crosstalk train`

Train a pairwise crosstalk model:

```bash
pynetphorest crosstalk train FASTA WITHIN_GZ BETWEEN_GZ \
    [--atlas ATLAS] \
    [--out PKL] \
    [--window-size N] \
    [--neg-ratio K]
```

**Positional arguments**

* `FASTA`
  FASTA file for sequence context (IDs must match PTMcode2).

* `WITHIN_GZ`
  PTMcode2 within-protein edges file (e.g. `within.gz`).

* `BETWEEN_GZ`
  PTMcode2 between-protein edges file (e.g. `between.gz`).

**Options**

* `--atlas ATLAS`
  NetPhorest atlas path. If omitted, the bundled atlas is used.

* `--out PKL`
  Output model filename (default: `crosstalk_model.pkl`).

* `--window-size N`
  Peptide window size around each STY site (odd number, default: `9`).

* `--neg-ratio, --negative-ratio K`
  Number of negative edges per positive (default: `3`).

Outputs (in the working directory):

* `crosstalk_model.pkl` – trained classifier
* `full_dataset.npz` – full feature matrix + labels
* `eval_data.npz` – held-out test split
* `edge_metadata.json` – JSON-lines metadata per edge

---

#### 2.2 `crosstalk predict`

Predict functional crosstalk for a new FASTA:

```bash
pynetphorest crosstalk predict FASTA \
    [--model PKL] \
    [--atlas ATLAS] \
    [--out TSV] \
    [--thresh P] \
    [--jobs N]
```

**Positional argument**

* `FASTA`
  Input FASTA whose STY sites you want to score.

**Options**

* `--model PKL`
  Trained model file (default: `crosstalk_model.pkl`).

* `--atlas ATLAS`
  NetPhorest atlas path. If omitted, the bundled atlas is used.

* `--out TSV`
  Output predictions file (default: `crosstalk_predictions.tsv`).

* `--thresh P`
  Base probability threshold for reporting a pair (default: `0.8`).
  Per-residue internal thresholds (S/S, Y/Y, mixed) still apply.

* `--jobs, --n-jobs N`
  Number of parallel processes (default: `-1`, use all cores).

**Output columns**

* `Protein`
* `Site1` (e.g. `S123`)
* `Site2` (e.g. `Y456`)
* `Crosstalk_Prob`

---

#### 2.3 `crosstalk eval`

Offline evaluation and plotting:

```bash
pynetphorest crosstalk eval \
    --model PKL \
    --eval-npz NPZ \
    --dataset-npz NPZ \
    [--predictions-tsv TSV] \
    [--metadata JSONL] \
    [--outdir DIR]
```

**Required options**

* `--model PKL`
  Trained model.

* `--eval-npz NPZ`
  `eval_data.npz` containing `X_test`, `y_test`, `w_test`.

* `--dataset-npz NPZ`
  `full_dataset.npz` containing the full dataset (`X`, `y`).

**Optional**

* `--predictions-tsv TSV`
  Predictions TSV from `crosstalk predict` for additional summaries.

* `--metadata JSONL`
  `edge_metadata.json` or `.jsonl` with edge annotations.

* `--outdir DIR`
  Output directory for figures/tables (default: `eval_output`).

Produces:

* PR / ROC curves
* Confusion matrix
* Feature-group importance
* rRCS summaries
* Optional prediction summaries

---

#### 2.4 `crosstalk model-thresh`

Threshold sweep and metrics:

```bash
pynetphorest crosstalk model-thresh \
    [--model PKL] \
    [--eval-npz NPZ] \
    [--dataset-npz NPZ] \
    [--metadata JSONL] \
    [--min-th FLOAT] \
    [--max-th FLOAT] \
    [--step FLOAT] \
    [--out-global TSV] \
    [--out-residues TSV]
```

**Options**

* `--model PKL`
  Trained model (default: `crosstalk_model.pkl`).

* `--eval-npz NPZ`
  Eval split (default: `eval_data.npz`).

* `--dataset-npz NPZ`
  Full dataset (default: `full_dataset.npz`).

* `--metadata JSONL`
  Edge metadata (default: `edge_metadata.json`).

* `--min-th FLOAT`
  Minimum threshold (default: `0.10`).

* `--max-th FLOAT`
  Maximum threshold (default: `0.90`).

* `--step FLOAT`
  Threshold step size (default: `0.05`).

* `--out-global TSV`
  Optional TSV for global metrics.

* `--out-residues TSV`
  Optional TSV for residue-level metrics.

Metrics include precision, recall, F1, MCC, and TP/FP/TN/FN counts across thresholds.

---

**Conceptual & data lineage**

This project builds on the ideas, datasets, and foundational work from:

- **PTMcode v2**
    - Minguez, P., Letunic, I., Parca, L., Garcia-Alonso, L., Dopazo, J., Huerta-Cepas, J., & Bork, P. (2015).
      *PTMcode v2: a resource for functional associations of post-translational modifications within and between
      proteins.*
      **Nucleic Acids Research**, 43(Database issue), D494–D502.
      https://doi.org/10.1093/nar/gku1081

- **KinomeXplorer / NetPhorest**
    - Horn, H., Schoof, E., Kim, J., et al. (2014).
      *KinomeXplorer: an integrated platform for kinome biology studies.*
      **Nature Methods**, 11, 603–604.
      https://doi.org/10.1038/nmeth.2968

- **Phosphorylation network discovery (NetworKIN foundations)**
    - Linding, R., Jensen, L. J., Ostheimer, G. J., van Vugt, M. A., Jørgensen, C., Miron, I. M., Diella, F., Colwill,
      K., Taylor, L., Elder, K., Metalnikov, P., Nguyen, V., Pasculescu, A., Jin, J., Park, J. G., Samson, L. D.,
      Woodgett, J. R., Russell, R. B., Bork, P., Yaffe, M. B., … Pawson, T. (2007).
      *Systematic discovery of in vivo phosphorylation networks.*
      **Cell**, 129(7), 1415–1426.
      https://doi.org/10.1016/j.cell.2007.05.052

--- 

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

--- 

## Acknowledgements

We thank the original authors of NetPhorest and PTMcode2 for their foundational work and datasets that made this project
possible.
We also acknowledge the open-source community for tools and libraries that facilitated this implementation.
Nevertheless, all code and implementations in this repository are original and developed independently.

--- 

## Contact

For questions, issues, or contributions, please open an issue.