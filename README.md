# pynetphorest
  
**A modern Python implementation of NetPhorest/NetworKIN for kinase–substrate prediction and phosphorylation crosstalk analysis.**

--- 

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Snakemake](https://img.shields.io/badge/workflow-Snakemake-1f77b4.svg)](https://snakemake.github.io) 
 
--- 

### **Understanding the biological problem**

Kinase signalling networks control almost every decision a cell makes — growth, stress response, DNA repair, apoptosis, migration.
These decisions are encoded in **phosphorylation events**, and each event depends on:

* which kinase recognizes a motif,
* the structural context of the site,
* and dynamic interactions between proteins (crosstalk).

Despite two decades of work, most phosphosites still lack an assigned kinase, and **crosstalk between phosphorylation events remains even more poorly mapped**.
Experimental methods cannot scale to the millions of possible site–kinase combinations.
Bioinformatics tools filled that gap — but many legacy implementations are slow, rigid, unmaintained, and difficult to extend to modern data.

### **Why this needed to be solved**

Researchers today work with:

* full human proteome FASTAs
* PTMcode2 co-modification networks
* deep phosphoproteomics datasets
* ML workflows and reproducible pipelines

Existing tools could not handle this scale or integrate modern ML approaches.
A modern, fast, clean, and extensible implementation was needed — something that could combine **classic motif-scoring (NetPhorest)** with **machine-learning models for kinase–kinase crosstalk**, and run end-to-end on real datasets without legacy constraints.

### **How pynetphorest solves it**

`pynetphorest` is a complete re-implementation of the NetPhorest/NetworKIN logic in modern Python, redesigned to be transparent, scalable, and extendable:

* **Fast motif-scoring of S/T/Y sites** using PSSMs and NN models
* **Causal “writer→reader” mode** for binder-mediated interactions
* **ML-based crosstalk prediction** (HistGradientBoosting) trained on PTMcode2
* **Unified CLI (`app`)** for scoring, training, predicting, and evaluation
* **Snakemake pipelines** for reproducible workflows
* **Full evaluation suite**: PR/ROC, Brier, MCC, per-residue metrics, subgroup analysis
* **Threshold sweeps** for downstream filtering and biological interpretability

Everything runs on standard Python 3.10+, with no external C dependencies, and can be integrated into any proteomics or systems-biology pipeline.

--- 

**Conceptual & data lineage**

This project builds on the ideas, datasets, and foundational work from:

- **PTMcode v2**
  - Minguez, P., Letunic, I., Parca, L., Garcia-Alonso, L., Dopazo, J., Huerta-Cepas, J., & Bork, P. (2015).
    *PTMcode v2: a resource for functional associations of post-translational modifications within and between proteins.*
    **Nucleic Acids Research**, 43(Database issue), D494–D502.
    https://doi.org/10.1093/nar/gku1081

- **KinomeXplorer / NetPhorest**
  - Horn, H., Schoof, E., Kim, J., et al. (2014).
    *KinomeXplorer: an integrated platform for kinome biology studies.*
    **Nature Methods**, 11, 603–604.
    https://doi.org/10.1038/nmeth.2968

- **Phosphorylation network discovery (NetworKIN foundations)**
  - Linding, R., Jensen, L. J., Ostheimer, G. J., van Vugt, M. A., Jørgensen, C., Miron, I. M., Diella, F., Colwill, K., Taylor, L., Elder, K., Metalnikov, P., Nguyen, V., Pasculescu, A., Jin, J., Park, J. G., Samson, L. D., Woodgett, J. R., Russell, R. B., Bork, P., Yaffe, M. B., … Pawson, T. (2007).
    *Systematic discovery of in vivo phosphorylation networks.*
    **Cell**, 129(7), 1415–1426.
    https://doi.org/10.1016/j.cell.2007.05.052
 
---