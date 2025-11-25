#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NetPhorest Python Implementation
================================

Author : Abhinav Mishra <mishraabhinav36@gmail.com>
Date   : 2025-06-15

Description
-----------
This Snakemake workflow implements the NetPhorest kinome analysis
and crosstalk prediction pipeline using the `pynetphorest` Python
package. It allows users to run classic and causal NetPhorest
analyses on a given FASTA file, as well as train and evaluate
a crosstalk prediction model based on PTMcode2 data.

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

configfile: "config.yaml"

import os

RESULTS = config.get("results_dir","results")
APP = config.get("pynetphorest_bin","pynetphorest")

FASTA = config["fasta"]
ATLAS = config.get("atlas","netphorest.db")

# NetPhorest options
NET_BASE = config.get("netphorest_out_basename","netphorest")
NET_CAUSAL = config.get("netphorest_causal",False)

# Crosstalk options
PTM_WITHIN = config.get("ptm_within",None)
PTM_BETWEEN = config.get("ptm_between",None)
CROSSTALK_MODEL_NAME = config.get("crosstalk_model_name","crosstalk_model.pkl")
CROSSTALK_PRED_NAME = config.get("crosstalk_predictions_name","crosstalk_predictions.tsv")
CROSSTALK_THRESH = config.get("crosstalk_threshold",0.30)
CROSSTALK_EVAL_DIR = config.get("crosstalk_eval_dir","eval")
stem = os.path.splitext(CROSSTALK_MODEL_NAME)[0]
mode = config.get("mode","crosstalk")

# -------------------------
# Targets per mode
# -------------------------

net_targets = []
if mode in ("netphorest", "both"):
    # classic NetPhorest
    net_targets.append(f"{RESULTS}/{NET_BASE}_classic.tsv")
    # optional causal NetPhorest
    if NET_CAUSAL:
        net_targets.append(f"{RESULTS}/{NET_BASE}_causal.tsv")

crosstalk_targets = []
if mode in ("crosstalk", "both"):
    # final crosstalk prediction + evaluation summary TSV
    crosstalk_targets.extend([
        f"{RESULTS}/{CROSSTALK_PRED_NAME}",
        f"{RESULTS}/{CROSSTALK_EVAL_DIR}/{os.path.splitext(CROSSTALK_MODEL_NAME)[0]}.subgroups.tsv",
        f"{RESULTS}/{CROSSTALK_EVAL_DIR}/{stem}.thresholds.global.tsv",
        f"{RESULTS}/{CROSSTALK_EVAL_DIR}/{stem}.thresholds.residues.tsv",
    ])

rule all:
    input:
        net_targets + crosstalk_targets

# -------------------------
# NetPhorest branch
# -------------------------

rule netphorest_classic:
    input:
        fasta=FASTA,
        atlas=ATLAS
    output:
        tsv=f"{RESULTS}/netphorest_classic.tsv"
    params:
        out_name=lambda w, input, output: os.path.basename(output.tsv)
    shell:
        r"""
        set -euo pipefail
        mkdir -p {RESULTS}
        cd {RESULTS}
        {APP} netphorest fasta ../{input.fasta} \
            --atlas ../{input.atlas} \
            --out {params.out_name}
        """

rule netphorest_causal:
    input:
        fasta=FASTA,
        atlas=ATLAS
    output:
        tsv=f"{RESULTS}/netphorest_causal.tsv"
    params:
        out_name=lambda w, input, output: os.path.basename(output.tsv)
    shell:
        r"""
        set -euo pipefail
        mkdir -p {RESULTS}
        cd {RESULTS}
        {APP} netphorest fasta ../{input.fasta} \
            --atlas ../{input.atlas} \
            --out {params.out_name} \
            --causal
        """

# -------------------------
# Crosstalk branch
# -------------------------

rule crosstalk_train:
    """
    Train crosstalk model using PTMcode2 + NetPhorest features.

    Inside {RESULTS}/ this will write:
      - crosstalk_model.pkl
      - full_dataset.npz
      - eval_data.npz
      - edge_metadata.json
    """
    input:
        fasta=FASTA,
        within=PTM_WITHIN,
        between=PTM_BETWEEN,
        atlas=ATLAS
    output:
        model=f"{RESULTS}/{CROSSTALK_MODEL_NAME}",
        dataset=f"{RESULTS}/full_dataset.npz",
        eval_npz=f"{RESULTS}/eval_data.npz",
        meta=f"{RESULTS}/edge_metadata.json"
    params:
        model_name=lambda wildcards, input, output: os.path.basename(output.model)
    shell:
        r"""
        set -euo pipefail
        mkdir -p {RESULTS}
        cd {RESULTS}
        {APP} crosstalk train \
            ../{input.fasta} \
            ../{input.within} \
            ../{input.between} \
            --atlas ../{input.atlas} \
            --out {params.model_name}
        """

rule crosstalk_predict:
    """
    Predict crosstalk edges using the trained model.
    Writes results/<CROSSTALK_PRED_NAME>.
    """
    input:
        model=f"{RESULTS}/{CROSSTALK_MODEL_NAME}",
        fasta=FASTA,
        atlas=ATLAS
    output:
        preds=f"{RESULTS}/{CROSSTALK_PRED_NAME}"
    params:
        model_name=lambda wildcards, input, output: os.path.basename(input.model),
        pred_name=lambda wildcards, input, output: os.path.basename(output.preds)
    shell:
        r"""
        set -euo pipefail
        mkdir -p {RESULTS}
        cd {RESULTS}
        {APP} crosstalk predict \
            ../{input.fasta} \
            --model {params.model_name} \
            --atlas ../{input.atlas} \
            --out {params.pred_name} \
            --thresh {CROSSTALK_THRESH}
        """

rule crosstalk_eval:
    """
    Evaluate the trained crosstalk model.

    Calls:
      app crosstalk eval --model ... --eval-npz ... --dataset-npz ... --metadata ... --predictions-tsv ... --outdir ...
    and we track the subgroup metrics TSV in results/<CROSSTALK_EVAL_DIR>/.
    """
    input:
        model=f"{RESULTS}/{CROSSTALK_MODEL_NAME}",
        eval_npz=f"{RESULTS}/eval_data.npz",
        dataset=f"{RESULTS}/full_dataset.npz",
        meta=f"{RESULTS}/edge_metadata.json",
        preds=f"{RESULTS}/{CROSSTALK_PRED_NAME}"
    output:
        subgroups=f"{RESULTS}/{CROSSTALK_EVAL_DIR}/{os.path.splitext(CROSSTALK_MODEL_NAME)[0]}.subgroups.tsv"
    params:
        model_name=lambda wildcards, input, output: os.path.basename(input.model),
        eval_name=lambda wildcards, input, output: os.path.basename(input.eval_npz),
        data_name=lambda wildcards, input, output: os.path.basename(input.dataset),
        meta_name=lambda wildcards, input, output: os.path.basename(input.meta),
        preds_name=lambda wildcards, input, output: os.path.basename(input.preds)
    shell:
        r"""
        set -euo pipefail
        mkdir -p {RESULTS}/{CROSSTALK_EVAL_DIR}
        cd {RESULTS}
        {APP} crosstalk eval \
            --model {params.model_name} \
            --eval-npz {params.eval_name} \
            --dataset-npz {params.data_name} \
            --metadata {params.meta_name} \
            --predictions-tsv {params.preds_name} \
            --outdir {CROSSTALK_EVAL_DIR}
        """

rule model_sweep_thresh:
    """
    Threshold sweep for the trained crosstalk model.

    Consumes:
      - results/crosstalk_model.pkl
      - results/eval_data.npz
      - results/full_dataset.npz
      - results/edge_metadata.json

    Produces:
      - results/<eval_dir>/<model_stem>.thresholds.global.tsv
      - results/<eval_dir>/<model_stem>.thresholds.residues.tsv
    """
    input:
        model=f"{RESULTS}/{CROSSTALK_MODEL_NAME}",
        eval_npz=f"{RESULTS}/eval_data.npz",
        full_npz=f"{RESULTS}/full_dataset.npz",
        meta=f"{RESULTS}/edge_metadata.json"
    output:
        global_tsv=f"{RESULTS}/{CROSSTALK_EVAL_DIR}/{os.path.splitext(CROSSTALK_MODEL_NAME)[0]}.thresholds.global.tsv",
        residues_tsv=f"{RESULTS}/{CROSSTALK_EVAL_DIR}/{os.path.splitext(CROSSTALK_MODEL_NAME)[0]}.thresholds.residues.tsv"
    params:
        model_name=lambda w, input, output: os.path.basename(input.model),
        eval_name=lambda w, input, output: os.path.basename(input.eval_npz),
        full_name=lambda w, input, output: os.path.basename(input.full_npz),
        meta_name=lambda w, input, output: os.path.basename(input.meta),
        global_name=lambda w, input, output: os.path.basename(output.global_tsv),
        residues_name=lambda w, input, output: os.path.basename(output.residues_tsv)
    shell:
        r"""
        set -euo pipefail
        mkdir -p {RESULTS}/{CROSSTALK_EVAL_DIR}
        cd {RESULTS}
        {APP} crosstalk model-thresh \
            --model {params.model_name} \
            --eval-npz {params.eval_name} \
            --dataset-npz {params.full_name} \
            --metadata {params.meta_name} \
            --min-th 0.10 \
            --max-th 0.90 \
            --step 0.05 \
            --out-global {CROSSTALK_EVAL_DIR}/{params.global_name} \
            --out-residues {CROSSTALK_EVAL_DIR}/{params.residues_name}
        """
