#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NetPhorest Python Implementation
================================

Author : Abhinav Mishra <mishraabhinav36@gmail.com>
Date   : 2025-06-15

Description
-----------
Command-line interface (CLI) for the pyNetPhorest package,
including NetPhorest kinase–substrate prediction and PTM crosstalk analysis.

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

import typer
import sys

app = typer.Typer(
    help="pyNetPhorest CLI: predict kinase–substrate sites from FASTA."
)
netphorest_app = typer.Typer(help="NetPhorest Prediction.")
app.add_typer(netphorest_app, name="netphorest")


@netphorest_app.command("fasta")
def netphorest(
        fasta: str = typer.Argument(
            ..., help="Input FASTA file (or '-' for stdin)."),
        out: str = typer.Option(None,
                                "--out", help="Output TSV path. Default: stdout."),
        atlas: str = typer.Option("netphorest.db",
                                  "--atlas", help="Atlas .db/.sqlite or .json."),
        causal: bool = typer.Option(False, "--causal",
                                    help="Enable Writer->Reader causal linking (Kinase recruits Binder).")):
    """
    Run NetPhorest prediction on FASTA sequences.
    Outputs a TSV file with predicted kinase–substrate relationships.

    Parameters
    ----------
    fasta : str
        Input FASTA file path (or '-' for stdin).
    out : str, optional
        Output TSV file path. If not provided, outputs to stdout.
    atlas : str, optional
        Path to netphorest.db or .json atlas file. Default is 'netphorest.db'.
    causal : bool, optional
        If True, enables Writer->Reader causal linking.

    Returns
    -------
    None
        Writes predictions to the specified output file or stdout.
    """
    from .pynetphorest import main

    sys.argv = ["netphorest-py", fasta]
    if out is not None:
        sys.argv += ["--out", out]
    if atlas is not None:
        sys.argv += ["--atlas", atlas]
    if causal:
        sys.argv.append("--causal")

    main()


crosstalk_app = typer.Typer(help="PTM Crosstalk (Functional Link) Analysis.")
app.add_typer(crosstalk_app, name="crosstalk")


@crosstalk_app.command("train")
def crosstalk_train(
        fasta: str = typer.Argument(..., help="FASTA file for sequence context."),
        within: str = typer.Argument(..., help="PTMcode2 'within.gz' file."),
        between: str = typer.Argument(..., help="PTMcode2 'between.gz' file."),
        atlas: str | None = typer.Option(
            None,
            "--atlas",
            help="Path to netphorest.db (defaults to package-local file)"
        ),
        model_out: str = typer.Option("crosstalk_model.pkl", "--out", help="Output model filename.")
):
    """
    Train a new pairwise crosstalk model using NetPhorest features + PTMcode2 labels.
    Generates a .pkl model file for later prediction.

    Parameters
    ----------
    fasta : str
        Input FASTA file path.
    within : str
        PTMcode2 'within.gz' file path.
    between : str
        PTMcode2 'between.gz' file path.
    atlas : str, optional
        Path to netphorest.db file. If None, uses package-local default.
    model_out : str, optional
        Output filename for the trained model. Default is 'crosstalk_model.pkl'.

    Returns
    -------
    None
        Saves the trained model to the specified output file.
    """
    try:
        import crosstalk
    except ImportError:
        try:
            from . import crosstalk
        except ImportError:
            typer.echo("Error: Could not import 'crosstalk.py'. Ensure it is in the same directory.")
            raise typer.Exit(code=1)

    if atlas is None:
        atlas_path = None
    else:
        atlas_path = atlas

    crosstalk.train_model(fasta, within, between, atlas, model_out)


@crosstalk_app.command("predict")
def crosstalk_predict(
        fasta: str = typer.Argument(..., help="Input FASTA file."),
        model: str = typer.Option("crosstalk_model.pkl", "--model", help="Trained crosstalk model."),
        atlas: str | None = typer.Option(
            None,
            "--atlas",
            help="Path to netphorest.db (defaults to package-local file)"
        ),
        out: str = typer.Option("crosstalk_predictions.tsv", "--out", help="Output prediction file."),
        threshold: float = typer.Option(0.8, "--thresh", help="Probability threshold.")
):
    """
    Predict functional links between phosphorylation sites in the input sequences.

    Parameters
    ----------
    fasta : str
        Input FASTA file path.
    model : str, optional
        Path to the trained crosstalk model. Default is 'crosstalk_model
.pkl'.
    atlas : str, optional
        Path to netphorest.db file. If None, uses package-local default.
    out : str, optional
        Output TSV file path for predictions. Default is 'crosstalk_predictions.tsv'.
    threshold : float, optional
        Probability threshold for predicting functional links. Default is 0.8 - conservative.

    Returns
    -------
    None
        Saves the predictions to the specified output file.
    """
    try:
        import crosstalk
    except ImportError:
        try:
            from . import crosstalk
        except ImportError:
            typer.echo("Error: Could not import 'crosstalk.py'. Ensure it is in the same directory.")
            raise typer.Exit(code=1)

    if atlas is None:
        atlas_path = None
    else:
        atlas_path = atlas

    crosstalk.predict(fasta, atlas, model, out, threshold)


@crosstalk_app.command("eval")
def crosstalk_eval(
        model: str = typer.Option(..., "--model", help="Path to trained .pkl model."),
        eval_npz: str = typer.Option(..., "--eval-npz", help="eval_data.npz containing X_test/y_test/w_test."),
        dataset_npz: str = typer.Option(..., "--dataset-npz", help="full_dataset.npz containing full X/y."),
        predictions_tsv: str = typer.Option(None, "--predictions-tsv", help="Optional predictions TSV file."),
        metadata: str = typer.Option(None, "--metadata", help="edge_metadata.json or .jsonl"),
        outdir: str = typer.Option("eval_output", "--outdir", help="Directory to write evaluation figures/tables.")
):
    """
    Evaluate a trained crosstalk model using saved test set + full dataset.
    Produces plots, metrics, and summaries.

    Parameters
    ----------
    model : str
        Path to the trained crosstalk model .pkl file.
    eval_npz : str
        Path to eval_data.npz file containing X_test, y_test, w_test.
    dataset_npz : str
        Path to full_dataset.npz file containing full X and y.
    predictions_tsv : str, optional
        Path to optional predictions TSV file.
    metadata : str, optional
        Path to edge_metadata.json or .jsonl file.
    outdir : str, optional
        Output directory to write evaluation results. Default is 'eval_output'.

    Returns
    -------
    None
        Saves evaluation results to the specified output directory.
    """
    from pathlib import Path
    from pynetphorest.evaluate import run_evaluation

    # Use outdir/modelname as prefix, e.g. eval_output/crosstalk_model
    out_prefix = str(Path(outdir) / Path(model).stem)

    run_evaluation(
        model_path=model,
        eval_npz_path=eval_npz,
        dataset_npz_path=dataset_npz,
        edge_metadata_path=metadata,
        predictions_tsv_path=predictions_tsv,
        out_prefix=out_prefix,
    )


@crosstalk_app.command("model-thresh")
def crosstalk_sweep_thresh(
        model: str = typer.Option(
            "crosstalk_model.pkl",
            "--model",
            help="Path to trained crosstalk model .pkl (default: crosstalk_model.pkl)",
        ),
        eval_npz: str = typer.Option(
            "eval_data.npz",
            "--eval-npz",
            help="Path to eval_data.npz with X_test/y_test (default: eval_data.npz)",
        ),
        dataset_npz: str = typer.Option(
            "full_dataset.npz",
            "--dataset-npz",
            help="Path to full_dataset.npz with full y (default: full_dataset.npz)",
        ),
        metadata: str = typer.Option(
            "edge_metadata.json",
            "--metadata",
            help="edge_metadata.json (JSON-lines, one dict per row).",
        ),
        min_th: float = typer.Option(
            0.10,
            "--min-th",
            help="Minimum decision threshold (default: 0.10)",
        ),
        max_th: float = typer.Option(
            0.90,
            "--max-th",
            help="Maximum decision threshold (default: 0.90)",
        ),
        step: float = typer.Option(
            0.05,
            "--step",
            help="Threshold step (default: 0.05)",
        ),
        out_global: str | None = typer.Option(
            None,
            "--out-global",
            help="Optional TSV path for global metrics.",
        ),
        out_residues: str | None = typer.Option(
            None,
            "--out-residues",
            help="Optional TSV path for per-residue metrics.",
        ),
):
    """
    Sweep probability thresholds and compute global + per-residue metrics
    (precision, recall, F1, MCC) on the held-out test set.

    Parameters
    ----------
    model : str
        Path to trained crosstalk model .pkl (default: crosstalk_model.pkl).
    eval_npz : str
        Path to eval_data.npz with X_test/y_test (default: eval_data.npz).
    dataset_npz : str
        Path to full_dataset.npz with full y (default: full_dataset.npz).
    metadata : str
        edge_metadata.json (JSON-lines, one dict per row).
    min_th : float
        Minimum decision threshold (default: 0.10).
    max_th : float
        Maximum decision threshold (default: 0.90).
    step : float
        Threshold step (default: 0.05).
    out_global : str, optional
        Optional TSV path for global metrics.
    out_residues : str, optional
        Optional TSV path for per-residue metrics.

    Returns
    -------
    None
        Prints tables to stdout and optionally saves TSV files.
    """

    from pynetphorest.model_thresh import run_sweep_thresh

    global_rows, residue_rows = run_sweep_thresh(
        model=model,
        eval_npz=eval_npz,
        full_npz=dataset_npz,
        meta_json=metadata,
        min_th=min_th,
        max_th=max_th,
        step=step,
        out_global=out_global,
        out_residues=out_residues,
    )

    from pynetphorest.model_thresh import print_global_table, print_residue_table

    print_global_table(global_rows)
    print_residue_table(residue_rows)


if __name__ == "__main__":
    app()
