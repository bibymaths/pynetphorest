#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyNetPhorest CLI
================

High-level command-line interface for the pyNetPhorest package.

Subcommands
-----------
- netphorest fasta       : Kinase–substrate site prediction from FASTA.
- crosstalk train        : Train PTM crosstalk model from PTMcode2 edges.
- crosstalk predict      : Predict crosstalk in new FASTA (with parallelism).
- crosstalk eval         : Offline evaluation / plots for a trained model.
- crosstalk model-thresh : Threshold sweep (precision/recall/F1/MCC tables).

Atlas handling
--------------
If you do *not* pass --atlas, the code internally calls
`core.get_default_atlas_path()` which searches inside the installed
package (typically `pynetphorest/models/`) in this order:

1. netphorest.db
2. netphorest.json

You can always pass an explicit path to a .db, .sqlite, or .json atlas
via --atlas to override the bundled one.
"""

import sys
import typer

app = typer.Typer(
    help="pyNetPhorest CLI: NetPhorest kinase prediction and PTM crosstalk tools.",
    no_args_is_help=True,
)

# -----------------------------
# NetPhorest prediction
# -----------------------------

netphorest_app = typer.Typer(help="NetPhorest kinase–substrate prediction.")
app.add_typer(netphorest_app, name="netphorest")


@netphorest_app.command("fasta")
def netphorest(
    fasta: str = typer.Argument(
        ...,
        help="Input FASTA file with protein sequences (or '-' for stdin).",
        metavar="FASTA",
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        metavar="TSV",
        help="Output TSV path. Default: write to stdout.",
    ),
    atlas: str | None = typer.Option(
        None,
        "--atlas",
        metavar="ATLAS",
        help=(
            "Path to NetPhorest atlas (.db/.sqlite/.json). "
            "If omitted, uses the atlas bundled with the package: "
            "first 'models/netphorest.db', then 'models/netphorest.json' "
            "inside the installed pyNetPhorest package."
        ),
    ),
    causal: bool = typer.Option(
        False,
        "--causal",
        help=(
            "Enable Writer→Reader causal linking "
            "(kinase recruits a binding-domain 'reader')."
        ),
    ),
    min_posterior: float = typer.Option(
        0.0,
        "--min-posterior",
        metavar="P",
        help="Minimum posterior probability to report a site (default: 0.0).",
    ),
    sigmoid_clamp: float = typer.Option(
        50.0,
        "--sigmoid-clamp",
        metavar="VAL",
        help=(
            "Absolute cap on sigmoid argument term (default: 50.0). "
            "Set to 0 to disable clamping."
        ),
    ),
):
    """
    Run NetPhorest prediction on protein sequences.

    Examples
    --------
    Basic usage (stdout):
        pynetphorest netphorest fasta proteins.fasta

    Save to file and specify atlas:
        pynetphorest netphorest fasta proteins.fasta --out preds.tsv --atlas netphorest.db

    Read FASTA from stdin:
        cat proteins.fasta | pynetphorest netphorest fasta - > preds.tsv
    """
    # We reuse the argparse-based `pynetphorest.main()` for now,
    # so we construct sys.argv accordingly.
    from .pynetphorest import main

    sys.argv = ["netphorest-py", fasta]
    if out is not None:
        sys.argv += ["--out", out]
    if atlas is not None:
        sys.argv += ["--atlas", atlas]
    if causal:
        sys.argv.append("--causal")
    # new knobs:
    sys.argv += ["--min-posterior", str(min_posterior)]
    sys.argv += ["--sigmoid-clamp", str(sigmoid_clamp)]

    main()


# -----------------------------
# Crosstalk subcommands
# -----------------------------

crosstalk_app = typer.Typer(help="PTM crosstalk (functional link) analysis.")
app.add_typer(crosstalk_app, name="crosstalk")


@crosstalk_app.command("train")
def crosstalk_train(
    fasta: str = typer.Argument(
        ...,
        help="FASTA file for sequence context (same IDs as PTMcode2 files).",
    ),
    within: str = typer.Argument(
        ...,
        help="PTMcode2 within-protein edges file (e.g. within.gz).",
        metavar="WITHIN_GZ",
    ),
    between: str = typer.Argument(
        ...,
        help="PTMcode2 between-protein edges file (e.g. between.gz).",
        metavar="BETWEEN_GZ",
    ),
    atlas: str | None = typer.Option(
        None,
        "--atlas",
        metavar="ATLAS",
        help=(
            "Path to NetPhorest atlas (.db/.sqlite/.json). "
            "If omitted, uses the package-bundled atlas as described above."
        ),
    ),
    model_out: str = typer.Option(
        "crosstalk_model.pkl",
        "--out",
        metavar="PKL",
        help="Output filename for the trained model (default: crosstalk_model.pkl).",
    ),
    window_size: int = typer.Option(
        9,
        "--window-size",
        metavar="N",
        help="Peptide window size around each STY site (odd number, default: 9).",
    ),
    negative_ratio: int = typer.Option(
        3,
        "--neg-ratio",
        "--negative-ratio",
        metavar="K",
        help="Number of negative edges per positive (default: 3).",
    ),
):
    """
    Train a pairwise crosstalk model using PTMcode2 + NetPhorest features.

    Outputs
    -------
    - crosstalk_model.pkl       : trained classifier
    - full_dataset.npz          : full feature matrix + labels (for summaries)
    - eval_data.npz             : held-out test split
    - edge_metadata.json        : JSON-lines with edge annotations
    """
    try:
        import crosstalk  # when running from source directory
    except ImportError:
        try:
            from . import crosstalk
        except ImportError:
            typer.echo(
                "Error: Could not import 'crosstalk.py'. "
                "Ensure it is installed or in the same package."
            )
            raise typer.Exit(code=1)

    atlas_path = atlas if atlas is not None else None
    crosstalk.train_model(
        fasta=fasta,
        within_file=within,
        between_file=between,
        atlas_path=atlas_path,
        output_model=model_out,
        window_size=window_size,
        negative_ratio=negative_ratio,
    )

@crosstalk_app.command("predict")
def crosstalk_predict(
    fasta: str = typer.Argument(
        ...,
        help="Input FASTA file whose STY sites you want to score for crosstalk.",
    ),
    model: str = typer.Option(
        "crosstalk_model.pkl",
        "--model",
        metavar="PKL",
        help="Trained crosstalk model (default: crosstalk_model.pkl).",
    ),
    atlas: str | None = typer.Option(
        None,
        "--atlas",
        metavar="ATLAS",
        help=(
            "Path to NetPhorest atlas (.db/.sqlite/.json). "
            "If omitted, uses the package-bundled atlas."
        ),
    ),
    out: str = typer.Option(
        "crosstalk_predictions.tsv",
        "--out",
        metavar="TSV",
        help="Output prediction file (default: crosstalk_predictions.tsv).",
    ),
    threshold: float = typer.Option(
        0.8,
        "--thresh",
        help=(
            "Base probability threshold for reporting a pair (default: 0.8). "
            "Note: per-residue internal thresholds (S/S, Y/Y, mixed) still apply."
        ),
    ),
    jobs: int = typer.Option(
        -1,
        "--jobs",
        "--n-jobs",
        help="Number of parallel processes for prediction (default: -1, all cores).",
    ),
):
    """
    Predict functional crosstalk links between phosphorylation sites.

    Output columns
    --------------
    - Protein
    - Site1 (e.g. S123)
    - Site2 (e.g. Y456)
    - Crosstalk_Prob
    """
    try:
        import crosstalk
    except ImportError:
        try:
            from . import crosstalk
        except ImportError:
            typer.echo(
                "Error: Could not import 'crosstalk.py'. "
                "Ensure it is installed or in the same package."
            )
            raise typer.Exit(code=1)

    atlas_path = atlas if atlas is not None else None
    crosstalk.predict(
        fasta=fasta,
        atlas_path=atlas_path,
        model_path=model,
        out=out,
        threshold=threshold,
        n_jobs=jobs,
    )


@crosstalk_app.command("eval")
def crosstalk_eval(
    model: str = typer.Option(
        ...,
        "--model",
        metavar="PKL",
        help="Path to trained .pkl model.",
    ),
    eval_npz: str = typer.Option(
        ...,
        "--eval-npz",
        metavar="NPZ",
        help="eval_data.npz containing X_test/y_test/w_test.",
    ),
    dataset_npz: str = typer.Option(
        ...,
        "--dataset-npz",
        metavar="NPZ",
        help="full_dataset.npz containing full X/y.",
    ),
    predictions_tsv: str | None = typer.Option(
        None,
        "--predictions-tsv",
        metavar="TSV",
        help="Optional predictions TSV (from crosstalk predict) for summary.",
    ),
    metadata: str | None = typer.Option(
        None,
        "--metadata",
        metavar="JSONL",
        help="Optional edge_metadata.json or .jsonl file.",
    ),
    outdir: str = typer.Option(
        "eval_output",
        "--outdir",
        metavar="DIR",
        help="Directory in which to write evaluation figures/tables.",
    ),
):
    """
    Offline evaluation and plotting for a trained crosstalk model.

    Produces
    --------
    - PR / ROC curves
    - Confusion matrix
    - Feature-group importance
    - rRCS summaries
    - Optional prediction TSV summaries
    """
    from pathlib import Path
    from pynetphorest.evaluate import run_evaluation

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
        metavar="PKL",
        help="Path to trained crosstalk model .pkl (default: crosstalk_model.pkl).",
    ),
    eval_npz: str = typer.Option(
        "eval_data.npz",
        "--eval-npz",
        metavar="NPZ",
        help="Path to eval_data.npz with X_test/y_test (default: eval_data.npz).",
    ),
    dataset_npz: str = typer.Option(
        "full_dataset.npz",
        "--dataset-npz",
        metavar="NPZ",
        help="Path to full_dataset.npz with full y (default: full_dataset.npz).",
    ),
    metadata: str = typer.Option(
        "edge_metadata.json",
        "--metadata",
        metavar="JSONL",
        help="edge_metadata.json (JSON-lines, one dict per row).",
    ),
    min_th: float = typer.Option(
        0.10,
        "--min-th",
        help="Minimum decision threshold (default: 0.10).",
    ),
    max_th: float = typer.Option(
        0.90,
        "--max-th",
        help="Maximum decision threshold (default: 0.90).",
    ),
    step: float = typer.Option(
        0.05,
        "--step",
        help="Threshold step (default: 0.05).",
    ),
    out_global: str | None = typer.Option(
        None,
        "--out-global",
        metavar="TSV",
        help="Optional TSV path for global metrics.",
    ),
    out_residues: str | None = typer.Option(
        None,
        "--out-residues",
        metavar="TSV",
        help="Optional TSV path for per-residue metrics.",
    ),
):
    """
    Sweep decision thresholds and compute global + per-residue metrics.

    Metrics
    -------
    - precision, recall, F1
    - MCC
    - TP / FP / TN / FN counts
    """
    from pynetphorest.model_thresh import (
        run_sweep_thresh,
        print_global_table,
        print_residue_table,
    )

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

    # If the user didn’t request TSVs, print the tables to stdout
    if out_global is None:
        print_global_table(global_rows)
    if out_residues is None:
        print_residue_table(residue_rows)


if __name__ == "__main__":
    app()