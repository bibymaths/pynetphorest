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

    # Mirror the script behaviour
    from pynetphorest.model_thresh import print_global_table, print_residue_table
    print_global_table(global_rows)
    print_residue_table(residue_rows)

if __name__ == "__main__":
    app()
