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
    threshold: float = typer.Option(0.5, "--thresh", help="Probability threshold.")
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

if __name__ == "__main__":
    app()
