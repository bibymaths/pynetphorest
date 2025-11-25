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


if __name__ == "__main__":
    app()
