from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import typer

app = typer.Typer(no_args_is_help=True)

# --- registry ---
RUNNERS: dict[str, Callable] = {}


def runner(name: str):
    def deco(fn):
        RUNNERS[name] = fn
        return fn

    return deco


def artifacts_dir() -> Path:
    # repo-level artifacts (adjust if you want it inside numpde/)
    d = Path(os.environ.get("NUMPDE_ARTIFACTS", "artifacts"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def in_kitty() -> bool:
    return "KITTY_WINDOW_ID" in os.environ or os.environ.get("TERM", "").startswith(
        "xterm-kitty"
    )


def configure_matplotlib(mode: str):
    """
    mode: 'auto'|'show'|'save'
    """
    import matplotlib

    if mode == "save":
        matplotlib.use("Agg")
        return "Agg"

    if mode == "show":
        # let matplotlib pick default GUI backend
        return matplotlib.get_backend()

    # auto
    if in_kitty():
        try:
            matplotlib.use("module://matplotlib-backend-kitty")
            return "kitty"
        except Exception:
            pass
    # fallback: show if interactive, else save
    return matplotlib.get_backend()


@app.command()
def list():
    """List available studies/runners."""
    for name in sorted(RUNNERS):
        typer.echo(name)


@app.command()
def run(
    name: str = typer.Argument(..., help="Runner name, see `pde list`"),
    plot: str = typer.Option("auto", help="auto|show|save"),
    out: Path = typer.Option(
        None, help="Output directory for artifacts (default: ./artifacts)"
    ),
):
    """Run a study/experiment."""
    if name not in RUNNERS:
        typer.echo(f"Unknown runner '{name}'. Try `pde list`.", err=True)
        raise typer.Exit(2)

    if out is None:
        out = artifacts_dir()
    else:
        out.mkdir(parents=True, exist_ok=True)

    backend = configure_matplotlib(plot)
    typer.echo(f"[matplotlib backend] {backend}")
    typer.echo(f"[artifacts] {out}")

    RUNNERS[name](out=out, plot=plot)


def main():
    app()


if __name__ == "__main__":
    main()
