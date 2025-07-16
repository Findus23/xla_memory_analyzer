from pathlib import Path

import click
from rich.console import Console

from .xla_memory_analyzer import main


@click.command()
@click.argument('dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--skip-small-modules", is_flag=True)
def cli(dir: Path, skip_small_modules: bool):
    interesting_modules = [
        "function_using_idx", "forward_model",
        "unnamed_wrapped_function", "scatter_gather_func",
        "scan_body", "jit_scan"
    ]
    # interesting_modules=["scatter"]
    # interesting_modules = ["delta_subset"]

    main(dir, interesting_modules, skip_small_modules)
