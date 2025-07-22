from pathlib import Path

import click
from rich.console import Console

from .cli_utils import BYTE_SIZE, module_name_completer
from .utils import pretty_byte_size
from .xla_memory_analyzer import main, load_all_modules


@click.group()
def cli():
    """XLA Memory Analyzer"""
    pass


def common_directory_arg(f):
    return click.argument(
        "directory",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
    )(f)


@cli.command("list-modules")
@common_directory_arg
@click.option(
    "-s", "--sort-by-size",
    is_flag=True,
    help="Sort modules by size"
)
@click.option(
    "-i", "--ignore-tiny",
    is_flag=True,
    help="Ignore modules smaller than 4KB"
)
@click.option(
    "-m", "--min-size",
    "min_size_bytes",
    type=BYTE_SIZE,
    default="4K",
    show_default=True,
    help="Minimum module size to keep (e.g. 4K, 1G)",
)
@click.option(
    "-b", "--bytes",
    "show_bytes",
    is_flag=True,
    help="Show allocation sizes in bytes"
)
def list_modules(directory, sort_by_size, ignore_tiny, min_size_bytes, show_bytes):
    """
    List all XLA modules and their allocation sizes.

    DIRECTORY must be the folder created via:
      XLA_FLAGS="--xla_dump_to=some_directory"
    """
    console = Console()

    xla_dump = load_all_modules(directory)
    all_modules = xla_dump.modules
    if ignore_tiny:
        all_modules = filter(lambda module: module.total_allocation > min_size_bytes, all_modules)
    if sort_by_size:
        all_modules = sorted(all_modules, key=lambda module: module.total_allocation)
    for mod in all_modules:
        if show_bytes:
            size = mod.total_allocation
        else:
            size = pretty_byte_size(mod.total_allocation)
        print(mod.id, mod.name, size)
    console.rule(f"all modules: {pretty_byte_size(xla_dump.total_size)}")


@cli.command("analyze-peaks")
@common_directory_arg
@click.argument(
    'modules',
    nargs=-1,
    shell_complete=module_name_completer,
)
@click.option("--only-main-peak", is_flag=True)
@click.option("--skip-small-modules", is_flag=True)
def main_command(
        directory: Path,
        modules: list[str],
        skip_small_modules: bool,
        only_main_peak: bool,
):
    if modules is None:
        modules = [
            "function_using_idx", "forward_model",
            "unnamed_wrapped_function", "scatter_gather_func",
            "scan_body", "jit_scan"
        ]
    print(modules)
    main(directory, modules, skip_small_modules, only_main_peak)
