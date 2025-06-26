from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from models import ModuleStats, Value
from utils import pretty_byte_size


def sourcefile_to_snippet(value: Value):
    if value.value_detailed.source is None:
        return None
    source_file, source_line = value.value_detailed.source
    file = Path(source_file)
    if not file.exists():
        return None
    lines = file.open().readlines()
    start_line = max(0, source_line - 3)
    end_line = min(len(lines), source_line + 1)
    code = "".join(lines[start_line:end_line])
    syntax = Syntax(code, "python",
                    dedent=True, line_numbers=True,indent_guides=True,
                    start_line=start_line + 1, highlight_lines={source_line})
    background_style = syntax._theme.get_background_style()
    title=f"{value.pretty_size} {value.name} {value.array_info} "
    panel=Panel(syntax,style=background_style,title=title,subtitle=value.value_detailed.short_source)
    console = Console()
    console.print(panel)


def print_stats(module_stats: ModuleStats):
    values = list(module_stats.values.values())
    # values = filter(lambda v: v.is_large_array, values)
    ordered_values: list[Value] = sorted(values, key=lambda x: -x.size)
    cumsize = 0
    table = Table(title="Memory Stats")
    table.add_column("Cumulative Size")
    table.add_column("Size")
    table.add_column("Name", no_wrap=True)
    table.add_column("op_name", no_wrap=True)
    table.add_column("source", no_wrap=True)
    for v in ordered_values:
        cumsize += v.size
        if not v.is_large_array:
            continue

        sourcefile_to_snippet(v)
        table.add_row(
            pretty_byte_size(cumsize),
            v.pretty_size,
            v.name,
            v.value_detailed.op_name,
            v.value_detailed.short_source
        )

    console = Console()
    console.print(table)
