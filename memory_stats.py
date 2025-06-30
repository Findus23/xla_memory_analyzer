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
        if "DISCO-DJ" in source_file:
            rel_path = source_file.split("DISCO-DJ/")[-1]
            disc_path_local = Path("/home/lukas/cosmoca/DISCO-DJ")
            file = disc_path_local / rel_path
        else:
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


def print_memory_stats(values: list[Value]):
    cumsize = 0
    table = Table(title="Memory Stats")
    table.add_column("Cumulative Size")
    table.add_column("Size")
    table.add_column("Name", no_wrap=False)
    table.add_column("op_name", no_wrap=False)
    table.add_column("source", no_wrap=False)
    for v in values:
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


def print_stats(module_stats: ModuleStats):
    # print(module_stats.times_of_largest_allocation)
    # exit()
    values = list(module_stats.values.values())
    values = filter(lambda v: v.is_large_array, values)
    ordered_values: list[Value] = sorted(values, key=lambda x: -x.size)
    print_memory_stats(ordered_values[:3])


def print_peak_stats(module_stats: ModuleStats):
    for peak in module_stats.allocation_peaks:
        time_map = module_stats.size_over_time[0]
        values_at_peak = sorted(time_map[peak], key=lambda x: -x.size)
        total_at_peak=sum(v.size for v in values_at_peak)
        print(f"peak at {peak} totalling {pretty_byte_size(total_at_peak)} bytes")
        print_memory_stats(values_at_peak)
