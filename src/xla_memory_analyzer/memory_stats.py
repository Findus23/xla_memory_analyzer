from collections import defaultdict
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .models import ModuleStats, Value
from .utils import pretty_byte_size


def sourcefile_to_snippet(value: Value):
    if value.value_detailed.source is None:
        return None
    source_file, source_line = value.value_detailed.source
    file = Path(source_file)
    if not file.exists():
        if "DISCO-DJ" in source_file:
            rel_path = source_file.split("DISCO-DJ/")[-1]
            disc_path_local = Path("/home/lukas/cosmoca/DISCO-DJ")
            #disc_path_local = Path("/home/lukas/cosmoca/DISCO-DJ/vsc_scripts/")
            file = disc_path_local / rel_path
        else:
            print(f"{file} not found")
            return None
    lines = file.open().readlines()
    start_line = max(0, source_line - 3)
    end_line = min(len(lines), source_line + 1)
    code = "".join(lines[start_line:end_line])
    syntax = Syntax(code, "python",
                    dedent=True, line_numbers=True,indent_guides=True, word_wrap=True,
                    start_line=start_line + 1, highlight_lines={source_line})
    background_style = syntax._theme.get_background_style()
    title=f"{value.pretty_size} | {value.name} | {value.array_info_without_order} "
    panel=Panel(syntax,style=background_style,title=title,subtitle=value.value_detailed.short_source +" | "                +value.value_detailed.op_name)
    console = Console()
    console.print(panel)


def print_memory_stats(values: list[Value]):
    cumsize = 0
    table = Table(title="Memory Stats", box=box.MARKDOWN)
    table.add_column("Cumulative Size")
    table.add_column("Size")
    table.add_column("Name", no_wrap=False)
    table.add_column("op_name", no_wrap=False)
    table.add_column("source", no_wrap=False)
    table.add_column("array_info", no_wrap=False)
    table.add_column("a", no_wrap=False)
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
            v.value_detailed.short_source,
            v.array_info.split("{")[0],
            str(v.allocation.alloc_id)
        )

    console = Console()
    console.print(table)


def print_stats(module_stats: ModuleStats):
    # print(module_stats.times_of_largest_allocation)
    # exit()
    values = list(module_stats.values.values())
    # values = filter(lambda v: v.is_large_array, values)
    ordered_values: list[Value] = sorted(values, key=lambda x: -x.size)
    print_memory_stats(ordered_values[:3])


def print_peak_stats(module_stats: ModuleStats):
    console = Console()
    for peak in module_stats.allocation_peaks:
        time_map = module_stats.size_over_time[0]
        values_at_peak = sorted(time_map[peak], key=lambda x: -x.size)
        total_at_peak=sum(v.size for v in values_at_peak)
        console.rule(f"peak at {peak} totalling {pretty_byte_size(total_at_peak)}")
        print_memory_stats(values_at_peak)


def memory_buffer_over_time(module_stats: ModuleStats):
    """
    show how the same memory buffer is reused within one module
    """
    by_buffer = defaultdict(list)
    by_buffer_size = {}
    for val in module_stats.values.values():
        key = val.allocation.alloc_id, val.size, val.offset
        by_buffer[key].append(val)
        by_buffer_size[key] = val.size

    by_buffer_size = {k: v for k, v in sorted(by_buffer_size.items(), key=lambda item: -item[1])}
    for key, size in list(by_buffer_size.items())[:25]:
        print(size, key, len(by_buffer[key]))
        v: Value
        for v in by_buffer[key]:
            print(v.sequence, v.pretty_size,
                  v.name,
                  v.value_detailed.op_name,
                  v.value_detailed.short_source,
                  v.array_info_without_order[0])
            print(v.allocation.alloc_id, pretty_byte_size(v.size), pretty_byte_size(v.offset),
                  pretty_byte_size(v.allocation.total_size))
            sourcefile_to_snippet(v)
        print("\n\n")


def vals_by_line_of_code(module_stats: ModuleStats):
    for val in module_stats.values.values():
        if val.value_detailed.source is None:
            continue
        source_file, source_line = val.value_detailed.source
        if source_file is None:
            continue
        if source_file.endswith("scatter_and_gather.py"):
            if source_line in [1172, 1350, 1175]:
                v = val
                print(v.sequence, v.pretty_size,
                      v.name,
                      v.value_detailed.op_name,
                      v.value_detailed.short_source,
                      v.array_info_without_order)
                print("alloc",v.allocation.alloc_id, pretty_byte_size(v.size), pretty_byte_size(v.offset),
                      pretty_byte_size(v.allocation.total_size))

                sourcefile_to_snippet(v)
