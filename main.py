import re
from pathlib import Path

from graph import make_graph
from memory_stats import print_stats, print_peak_stats
from models import Value, Allocation, ValueDetailed, ModuleStats
from parse_mlir import parse_mlir_line

# dir = Path("/home/lukas/cosmoca/DISCO-DJ/vsc_scripts/scripts/data/dump_host_20599_119")
# dir = Path("local_test")
dir = Path("dump_host_20599_126")

header_re = re.compile(
    r'^allocation\s+'
    r'(?P<alloc_id>\d+):\s*'
    r'size\s+(?P<total_size>\d+),'
)

value_re = re.compile(
    r'^\s*value:\s+'  # indent + “value: ”
    r'<(?P<id>\d+)\s+'  # “<737 ”
    r'(?P<name>[^@>]+?)\s*?(?P<opt_name>\(\w+\))? '  # “all-to-all.5.1 ”
    r'@(?P<at>\d+)>'  # “@0>”
    r'\s*\(size=(?P<size>\d+),offset=(?P<offset>\d+)\):\s*'
    r'(?P<array_info>.+)$'  # “f32…”
)
used_header_re = re.compile(r'^Used values:')

used_id_re = re.compile(r'^<(?P<id>\d+)\s+(?P<name>[^\s@]+) ?(?P<opt_name>\(\w+\))? @(?P<at>\d+)')


def analyze_module(memory_report_file: Path):
    buffer_assignment_file = memory_report_file.parent / memory_report_file.name.replace("memory-usage-report",
                                                                                         "buffer-assignment")
    with buffer_assignment_file.open("r") as f:
        mode = "alloc"
        uses_mode = None
        current_used_id = None
        module_stats = ModuleStats()
        for line in f:
            line = line.strip()
            if used_header_re.match(line):
                mode = "used"
                continue
            if line.startswith("HloLiveRange"):
                continue
            if line.startswith("InstructionSequence"):
                mode = "InstructionSequence"
                continue
            if line.startswith("BufferLiveRange"):
                mode = "BufferLiveRange"
                continue
            if line.startswith("Live ranges at"):
                mode = "LiveRangesPeak"
                continue
            if mode == "alloc":
                if line.startswith("allocation"):
                    m = header_re.search(line)
                    if not m:
                        raise ValueError("malformed allocation header")
                    d = m.groupdict()
                    for k in d.keys():
                        d[k] = int(d[k])
                    alloc_id = d["alloc_id"]
                    alloc = Allocation(**d)
                    module_stats.allocations[alloc_id] = alloc

                elif line.startswith("value"):
                    m = value_re.search(line)
                    if not m:
                        raise ValueError("malformed allocation value")
                    d = m.groupdict()
                    for k in ["id", "at", "size", "offset"]:
                        d[k] = int(d[k])
                    d["allocation"] = alloc
                    value = Value(**d)
                    module_stats.values[value.id] = value
                    if value.id in module_stats.value_id_to_name:
                        raise ValueError("duplicate value id")
                    module_stats.value_id_to_name[value.id] = value.name
                    if value.name in module_stats.value_name_to_id:
                        raise ValueError("duplicate value name")
                    module_stats.value_name_to_id[value.name] = value.id
            elif mode == "used":
                m = used_id_re.search(line)
                if m:
                    d = m.groupdict()
                    for k in ["id", "at"]:
                        d[k] = int(d[k])
                    value_uses = ValueDetailed(**d)
                    module_stats.values[value_uses.id].value_detailed = value_uses
                    current_used_id = value_uses.id
                    module_stats.used_values[current_used_id] = value_uses
                    continue
                if line.startswith("positions"):
                    uses_mode = "positions"
                    continue
                if line.startswith("uses"):
                    uses_mode = "uses"
                    continue
                if line.startswith("from instruction"):
                    instruction_raw = line.split(':', 1)[1].strip()
                    instruction = parse_mlir_line(instruction_raw)

                    module_stats.used_values[current_used_id].instruction = instruction
                    continue
                if uses_mode == "positions":
                    module_stats.used_values[current_used_id].positions.append(line)
                    continue
                if uses_mode == "uses":
                    module_stats.used_values[current_used_id].uses.append(line)
            elif mode == "BufferLiveRange":
                val, rangestr = line.split(":")
                name = val.rstrip("{}")
                range = tuple(map(int, rangestr.strip().split("-")))
                try:
                    value = module_stats.values[module_stats.value_name_to_id[name]]
                except KeyError:
                    value = module_stats.values[module_stats.value_name_to_id[val]]
                value.live_range = range
            elif mode == "InstructionSequence":
                order_str, name = line.split(":")
                order = int(order_str.strip())
                try:
                    value = module_stats.values[module_stats.value_name_to_id[name]]
                except KeyError:
                    continue
                value.sequence = order
    # print(module_stats.values[9].model_dump_json(indent=2))
    # print(json.dumps(value_name_to_id, indent=2))
    make_graph(module_stats)
    print_peak_stats(module_stats)


for file in sorted(dir.glob("*memory-usage-report.txt")):
    module_id = int(file.stem.split(".")[0].split("_")[1])
    module_name = file.stem.split(".")[1]

    if module_id == 447:
    # if module_id == 5:
        analyze_module(file)
