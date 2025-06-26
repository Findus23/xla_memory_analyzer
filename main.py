import json
import re
from pathlib import Path

from graph import make_graph
from models import Value, Allocation, ValueDetailed
from parse_mlir import parse_mlir_line

# dir = Path("/home/lukas/cosmoca/DISCO-DJ/vsc_scripts/scripts/data/dump_host_20599_119")
dir = Path("local_test")

header_re = re.compile(
    r'^allocation\s+'
    r'(?P<alloc_id>\d+):\s*'
    r'size\s+(?P<total_size>\d+),'
)

value_re = re.compile(
    r'^\s*value:\s+'  # indent + “value: ”
    r'<(?P<id>\d+)\s+'  # “<737 ”
    r'(?P<name>[^@>]+?)\s*'  # “all-to-all.5.1 ”
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
        value_name_to_id: dict[str, int] = {}
        values: dict[int, Value] = {}
        used_values: dict[int, ValueDetailed] = {}
        allocations: dict[int, Allocation] = {}
        for line in f:
            line = line.strip()
            if used_header_re.match(line):
                mode = "used"
                continue
            if line.startswith("HloLiveRange"):
                mode = "HloLiveRange"
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
                    print(alloc)

                    allocations[alloc_id] = alloc

                elif line.startswith("value"):
                    m = value_re.search(line)
                    if not m:
                        raise ValueError("malformed allocation value")
                    d = m.groupdict()
                    for k in ["id", "at", "size", "offset"]:
                        d[k] = int(d[k])
                    d["allocation"] = alloc
                    value = Value(**d)
                    values[value.id] = value
                    if d["name"] in value_name_to_id:
                        raise ValueError("duplicate value name")
                    value_name_to_id[value.name] = value.id
            elif mode == "used":
                m = used_id_re.search(line)
                if m:
                    d = m.groupdict()
                    for k in ["id", "at"]:
                        d[k] = int(d[k])
                    value_uses = ValueDetailed(**d)
                    values[value_uses.id].value_detailed = value_uses
                    current_used_id = value_uses.id
                    used_values[current_used_id] = value_uses
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

                    used_values[current_used_id].instruction = instruction
                    continue
                if uses_mode == "positions":
                    used_values[current_used_id].positions.append(line)
                    continue
                if uses_mode == "uses":
                    used_values[current_used_id].uses.append(line)
            # elif mode == "HloLiveRange":
            #     val,name=line.split(":")


    print(used_values[24].model_dump_json(indent=2))
    # print(json.dumps(value_name_to_id, indent=2))
    make_graph(values, value_name_to_id)


for file in sorted(dir.glob("*memory-usage-report.txt")):
    module_id = int(file.stem.split(".")[0].split("_")[1])
    module_name = file.stem.split(".")[1]

    # if module_id == 447:
    if module_id == 5:
        analyze_module(file)
