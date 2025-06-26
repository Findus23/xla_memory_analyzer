from graphviz import Digraph

from models import Value


def make_graph(values: dict[int, Value], value_name_to_id: dict[str, int]):
    dot = Digraph('memory-graph', comment='Memory Graph')
    for val in values.values():
        label = val.name + "\n" + val.pretty_size + "\n" + val.array_info
        dot.node(str(val.id), label=label, fontsize="14" if val.is_large_array else "8")
        for uses in val.value_detailed.uses:
            if not uses:
                continue
            print(uses)
            print(uses.split(","))
            uses_name, operand_str = uses.split(",")
            uses_name = uses_name.strip()
            operand = operand_str.split()[-1]
            print(operand)
            if uses_name in value_name_to_id:
                dot.edge(str(val.id), str(value_name_to_id[uses_name]), headlabel=operand, labeldistance="2")
            else:
                dot.edge(str(val.id), str(uses_name), headlabel=operand, labeldistance="2")
    print(dot.source)
    dot.render(directory='.')
