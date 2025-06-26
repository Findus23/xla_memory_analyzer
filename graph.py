from graphviz import Digraph
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from models import ModuleStats


def make_graph(module_stats: ModuleStats):
    dot = Digraph('memory-graph', comment='Memory Graph')
    for val in module_stats.values.values():
        label = val.name + "\n" + val.pretty_size + "\n" + val.array_info
        if val.sequence is not None:
            cmap = plt.get_cmap("plasma")
            color = cmap(val.sequence / module_stats.largest_sequence_value)
            print(to_hex(color))
        else:
            color = "green"

        dot.node(str(val.id), label=label, fontsize="14" if val.is_large_array else "8", style="filled",
                 fillcolor=to_hex(color) + "33", )
        for uses in val.value_detailed.uses:
            if not uses:
                continue
            uses_name, operand_str = uses.split(",", maxsplit=1)
            uses_name = uses_name.strip()
            operand = operand_str.split()[-1]
            if uses_name in module_stats.value_name_to_id:
                dot.edge(str(val.id), str(module_stats.value_name_to_id[uses_name]), headlabel=operand,
                         labeldistance="2")
            else:
                dot.node(uses_name, fontsize="10", shape="box")
                dot.edge(str(val.id), str(uses_name), headlabel=operand, labeldistance="2")
    # print(dot.source)
    dot.render(directory='.', format='svg')
