import json
import re

base_re = re.compile(r'''
    ^
    (?P<var>%[^\s=]+)           # the %variable
    \s*=\s*
    (?P<dtype>.*?)              # anything (non-greedy) up to the op
    \s+
    (?P<op>\w+)                 # the operation name
    \(
      (?P<operands>[^)]*)      # comma-sep list of operands
    \)
    (?:\s*,\s*(?P<rest>.*))?    # the rest of the line (attrs + metadata)
    $
''', re.VERBOSE)

meta_kv_re = re.compile(r'(\w+)=("([^"]*)"|(\d+))')


def parse_mlir_line(line):
    m = base_re.match(line.strip())
    if not m:
        return None
    d = m.groupdict()

    # tidy up
    result = {
        'var': d['var'],
        'dtype': d['dtype'].strip(),
        'op': d['op'],
        'operands': [op.strip() for op in d['operands'].split(',') if op.strip()],
        'attrs': {},
        'metadata': {}
    }

    rest = d.get('rest') or ''

    # extract metadata={} block
    meta_match = re.search(r'metadata=\{(.*)\}', rest)
    if meta_match:
        meta_content = meta_match.group(1)
        for key, full, strval, numval in meta_kv_re.findall(meta_content):
            val = strval if strval else numval
            if numval:
                val = int(numval)
            result['metadata'][key] = val

    # strip out metadata from rest, then split remaining attrs on commas
    attrs_str = re.sub(r'metadata=\{.*\}', '', rest).strip(' ,')
    if attrs_str:
        for kv in attrs_str.split(','):
            if '=' in kv:
                k, v = kv.split('=', 1)
                result['attrs'][k.strip()] = v.strip()
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


if __name__ == '__main__':
    lines = """
    %constant_1513_0 = f32[8]{0} constant({0.0608953461, 0.0608953387, 0.060895294, 0.0608953536, 0.0608953238, 0.0608952641, 0.0608952641, 0.0608953238})
    %loop_dynamic_slice_fusion.3 = f32[1]{0} fusion(%constant_1513_0, %param.42), kind=kLoop, calls=%fused_dynamic_slice.3, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/dynamic_slice" source_file="/home/luwi100116/DISCO-DJ/src/discodj/nbody/steppers/dkd_leapfrog.py" source_line=37 deduplicated_name="loop_dynamic_slice_fusion.2"}
    %loop_add_fusion.1 = f32[226492416,3]{1,0} fusion(%input_concatenate_fusion.4, %loop_dynamic_slice_fusion.3, %param.41, %param.43, %get-tuple-element.275), kind=kLoop, calls=%fused_add.1, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/add" source_file="/weka/p201009/luwi100116/venvs/discodj/lib/python3.11/site-packages/equinox/internal/_omega.py" source_line=100}
    %tuple.131.0 = (f32[226492416,3]{1,0}, f32[226492416,3]{1,0}) tuple(%loop_add_fusion.1, %input_concatenate_fusion.4)
    %conditional = () conditional(%loop_compare_fusion, %tuple.109.0, %tuple.109.0), true_computation=%true_computation, false_computation=%region_5.368_spmd, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/cond/branch_1_fun/debug_callback" source_file="/home/luwi100116/DISCO-DJ/src/discodj/core/scatter_and_gather.py" source_line=1330}
    %conditional.1 = () conditional(%loop_convert_fusion.2, %tuple.120.0, %tuple.121.0), branch_computations={%region_5.368_spmd.clone, %region_6.371_spmd}, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/cond" source_file="/home/luwi100116/DISCO-DJ/src/discodj/core/scatter_and_gather.py" source_line=1328}
    %conditional.2 = (c64[3072,24,1537]{2,1,0}) conditional(%bitcast.18.0, %tuple.122.0, %tuple.123.0, %tuple.124.0), branch_computations={%region_15.613_spmd, %region_16.620_spmd, %region_17.627_spmd}, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/cond" source_file="/home/luwi100116/DISCO-DJ/src/discodj/nbody/acc.py" source_line=380}
    """.strip()

    for line in lines.splitlines():
        parse_mlir_line(line)
