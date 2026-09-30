"""Ordinary Python YAML -> immutable, checksummed Abaqus build request.
No CASE is allocated here. --case-json consumes a future existing Stage4Case.
"""
import argparse, hashlib, json, os
from pathlib import Path
import yaml


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def legal(order, n):
    if type(n) is not int or n != 32:
        raise ValueError('This geometry adapter supports N=32 only')
    if not isinstance(order, list) or any(type(i) is not int for i in order):
        raise ValueError('scan_order must contain integers, not bool/float/string')
    if len(order) != n or sorted(order) != list(range(1, n + 1)):
        raise ValueError('scan_order must be a permutation of 1..N')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--scan-order', help='JSON array')
    p.add_argument('--case-json')
    p.add_argument('--process-config-id', help='Explicit immutable ID binding for future CASE consumption')
    p.add_argument('--base-cae-path', required=True)
    p.add_argument('--output-cae-path', required=True)
    p.add_argument('--request', required=True)
    a = p.parse_args()
    config = yaml.safe_load(Path(a.config).read_text(encoding='utf-8-sig'))
    case = None
    if a.case_json:
        if a.scan_order:
            p.error('Do not override a CASE scan_order')
        import jsonschema
        case = json.loads(Path(a.case_json).read_text(encoding='utf-8-sig'))
        schema = Path(__file__).resolve().parents[3] / 'configs/schemas/case.schema.json'
        jsonschema.Draft202012Validator(json.loads(schema.read_text(encoding='utf-8-sig')),
                                      format_checker=jsonschema.FormatChecker()).validate(case)
        if case['example_only'] or not a.process_config_id or case['process_config_id'] != a.process_config_id:
            raise ValueError('Real CASE requires explicit matching registered configuration binding')
        order = case['scan_order']
        if case['N'] != config['model']['N']:
            raise ValueError('CASE/config geometry mismatch')
    else:
        order = json.loads(a.scan_order or 'null')
    legal(order, config['model']['N'])
    base, out = Path(a.base_cae_path).resolve(), Path(a.output_cae_path).resolve()
    if base == out or out.exists():
        raise ValueError('Output must be a new derived CAE, never overwrite input/output')
    request = {'schema_version': '1.0', 'diagnostic_only': case is None,
               'case_id': case['case_id'] if case else None,
               'process_config_id': case['process_config_id'] if case else None,
               'N': config['model']['N'], 'scan_order': order, 'config': config,
               'config_path': str(Path(a.config).resolve()), 'config_sha256': sha(a.config),
               'base_cae_path': str(base), 'base_sha256': sha(base),
               'output_cae_path': str(out)}
    dest = Path(a.request)
    if dest.exists():
        raise ValueError('Request already exists')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(request, indent=2), encoding='utf8')
    os.replace(str(tmp), str(dest))
    print(str(dest))


if __name__ == '__main__':
    main()
