"""Independent CAE snapshot and lifecycle audit; never submits a job.
Abaqus: cae noGUI=this.py -- --request request.json --report audit.json
Checks actual saved CAE states, not the generator's planned operations.
"""
from builtins import sum, map, any, all
import argparse
import hashlib
import json
import os
import sys

PHASES = ('preheat', 'react', 'coheat', 'cool')
STEP_ATTRS = ('previous', 'timePeriod', 'procedureType', 'suppressed', 'nlgeom',
              'initialInc', 'maxInc', 'minInc', 'maxNumInc', 'deltmx', 'amplitude',
              'response', 'timeIncrementationMethod', 'matrixStorage', 'solutionTechnique')
STATE_ATTRS = ('status', 'magnitude', 'amplitude', 'filmCoeff', 'filmCoeffAmplitude',
               'sinkTemperature', 'sinkAmplitude', 'emissivity', 'ambientTemperature',
               'ambientTemperatureAmp', 'activeInStep', 'includeStrain')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, data):
    with open(path + '.tmp', 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(path + '.tmp', path)


def attrs(obj, names):
    out = {}
    for key in names:
        try:
            value = getattr(obj, key)
            out[key] = value if type(value) in (bool, int, float, str) else str(value)
        except Exception:
            pass
    return out


def membership(region):
    return {'cells': sorted(c.index for c in region.cells),
            'elements': sorted(e.label for e in region.elements),
            'nodes': sorted(n.label for n in region.nodes)}


def snapshot(m):
    d = {'step_order': list(m.steps.keys()), 'steps': {}, 'loads': {}, 'interactions': {},
         'states': {}, 'sets': {}, 'surfaces': {}, 'preserved': {}}
    for name, obj in m.steps.items():
        d['steps'][name] = attrs(obj, STEP_ATTRS)
        d['states'][name] = {repo: {k: attrs(v, STATE_ATTRS) for k, v in getattr(obj, repo).items()}
                             for repo in ('loadStates', 'interactionStates')}
    for repo in ('loads', 'interactions'):
        for name, obj in getattr(m, repo).items():
            d[repo][name] = attrs(obj, ('suppressed', 'distributionType', 'definition', 'radiationType', 'regionType'))
            d[repo][name]['type'] = type(obj).__name__
            d[repo][name]['region'] = list(obj.region)
    inst = m.rootAssembly.instances['inst_plate']
    for name, obj in inst.sets.items():
        d['sets'][name] = membership(obj)
    for name, obj in inst.surfaces.items():
        d['surfaces'][name] = {'faces': sorted(f.index for f in obj.faces)}
    d['bottom_layer'] = membership(m.rootAssembly.sets['set_bottom_layer'])
    for repo in ('materials', 'sections', 'boundaryConditions', 'predefinedFields', 'fieldOutputRequests', 'historyOutputRequests'):
        d['preserved'][repo] = {k: str(v) for k, v in getattr(m, repo).items()}
    d['mesh'] = {'nodes': [[n.label, list(n.coordinates)] for n in inst.nodes],
                 'elements': [[e.label, str(e.type), list(e.connectivity)] for e in inst.elements]}
    return d


def audit(d, req, baseline=None, template=False):
    errors = []
    checks = [0]
    def check(ok, msg):
        checks[0] += 1
        if not ok:
            errors.append(msg)
    config = req['config']
    order = [1] if template else req['scan_order']
    n = req['N']
    check(n == 32 and type(n) is int, 'N32 required')
    full = req['scan_order']
    check(isinstance(full, list) and all(type(x) is int for x in full) and
          len(full) == n and sorted(full) == list(range(1, n + 1)), 'illegal scan order')
    steps = ['Initial', 'Step-initial-remove', 'Step-initial-equilibration']
    steps += ['Step-%02d-%s' % (k, p) for k in range(1, len(order) + 1) for p in PHASES]
    steps += ['Step-final-cooling']
    check(d['step_order'] == steps, 'exact ordered step sequence')
    if d['step_order'] != steps:
        return {'status': 'failed', 'errors': errors, 'checks': checks[0]}
    for j, s in enumerate(steps):
        check(not d['steps'][s]['suppressed'], 'suppressed step ' + s)
        if j:
            check(d['steps'][s]['previous'] == steps[j-1], 'step predecessor ' + s)
    for k in range(1, len(order)+1):
        for phase in PHASES:
            s = 'Step-%02d-%s' % (k, phase)
            check(d['steps'][s]['timePeriod'] == config['timing'][phase], 'duration ' + s)
            check(d['steps'][s]['procedureType'] == 'COUPLED_TEMP_DISPLACEMENT', 'procedure ' + s)
    check(d['steps'][steps[-1]]['timePeriod'] == config['timing']['final_cooling'], 'final duration')
    loads, interactions = {}, {}
    env = config['environment']
    for i in range(1, n+1):
        for family, typ in [('film', 'FilmCondition'), ('rad', 'RadiationToAmbient')]:
            interactions['Int-initial-%s-%02d' % (family, i)] = (typ, 'Surf_in_%d' % i)
    for family, typ in [('film', 'FilmCondition'), ('rad', 'RadiationToAmbient')]:
        interactions['Int-initial-%s-fixall' % family] = (typ, 'Surf_fix_all')
    interactions['Int-initial-mc-remove'] = ('ModelChange', 'set_addi_all')
    for k, i in enumerate(order, 1):
        for suffix, region in [('pre-pool', 'pool'), ('co-pool', 'pool'), ('co-drop', 'addi')]:
            loads['load-%02d-%s' % (k, suffix)] = ('BodyHeatFlux', 'set_%s_%d' % (region, i))
        for suffix, typ, region in [('mc', 'ModelChange', 'set_addi'), ('film', 'FilmCondition', 'Surf_addi'), ('rad', 'RadiationToAmbient', 'Surf_addi')]:
            interactions['Int-addi-%s-%02d' % (suffix, k)] = (typ, '%s_%d' % (region, i))
    for repo, expected in [('loads', loads), ('interactions', interactions)]:
        check(set(d[repo]) == set(expected), 'exact names/counts: ' + repo)
        for name, (typ, region) in expected.items():
            if name not in d[repo]:
                continue
            obj = d[repo][name]
            check(obj['type'] == typ and obj['region'][:3] == [region, 'part_plate', 'inst_plate'], 'mapping/type ' + name)
            check(obj.get('suppressed') is False, 'suppressed object ' + name)
    if set(d['loads']) != set(loads) or set(d['interactions']) != set(interactions):
        return {'status': 'failed', 'errors': errors, 'checks': checks[0]}
    def lifecycle(repo, name, start, end=None, parameters=None):
        for pos, step in enumerate(steps):
            state = d['states'][step][repo][name]
            expected = ('NOT_YET_ACTIVE' if pos < start else 'CREATED' if pos == start
                        else 'PROPAGATED' if end is None or pos < end
                        else 'DEACTIVATED' if pos == end else 'NO_LONGER_ACTIVE')
            # Abaqus does not expose a load status in Initial; all analysis steps are strict.
            if not (repo == 'loadStates' and pos == 0):
                check(state.get('status') == expected, 'lifecycle %s @ %s: %s != %s' % (name, step, state.get('status'), expected))
            if start <= pos and (end is None or pos < end):
                for key, value in (parameters or {}).items():
                    check(state.get(key) == value, 'parameter %s.%s @ %s' % (name, key, step))
    def cooling(name, start, end, family):
        params = ({'filmCoeff': env['film_coefficient'], 'sinkTemperature': env['temperature'],
                   'filmCoeffAmplitude': '', 'sinkAmplitude': ''} if family == 'film' else
                  {'emissivity': env['emissivity'], 'ambientTemperature': env['temperature'], 'ambientTemperatureAmp': ''})
        lifecycle('interactionStates', name, start, end, params)
    lifecycle('interactionStates', 'Int-initial-mc-remove', 1, parameters={'activeInStep': 'OFF', 'includeStrain': 'OFF'})
    position = {i: k for k, i in enumerate(order, 1)}
    for i in range(1, n+1):
        k = position.get(i)
        react = steps.index('Step-%02d-react' % k) if k else None
        for family in ('film', 'rad'):
            cooling('Int-initial-%s-%02d' % (family, i), 2, react, family)
    for family in ('film', 'rad'):
        cooling('Int-initial-%s-fixall' % family, 2, None, family)
    rows = []
    for k, i in enumerate(order, 1):
        pre, react, co, cool = [steps.index('Step-%02d-%s' % (k, p)) for p in PHASES]
        lifecycle('interactionStates', 'Int-addi-mc-%02d' % k, react,
                  parameters={'activeInStep': 'ON', 'includeStrain': 'OFF'})
        for family in ('film', 'rad'):
            cooling('Int-addi-%s-%02d' % (family, k), co, None, family)
        for suffix, start, end, key in [('pre-pool', pre, react, 'preheat_pool_mag'),
                                      ('co-pool', co, cool, 'coheat_pool_mag'),
                                      ('co-drop', co, cool, 'coheat_drop_mag')]:
            lifecycle('loadStates', 'load-%02d-%s' % (k, suffix), start, end,
                      {'magnitude': config['heat'][key], 'amplitude': ''})
        rows.append({'event_k': k, 'spatial_i': i, 'react_step': steps[react], 'coheat_step': steps[co]})
    check(config['model_change']['include_strain'] is False, 'strain-free config')
    # Initial removal must cover exactly the disjoint deposit regions, not substrate.
    for category in ('cells', 'elements'):
        groups = [set(d['sets']['set_addi_%d' % i][category]) for i in range(1, n+1)]
        union = set().union(*groups)
        check(bool(union) or category == 'cells', 'nonempty deposit elements')
        check(sum(map(len, groups)) == len(union), 'disjoint deposit ' + category)
        check(union == set(d['sets']['set_addi_all'][category]), 'initial removal coverage ' + category)
    check(bool(d['bottom_layer']['elements']), 'bottom layer nonempty')
    if baseline:
        for key in ('sets', 'surfaces', 'bottom_layer', 'mesh', 'preserved'):
            check(d[key] == baseline[key], 'preserved master ' + key)
        for s in ['Initial', 'Step-initial-remove', 'Step-initial-equilibration'] + ['Step-01-' + p for p in PHASES]:
            check(d['steps'][s] == baseline['steps'][s], 'preserved step01/initial ' + s)
        final0 = dict(baseline['steps']['Step-final-cooling'])
        final0['previous'] = 'Step-%02d-cool' % len(order)
        check(d['steps']['Step-final-cooling'] == final0, 'final settings preserved')
        for k in range(2, len(order)+1):
            for p in PHASES:
                ref = dict(baseline['steps']['Step-01-' + p])
                name = 'Step-%02d-%s' % (k, p)
                ref['previous'] = d['steps'][name]['previous']
                check(d['steps'][name] == ref, 'cloned step controls ' + name)
        if full[0] == 1:
            for s in baseline['step_order'][:-1]:
                for repo in ('loadStates', 'interactionStates'):
                    for name, state in baseline['states'][s][repo].items():
                        check(d['states'][s][repo][name] == state, 'master prefix unchanged %s/%s' % (s, name))
    final = d['states'][steps[-1]]
    final_counts = {
        'addi_active': sum(final['interactionStates']['Int-addi-mc-%02d' % k].get('activeInStep') == 'ON' for k in range(1, len(order)+1)),
        'in_film_off': sum(final['interactionStates']['Int-initial-film-%02d' % i]['status'] == 'NO_LONGER_ACTIVE' for i in range(1, n+1)),
        'in_rad_off': sum(final['interactionStates']['Int-initial-rad-%02d' % i]['status'] == 'NO_LONGER_ACTIVE' for i in range(1, n+1)),
        'addi_film_on': sum(final['interactionStates']['Int-addi-film-%02d' % k]['status'] == 'PROPAGATED' for k in range(1, len(order)+1)),
        'addi_rad_on': sum(final['interactionStates']['Int-addi-rad-%02d' % k]['status'] == 'PROPAGATED' for k in range(1, len(order)+1)),
        'heat_loads_on': sum(v.get('status') in ('CREATED', 'PROPAGATED', 'MODIFIED') for v in final['loadStates'].values())}
    if not template:
        check(all(v == (0 if k == 'heat_loads_on' else n) for k,v in final_counts.items()), 'final state')
    return {'status': 'passed' if not errors else 'failed', 'errors': errors, 'checks': checks[0],
            'counts': {'steps_including_initial': len(steps), 'analysis_steps': len(steps)-1,
                       'events': len(order), 'reactivations': len(order), 'model_change_total': len(order)+1,
                       'film': n+len(order)+1, 'radiation': n+len(order)+1, 'heat_loads': len(loads)},
            'mapping': rows, 'final_state': final_counts, 'bottom_layer_elements': len(d['bottom_layer']['elements']),
            'scope': 'All actual CAE states at every analysis step; removal is a one-time command, propagated removal is not re-applied.'}


def main():
    from abaqus import openMdb
    import part, material, section, assembly, step, interaction, load, mesh, job
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--report', required=True)
    args = p.parse_args(sys.argv[sys.argv.index('--request'):])
    req = json.load(open(args.request))
    report = {'status': 'failed'}
    try:
        db = openMdb(pathName=req['output_cae_path'])
        observed = snapshot(db.models['Model-1'])
        db.close()
        baseline = json.load(open(req['output_cae_path'] + '.baseline.json'))
        report = audit(observed, req, baseline)
        report['cae_sha256'] = sha(req['output_cae_path'])
        report['config_sha256'] = sha(req['config_path'])
        if sha(req['base_cae_path']) != req['base_sha256'] or report['config_sha256'] != req['config_sha256']:
            report['errors'].append('Input checksum changed')
            report['status'] = 'failed'
        atomic(args.report + '.snapshot.json', observed)
    except Exception:
        import traceback
        report = {'status': 'failed', 'errors': [traceback.format_exc()]}
    atomic(args.report, report)
    print('AUDIT: ' + report['status'])
    if report['status'] != 'passed':
        raise RuntimeError('Independent audit failed: ' + str(report['errors'][:5]))


if __name__ == '__main__':
    main()







