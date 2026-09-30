"""Build-only N32 deposition events from a verified single-drop CAE copy.
Usage: abaqus cae noGUI=this.py -- --request request.json
Prepare request with stage4_prepare_deposition_request.py (ordinary Python).
Identity order preserves all event-01 objects and states. Other permutations retain
its steps/controls and remap only event-01 regions/cooling switch; audit is mandatory.
Never submits a solver, writes ODB, allocates CASE, or overwrites an existing CAE.
"""
import argparse
import json
import os
import shutil
import sys

SCRIPT_PATH = globals().get('__file__') or sys.argv[sys.argv.index('-noGUI') + 1]


def validate_request(req):
    n, order = req['N'], req['scan_order']
    if type(n) is not int or n != 32 or not isinstance(order, list):
        raise ValueError('Expected N32 request')
    if any(type(i) is not int for i in order) or len(order) != n or sorted(order) != list(range(1, n+1)):
        raise ValueError('scan_order must be a permutation of integers 1..32')
    if req['config']['model']['N'] != n or req['config']['model_change']['include_strain'] is not False:
        raise ValueError('Geometry/strain configuration mismatch')
    base, out = [os.path.normcase(os.path.realpath(req[k])) for k in ('base_cae_path', 'output_cae_path')]
    master = os.path.normcase(os.path.realpath(os.path.join(os.path.dirname(SCRIPT_PATH), '..', '..', 'Models', 'N32', 'Sanity', 'N32_sanity.cae')))
    if base == out or out == master or os.path.exists(out):
        raise ValueError('Refusing to overwrite master, input or existing output')


def generate(req):
    from abaqus import openMdb
    from abaqusConstants import EMBEDDED_COEFF, UNIFORM, AMBIENT
    import part, material, section, assembly, step, interaction, load, mesh, job
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(SCRIPT_PATH), '..', 'audit')))
    from stage4_audit_n32_deposition_model import snapshot, audit, sha, atomic, PHASES
    validate_request(req)
    base, out = req['base_cae_path'], req['output_cae_path']
    if sha(base) != req['base_sha256'] or sha(req['config_path']) != req['config_sha256']:
        raise ValueError('Request provenance checksum mismatch')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Exclusive file creation closes the check/copy race and protects existing outputs.
    with open(base, 'rb') as source, open(out, 'xb') as dest:
        shutil.copyfileobj(source, dest)
    db = None
    result = {'status': 'failed', 'solver_status': 'not_run', 'case_id': req.get('case_id'),
              'diagnostic_only': req['diagnostic_only'], 'scan_order': req['scan_order'],
              'strategy': 'preserve_event_01_extend_02_32', 'warnings': []}
    try:
        db = openMdb(pathName=out)
        if list(db.models.keys()) != ['Model-1']:
            raise ValueError('Expected exactly Model-1')
        m = db.models['Model-1']
        baseline = snapshot(m)
        template_report = audit(baseline, req, template=True)
        atomic(out + '.template_audit.json', template_report)
        atomic(out + '.baseline.json', baseline)
        if template_report['status'] != 'passed':
            raise ValueError('Master template conflicts with frozen physics: ' + str(template_report['errors'][:8]))
        config, order = req['config'], req['scan_order']
        inst = m.rootAssembly.instances['inst_plate']
        # Insert steps before the existing final cooling; Abaqus updates predecessors.
        step_keys = ('initialInc', 'maxInc', 'minInc', 'maxNumInc', 'deltmx', 'nlgeom',
                     'amplitude', 'response', 'timeIncrementationMethod', 'matrixStorage', 'solutionTechnique')
        controls = {p: {key: getattr(m.steps['Step-01-'+p], key) for key in step_keys} for p in PHASES}
        previous = 'Step-01-cool'
        for k in range(2, len(order)+1):
            for phase in PHASES:
                name = 'Step-%02d-%s' % (k, phase)
                m.CoupledTempDisplacementStep(name=name, previous=previous,
                    timePeriod=config['timing'][phase], **controls[phase])
                previous = name
        # For a future nonidentity permutation, change only event-01 spatial targets.
        # The documented diagnostic build never enters this branch.
        if order[0] != 1:
            i = order[0]
            result['strategy'] += '_remap_first_event'
            for suffix, region in [('pre-pool', 'pool'), ('co-pool', 'pool'), ('co-drop', 'addi')]:
                m.loads['load-01-'+suffix].setValues(region=inst.sets['set_%s_%d' % (region, i)])
            m.interactions['Int-addi-mc-01'].setValues(region=inst.sets['set_addi_%d' % i])
            for family in ('film', 'rad'):
                m.interactions['Int-addi-%s-01' % family].setValues(surface=inst.surfaces['Surf_addi_%d' % i])
                m.interactions['Int-initial-%s-01' % family].reset('Step-01-react')
                m.interactions['Int-initial-%s-%02d' % (family, i)].deactivate('Step-01-react')
        env = config['environment']
        for k, i in enumerate(order, 1):
            if k == 1:
                continue
            pre, react, co, cool = ['Step-%02d-%s' % (k, phase) for phase in PHASES]
            specs = [('pre-pool', 'pool', pre, react, 'preheat_pool_mag'),
                     ('co-pool', 'pool', co, cool, 'coheat_pool_mag'),
                     ('co-drop', 'addi', co, cool, 'coheat_drop_mag')]
            for suffix, region, start, end, value in specs:
                name = 'load-%02d-%s' % (k, suffix)
                m.BodyHeatFlux(name=name, createStepName=start, region=inst.sets['set_%s_%d' % (region, i)],
                               magnitude=config['heat'][value])
                m.loads[name].deactivate(end)
            for family in ('film', 'rad'):
                m.interactions['Int-initial-%s-%02d' % (family, i)].deactivate(react)
            m.ModelChange(name='Int-addi-mc-%02d' % k, createStepName=react,
                          region=inst.sets['set_addi_%d' % i], activeInStep=True,
                          includeStrain=config['model_change']['include_strain'])
            surface = inst.surfaces['Surf_addi_%d' % i]
            m.FilmCondition(name='Int-addi-film-%02d' % k, createStepName=co, surface=surface,
                            definition=EMBEDDED_COEFF, filmCoeff=env['film_coefficient'],
                            filmCoeffAmplitude='', sinkTemperature=env['temperature'], sinkAmplitude='',
                            sinkDistributionType=UNIFORM, sinkFieldName='')
            m.RadiationToAmbient(name='Int-addi-rad-%02d' % k, createStepName=co, surface=surface,
                                 radiationType=AMBIENT, emissivity=env['emissivity'],
                                 ambientTemperature=env['temperature'], ambientTemperatureAmp='',
                                 distributionType=UNIFORM, field='')
        built = snapshot(m)
        pre_save = audit(built, req, baseline)
        atomic(out + '.presave_audit.json', pre_save)
        if pre_save['status'] != 'passed':
            raise ValueError('Built model failed audit: ' + str(pre_save['errors'][:8]))
        db.save()
        result['status'] = 'built'
        result['cae_sha256'] = sha(out)
    except Exception:
        import traceback
        result['error'] = traceback.format_exc()
        raise
    finally:
        if db is not None:
            db.close()
        result['master_unchanged'] = sha(base) == req['base_sha256']
        atomic(out + '.build.json', result)
    if not result['master_unchanged']:
        raise RuntimeError('Master changed during build')
    print('BUILD_ONLY_COMPLETED ' + out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    args = p.parse_args(sys.argv[sys.argv.index('--request'):])
    with open(args.request) as f:
        generate(json.load(f))


if __name__ == '__main__':
    main()



