"""Regression checks on observed CAE evidence; no Abaqus/solver execution.
Mutations deliberately corrupt mapping, lifecycle, final state and legality.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'CAE/scripts/audit'))
sys.path.insert(0, str(ROOT / 'CAE/scripts/generate'))
from stage4_audit_n32_deposition_model import audit
from stage4_prepare_deposition_request import legal
from stage4_generate_n32_deposition_model import validate_request


def main():
    folder = Path(sys.argv[1])
    req = json.loads((folder/'build_request.json').read_text())
    d = json.loads((folder/'deposition_audit.json.snapshot.json').read_text())
    baseline = json.loads(Path(req['output_cae_path']+'.baseline.json').read_text())
    assert audit(d, req, baseline)['status'] == 'passed'
    results = []
    def mutation(label, change):
        x = copy.deepcopy(d)
        change(x)
        verdict = audit(x, req, baseline)
        assert verdict['status'] == 'failed', label
        results.append({'test': label, 'passed': True, 'detected_errors': verdict['errors'][:3]})
    mutation('wrong_spatial_mapping', lambda x: x['loads']['load-02-pre-pool']['region'].__setitem__(0, 'set_pool_3'))
    mutation('late_previous_heat_reset', lambda x: x['states']['Step-32-cool']['loadStates']['load-01-pre-pool'].__setitem__('status', 'PROPAGATED'))
    mutation('previous_drop_cooling_reset', lambda x: x['states']['Step-10-preheat']['interactionStates']['Int-addi-film-01'].__setitem__('status', 'DEACTIVATED'))
    mutation('future_surface_cooling_early_off', lambda x: x['states']['Step-02-react']['interactionStates']['Int-initial-rad-32'].__setitem__('status', 'DEACTIVATED'))
    mutation('strain_reactivation', lambda x: x['states']['Step-02-react']['interactionStates']['Int-addi-mc-02'].__setitem__('includeStrain', 'ON'))
    mutation('final_material_off', lambda x: x['states']['Step-final-cooling']['interactionStates']['Int-addi-mc-12'].__setitem__('activeInStep', 'OFF'))
    mutation('duplicate_activation', lambda x: x['interactions'].__setitem__('duplicate', copy.deepcopy(x['interactions']['Int-addi-mc-01'])))
    mutation('missing_bottom_layer_elements', lambda x: x['bottom_layer'].__setitem__('elements', []))
    mutation('heat_magnitude_changed', lambda x: x['states']['Step-05-coheat']['loadStates']['load-05-co-drop'].__setitem__('magnitude', 1.0))
    mutation('reactivated_twice', lambda x: x['states']['Step-10-react']['interactionStates']['Int-addi-mc-01'].__setitem__('status', 'MODIFIED'))
    mutation('initial_removal_incomplete', lambda x: x['sets']['set_addi_all']['elements'].pop())
    mutation('wrong_step_duration', lambda x: x['steps']['Step-03-react'].__setitem__('timePeriod', 1.0))
    invalid = [list(range(1,32)), list(range(1,32))+[31], list(range(2,34)), [True]+list(range(2,33)), [1.0]+list(range(2,33))]
    for order in invalid:
        bad = dict(req, scan_order=order)
        for fn in (lambda: legal(order, 32), lambda: validate_request(bad)):
            try:
                fn()
            except ValueError:
                pass
            else:
                raise AssertionError('invalid order accepted')
    results.append({'test': 'five_illegal_orders_rejected_in_both_entry_points', 'passed': True})
    # Re-index actual spatial regions and initial surface switching histories to
    # verify the auditor does not confuse event k with spatial i. This is an
    # in-memory semantic fixture, not an additional CAE or scientific candidate.
    x = copy.deepcopy(d)
    order = list(range(32, 0, -1))
    for k, i in enumerate(order, 1):
        for suffix, region in [('pre-pool','pool'),('co-pool','pool'),('co-drop','addi')]:
            x['loads']['load-%02d-%s'%(k,suffix)]['region'][0] = 'set_%s_%d'%(region,i)
        for family, prefix in [('mc','set_addi'),('film','Surf_addi'),('rad','Surf_addi')]:
            x['interactions']['Int-addi-%s-%02d'%(family,k)]['region'][0] = '%s_%d'%(prefix,i)
        for s in x['states']:
            for family in ('film','rad'):
                x['states'][s]['interactionStates']['Int-initial-%s-%02d'%(family,i)] = copy.deepcopy(d['states'][s]['interactionStates']['Int-initial-%s-%02d'%(family,k)])
    assert audit(x, dict(req, scan_order=order))['status'] == 'passed'
    results.append({'test': 'nonidentity_k_i_auditor_fixture', 'passed': True, 'actual_cae_build': False})
    out = {'status': 'passed', 'results': results, 'nonidentity_cae_build_tested': False}
    (folder/'audit_regression_tests.json').write_text(json.dumps(out, indent=2))
    print('PASS',len(results),'regression groups')


if __name__ == '__main__':
    main()
