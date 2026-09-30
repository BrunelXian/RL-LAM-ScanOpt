# N32 deposition-event generator

Authority: `docs/Stage4_Drop_Deposition_Event_Template.md` and
`configs/stage4_sanity.yaml`. Run-003 diagnostic is not a scientific CASE.

## Build-only usage (PowerShell)

```powershell
$root = 'E:\Projects\Stage-4'
$order = '[' + ((1..32) -join ',') + ']'
& 'D:\XianLab\envs\Chuzi-PY\Scripts\python.exe' "$root\CAE\scripts\generate\stage4_prepare_deposition_request.py" --config "$root\configs\stage4_sanity.yaml" --scan-order $order --base-cae-path "$root\CAE\Models\N32\Sanity\N32_sanity.cae" --output-cae-path '<NEW derived CAE path>' --request '<NEW request.json path>'
& 'D:\ABAQUS2024\commands\abaqus.bat' cae noGUI=E:\Projects\Stage-4\CAE\scripts\generate\stage4_generate_n32_deposition_model.py -- --request '<request.json path>'
& 'D:\ABAQUS2024\commands\abaqus.bat' cae noGUI=E:\Projects\Stage-4\CAE\scripts\audit\stage4_audit_n32_deposition_model.py -- --request '<request.json path>' --report '<audit.json path>'
```

Run commands from a dedicated CAE report directory so Abaqus replay/log files stay
inside Stage 4. Existing output CAE/request paths are rejected. A failed build may
leave an unvalidated derived copy; consult `.build.json`, retain it for diagnosis,
and use a new output location for a retry. Abaqus launcher exit code alone is not
reliable: require build status `built`, independent audit `passed`, current hashes,
and the Run's validation/handoff gate. No job is submitted and no INP/ODB is needed.

## Design and validation

The bridge parses YAML with ordinary Python/PyYAML; Abaqus reads the checksummed JSON.
No process_config_id is allocated for a diagnostic. Future `--case-json` accepts a
schema-valid existing Stage4Case, rejects example_only, validates N/order and requires
an explicit matching `--process-config-id`. Formal immutable configuration registry
binding and CASE workflow integration remain future work; this adapter does not
allocate CASE/JOB identities or emit a fake CAE Job Manifest.

The generator copies the base before opening it and checks the single-drop template
against the frozen physics. For identity order, existing event 01, initial removal,
initial cooling, geometry, mesh and bottom layer are preserved. Events 02–32 are
inserted before the existing final cooling and inherit event-01 phase controls.
Only current spatial drop i is switched at event k. New drop cooling propagates.

Other legal permutations use event k for naming and spatial i for region lookup.
If the first spatial drop is not 1, the retained event-01 objects are remapped and
its original substrate cooling deactivation is reset. This branch is implemented
but has NOT received an actual nonidentity CAE build in run_003; validate it in a
separately authorized future diagnostic before production use. The auditor's k/i
logic has an in-memory nonidentity fixture; that is not an additional built model.

The independent audit reopens the saved CAE and examines every actual step state:
exact object inventories/types, spatial mappings, magnitude/timing/environment,
creation/deactivation/propagation, strain-free activation, initial removal element
coverage, disjoint deposit sets, final state and master-prefix preservation. It also
compares mesh/set/bottom-layer membership against the source snapshot. The source
contains mesh-based deposit sets with empty geometry cell arrays; element labels
are the authoritative nonempty coverage check.

Counts: 132 steps including Initial (131 analysis steps), 32 deposition ModelChange
activations plus one initial removal, 65 Film and 65 Radiation objects (32 substrate,
32 deposit and one retained fixed external surface each), 96 heat loads. Removal
PROPAGATED status is not a repeated remove command; event activations are persistent.

Regression test: ordinary Python `CAE/scripts/audit/stage4_test_deposition_audit.py
<report directory>`. It deliberately corrupts observed states and verifies rejection.
Build validation proves CAE state-machine construction, not solver convergence,
thermal predictions, experimental calibration or teacher-data readiness.
