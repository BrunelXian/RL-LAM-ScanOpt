# run_003 — N32 deposition-event generator

Agent: stage4_cae. Operation: build_n32_deposition_event_generator.
Inputs: run_001 + run_002; their manifests resolved and files verified. case_ids=[].
The identity diagnostic is not a scientific candidate; S4C000001 remains unallocated.

## Strategy and engineering work

Opened only a byte copy of the N32 sanity master. Its actual event-01 physics matches
frozen YAML/template. Retained event-01 steps, loads, interactions and state prefix;
created events 02–32 and inserted them before the existing final-cooling step. This
preserves manual event-01 verification while extending exactly the same physical
sequence. k controls event naming; i controls region lookup. No master save occurred.
Geometry, mesh, set memberships, boundary conditions and predefined fields preserved.
No frozen heat/timing/environment parameter changed. YAML is parsed once by the
ordinary-Python bridge into a checksummed request; no scientific config ID invented.

Primary generator: ../../CAE/scripts/generate/stage4_generate_n32_deposition_model.py
Independent audit: ../../CAE/scripts/audit/stage4_audit_n32_deposition_model.py
Usage: ../../CAE/scripts/generate/README.md
CAE: ../../CAE/Models/N32/Generated/diagnostic/stage4_N32_diag_identity_order.cae
Request/audit/safety/regression evidence: ../../CAE/Reports/N32/diagnostic/run_003/
Run contains metadata and references only, not engineering file copies.

## Actual build and validation

Abaqus 2024 noGUI build-only executed. Saved CAE reopened in a separate invocation.
Independent audit passed 70,276 assertions across actual saved step/object states.
132 total steps including Initial (131 analysis steps): Initial + 2 initial analysis
steps + 32 each of preheat/react/coheat/cool + final cooling (1200 s).
32 unique deposition activations, one per set_addi_i; plus initial all-deposit removal,
so total ModelChange objects = 33. No duplicate activation or cooling/heat objects.
Film=65 and Radiation=65: 32 Surf_in + 32 Surf_addi + 1 retained Surf_fix_all each.
Heat=96: 32 preheat pool + 32 coheat pool + 32 coheat drop. All mappings/lifecycles pass.
Every future drop remains removed until its REACT; its Surf_in remains on until then.
Every deposited drop remains active; its in-surface cooling remains off and its
addi-surface cooling persists from COHEAT through every later step including final.
Initial removal propagation is not interpreted as a repeatedly applied removal.
Final: addi active 32/32; in Film/Rad off 32/32 each; addi Film/Rad on 32/32 each;
all 96 heat loads off. set_bottom_layer has 136 elements and unchanged membership.

Regression suite passed 14 groups: 12 deliberate corruptions rejected, five illegal
orders rejected by both entry points, and one nonidentity in-memory auditor fixture.
These are audit tests, not another CAE build or CASE. No solver/analysis/INP/ODB generated.
No ODB metrics/teacher pipeline/dataset/RL/surrogate changes performed.

## Safety and provenance

Master SHA256 before = after:
1500483e486c983ef3d2c583655368a5bc37ca5147a2b7318efd2f4172a8ce1e
Generated CAE SHA256: 2c554c33fccb4894cb7bc9c33a834ed20d4e4d472acca7c6dd1903da351d2197
Stage 3: 17,209-file inventory, sizes and nanosecond mtimes unchanged; 7,837
text/script/config content hashes checked and unchanged. Large Stage 3 binaries were
not all content-hashed; no claim of a full binary hash audit is made.
Stage 4 pre-existing original engineering files retained; only this Run's files and
authorized CAE_STATUS / minimal MASTER latest-run lines / Run index updated.

## Resolved implementation issues and limitations

During development, Abaqus noGUI exposed its launcher flags in argv, did not define
__file__, and raises errors when inactive attributes are accessed. The independent
entrypoint also inherited an overridden sum from Abaqus startup. These compatibility
issues were fixed; active parameter absence still fails validation. Mesh-based deposit
sets have empty cell arrays, so initial removal coverage requires nonempty element
coverage, while cell coverage/disjointness is still compared. Failed copies/logs are
retained inside CAE/Reports/N32/diagnostic/run_003; they are not handoff models.
Warnings: none unresolved within requested identity-order scope.
Nonidentity first-event remapping is implemented but not actual-CAE-tested this Run.
Future formal CASE/config registry integration is not completed; no fake Job Manifest
or config registration was produced. Build audit does not establish solver convergence,
thermal predictions or experimental calibration.

## Handoff and next recommendation

Execution=completed; validation=passed; handoff.ready=true.
Scope: validated N32 identity diagnostic model/generator state-machine construction.
Not authorization to submit solver or feed teacher data. Suggested next action only:
manually review this diagnostic CAE, then separately authorize a nonidentity diagnostic
to validate first-event remapping before formal candidate integration. No next action
executed automatically.

PASS_STAGE4_N32_DEPOSITION_EVENT_GENERATOR_READY
