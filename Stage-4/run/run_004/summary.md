# run_004 — N32 custom-order diagnostic CAE

Agent: stage4_cae. Operation: build_n32_custom_order_diagnostic_model.
Input runs: [run_003]. case_ids=[]. No later CAE/debug Run existed at allocation.
Execution completed; validation passed; handoff ready for diagnostic model only.

## Input and method

scan_order = [12, 2, 4, 16, 22, 24, 32, 26, 30, 20, 28, 18, 14, 10, 6, 1, 8, 25, 29, 21, 23, 31, 27, 19, 17, 7, 5, 3, 9, 11, 15, 13]

Used existing generator, config bridge and independent audit without code changes.
Frozen parameters unchanged. Source: CAE/Models/N32/Sanity/N32_sanity.cae.
Only derived output opened for building. Existing event-01 step controls retained;
existing nonidentity branch remapped its loads/activation/deposit cooling to drop 12,
reset drop-1 substrate cooling switch, and switched drop-12 substrate cooling at
Step-01-react. Remaining event k uses scan_order[k-1] as actual spatial i. Drop 1
is deposited at event 16; final event 32 deposits drop 13. No physics redesign.

Generated CAE: ../../CAE/Models/N32/Generated/diagnostic/stage4_N32_diag_custom_order_001.cae
Build request: ../../CAE/Reports/N32/diagnostic/run_004/build_request.json
Independent audit: ../../CAE/Reports/N32/diagnostic/run_004/deposition_audit.json
Observed mapping: ../../CAE/Reports/N32/diagnostic/run_004/mapping_summary.json
Validation: ../../CAE/Reports/N32/diagnostic/run_004/run_validation.json

## Actual execution and validation

Abaqus 2024 noGUI build-only succeeded; saved CAE reopened in separate audit invocation.
Independent audit passed 69,765 checks; no errors. Counts derived from exact
inventory and state checks: 132 steps including Initial (131 analysis steps); 2 initial
analysis steps, 32 each of preheat/react/coheat/cool, one final cooling of 1200 seconds.
32 deposit activation ModelChange objects plus one initial removal = 33 total.
96 heat loads: 32 preheat pool, 32 coheat pool and 32 coheat drop.
65 Film and 65 Radiation: 32 in surfaces, 32 deposit surfaces and one preserved fixed
external surface each. No duplicate steps, loads or interactions.
All 32 k-to-i mappings checked for ModelChange, all three load classes and cooling.
In-surface Film/Rad deactivate at matching react; deposit Film/Rad start at matching
coheat and persist through every later step. All heat is off already at Step-32-cool.
Final: 32 addi active; 32 in Film and 32 in Radiation off; 32 deposit Film and 32 deposit
Radiation on; zero heat loads active. set_bottom_layer retained: 136 elements with
exact unchanged membership. Geometry/mesh/sets and inherited controls preserved.

## Safety and provenance

Master SHA256 unchanged: 1500483e486c983ef3d2c583655368a5bc37ca5147a2b7318efd2f4172a8ce1e
Generated CAE SHA256 after reopen: 36d970255d1f43027b028c95fedb8b1eab4aaaebf23e70dd77ade2561648f47a
Sanity YAML, existing generator/audit/bridge and RL/ODB status documents unchanged.
Stage 3 inventory, sizes/mtimes and 6,829 text/script/config SHA256 values
unchanged across 17,209 files. ODB contents were not opened; large binary content
hashing is not claimed. No solver, INP generation, ODB reading, teacher extraction,
PPO/surrogate or CASE creation occurred. S4C000001 was not allocated.

Pre-existing provenance notice: Existing identity CAE differed from run_003 handoff SHA256 before this run; not used as build input. Its current bytes are unchanged by run_004. Cause not investigated.
Identity SHA256 at run_003 handoff:
2c554c33fccb4894cb7bc9c33a834ed20d4e4d472acca7c6dd1903da351d2197
Identity SHA256 at start and end of run_004:
0b612e7baac43a4f3aa076673bb057c8ecb41924d6b9786fb30beaf608a580c6
This does not invalidate the new model, which was built from the unchanged master.
Warnings affecting current model validation: none. No automatic future action taken.

## Handoff

Scope is this specific custom-order diagnostic CAE; not a formal scientific candidate,
solver convergence result, teacher data or blanket validation of every permutation.
Suggested next step: manual CAE review; solver/candidate work requires a separate task.

PASS_STAGE4_N32_CUSTOM_ORDER_CAE_READY
