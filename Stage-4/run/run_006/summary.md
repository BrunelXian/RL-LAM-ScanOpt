# run_006 — Final Stage 4 archive documentation

## Objective
Archive the tested CAE production route while retaining the positive, bounded Agent diagnostic-workflow result. No Stage 5 execution.

## Inputs and evidence
run_003 build/state audit; run_004 build/provenance notice; run_005 summary, matrix, native A MSG, plan, A/B/C/D input-diff audits and safety reports. Required project documents read. The three archive documents did not previously exist and were created rather than overwriting earlier analysis.

## Changes
README rewritten with CAE/Agent outcomes distinguished, Agent capability in its first half, A/B/C/D results, different-element angle evidence, stop criteria, provenance and claim boundaries.
Created docs/Stage4_Failure_Analysis.md, docs/Stage4_to_Stage5_Handoff.md and status/STAGE4_ARCHIVED.md.
Updated AGENTS, MASTER/CAE status and Run Index to prevent stale implementation priorities from restarting an archived route.
No CAE/JNL/physical parameter/solver/script changes. No ODB opened or solver/training launched.

## Validation
Passed: document links, evidence claims and source-file preservation verified. Stage 3 path/size/mtime inventory unchanged; Stage 4 engineering files and parameters retain recorded SHA256/mtime.
Native A MSG verified Step 24/28/32/36/40 angle context and Step 40 area error; matrix confirms all solver statuses failed and valid arm fatal steps 40/42/6/6.
Different angle-warning groups are not a same-element history; shared/dormant-node mechanism is not promoted to a proved fact.
The invalid B attempt and strengthened lifecycle audit are disclosed, not hidden.

## Git publication
Source project has no .git. User authorized branch publication to https://github.com/BrunelXian/RL-LAM-ScanOpt.
Independent checkout: E:/Projects/Stage4-publication-run006.
Target branch: stage4-cae-failure-agent-diagnostics; no force push.
Stage-4/ contains a lightweight documentation/evidence archive; remote original tree is preserved, with a root README entry point.
Formal CAE/ODB remain local and are not published; historical Run manifests can reference local-only files.

## Handoff
Local documentation and remote publication are separate deliverables. Archive commit 8e02d2c62494216f939df280c430151c592d21e3 was successfully pushed to branch stage4-cae-failure-agent-diagnostics; no force push. Closure metadata is a subsequent documentation commit on the same branch. No new scientific CASEs (case_ids=[]). No Stage 5 task started.

Archive scope: docs/Stage4_Git_Archive_Scope.md. Baseline MSG excerpts carry source hash and line numbers. Full solver files remain local.
