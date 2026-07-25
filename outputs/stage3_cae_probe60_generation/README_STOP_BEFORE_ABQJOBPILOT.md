# STOP Before Abqjobpilot

The Stage 3 true variable-N probe60 cases must have generated INP files before
any enqueue step.

Do not run abqjobpilot until:

`PASS_PROBE60_60_INPS_EXIST_READY_TO_ENQUEUE`

is reported by:

`E:\Projects\RL-LAM-ScanOpt\scripts\stage3\check_probe60_generated_inps.py`

The fixed abqjobpilot command file for later use is:

`E:\Projects\RL-LAM-ScanOpt\outputs\stage3_manual_probe60_handoff\variable_N_probe60_abqjobpilot_commands_FIXED.txt`

Current guardrail: abqjobpilot remains blocked while INP count is below 60 or
while heat-load scan-order mapping is not safely implemented.
