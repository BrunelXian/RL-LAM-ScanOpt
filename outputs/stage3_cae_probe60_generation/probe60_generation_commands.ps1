& "D:\XianLab\envs\conda\torch-gpu\python.exe" "E:\Projects\RL-LAM-ScanOpt\scripts\stage3\preflight_probe60_generation_inputs.py"

& "D:\XianLab\envs\conda\torch-gpu\python.exe" "E:\Projects\RL-LAM-ScanOpt\scripts\stage3\audit_probe60_base_jnl_names.py"

abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\generate_probe60_from_sanity_base_nogui.py"

& "D:\XianLab\envs\conda\torch-gpu\python.exe" "E:\Projects\RL-LAM-ScanOpt\scripts\stage3\check_probe60_generated_inps.py"

# Do not run abqjobpilot from this command file.
# abqjobpilot can be used later only after:
# PASS_PROBE60_60_INPS_EXIST_READY_TO_ENQUEUE
