# Probe60 Pilot-Only Generation Instructions

Do not generate all 60 cases immediately.

1. Review:

   `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\heat_mapping_jnl_snippets.md`

2. Review and edit if needed:

   `E:\Projects\RL-LAM-ScanOpt\outputs\stage3_cae_probe60_generation\probe60_heat_mapping_config_TEMPLATE.json`

3. Manually confirm the heat-load mapping:

   - exact Abaqus load type is `BodyHeatFlux` or equivalent
   - load magnitude is inherited from `load_body_hflux_00`
   - scan step time period is inherited from `step_scan_00`
   - cool step time period is inherited from `step_cool_00`
   - final cooling inherits the cooling template procedure type
   - `step_final_cooling` duration is 1200.0 seconds
   - track region objects exist as `set_body_heat_XX` for every N

4. Only after manual confirmation, edit:

   `E:\Projects\RL-LAM-ScanOpt\scripts\stage3\generate_probe60_from_sanity_base_nogui.py`

   Set:

   ```python
   ALLOW_USER_CONFIRMED_HEAT_MAPPING = True
   ONLY_GENERATE_ONE_PILOT_CASE = True
   ```

5. Run Abaqus noGUI to generate one pilot case only:

   ```powershell
   abaqus cae noGUI="E:\Projects\RL-LAM-ScanOpt\scripts\stage3\generate_probe60_from_sanity_base_nogui.py"
   ```

   Pilot case:

   `N12_A01_raster_left_to_right`

6. Run the generated-file checker:

   ```powershell
   & "D:\XianLab\envs\conda\torch-gpu\python.exe" "E:\Projects\RL-LAM-ScanOpt\scripts\stage3\check_probe60_generated_inps.py"
   ```

7. Run the pilot INP heat-order checker:

   ```powershell
   & "D:\XianLab\envs\conda\torch-gpu\python.exe" "E:\Projects\RL-LAM-ScanOpt\scripts\stage3\check_probe60_pilot_inp_heat_order.py"
   ```

8. Open the generated pilot INP and verify:

   - N scan steps exist
   - N cool steps exist if expected
   - one additional 1200-second `step_final_cooling` exists after the last cooling step
   - heat loads follow `scan_order`
   - no heat loads are active during `step_final_cooling`
   - no solver job was submitted

Only after pilot inspection should the user consider generating all 60 cases.
