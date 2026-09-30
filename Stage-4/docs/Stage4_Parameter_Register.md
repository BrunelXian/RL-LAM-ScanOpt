# Stage 4 Parameter Register

记录日期：2026-09-27。来源：用户当前指令及既有 deposition-event 模板。
状态：当前 sanity / effective heat-source parameters，尚未经过最终实验 IR 校准，
不应写成永久物理常数。本文登记不代表已应用到 master CAE。

| 参数 | 当前值 |
|---|---:|
| PREHEAT pool MAG | 8.0E10 |
| COHEAT pool MAG | 4.0E10 |
| COHEAT drop MAG | 4.0E10 |
| preheat time | 0.1 s |
| react time | 1.0E-6 s |
| coheat time | 0.1 s |
| cool time | 4.0 s |
| final cooling | 1200 s |
| film coefficient | 46.3 |
| ambient temperature | 25 °C |
| emissivity | 0.285 |
| ModelChange includeStrain | False |

MAG 与 film coefficient 使用模型一致单位制；这里不引入设备功率、absorptivity
或未经确认的实验参数，不建立 MAG 与激光功率的一一换算。
final cooling 指完整制造后的最终冷却；本次没有添加或修改任何 CAE step。

## Current sanity observations
- preheat interface temperature ≈ 1300 °C
- single-drop coheat Tmax ≈ 1813 °C

以上为用户提供、既有模板记载的观察值；本次未运行求解或 ODB extraction，
未独立核验其原始结果。不得标记为本次生成或冻结的新证据。

配置对应：`configs/stage4_sanity.yaml`。
状态机依据：`docs/Stage4_Drop_Deposition_Event_Template.md`。
