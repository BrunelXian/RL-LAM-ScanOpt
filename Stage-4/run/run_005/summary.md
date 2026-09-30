# Stage 4 Numerical Stability Diagnostic

Run: **run_005** · Agent: **stage4_cae** · Operation: **diagnose_n32_numerical_stability_ablation**

## Objective

完成 A baseline、B ALE、C automatic stabilization、D ALE + stabilization 的单次 2×2 数值诊断。有效 Job 按 A → B → C → D 顺序执行，每项结束并读取日志及 ODB 后才开始下一项。四个有效 Job 均 FAILED；完整矩阵已完成，不再调参或提交求解。

输入为用户指定的当前 identity baseline：`CAE/Models/N32/Generated/diagnostic/stage4_N32_diag_identity_order.cae`。扫描顺序为 `[1,2,...,32]`。B/C/D 分别由该文件独立复制；没有从 B 派生 C/D。A 保持 baseline 模型设置。A/B/C/D 仅是 diagnostic arm，不是 Stage4Case；`case_ids=[]`，未占用 S4C000001。

使用 Abaqus 2024 / Standard、14 CPUs、1 GPU；各 Job 独立工作目录及名称。正式构建脚本：[stage4_numerical_stability_ablation.py](../../CAE/scripts/debug/stage4_numerical_stability_ablation.py)。参数计划：[plan.json](../../CAE/Reports/N32/numerical_stability/run_005/plan.json)。构建脚本只生成 CAE/INP，不提交 solver；执行脚本单独运行，并拒绝在矩阵目录有活动 lock 时启动下一项。重新执行应使用新的 Run/输出路径，脚本拒绝覆盖已有模型及 INP。

## Baseline failure

A 重新求解确认用户描述的故障。最后完整完成 Step 39 / `Step-10-preheat`，在 Step 40 / `Step-10-react`、increment 1 的 reactivation geometry check 失败。该 step 尚无 equilibrium convergence check，step time = 0，total time 约 42.90001 s。

`INST_PLATE.721 / 723 / 725` very small / negative area；有 `ErrElemAreaSmallNegZero`。721 报告的最小等参线夹角为 0.27893°；725 为 1.9637°、727 为 4.5306°，出现 `WarnElemDistorted`。此前已有塑性计算困难、divergence、negative eigenvalue、excessive distortion 与 cutback。最后成功 reactivation 为 Step 36 / event 09。

本次 baseline SHA256 为 `0b612e7baac43a4f3aa076673bb057c8ecb41924d6b9786fb30beaf608a580c6`。这是当前指定文件的诊断，不把它冒充为 run_003 原始交付字节；run_004 已记录该文件的历史 hash 差异。

## ALE capability audit

结论：**SUPPORTED_WITH_REGION_RESTRICTIONS**，不是 `ALE_UNSUPPORTED_FOR_CURRENT_PROCEDURE`。

实际模型为 Abaqus/CAE 2024（MDB 2405），Standard、NLGEOM、CoupledTempDisplacementStep，1072 个 CPE4T 单元。该过程及单元类型支持 Standard ALE；但 adaptive domain 的单元不能被移除／再激活。因此，直接对 `set_addi_all` 或 721/723/725 所属沉积单元使用 ALE 不符合限制。

本轮唯一 ALE 域 `diag_ALE_pool_band` 为 `set_pool_1 ... set_pool_32` 的并集，共 256 个始终激活的 CPE4T 单元，与 `set_addi_all` 的交集为零。这是与再激活界面相邻的 pool strip，不是整个 substrate。具体 labels 与 eligibility 检查见 [capability audit](../../CAE/Reports/N32/numerical_stability/run_005/ale_capability_audit.json)。未改变 solver、procedure、element type 或 ModelChange。

ALE 在 Step-01-preheat 至 Step-final-cooling 的全部 129 个 step 启用；initial remove / equilibration 不启用。CAE API 调用 `frequency=10, meshSweeps=1`、默认 controls；CAE 导出实际为 `*Adaptive Mesh, elset=diag_ALE_pool_band, op=NEW`，省略 frequency/sweeps 参数。**实际 Standard 日志每次报告 5 mesh sweeps**，不可将 API 的 1 误报为实际 sweep 数。第一个平滑发生在 Step-01-preheat 第 10 个增量后；B 实际 103 次平滑，D 5 次。均无识别到的 ALE/advection/smoothing 专属 warning/error block；仍有普通畸变与收敛警告。

能力依据为 Dassault 文档及安装版本的真实执行：

- [Standard ALE domains](https://abaqus.uclouvain.be/English/SIMACAEANLRefMap/simaanl-c-alestd.htm)：支持范围及不得移除／添加域内单元的限制。
- [AdaptiveMeshDomain API](https://abaqus.uclouvain.be/English/SIMACAEKERRefMap/simaker-c-adaptivemeshdomainpyc.htm)：CAE API 参数。
- [CoupledTempDisplacementStep API](https://abaqus.uclouvain.be/English/SIMACAEKERRefMap/simaker-c-coupledtempdisplacementsteppyc.htm)：automatic stabilization 功能及默认级参数。

2024 文档网页访问返回 403，使用可访问的 Dassault 文档镜像并交叉核对新版文档；安装的 2024 input processor 接受输入且 Standard 实际执行平滑，构成版本直接证据。

## Case A — Baseline

Verdict: **BASELINE_FAILURE_CONFIRMED**。

- 模型：[A baseline CAE](../../CAE/Models/N32/Generated/diagnostic/numerical_stability/run_005/stage4_N32_diag_A_baseline.cae)。A 未修改物理或数值模型设置；Job 定义在 build-only 会话及独立 INP 中创建。
- Job：`stage4_diag_A_baseline_run_005`，目录 `CAE/Jobs/run_005/A_baseline/`。
- Solver FAILED；最后完整 step 39；fatal step 40 / increment 1；reactivation geometry check、平衡迭代前失败。
- 732 native total increments，732 accepted increments，59 native cutbacks，2247 native total iterations，9 negative-eigenvalue warnings，5 excessive-distortion warning blocks。Standard wallclock 342 s。
- 88 material/plasticity diagnostic lines、13 divergence messages；未达到 minimum time increment。
- 详细证据：[A_logs.json](../../CAE/Reports/N32/numerical_stability/run_005/A_logs.json)、[A ODB read-only audit](../../CAE/Reports/N32/numerical_stability/run_005/A_odb_diagnostic.json)。

## Case B — ALE only

Verdict: **ALE_GEOMETRY_IMPROVED_BUT_NOT_SOLVED**。

- 有效模型：[B ALE CAE](../../CAE/Models/N32/Generated/diagnostic/numerical_stability/run_005/stage4_N32_diag_B_ALE_02.cae)。无 stabilization。
- 有效 Job：`stage4_diag_B_ALE_02_run_005`，目录 `CAE/Jobs/run_005/B_ALE_02/`。
- 通过 Step 40 / event 10 reactivation；最后完整完成 Step 41。在 Step 42 / `Step-10-cool`、increment 131 因 **TOO MANY ATTEMPTS MADE FOR THIS INCREMENT** 失败。最后收敛 step time 0.170498982 s，total time 43.170510012 s。
- 本次没有 `ErrElemAreaSmallNegZero`；但 721/723/725/727 仍有严重角度畸变，Step 40 对应 min/max 角分别 1.6861° / 0.94577° / 2.7265° / 6.9116°。失败附近仍有 723 distorting excessively。完整 labels 及角度见日志报告。
- 914 native total increments（含失败增量），913 accepted；105 native cutbacks；2923 native total iterations；15 negative-eigenvalue warnings；17 excessive-distortion warning blocks；wallclock 448 s。134 material/plasticity lines、26 divergence messages；未达到 minimum time increment。
- 103 次实际 ALE 平滑；ALLSD=0。最后成功 reactivation 为 Step 40，未运行到 event 11 的 reactivation。不能声称已解决后续再激活或完整 32-event 模型。
- 详细证据：[B_logs.json](../../CAE/Reports/N32/numerical_stability/run_005/B_logs.json)、[B ODB audit](../../CAE/Reports/N32/numerical_stability/run_005/B_odb_diagnostic.json)。

首次 B 构建错误地仅在第一个 preheat 定义 ALE，后续 step 导出 reset。发现后终止该无效尝试，保留原始 B_ALE 目录、模型及 `B_attempt001_*` 报告。修复仅使相同域、相同默认参数在全部目标 step 保持启用；有效运行使用新文件名 `_02`。这不是参数 sweep；无效尝试不进入对照统计。增强后的 input audit 独立检查全部 ALE step lifecycle。

## Case C — Automatic Stabilization only

Verdict: **STABILIZATION_NO_MEANINGFUL_IMPROVEMENT**。

- 模型：[C stabilization CAE](../../CAE/Models/N32/Generated/diagnostic/numerical_stability/run_005/stage4_N32_diag_C_stabilization.cae)。从原始 baseline 独立复制，无 ALE。
- Job：`stage4_diag_C_STAB_run_005`，目录 `CAE/Jobs/run_005/C_STAB/`。
- 方法 `DISSIPATED_ENERGY_FRACTION`，magnitude `0.0002`，adaptiveDampingRatio `0.05`，continueDampingFactors=False。实际关键字 `stabilize=0.0002, allsdtol=0.05, continue=NO`。没有手工指定巨大 damping factor，也没有调参。
- 仅基于 A 实际 cutback / negative eigenvalue / plasticity / divergence 证据，预先选择 23 个 step；清单及逐项依据见 [stabilization_selection.json](../../CAE/Reports/N32/numerical_stability/run_005/stabilization_selection.json)。包括 01-cool；02/03/04 的 preheat、coheat、cool；05/06 的 preheat、cool；07 的 preheat、coheat、cool；08 的 preheat、cool；09 的 preheat、coheat、cool；10-preheat。所有 react 均不启用。
- 最后完整完成 Step 5；Step 6 / `Step-01-cool`、increment 2 在平衡迭代过程中因 too many attempts 失败。最后收敛 step time 0.01 s、total time 5.210002003 s。最后成功 reactivation 为 Step 4 / event 01。
- 60 native total increments、59 accepted，5 cutbacks，132 native total iterations，0 negative-eigenvalue warnings，0 excessive-distortion warning blocks；2 material/plasticity lines、4 divergence messages；wallclock 22 s。无 area error，未达到 minimum time increment，**没有到达 Step 40**。
- 自动求得的 initial dimensional damping factor 为 `7.20271E11`；这是 solver 从小耗散目标计算出的有量纲数值，不是人为设置的耗散能比例，也不能仅凭量纲数值判断能量主导。
- 最后 ALLSD=0.2087134123、ALLIE=42.81811523，比例 **0.48744185%**；与 `(abs(ALLSE)+abs(ALLPD))` 的比例相同量级。尚未出现稳定化能量主导，但求解更早失败。
- 证据：[C_logs.json](../../CAE/Reports/N32/numerical_stability/run_005/C_logs.json)、[C ODB audit](../../CAE/Reports/N32/numerical_stability/run_005/C_odb_diagnostic.json)。

## Case D — ALE + Stabilization

Verdict: **ALE_STAB_NO_MEANINGFUL_IMPROVEMENT**。

- 模型：[D ALE + stabilization CAE](../../CAE/Models/N32/Generated/diagnostic/numerical_stability/run_005/stage4_N32_diag_D_ALE_stabilization.cae)，独立源于原始 baseline。
- Job：`stage4_diag_D_ALE_STAB_run_005`，目录 `CAE/Jobs/run_005/D_ALE_STAB/`。
- 仅组合 B 的 129-step ALE 域与 C 的 23-step stabilization，未加入第三种修改。实际执行 5 次 ALE 平滑。
- 最后完整完成 Step 5；Step 6 / first cool、increment 2、too many attempts，在平衡迭代期间失败。最后收敛 step time 0.01 s、total time 5.210002003 s；最后成功 reactivation 为 Step 4。
- 60 native total increments、59 accepted，5 cutbacks，134 native total iterations，0 negative-eigenvalue warnings，1 excessive-distortion warning block（包括 INST_PLATE.1016）；2 material/plasticity lines、4 divergence messages；wallclock 23 s。无 area error，未达到 minimum time increment，未到达 Step 40。
- 自动 initial damping factor `7.20509E11`。最后 ALLSD=0.04645754769、ALLIE=42.10679626，比例 **0.11033266%**。已收敛历史未显示稳定化能量主导；不能把低能量占比解释成求解成功。
- 证据：[D_logs.json](../../CAE/Reports/N32/numerical_stability/run_005/D_logs.json)、[D ODB audit](../../CAE/Reports/N32/numerical_stability/run_005/D_odb_diagnostic.json)。

## Comparison table

| CASE | Configuration | Solver | Last completed / fatal step | Area error | Distortion warnings | Cutbacks | Neg. eigen | Stabilization energy | Verdict |
|---|---|---|---|---|---:|---:|---:|---|---|
| A | baseline | FAILED | 39 / 40 | 721/723/725 | 5 | 59 | 9 | N/A | BASELINE_FAILURE_CONFIRMED |
| B | ALE | FAILED | 41 / 42 | 未出现 | 17 | 105 | 15 | N/A | ALE_GEOMETRY_IMPROVED_BUT_NOT_SOLVED |
| C | stabilization | FAILED | 5 / 6 | 未出现；未到 Step 40 | 0 | 5 | 0 | ALLSD/ALLIE 0.4874% | STABILIZATION_NO_MEANINGFUL_IMPROVEMENT |
| D | ALE + stabilization | FAILED | 5 / 6 | 未出现；未到 Step 40 | 1 | 5 | 0 | ALLSD/ALLIE 0.1103% | ALE_STAB_NO_MEANINGFUL_IMPROVEMENT |

Distortion 为 `.msg` 的 excessive-distortion warning block 次数，不是单元数；完整 distorted labels、等参角、Jacobian/area 证据及 reconstructed geometry 在各 JSON 中。Cutbacks 采用 solver 最终摘要的定义；`.sta` 最终 abort 行可能使 unsuccessful-attempt 数比实际 cutback 多一。Native iteration summary、原始 `.sta` equilibrium 列求和、打印的 equilibrium convergence-check 次数分别保留，不能混作同一统计口径。

相同已完成区间 Step 1–39：A/B 的 retry 次数 59/54，negative-eigenvalue warnings 9/7，equilibrium convergence-check 次数 2226/2141。B 有小幅前段改善及跨过 Step 40 的证据，但全程仍未完成。C/D 在 Step 6 就失败，累计 cutback 较少完全不能证明改善。四项在 Step 1–5 的共同区间均无 retry；该区间还没有 stabilization，因此也不能据此评价其作用。

## What ALE changed

仅新增相邻、始终激活 pool strip 的 adaptive domain。实际改变了平滑后的网格／场演化，足以让第 10 次 reactivation 通过；随后第 10 次冷却仍因畸变和收敛困难失败。没有直接对被 ModelChange 移除／添加的沉积单元做 ALE，结论不外推到任意 ALE 区域或控制设置。

## What stabilization changed

仅改变预定 23 个 coupled steps 的自动稳定化控制。C/D 都在首次启用后的第二个增量失败，未体现 convergence 改善；更未建立对 Step 40 negative-area fatal error 的直接证据。没有因失败扩大 step 范围、增加 damping 或进行 sweep。

## Direct evidence

所有工程实体均位于 CAE 目录；Run artifacts 仅引用，不复制 CAE。入口为 [artifacts manifest](artifacts/artifacts.json)、[matrix_results.json](../../CAE/Reports/N32/numerical_stability/run_005/matrix_results.json)。各有效 Job 目录保存 INP、MSG、DAT、STA、ODB；B/C/D 保存 launcher `.log`。A 的直接 interactive 启动没有生成原生 `.log`，其工具输出转录另存 [A_launcher_console.txt](../../CAE/Reports/N32/numerical_stability/run_005/A_launcher_console.txt)，明确标注来源，不冒充原生日志。

四份 `*_input_diff_audit.json` 全部通过：除 job 注释、允许新增的 ALE elset/keyword 及 stabilization 参数外，导出的输入逐行一致。另有 build 前后 physical snapshot 一致检查。保持所有既有 mesh、geometry、材料、热源、热膨胀、塑性、BC、交互、ModelChange 顺序、scan order、step duration、输出请求和 bottom-layer membership；没有增加能量输出请求，因为 baseline 已有 ALLSD/ALLIE/ALLSE/ALLPD 等历史。

构建／执行／只读诊断脚本位于 `CAE/scripts/debug/`。安装版本实际运行证明不是仅检查对象存在。ODB 均使用 `readOnly=True`；未提取 teacher metrics、未进入 dataset。

[Safety after](../../CAE/Reports/N32/numerical_stability/run_005/safety_after.json) 确认：baseline、sanity master、YAML、RL_STATUS、ODB_STATUS 的 SHA256 与本轮开始一致。当前 sanity master hash 为 `407a36afc3f0687bf611869e9f3d54a631e5072076adff720036377f5af23fed`。正式 generator/auditor hash 也保持 run_004 版本。Stage 3 共 17,209 个文件的路径／大小／mtime_ns 清单无变化；这是 metadata inventory 检查，未声称全树内容 hash。Stage 3 无写操作。

## Numerical credibility

C/D 的 ODB 实际提供 ALLSD、ALLIE、ALLSE、ALLPD、ALLAE、ALLKE、ALLWK、ETOTAL。采用 `abs(ALLSD)/abs(ALLIE)`，并交叉检查 `abs(ALLSD)/(abs(ALLSE)+abs(ALLPD))`。计算最大比例时剔除 `abs(ALLIE) <= 1% peak` 的近零能量起始点；曲线和原始值全部保留。5% 仅是诊断筛查阈值，不是完整物理可信性的认证标准。

C/D 筛查后最大比例分别 0.4874% / 0.1103%，均为最后收敛点；未触发 STABILIZATION_DOMINATED。但这只覆盖第一段稳定化的一个收敛增量，失败迭代的能量未被这些历史值证明。两项既不 completed，也没有完整过程的数值可信性证据。

A/B 无 stabilization，ALLSD=0。B 的极小等参角仍显示几何退化。ODB 几何重建采用 initial coordinates + saved U 的 Q4 角点／中心 Jacobian，仅作为旁证；不把它等同于 solver 内部 reactivation geometry check。

Run execution = completed；validation = passed **仅指矩阵执行、差分隔离、证据及保护检查**；所有 solver/model completion 均 failed。handoff.ready=true **仅交付诊断报告供审阅**，formal_model_ready=false，teacher_candidate_ready=false。这不是正式模型修复 PASS。

## Final diagnostic verdict

**NO_COMPLETE_NUMERICAL_RESCUE_IN_TESTED_MATRIX**。

1. **ALE 是否支持？** 支持当前 Standard coupled/CPE4T 组合，但域内单元不得参与移除／再激活；本次使用合法相邻 pool 域。
2. **ALE 是否解决／推迟／未改善？** 通过原来的 Step 40 geometry failure，最终失败从 Step 40 推迟到 Step 42，且失败类型变为冷却阶段重试耗尽；属于改善但未解决。没有观察到后续 reactivation 的完整通过。
3. **Stabilization 是否改善 convergence？** 本次设置没有；C 在 Step 6 提前失败，D 同样如此。
4. **对 Step 40 negative-area fatal 是否有实际作用？** C/D 未到 Step 40，无法直接判定；不能把“没观察到 area error”写成已解决，也不能声称已经在该 geometry check 上测试无效。
5. **组合是否真正解决？** 没有，D FAILED。
6. **是否足以判定整个 pre-meshed + ModelChange + NLGEOM architecture 不值得继续数值修补？** 不足以作普遍否定。本轮足以说明这一固定 2×2 实现没有完整挽救当前模型，且 stabilization 组合的单次实现表现更差；同时 B 的局部改善反驳“这些机制完全没有作用”。受限 ALE 区域、单一设置、C/D 提前终止限制了结论范围。未宣称正式 architecture 已修复，也未提出或启动新的研究路线。

四项测试与诊断已停止。没有新物理参数研究、RL、surrogate、真实 CASE 或额外 solver 任务。
