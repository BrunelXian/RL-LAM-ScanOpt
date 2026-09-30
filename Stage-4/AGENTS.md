# RL-LAM-ScanOpt Stage 4 — Agent rules

## Archive boundary (2026-09-30)
Stage 4 tested CAE route is ARCHIVED. Read `status/STAGE4_ARCHIVED.md` and the final README first.
Prior implementation priorities below are historical; do not resume tuning, solver jobs or training without an explicit new task.
Preserve the positive, bounded Agent diagnostic-workflow result and its claim limits. This archive does not start Stage 5.

## Boundary and priority
- 根目录：`E:\Projects\Stage-4`；Stage 4 是独立工程，不是 Stage 3 continuation folder。
- `E:\Projects\RL-LAM-ScanOpt` 是 READ ONLY reference。不得修改、移动或删除 Stage 1–3 文件，不复制历史 outputs，不覆盖 frozen evidence，不把历史结果当作 Stage 4 新证据。
- 当前优先级：Abaqus sanity → progressive material activation → deposition-event generator。未获明确指令，不运行 large-scale FEA、RL/surrogate retraining 或 descriptor batch analysis。

## Mandatory reading and communication
- 任何 Stage 4 Agent 工作前必须完整读取 `docs/Agent_File_Communication_Protocol.md`，这是通信协议权威依据。
- 再读 `docs/Stage4_Orchestration_Architecture.md`、`status/MASTER_STATUS.md`、对应模块 STATUS 及显式上游 Run。
- 固定 Agent 名称：`stage4_master`、`stage4_rl`、`stage4_cae`、`stage4_odb`；任务区别写入 `operation`，不另起 Agent 别名。
- `/docs` 保存长期知识，`/status` 保存当前状态，`/run` 保存执行历史；正式实体留在 CAE/RL/surrogate/descriptors/outputs 的所属位置。
- 每次独立执行创建全项目唯一递增 `run_XXX`；人工串行分配，不覆盖，不按模块单独编号。
- 每 Run 必须有 `run.json`、`artifacts/artifacts.json`、`checkpoints/`、`summary.md`；完成后更新 `run/README.md` 与相关状态。
- Run 只保存记录、checkpoint metadata 和 artifact manifest，不复制正式工程文件。
- Manifest 的 `path` 相对于所属 Run 目录（明确 `path_base: run_directory`）；`artifacts_manifest` 相对于 run.json 所在目录。逐条验证路径，不凭编号猜依赖。
- 只有 completed + validation passed + handoff ready 才能正式交接。失败或 warning 必须如实记录，不伪造 PASS。
- RUN 是执行身份，CASE 是物理/FEA candidate 身份（预留 `S4C000001` 起），JOB 是 Abaqus 求解执行。无正式 case 则 `case_ids: []`。

- 跨模块数据操作前必须读取 `docs/Stage4_Interface_Contract.md` 及 `configs/schemas/` 对应 schema；示例不得进入生产数据，schema 通过不等于 teacher 已验证。

## Modelling and safety
- 建模前必读 `docs/Stage4_Drop_Deposition_Event_Template.md`；未经明确指令，不改变 preheat → react → coheat → cool、ModelChange、Film/Radiation、Surf_in/Surf_addi 和热源生命周期。
- 参数依据 `docs/Stage4_Parameter_Register.md`、`configs/stage4_sanity.yaml`；冲突先报告，区分事件 k 与空间编号 i。
- master：`CAE/Models/N32/Sanity/N32_sanity.cae`。不直接覆盖或打开 master 执行修改；获准操作使用副本，优先 build-only/audit，不删除历史模型。
- 未获明确要求不提交大型 Abaqus job、不批量生成、不执行训练。编排交接就绪不授权执行工程任务。
- 所有新输出使用 Stage 4 独立路径；冻结证据附来源和校验清单，不覆盖。大型 CAE/ODB/checkpoint/缓存不默认纳入版本控制。
- 保留现有工程路径；legacy `E:\Projects\Step-4` references exist，仅在真正使用相应脚本时逐项迁移，禁止全局替换。

## Environments
- 轻量 Python：`D:\XianLab\envs\Chuzi-PY`。
- PyTorch/CUDA：`D:\XianLab\envs\conda\torch-gpu`（Torch-GPU）。
- Abaqus API 使用其内置 Python；环境选择不是运行授权。
