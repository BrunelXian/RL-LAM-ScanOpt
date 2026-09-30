# run_001 — Stage 4 architecture migration

## Objective
按正式 Agent 文件通信协议建立 docs/status/run/工程实体分工，以及 Run/Artifact/Handoff 机制。

## Input
- docs/Agent_File_Communication_Protocol.md（已完整读取，原样保留）
- 根目录 AGENTS/README、既有模板、参数登记与配置。
- 修改前审计：../../outputs/diagnostics/stage4_run_001_layout_before.json。
- input_runs=[]；case_ids=[]。此前协议空文件导致的预审计保留，但不伪造为已完成 Run。

## Actions and outputs
新增 run/status 结构、四个 STATUS、长期架构说明；更新根 AGENTS 与 README。
全项目统一 run_001；固定 stage4_master/stage4_rl/stage4_cae/stage4_odb。
manifest 以 Run 目录为 path base，引用真实项目文件，不复制工程实体。
只预留 CASE identity；不分配虚假 CASE，不启动其他 Agent/模块任务。

## Validation
PASS：JSON、Run 固定结构、所有 manifest 路径、四份 STATUS 与索引引用通过。
Stage 3 全部 17,209 个文件 size/mtime 与文件清单保持，5,787 个文档/脚本/配置 SHA256 保持；未声称对所有大型工程文件做全量 hash。
Stage 4 原有文件除授权修改的 AGENTS.md/README.md 外 hash/mtime 保持；master CAE SHA256 保持。
报告：../../outputs/diagnostics/stage4_run_001_validation.json。Warnings: none（就本轮架构范围）。

## Boundaries and next action
未运行 Abaqus/solver/INP generation/ODB extraction/generator/RL/surrogate。
未移动工程文件、未替换历史路径、未改模型或参数。
模型 established 和 set_bottom_layer 来自用户确认；本轮只验证 master 文件存在及字节不变。
既有 handoff TODO、legacy path references 和未完成 IR 校准是研究待办；不作为新研究结果。
下一步建议：分别定义 RL/CAE/ODB 的接口与独立任务，经用户明确安排后登记新的 Run。

## Handoff
ready=true；目标 stage4_master、stage4_rl、stage4_cae、stage4_odb。范围仅为 orchestration architecture，不自动授权工程运行。
Verdict: PASS_STAGE4_AGENT_FILE_ENGINE_ARCHITECTURE_READY
