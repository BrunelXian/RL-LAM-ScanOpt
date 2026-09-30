# Stage 4 orchestration / file-engine architecture

Authority: `docs/Agent_File_Communication_Protocol.md`（原文件保留）。
本文补充本项目 Agent 名称、status、路径解析与 CASE/JOB 约定，不改变正式协议。

## Roles and durable locations
| Agent | Responsibilities | Current-state file |
|---|---|---|
| stage4_master | 项目编排、Run 索引、跨模块交接 | status/MASTER_STATUS.md |
| stage4_rl | surrogate、PPO/RL、candidate generation、teacher-data consumption | status/RL_STATUS.md |
| stage4_cae | deposition-event generator、模型 build/audit、获准求解工作 | status/CAE_STATUS.md |
| stage4_odb | solver completion audit、ODB extraction、teacher metrics、thermal history、bottom-layer response | status/ODB_STATUS.md |

正式实体留在 CAE/RL/surrogate/descriptors/outputs；Run 不复制实体。
运行记录不替代当前状态；当前状态不能替代历史证据。

## Run lifecycle and allocation
人工串行调度下检查 `run/run_数字`，取最大已用编号 + 1，以独占创建方式分配。
至少三位补零（run_001）；不复用取消/失败 Run，不按模块编号。
开始写 status=running、validation.status=not_run、handoff.ready=false。
中途 Save 不创建新 Run；恢复点写 checkpoints metadata，同一任务恢复沿用该 Run。
结束分别登记 execution、validation、handoff；失败记录 error，warning 不冒充 passed。
完成时更新 run/README.md 的简短记录和相关 STATUS。status 文件允许持续更新，历史 Run 不当作活动便笺改写。
机器可读 JSON 更新应先写同目录临时文件再原子替换，避免下游读到半写内容。

## Path resolution
正式协议例子的 `../../CAE/...` 采用 **所属 Run 目录** 为基准，而非 artifacts 子目录。
manifest 明确写 `path_base: "run_directory"`；所有 artifact.path 以 run/run_XXX 解析。
例如 run/run_001 中 `../../docs/Agent_File_Communication_Protocol.md` 指向根目录 docs。
run.json 的 `artifacts_manifest: "./artifacts/artifacts.json"` 也相对于 Run 目录。
解析后确认真实文件存在，并且项目内引用不越过 Stage 4 根目录。
artifact 使用稳定 artifact_id、type、role、path，可补充 change 和 sha256_at_handoff。
状态文档后续可变，记录的 hash 是交接时快照，不代表文件永远不可更新。

## Handoff consumption
读取 AGENTS、正式协议、MASTER 与模块 STATUS；显式选择 input_runs。
对每个上游读取 run.json、manifest 和 summary，检查 completed + passed + ready。
依照 path_base 解析并验证 artifact 实体；不得从 Run 顺序推定科学依赖。
收到 handoff 只表示产物在声明范围内可用，不自动授权训练、批量任务或 solver。
本轮 handoff 范围只有 orchestration readiness，不包含物理正确性或 CAE 内部审计。

## RUN / CASE / JOB
RUN = Agent execution identity。
CASE = 一个独立的物理 / FEA candidate identity；预留 S4C000001、S4C000002……。
JOB = Abaqus solver execution identity；同一 CASE 可以对应不同 JOB 或处理 Run。
一个 Run 可涉及零个或多个 CASE；case_ids 必须显式记录，不能用 Run ID 代替。
本轮没有正式 CASE，case_ids=[]；不创建空壳 CASE registry 或虚假数据。
以后首次正式分配 CASE 时建立可审计注册记录，不能把 Stage 3 结果重新编号成新证据。

## Known boundaries
legacy path references exist：历史 temp 中旧 Step-4 路径本轮保持。
实验 IR 校准尚未完成；交接文档仍有 TODO；这些是已知项目待办，不是本轮架构验证失败。
CAE 内容（含 set_bottom_layer）本轮不打开核验；状态文件必须注明用户确认来源。
