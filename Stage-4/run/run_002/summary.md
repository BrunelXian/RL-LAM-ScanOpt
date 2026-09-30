# run_002 — Cross-agent interface contract

## Objective
定义 Stage4 RL → CAE → ODB → RL 的身份、schema、interface 与 handoff contract；不执行科学计算。

## Inputs
input_runs=[run_001]，case_ids=[]。
已完整读取 AGENTS、正式通信协议、架构文档、四份 STATUS、run_001 run.json/summary、
Drop Deposition Event Template、Parameter Register 与 stage4_sanity.yaml，并检查上游 manifest。
预审计：../../outputs/diagnostics/stage4_run_002_layout_before.json。

## Files created
- docs/Stage4_Interface_Contract.md
- configs/schemas/case.schema.json
- configs/schemas/cae_job_manifest.schema.json
- configs/schemas/teacher_metrics.schema.json
- docs/examples/case_example.json
- docs/examples/cae_job_manifest_example.json
- docs/examples/teacher_metrics_example.json
- outputs/diagnostics/stage4_run_002_layout_before.json
- outputs/diagnostics/stage4_run_002_validation.json

另建 run/run_002 的 run.json、summary.md、artifacts/artifacts.json 与 checkpoints metadata。

## Files modified
- AGENTS.md
- README.md
- status/MASTER_STATUS.md
- status/RL_STATUS.md
- status/CAE_STATUS.md
- status/ODB_STATUS.md
- run/README.md

## Schema decisions
Draft 2020-12；schema_version=1.0；metric_schema_version=1.0 为封装版本，不宣称科学 reduction 已最终确定。
核心层严格，metadata/extensions/metric 扩展层开放。字段 null 表示未知，无伪造数值默认。
核心指标是带 unit/category/definition/aggregation/region/time_selection 的 physical envelope；派生 normalized/ranking 需要变换来源。
build、solver、extraction、teacher validation 各自独立；teacher_ready 有显式 gate。

## Identity rules
RUN=execution；CASE=scientific identity；JOB=Abaqus execution。
真实 CASE 固定 S4C+6 digits；S4C000000 无效，S4C999999 保留给 example_only。
示例 process_config_id=stage4_example_only_v00，假想生产 Run run_999991/999992/999993，未创建这些实体。
process_config_id 未来绑定不可变工况快照；本轮未注册 stage4_sanity_v01，原 YAML 不变。
载荷路径相对项目根；Run artifact.path 相对 Run 目录。

## Validation
三个 schema 解析并通过 metaschema 检查；三个 example 在 jsonschema 4.25.1 + FormatChecker 下通过。
示例 case_id/N/process_config_id/scan_order/JOB/ODB 路径与 manifest 引用一致；预留测量值均 null。
10 项 schema 反例检查通过；扩展字段正例通过。
未运行 len=N/编号<=N 的 semantic legality audit；其要求已写入契约，未声称通用 schema 能跨字段验证。
Stage 3 文件清单、size/mtime 与所记录文本类 SHA256 均保持；Stage 4 仅修改授权的七个项目文档。
N32_sanity.cae SHA256 保持，run_001 历史文件保持，未创建真实 CASE。
最终报告：../../outputs/diagnostics/stage4_run_002_validation.json。

## Warnings
无本轮未解决 warning。jsonschema 原先未安装，已安装到用户指定 Chuzi-PY（4.25.1，含 jsonschema-specifications 依赖）。
未最终冻结 metric reduction/实验校准、未做 candidate legality audit 是本轮明确边界，不是已完成工作。
旧 Step-4 路径保持不动；既有研究 TODO 保留。

## Frozen decisions
身份/来源规则、1-based drop 索引、三种载荷结构、版本与扩展策略、路径基准、示例隔离、状态分离和 teacher 放行条件。
本轮只冻结接口；未实现 generator、extractor、workflow engine 或训练。

## Handoff
ready=true；stage4_rl、stage4_cae、stage4_odb。消费入口为 AGENTS + 对应 STATUS + Interface Contract + schemas。
Run artifacts 只引用正式文档和配置，不复制工程实体。后续执行需要独立任务授权。

## Next recommended action
分别建立 CAE/ODB/RL 工作线并读取本轮接口；优先定义 generator 的输入核验与 ODB metric definition，
另开正式 Run 处理。本轮没有创建新对话或启动模块工作。

PASS_STAGE4_CROSS_AGENT_INTERFACE_CONTRACT_READY
