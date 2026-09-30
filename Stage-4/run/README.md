# Run Index

全 Stage 4 统一递增编号；Agent 使用固定名称，通过 operation 区分任务。
编号只用于历史排序，依赖由 input_runs 显式声明。Run 不存放工程实体副本。
详见 `../docs/Agent_File_Communication_Protocol.md` 与 `../docs/Stage4_Orchestration_Architecture.md`。

## run_001

**Agent:** stage4_master  
**Operation:** migrate_project_orchestration_architecture  
**Status:** completed  
**Validation:** passed  
**Handoff:** ready  
**Input runs:** []  
**Case IDs:** []  
**Result:** 建立共享 Run/Artifact/Handoff 与 status 架构；文件完整性与协议一致性检查通过，未运行工程任务。  
**Main output:** [MASTER_STATUS](../status/MASTER_STATUS.md)  
**Record:** [run.json](run_001/run.json) · [summary](run_001/summary.md) · [artifacts](run_001/artifacts/artifacts.json)

## run_002

**Agent:** stage4_master  
**Operation:** define_cross_agent_interface_contract  
**Status:** completed  
**Validation:** passed  
**Handoff:** ready — stage4_rl / stage4_cae / stage4_odb；仅接口规范  
**Input runs:** [run_001]  
**Case IDs:** []  
**Result:** 冻结身份/三类载荷接口；三个 schema 和示例验证通过，无真实 CASE/工程执行。  
**Main output:** [Interface Contract](../docs/Stage4_Interface_Contract.md)  
**Record:** [run.json](run_002/run.json) · [summary](run_002/summary.md) · [artifacts](run_002/artifacts/artifacts.json)

## run_003

**Agent:** stage4_cae  
**Operation:** build_n32_deposition_event_generator  
**Status:** completed  
**Validation:** passed  
**Handoff:** ready — N32 identity diagnostic build only; no solver authorization  
**Input runs:** [run_001, run_002]  
**Case IDs:** []  
**Result:** Actual Abaqus build; 32 complete deposition events; reopened-CAE mapping/lifecycle/final-state audit passed. Master unchanged.  
**Main output:** [diagnostic CAE](../CAE/Models/N32/Generated/diagnostic/stage4_N32_diag_identity_order.cae)  
**Record:** [run.json](run_003/run.json) · [summary](run_003/summary.md) · [artifacts](run_003/artifacts/artifacts.json)

## run_004

**Agent:** stage4_cae  
**Operation:** build_n32_custom_order_diagnostic_model  
**Status:** completed  
**Validation:** passed  
**Handoff:** ready — specified custom-order diagnostic CAE only  
**Input runs:** [run_003]  
**Case IDs:** []  
**Result:** Specified 32-drop custom order built; separate reopen audit passed 69,765 checks. Master/parameters unchanged; no solver.  
**Main output:** [custom diagnostic CAE](../CAE/Models/N32/Generated/diagnostic/stage4_N32_diag_custom_order_001.cae)  
**Record:** [run.json](run_004/run.json) · [summary](run_004/summary.md) · [artifacts](run_004/artifacts/artifacts.json)

## run_005

**Agent:** stage4_cae  
**Operation:** diagnose_n32_numerical_stability_ablation  
**Status:** completed  
**Validation:** passed — diagnostic execution/isolation/evidence only; all solver cases failed  
**Handoff:** ready for diagnostic report review; formal model/teacher ready = false  
**Input runs:** [run_003, run_004] (history; current identity CAE is authoritative input)  
**Case IDs:** []  
**Result:** A failed Step40; B ALE passed Step40 then failed Step42; C/D failed Step6. No complete numerical rescue.  
**Record:** [run.json](run_005/run.json) · [summary](run_005/summary.md) · [artifacts](run_005/artifacts/artifacts.json)

## run_006

**Agent:** stage4_master  
**Operation:** archive_stage4_cae_route_and_document_agent_diagnostic_evidence  
**Status:** completed  
**Validation:** passed (documentation/evidence/preservation)  
**Publication:** archive commit pushed; branch stage4-cae-failure-agent-diagnostics  
**Input runs:** run_003, run_004, run_005  
**Case IDs:** []  
**Result:** CAE route archived; bounded Agent diagnostic capability documented with evidence and limitations.  
**Main output:** [README](../README.md), [Failure Analysis](../docs/Stage4_Failure_Analysis.md), [Stage 5 handoff](../docs/Stage4_to_Stage5_Handoff.md)  
**Record:** [summary](run_006/summary.md) · [run.json](run_006/run.json)
