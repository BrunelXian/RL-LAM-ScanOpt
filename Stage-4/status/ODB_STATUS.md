# ODB STATUS

Current stage: not started under Stage 4 Run protocol。
Latest module run: none。Architecture baseline: run_001。

Responsibilities:
- solver completion audit
- ODB extraction
- teacher metrics
- thermal history
- bottom-layer response

Reference master: `CAE/Models/N32/Sanity/N32_sanity.cae`。
set_bottom_layer 已存在于 N32 sanity master（用户确认；本轮未打开 CAE 或独立核验）。
Current next action: define detailed metric extraction/validation design in a separately authorized Run; no extraction started。
本轮未读取 ODB、未提取数据、未生成 teacher metrics。

## Cross-agent interface
Cross-agent interface contract established in run_002; schema/examples validated; handoff ready (interface scope only).
Reference: `docs/Stage4_Interface_Contract.md` and `configs/schemas/`.
未来输入：CAE job manifest + ODB；输出：teacher metrics。extraction 与 teacher validation 分开；本轮不读取 ODB。
真实 CASE 尚未分配；case_ids=[]。示例不是研究结果。
