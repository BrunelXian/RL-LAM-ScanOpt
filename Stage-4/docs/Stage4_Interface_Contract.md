# Stage 4 Cross-Agent Interface Contract

Contract version: 1.0；定义 Run：run_002；所有示例均非真实 CASE。
Authority: `docs/Agent_File_Communication_Protocol.md`；路径与 Run 规则继承
`docs/Stage4_Orchestration_Architecture.md`。本契约冻结身份、载荷和交接边界，不冻结最终科研指标算法。

## 1. 数据流和身份
stage4_rl → Stage4Case → stage4_cae → CAE Job Manifest → stage4_odb → Teacher Metrics → stage4_rl。

- RUN = Agent execution identity；格式 `run_` + 至少三位数字，零编号无效。
- CASE = scientific / FEA candidate identity；格式 `S4C` + 六位十进制数字。
- JOB = Abaqus solver execution identity；一次实际 solver 执行，与 CASE/Run 分开。

S4 表示 Stage 4，C 表示 Case。实际分配从 S4C000001 起，S4C000000 无效；
S4C999999 专门保留给 example_only=true 的文档示例，不进入真实 CASE 分配池。
CASE 创建后不重命名，不因文件迁移或 JOB 后缀改变科学身份。
CASE 贯穿 candidate、CAE、INP、JOB、ODB、teacher metrics、surrogate dataset。
科学输入（N、scan_order、process_config_id）改变应建立新 CASE；相同科学输入的 solver 重跑保留 CASE，使用不同 JOB 标识避免覆盖。

一个 Run 可处理 0、1 或多个 Case；一个 Case 可由多个 Run 构建、求解、提取或验证。
例如未来一个 RL Run 生成 60 个 CASE，下一个 CAE Run 处理同一批 CASE，这不是本轮已完成工作。
Run 依赖用 input_runs；科学身份用 case_ids。source_run_id/cae_run_id/odb_run_id 是各生产阶段的执行来源，不能替代 case_id。
本轮 run_002.case_ids=[]；没有建立 CASE registry，没有占用 S4C000001。

## 2. 机器载体、版本与路径
JSON/JSONL = identity + structured metadata，作为机器主接口。
单 Case 用 JSON，批量可用 JSONL（每行独立符合对应 schema；不同记录不可随意混合类型）。
CSV = batch analysis / tabular convenience，例如 candidate_table.csv、teacher_metrics.csv，
是派生表，不是唯一真相源；保留 case_id/process_config_id/来源 Run 与 schema version。
本轮不实现 CSV pipeline 或 workflow engine。

三个 schema 使用 JSON Schema Draft 2020-12，schema_version='1.0'，teacher 的 metric_schema_version='1.0'。
后者版本化指标封装结构，不宣称所有 metric definition 已经最终确认。
根级 additionalProperties=false 防止拼写错误；extensions、candidate_metadata、指标扩展区允许扩展。
不在既有字段中悄悄更换单位、归一化或统计定义；有破坏性变更发布新版本并安排显式迁移。
消费者需核对支持的 schema_version；日期要求有时区的 RFC3339 date-time，并启用 validator FormatChecker。
JSON 不使用 NaN/Infinity；未知数据用 null，不能用 0 代替缺失。

载荷中的所有 *_path 使用相对于 Stage 4 项目根目录的正斜线路径，不允许绝对路径或 .. 路径穿越。
这与 Run artifact manifest 的 run_directory 基准不同，读取时不得混用。
planned path 只声明计划位置，不证明文件存在；交接执行前另行核验正式路径与输入来源。
正式工程实体保留 CAE/RL/surrogate 等所属目录；Run 只保存 manifest 引用。

## 3. RL → CAE：Stage4Case
Schema: `configs/schemas/case.schema.json`。

| Field | Meaning |
|---|---|
| schema_version | 载荷结构版本 1.0 |
| case_id | 不变的科学 candidate identity |
| N | 正整数，不限于 N32；12/16/24/32/40 是现阶段规模例子，不是永久枚举 |
| scan_order | 1-based 空间 drop 编号排列 |
| source | baseline/heuristic/surrogate_search/ppo/manual/diagnostic；扩展来源使用 x_ 前缀并在 metadata 解释 |
| source_run_id | 实际创建 candidate 的 Run；真实数据必须可追溯到存在的 Run |
| process_config_id | 不可变工况配置身份 |
| parent_case_ids | 来源 CASE 列表，无 parent 时 []；不得自引用，不得把示例当真实 parent |
| candidate_metadata | surrogate score/PPO reward/novelty/strategy family/notes 等；不是 FEA teacher labels |
| created_at | 创建时刻 |
| example_only | 必填；真实记录 false，文档示例 true，生产消费者必须拒绝 true |

scan_order 必须满足长度=N、无重复、整数范围1..N；事件位置 k 与空间编号 i 严格区分。
标准 JSON Schema 能直接验证整数、正数和 uniqueItems，但不能对任意 N 直接比较数组长度或元素上界与同级 N。
因此 len(scan_order)==N、max(scan_order)<=N 及物理 mapping/geometry legality 属于后续 semantic legality audit。
本轮只做示例 schema validation，不实现或运行 candidate legality audit。

## 4. process_config_id 与 provenance
三个载荷都必须保留 process_config_id，不可从 CASE ID 推断工况。
命名形态：stage4_<name>_vNN，例如未来可用 stage4_sanity_v01、stage4_expcal_v01。
本轮不注册这两个正式版本，不改名或修改 configs/stage4_sanity.yaml。
仅示例使用 stage4_example_only_v00，它不对应正式配置。
未来正式注册时应绑定不可变配置快照、内容 checksum 和适用模型/几何版本，
覆盖 MAG/功率/时序/材料激活/film-radiation/实验校准/几何等；不得同 ID 悄悄改变参数。
source 与 source_run_id 保留生成方法和执行来源；parent_case_ids 支持 mutation、repair、PPO/surrogate variants。
下游保留原 CASE 的科学字段，不在 CAE/ODB 阶段重新解释 scan_order 或重编号。

## 5. CAE → ODB：CAE Job Manifest
Schema: `configs/schemas/cae_job_manifest.schema.json`。
必填 schema_version、case_id、N、scan_order、process_config_id、cae_run_id、
candidate_manifest_path、cae_model_path、inp_path、abaqus_job_name、odb_path、
build_status、solver_status、model_template、created_at、example_only。

- candidate_manifest_path 指向生产输入，便于与 Stage4Case 比对。
- model_template 记录版本化模板标识或模板路径；真正使用时必须能解析到已记录的模板版本。
- cae_model_path/inp_path/odb_path 在未计划时可为 null，也可提前填写计划路径。
- build_status: not_built / built / build_warning / build_failed。
- solver_status: not_run / submitted / running / completed / failed / cancelled。
- built 表示正式 CAE 与 INP 已构建并有非空路径，不表示 solver completed。
- 本 v1 只有 built/build_warning 才可进入非 not_run 的 solver 状态，odb_path 此时必填；submitted/running 的 ODB 路径仍可能尚无文件。
- completed 必须由未来 solver completion audit 的实际证据支持，不能仅靠 ODB 文件存在认定。
- build_warning 必须说明问题，不能绕过后续 teacher 验证；构建失败不交付为可运行模型。

默认 abaqus_job_name=case_id，如 S4C000127，对应同名 .inp/.odb/.msg/.dat/.sta。
重跑等可使用 S4C000127_v01 等后缀；所有派生产物始终携带 case_id=S4C000127。
JOB schema 只检查命名形态，前缀与 case_id 相等、文件归属一致由跨记录语义核验负责。

## 6. ODB → RL：Teacher Metrics
Schema: `configs/schemas/teacher_metrics.schema.json`。
必填 schema_version、metric_schema_version、case_id、N、process_config_id、odb_run_id、
cae_job_manifest_path、abaqus_job_name、odb_path、solver_status、extraction_status、
u2、peeq、surface_t、mises、thermal_metrics、bottom_layer_metrics、warnings、
teacher_validation、teacher_ready、created_at、example_only。

extraction_status = not_run / running / completed / partial / failed。
teacher_validation.status = not_run / passed / warning / failed；passed 需要 report_path。
teacher_ready=true 必须满足：真实记录、solver completed、extraction completed、
teacher_validation passed、ODB 路径存在于记录、四个核心指标均为非空 physical metric。
Schema 能检查声明状态及结构，不能证明 solver 真正完成、路径文件存在或指标科学有效。
未来 teacher validation 必须核验来源、完整性、单位/region/time/reduction、配置一致性及异常，生成真实报告。
partial/failed solver 的提取可以作为 diagnostic 数据保存，但不能成为 teacher_ready 训练样本。
warning 不自动放行；问题解决并有新的验证记录前 teacher_ready=false。
RL 默认仅消费 example_only=false 且 teacher_ready=true 的已验证数据。

核心 U2/PEEQ/SurfaceT/Mises 的 JSON 键固定为 u2/peeq/surface_t/mises。
四个核心键只承载 raw_physical envelope 或 null；绝不把 surrogate prediction/PPO reward 填入这些键。

## 7. 指标封装与可扩展性
每个已提取的非空标量用对象表示，必填：
value（数值）、unit、category、definition_id、aggregation、region、time_selection。
category 区分 raw_physical、normalized、ranking、diagnostic；四类不能混用。
unit 使用实际记录单位（例如无量纲用 '1'），不可从文件名猜测；单位制和 metric definition 在后续提取设计中明确。
U2 的有符号/绝对值、PEEQ 的空间统计、SurfaceT 的表面选择、Mises 的帧/区域等尚未冻结为唯一算法，
必须由 definition_id + aggregation + region + time_selection 明确，不靠模糊指标名作比较。
normalized/ranking metric 还必须给出 transform_id、reference_id、basis_metric_ids，
记录变换和归一化参考，放 derived_metrics 或明确的扩展指标，不覆盖 raw 核心字段。
扩展指标也使用 envelope，避免裸数值失去单位和含义。

thermal_metrics 已预留 tmax、preheat_interface_temperature、coheat_peak_temperature、cooling_rate，均可 null。
未来可增加高温区/熔池面积等标量；thermal history 的文件索引可通过 extensions 记录正式 artifact 路径，
必要时以后版本再增加专门结构，不把时间序列强塞成一个标量。
1300 °C 与 1813 °C 是已有 sanity observations，不是 CASE 默认值；schema 无此类默认值。

bottom_layer_metrics 预留 s11_max、s22_max、mises_max、peeq_max（envelope 或 null），
以及 spatial_profile_path、history_path（root-relative path 或 null）。
set_bottom_layer 已存在于 N32 master 为用户确认，本轮没有打开 CAE 验证或实施提取。
后续可扩展 bottom-layer stress response、residual stress 指标，并显式标明取样时刻和定义。

## 8. Claims、状态链和验证分层
surrogate prediction != teacher metric。
candidate generated != CAE built != solver completed != ODB extracted != teacher validated。
概念状态链：candidate_created → cae_built → solver_completed → odb_extracted → teacher_ready。
失败、取消、部分结果和 warning 可中断链；这是语义契约，不是本轮实现的 workflow engine。

1. Schema validation：结构、类型、格式、枚举、nullable 和局部状态约束。
2. Cross-record identity：case_id/N/process_config_id 一致；CASE/CAE scan_order 不改；JOB/ODB 归属一致；上游 Run 和真实配置可追溯。
3. Semantic legality：排列长度/范围、parent 非自身、物理 mapping；本轮不运行。
4. Engineering/scientific validation：构建、solver 完整性、ODB 内容、指标定义和 teacher readiness；本轮不运行。
结构验证 PASS 仅说明载荷合规，不证明科学或工程正确。

## 9. 示例和本轮冻结范围
`docs/examples/case_example.json` → `cae_job_manifest_example.json` → `teacher_metrics_example.json`。
三者 case_id=S4C999999，process_config_id=stage4_example_only_v00，example_only=true。
run_999991/999992/999993 是说明各生产阶段来源字段的假想执行 ID，不创建相应 Run。
工程路径都带 __EXAMPLE_ONLY__；它们不应存在，也不会被创建或读取。
teacher 示例 solver/extraction=not_run，teacher_ready=false，所有预留测量值为 null。
示例内说明用 created_at 不代表实际执行时刻，不是新 teacher data。

本轮冻结：身份与编号规范、1-based spatial ID、三种载荷字段、schema version、扩展方式、
路径基准、生产阶段状态分离、示例隔离与 teacher 放行条件。
不冻结最终 metric reduction/实验参数，不注册正式工况、不分配 CASE、不执行 legality audit 或工程任务。
