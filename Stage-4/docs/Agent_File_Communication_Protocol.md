# Agent 文件通讯协议

## 1. 协议目的

本协议用于规范项目中多个子 Agent 之间的文件通信、运行记录、结果交接与状态追踪。

本项目采用：

> **文件系统作为 Agent 间的主要共享状态和交接媒介。**

核心原则：

- 每一次独立 Agent 执行都形成一个唯一的 `run_XXX`；
- 所有 Run 按编号统一管理；
- `/run` 记录运行历史与 Agent 通信状态；
- `/docs` 保存需要长期统筹和跨 Agent 使用的项目文档；
- `/CAE` 保存所有实际 CAE 模型、求解文件和工程结果；
- Run 只引用 CAE 文件，不在 Run 中复制正式 CAE 文件；
- Save、Checkpoint、Run、Artifact、Handoff 必须明确区分。

---

# 2. 项目顶层目录

项目根目录至少包含：
```
Project/
├── run/
├── docs/
└── CAE/
```

三个目录职责必须严格分离。

---

# 3. `/run` —— Agent 运行记录与通信中心

所有 Agent 的每一次独立执行，都必须在：
```
/run
```

下面创建一个新的 Run。

采用统一编号：
```
run_001
run_002
run_003
...
run_XXX
```

不允许因为功能不同重新建立独立编号体系。

Run 编号表示：

> **整个项目 Agent 执行历史中的唯一运行序号。**

例如：
```
run/
├── README.md
├── run_001/
├── run_002/
├── run_003/
├── run_004/
└── ...
```

---

# 4. `/run/README.md` —— Run 总索引

`/run/README.md` 是整个运行历史的总说明文件。

每完成一个 Run，都必须在其中增加一条简短记录。

例如：
```
# Run Index

## run_001

**Agent:** model_builder  
**Operation:** build Stage 4 pre-sanity model  
**Status:** completed  
**Result:** 创建 N32 pre-sanity CAE 模型，并完成基础检查。  
**Main output:** `../CAE/Models/N32/stage4_N32_presanity.cae`

---

## run_002

**Agent:** model_validator  
**Operation:** validate Stage 4 pre-sanity model  
**Status:** completed  
**Result:** 模型结构检查通过，未发现 reactivation。  
**Input:** `run_001`  
**Main output:** `../CAE/Reports/N32/presanity_validation.json`
```

这里仅保存：

> **每次 Run 的短摘要和导航信息。**

详细信息仍然放在对应的 `run_XXX` 目录中。

因此：
```
/run/README.md
```

是人类快速阅读的运行总目录，

而：
```
/run/run_XXX/
```

才是某一次执行的正式记录。

---

# 5. 每一个 `run_XXX` 的固定结构

每个 Run 必须采用以下基本结构：
```
run/
└── run_XXX/
    ├── run.json
    ├── artifacts/
    ├── checkpoints/
    └── summary.md
```

例如：
```
run/
└── run_017/
    ├── run.json
    ├── artifacts/
    │   └── artifacts.json
    ├── checkpoints/
    │   ├── checkpoint_001.json
    │   └── checkpoint_002.json
    └── summary.md
```

---

# 6. `run.json` —— 机器可读的核心通信记录

`run.json` 是该次 Run 的正式机器可读状态文件。

推荐基本结构：
```
{
  "schema_version": "1.0",

  "run_id": "run_017",

  "agent": "model_builder",

  "operation": "build_presanity_model",

  "status": "completed",

  "started_at": "2026-09-27T20:10:00+08:00",

  "completed_at": "2026-09-27T20:16:00+08:00",

  "input_runs": [
    "run_016"
  ],

  "artifacts_manifest": "./artifacts/artifacts.json",

  "validation": {
    "status": "passed"
  },

  "handoff": {
    "ready": true
  },

  "error": null
}
```

每个 `run.json` 至少必须回答：

1. 这是哪个 Run？
2. 哪个 Agent 执行？
3. 做了什么？
4. 使用了哪些上游 Run？
5. 是否正常完成？
6. 输出记录在哪里？
7. 是否通过验证？
8. 是否允许后续 Agent 使用？
9. 如果失败，失败原因是什么？

---

# 7. `artifacts/` —— 交付物清单，而不是正式文件存储区

这是本协议的一个核心规定。

`artifacts/` 的作用不是保存 CAE 文件副本。

它负责记录：

> **这个 Run 产生、修改或正式交付了哪些工程文件，以及这些文件实际位于项目中的什么位置。**

例如：
```
run_017/
└── artifacts/
    └── artifacts.json
```

内容可以是：
```
{
  "run_id": "run_017",

  "artifacts": [
    {
      "artifact_id": "model_main",
      "type": "abaqus_cae",
      "role": "primary_output",
      "path": "../../CAE/Models/N32/stage4_N32_presanity.cae"
    },

    {
      "artifact_id": "build_script",
      "type": "python_script",
      "role": "supporting_output",
      "path": "../../CAE/Models/N32/Scripts/stage4_build_presanity.py"
    },

    {
      "artifact_id": "audit_report",
      "type": "validation_report",
      "role": "supporting_output",
      "path": "../../CAE/Reports/N32/stage4_presanity_audit.json"
    }
  ]
}
```

因此：
```
artifacts/
```

本质上是一个：

> **Artifact Manifest / 交付物索引。**

而不是一个用于复制所有成果文件的目录。

---

# 8. 为什么 CAE 文件不能放进 Run

所有正式 CAE 文件统一位于：
```
/CAE
```

其目的不仅是目录整洁，更重要的是：

> **保证 CAE 模型拥有独立、稳定、统一的版本管理体系。**

例如：
```
CAE/
├── Models/
│   ├── N12/
│   ├── N16/
│   ├── N24/
│   ├── N32/
│   └── N40/
│
├── Jobs/
├── ODB/
├── INP/
├── Scripts/
└── Reports/
```

实际运行的：
```
.cae
.inp
.odb
```

以及相关脚本、结果，应按照 CAE 项目自己的组织规则统一管理。

因此：
```
/run
```

解决的是：

> 谁在什么时候做了什么。

而：
```
/CAE
```

解决的是：

> 当前工程模型和仿真文件究竟在哪里、属于哪个版本。

两者不能混为一谈。

---

# 9. Run 与 CAE 文件之间采用引用关系

Run 不拥有 CAE 文件。

Run 只引用 CAE 文件。

例如：
```
run_017
    │
    └── artifact reference
              │
              ▼
CAE/Models/N32/stage4_N32_presanity.cae
```

因此下游 Agent 不应该：
```
进入上一个 Run
→ 找一个 .cae
→ 复制出来继续修改
```

而应该：
```
读取上一个 Run
→ 读取 artifacts manifest
→ 获得正式 CAE 文件地址
→ 直接访问 /CAE 中对应模型
```

这样可以保证：

> **任何时刻，正式 CAE 文件只有明确的工程位置。**

---

# 10. `checkpoints/` —— Run 内部恢复状态

一个 Run 执行过程中可能进行多次 Save。

这些 Save 不应该产生新的 Run。

例如：
```
run_017
    │
    ├── 修改材料
    │      ↓
    │    Save
    │
    ├── 创建 Step
    │      ↓
    │    Save
    │
    ├── 创建 Interaction
    │      ↓
    │    Save
    │
    └── Validation
```

整个过程仍然属于：
```
run_017
```

必要时，可以将关键恢复信息记录为：
```
checkpoints/
├── checkpoint_001.json
├── checkpoint_002.json
└── checkpoint_003.json
```

Checkpoint 用于：

- 崩溃恢复；
- 中断后继续运行；
- 记录关键状态；
- 定位失败位置。

Checkpoint 不是正式交付。

---

# 11. Save、Checkpoint、Run、Artifact 的区别

四者必须明确区分。

### Save

工程软件内部的一次保存行为。

例如：
```
mdb.save()
```

它只是保存当前工程状态。

---

### Checkpoint

Agent 执行过程中的可恢复节点。

例如：
```
模型已经创建完成
但 Validation 尚未执行
```

---

### Run

一次完整的 Agent 执行任务。

例如：
```
run_017 = 构建 Stage 4 N32 pre-sanity model
```

一个 Run 可以包含很多 Save 和多个 Checkpoint。

---

### Artifact

Run 最终声明产生或修改的正式交付物。

例如：
```
CAE/Models/N32/stage4_N32_presanity.cae
```

---

# 12. `summary.md` —— 人类可读的 Run 总结

每个 Run 必须包含：
```
summary.md
```

它用于让人快速理解：

- 为什么执行这个 Run；
- Agent 做了什么；
- 输入是什么；
- 修改了什么；
- 输出是什么；
- 验证结果如何；
- 有没有异常；
- 下一步建议是什么。

例如：
```
# run_017

## Objective

建立 Stage 4 N32 pre-sanity CAE 模型。

## Input

- run_016
- `../../CAE/Models/N32/Sanity/sanity_base.cae`

## Actions

- 创建 Stage 4 thermal step
- 保留现有 mesh
- 建立 element removal
- 未创建 element reactivation

## Outputs

- `../../CAE/Models/N32/stage4_N32_presanity.cae`
- `../../CAE/Reports/N32/stage4_presanity_audit.json`

## Validation

PASS

## Handoff

模型可以交给下一 Agent 继续处理。
```

`summary.md` 用于人类阅读。

`run.json` 用于 Agent 和程序读取。

两者不能互相替代。

---

# 13. `/docs` —— 项目长期统筹文档

`/docs` 保存跨 Run、跨 Agent、长期有效的项目级文档。

例如：
```
docs/
├── agent_protocol.md
├── project_overview.md
├── workflow.md
├── modelling_rules.md
├── validation_rules.md
├── architecture.md
└── stage_handoff.md
```

判断标准非常简单：

> **如果这个内容描述的是“整个项目应该怎么做”，放 `/docs`。**

例如：

- Agent 通讯规则；
- CAE 建模原则；
- 参数定义；
- Validation 标准；
- Stage 3 → Stage 4 交接；
- 系统架构；
- 长期设计决策。

---

# 14. `/run` 与 `/docs` 的边界

例如：
```
“Stage 4 使用 element activation”
```

属于长期项目设计：
```
/docs
```

而：
```
“run_017 成功创建 element removal，
但尚未加入 reactivation”
```

属于具体运行结果：
```
/run/run_017
```

因此：

> `/docs` = Project Knowledge

> `/run` = Execution History

---

# 15. Agent 间通信方式

Agent 之间原则上通过：
```
run.json
+
artifacts manifest
+
summary.md
```

完成交接。

例如：
```
Model Builder Agent
        │
        ▼
     run_017
        │
        ├── run.json
        ├── artifacts/artifacts.json
        └── summary.md
        │
        ▼
Validator Agent
        │
        ▼
     run_018
```

Validator Agent 不需要根据目录顺序猜测任务。

它应明确声明：
```
{
  "run_id": "run_018",

  "input_runs": [
    "run_017"
  ]
}
```

因此：

> **编号用于历史排序，显式引用用于描述依赖关系。**

不能因为：
```
run_018
```

排在：
```
run_017
```

后面，就默认两者一定具有依赖关系。

---

# 16. Run 的正式完成条件

一个 Run 不能仅因为程序执行结束，就被视为可交付。

推荐区分：
```
Execution Status
Validation Status
Handoff Status
```

例如：
```
{
  "status": "completed",

  "validation": {
    "status": "passed"
  },

  "handoff": {
    "ready": true
  }
}
```

只有：
```
completed
+
validation passed
+
handoff ready
```

才表示：

> **这个 Run 可以正式交给下游 Agent 使用。**

---

# 17. 推荐状态

Run 的 `status` 建议至少支持：
```
pending
running
completed
failed
cancelled
```

Validation 建议：
```
not_run
passed
warning
failed
```

Handoff：
```
ready: true
ready: false
```

这样可以避免：

> “脚本运行成功”

被错误理解成：

> “工程结果已经正确，可以继续使用”。

---

# 18. 推荐完整目录

最终推荐结构：
```
Project/
│
├── run/
│   ├── README.md
│   │
│   ├── run_001/
│   │   ├── run.json
│   │   ├── artifacts/
│   │   │   └── artifacts.json
│   │   ├── checkpoints/
│   │   └── summary.md
│   │
│   ├── run_002/
│   │   ├── run.json
│   │   ├── artifacts/
│   │   │   └── artifacts.json
│   │   ├── checkpoints/
│   │   └── summary.md
│   │
│   └── ...
│
├── docs/
│   ├── agent_protocol.md
│   ├── project_overview.md
│   ├── architecture.md
│   ├── modelling_rules.md
│   └── validation_rules.md
│
└── CAE/
    ├── Models/
    ├── Jobs/
    ├── ODB/
    ├── INP/
    ├── Scripts/
    └── Reports/
```

---

# 19. 最终原则

整个协议可以压缩成三句话：

> **`/run` 记录 Agent 做过什么。**

> **`/docs` 记录整个项目应该怎么做。**

> **`/CAE` 保存真正被创建、修改和运行的 CAE 工程实体。**

进一步：

> **Run 用 `run_XXX` 统一编号，用显式引用表达依赖，用 Artifact Manifest 指向 `/CAE` 中的正式交付物。**

因此：
```
Run ≠ Save
Run ≠ CAE Model
Artifact Record ≠ Artifact Copy
Run Number ≠ Dependency
```

这种结构保证整个 Agent 系统具备：

- 可追溯性；
- 可恢复性；
- 清晰的运行历史；
- 明确的 Agent 交接；
- CAE 文件统一管理；
- 后续扩展 A2A / MCP 等外部协议的空间。