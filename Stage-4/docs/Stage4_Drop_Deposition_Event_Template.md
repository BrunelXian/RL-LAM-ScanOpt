# RL-LAM-ScanOpt Stage 4：Drop 沉积事件模板

> 用途：作为 Stage 4 Abaqus 模型生成的固定参考。  
> 目标：每生成一个新的 drop，都按同一套 `preheat → react → coheat → cool` 物理流程执行，避免漏掉 Model Change、Film、Radiation、Surface 状态切换或 Heat Load。

---

## 1. 基本概念

Stage 4 中，一个扫描动作不再只是“在某个位置加热”。

每一个扫描动作对应一个完整的 **deposition event**：

```text
PREHEAT
→ REACT
→ COHEAT
→ COOL
```

设：

- `k`：当前是第几个加工事件，即扫描顺序中的位置；
- `i`：当前实际加工的 drop 空间编号。

例如：

```text
scan_order = [12, 3, 27, ...]
```

则：

```text
Step-01-* → drop 12
Step-02-* → drop 3
Step-03-* → drop 27
```

因此必须区分：

```text
加工顺序编号 k
≠
空间 drop 编号 i
```

---

## 2. 每个 Drop 必须执行的四个 Step

### 2.1 PREHEAT

Step：

```text
Step-k-preheat
```

当前 drop `i` 仍未激活。

需要执行：

```text
Heat:
    set_pool_i
    MAG = 8.0E10
```

此时：

```text
set_addi_i       = inactive

Surf_in_i:
    Film = ON
    Radiation = ON

Surf_addi_i:
    Film = OFF
    Radiation = OFF
```

物理意义：在材料加入之前，先加热该 drop 下方的 substrate / pool 区域。

### 2.2 REACT

Step：

```text
Step-k-react
```

这是 deposition event 的状态切换步骤，需要同时完成：

```text
1. load-k-pre-pool → OFF
2. Surf_in_i Film → OFF
3. Surf_in_i Radiation → OFF
4. Model Change → ADD / Reactivate set_addi_i
5. Reactivate elements with strain = OFF
```

原因：`Surf_in_i` 在 drop 加入前是自由表面；drop 加入后成为 substrate–deposit 内部界面，不能继续与环境进行 Film/Radiation 换热。

### 2.3 COHEAT

Step：

```text
Step-k-coheat
```

创建两个热源：

```text
load-k-co-pool
    region = set_pool_i
    MAG = 4.0E10

load-k-co-drop
    region = set_addi_i
    MAG = 4.0E10
```

同时开启新 drop 外表面的环境散热：

```text
Surf_addi_i Film → ON
Surf_addi_i Radiation → ON
```

当前参数：

```text
Film coefficient = 46.3
Sink temperature = 25 °C

Emissivity = 0.285
Ambient temperature = 25 °C
```

这些 Film/Radiation 从此以后继续存在。

### 2.4 COOL

Step：

```text
Step-k-cool
```

关闭：

```text
load-k-co-pool → OFF
load-k-co-drop → OFF
```

但保持：

```text
Surf_addi_i Film = ON
Surf_addi_i Radiation = ON
```

当前 drop 在完成 COOL 后成为模型中的永久已沉积材料。

---

## 3. 一个 Drop 完成后的永久状态

```text
set_addi_i       = ACTIVE

Surf_in_i:
    Film = OFF
    Radiation = OFF

Surf_addi_i:
    Film = ON
    Radiation = ON
```

这个状态后续不得被 reset。

---

## 4. 未沉积 Drop 的状态

```text
set_addi_j       = INACTIVE

Surf_in_j:
    Film = ON
    Radiation = ON

Surf_addi_j:
    Film = OFF
    Radiation = OFF
```

当轮到 drop `j` 时，再按完整四步流程切换。

---

## 5. 状态表

| 对象 | PREHEAT | REACT | COHEAT | COOL |
|---|---:|---:|---:|---:|
| `set_addi_i` | OFF | **ON** | ON | ON |
| `Surf_in_i Film` | ON | **OFF** | OFF | OFF |
| `Surf_in_i Radiation` | ON | **OFF** | OFF | OFF |
| `pool preheat load` | ON | **OFF** | OFF | OFF |
| `pool coheat load` | OFF | OFF | **ON** | **OFF** |
| `drop coheat load` | OFF | OFF | **ON** | **OFF** |
| `Surf_addi_i Film` | OFF | OFF | **ON** | ON |
| `Surf_addi_i Radiation` | OFF | OFF | **ON** | ON |

---

## 6. 当前热源参数

```text
PREHEAT:
    pool = 8.0E10

COHEAT:
    pool = 4.0E10
    drop = 4.0E10
```

当前单 drop sanity 结果：

```text
preheat 末端 interface temperature ≈ 1300 °C
coheat Tmax ≈ 1813 °C
```

这些值当前作为 **sanity / effective heat-source calibration anchor** 使用。暂时不要把 MAG 直接解释成设备激光功率的严格一一换算；后续实验完成后，再依据 IR 温度历史和高温区/熔池尺寸校准。

---

## 7. Step 时间结构

当前单 drop 模板：

```text
Step-k-preheat    = 0.1 s
Step-k-react      = 1.0E-6 s
Step-k-coheat     = 0.1 s
Step-k-cool       = 4.0 s
```

完整制造完成后：

```text
Step-final-cooling = 1200 s
```

---

## 8. Model Generator 的核心逻辑

```python
for k, i in enumerate(scan_order, start=1):

    # PREHEAT
    create Step-k-preheat
    turn ON heat on set_pool_i

    # REACT
    create Step-k-react
    turn OFF preheat load
    turn OFF Film on Surf_in_i
    turn OFF Radiation on Surf_in_i
    reactivate set_addi_i with strain-free option

    # COHEAT
    create Step-k-coheat
    turn ON pool coheat
    turn ON drop coheat
    turn ON Film on Surf_addi_i
    turn ON Radiation on Surf_addi_i

    # COOL
    create Step-k-cool
    turn OFF pool coheat
    turn OFF drop coheat
```

关键原则：

> **scan order 只决定本次事件中的 `i` 是谁；物理模板本身不能随策略改变。**

---

## 9. 每次生成 Job 前必须检查

- [ ] 每个扫描事件都有 `preheat/react/coheat/cool` 四个 Step；
- [ ] Step 编号按加工顺序 `k` 排列；
- [ ] `set_pool_i` 与 `set_addi_i` 和当前扫描策略一致；
- [ ] PREHEAT 使用 `8E10`；
- [ ] COHEAT pool 使用 `4E10`；
- [ ] COHEAT drop 使用 `4E10`；
- [ ] REACT 中关闭对应 `Surf_in_i` 的 Film；
- [ ] REACT 中关闭对应 `Surf_in_i` 的 Radiation；
- [ ] REACT 中存在 `set_addi_i` 的 Model Change；
- [ ] Model Change 使用 strain-free reactivation；
- [ ] COHEAT 开启 `Surf_addi_i` 的 Film；
- [ ] COHEAT 开启 `Surf_addi_i` 的 Radiation；
- [ ] COOL 关闭 pool/drop 两个热源；
- [ ] 已完成 drop 的 Film/Radiation 状态不会被后续 Step reset；
- [ ] 未沉积 drop 的 `Surf_in` 仍保持环境散热；
- [ ] 最后存在 `Step-final-cooling`；
- [ ] `set_bottom_layer` 保留，用于底板应力演化后处理。

---

## 10. 一句话定义

Stage 4 的扫描策略不是：

```text
scan order
→ ordered heat loads
```

而是：

```text
scan order
→ ordered deposition events
```

其中每个 deposition event 都必须完整执行：

```text
preheat
→ surface-state switch + material reactivation
→ coheat
→ cool
```

这套模板是 Stage 4 N32 及后续 variable-N 模型生成的基础。
