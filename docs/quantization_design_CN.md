# YOLO26 全算子量化训练框架 — 完整设计思路

> English version: [quantization_design_en.md](quantization_design_en.md)

> 适用范围：YOLO26 检测 / 分割 / 姿态 / 分类 / 旋转框 / 深度 6 任务，
> float → PTQ → QAT → compare 全流程，面向嵌入式 NPU / TPU / 地平线等加速器真实部署。
>
> 核心理念一句话：**板端实际执行什么，训练侧就量化什么**——量化范围覆盖几乎所有算子，
> 而不只是 conv + linear + relu。
>
> 本文基于对 `quantization/` 全部 8 个文件与 `networks_yolo26-*.py` 的完整源码通读写成。

---

## 目录

1. [为什么必须量化几乎所有算子](#1-为什么必须量化几乎所有算子)
2. [量化算子全集与分类](#2-量化算子全集与分类)
3. [总体架构：三层结构](#3-总体架构三层结构)
4. [7 个量化后端与量化器数学](#4-7-个量化后端与量化器数学)
5. [激活量化算子的 forward 精确行为](#5-激活量化算子的-forward-精确行为)
6. [PTQ 与 QAT 的生命周期（freeze/reset 状态机）](#6-ptq-与-qat-的生命周期)
7. [部署产物：三件套与 clip 折叠](#7-部署产物三件套与-clip-折叠)
8. [unsigned/signed 与 per_channel/per_tensor 设计](#8-unsignedsigned-与-per_channelper_tensor-设计)
9. [网络脚本接入机制](#9-网络脚本接入机制)
10. [关键工程决策与踩坑记录](#10-关键工程决策与踩坑记录)
11. [实验验证结论](#11-实验验证结论)
12. [局限与后续方向](#12-局限与后续方向)

---

## 1. 为什么必须量化几乎所有算子

### 1.1 教科书 vs 真实板端

很多量化入门示例只量化 `conv + linear + relu`。这在 GPU 上做 PTQ/QAT 研究够用，
但**与嵌入式 NPU/TPU 的真实执行方式严重脱节**：

| 位置 | 教科书做法 | 板端实际 |
|---|---|---|
| 算子范围 | conv/linear 量化，其余全 FP32 | 图上**每个算子**都有 int 输入/输出 scale |
| 逐元素 | add/mul/sub/div 在 FP32 域算，再量化回 | 直接 int8 计算（requant 融合进硬件） |
| concat | FP32 拼接 | 各输入分支按各自 scale 拼接，输出统一 scale |
| 激活 | silu/sigmoid/softmax 走 FP32 | LUT（查找表）量化实现，输入输出都是 int |

如果训练侧只量化 conv+linear，板端执行时 add/concat/silu 等算子的量化误差是
**训练时从未见过的**——这是部署"训练好、上板崩"的最常见根因。

### 1.2 本项目对策

训练图在**每一个**会产生量化误差的算子位置都插入伪量化（fake quant），让 QAT 的
损失函数完整感知板端噪声：

- 卷积/全连接的**权重**与**输入激活**都量化
- 残差 `add`、shortcut `sub`、逐元素 `mul`/`div` 的两输入与输出都量化
- `concat` 每个输入分支独立量化（板端各分支 scale 本就不同）
- `silu`/`sigmoid`/`softmax` 等非线性输入量化（复现板端 LUT 的截断）
- `maxpool` 输出量化

代价是实现复杂度高（每个后端要实现全套 ~15 个算子类），但这是部署保真的唯一正确路径。

---

## 2. 量化算子全集与分类

**关键事实：7 个后端每一个都在自己的模块文件里完整实现了同一套 ~15 个算子类**
（不共享基类、不跨文件复用，只有 PACT 的权重量级化器是可插拔工厂）。算子分三类：

### 2.1 带权重量级化器（weight-carrying ops）

| 算子类 | 说明 | 量化内容 |
|---|---|---|
| `QuantConv2d` | 2D 卷积 | 输入激活 + 权重（per_channel/per_tensor） |
| `QuantConvTranspose2d` | 转置卷积（seg 上采样 proto 头） | 同上 |
| `QuantLinear` | 全连接 | 同上 |
| `QuantMatMul` | 矩阵乘（attention/head） | 输入激活 ×2 |

鸭子类型标识：`hasattr(module, "weight_quantizer")`
（[quantization/__init__.py](../quantization/__init__.py) `is_weight_quant_module`）。
训练时权重每次前向实时量化产生梯度；推理时烘焙后的量化权重直接存入 `.weight`，
权重量级化器被 `quant_inference=True` 短路。

### 2.2 纯激活量化算子（activation-only ops）

无权重，只有激活量化器，只有 `all_positive` 配置（无权重概念）：

| 算子类 | 典型位置 | 量化点 |
|---|---|---|
| `QuantAdd` / `QuantSub` | 残差/shortcut、特征差 | 两输入（各一个量化器）+ 输出 |
| `QuantMultiply` / `QuantDiv` | 门控/注意力、归一化 | 同上；`QuantDiv` 分母量化后 `clamp(min=1e-6)` 防 0/0 |
| `QuantConcat` | FPN 两路拼接 | 两输入各一个量化器（`activation_quantizer0/1`）|
| `QuantCat` | FPN 多尺度拼接 | **每输入分支一个量化器**（`self.quantizers = nn.ModuleList`）|
| `QuantMaxPool` | 下采样 | 输入 |
| `QuantSiLU` / `QuantSigmoid` / `QuantReLU` / `QuantSoftmax` | 激活/归一化 | 输入 |

**为什么 concat 每分支独立量化**：板端拼接时各输入张量各有自己的 scale，拼接核按
各自 scale 搬运、输出再统一。`QuantCat` 用 `ModuleList` 每分支一个量化器精确复现。

**为什么两输入算子各自量化输入**：逐元素算子在 int 域计算时，两个输入 scale 必须
对齐（requant 到公共 scale），requant 本身就是噪声源。训练时量化输入，QAT 才能学到
对该噪声鲁棒的权重。

### 2.3 量化器（quantizer）本体

与算子分离的底层模块，真正做 fake-quant（`round + clamp + dequant`）：

- **激活量化器**：`*ActivationQuantizer`，持有可学习 scale / beta / alpha，
  受 `all_positive` 控制 signed/unsigned
- **权重量级化器**：`*WeightQuantizer`，受 `w_all_positive` / `per_channel` 控制

---

## 3. 总体架构：三层结构

```
┌─────────────────────────────────────────────────┐
│ 网络脚本 networks_yolo26-{detect,seg,pose,...}.py │
│  · set_quant_method()  把后端算子类绑定为全局名     │
│  · 搭建网络时直接用 QuantConv2d/QuantAdd/... 建层   │
│  · float/ptq/qat/compare 各 stage CLI              │
└──────────────────┬──────────────────────────────┘
                   │ quant_pkg.load_quant_backend(method)
┌──────────────────▼──────────────────────────────┐
│ quantization/__init__.py（跨后端公共层）           │
│  · _BACKEND_MODULES / QUANT_METHODS             │
│  · freeze_batch_init / reset_quantizer_states   │
│  · collect_quant_params（导出 scale/zp 三件套）   │
│  · build_float_model（部署用干净浮点图，含 clip 折叠）│
└──────────────────┬──────────────────────────────┘
                   │ importlib 动态加载
┌──────────────────▼──────────────────────────────┐
│ 7 个后端模块 quantization/{backend}.py            │
│  · 每文件独立实现全套 ~15 个算子类                 │
│  · 每文件的 Activation/Weight quantizer 实现      │
└─────────────────────────────────────────────────┘
```

要点：
- **后端可插拔**：`load_quant_backend("lsqplus_v1")` 用 `importlib` 返回对应模块，
  网络脚本拿到的算子类属性名跨后端完全一致 → 换后端只改一个字符串
- **公共逻辑集中**：量化器状态管理、参数导出、浮点图构建都放 `__init__.py`，
  避免 7 个后端重复实现
- **鸭子类型而非继承**：用 `hasattr(m, "weight_quantizer")` /
  `hasattr(m, "clip_bounds")` 判断模块角色。7 个后端各自独立实现全套算子
  （**没有**共享基类），鸭子类型提供了跨后端的最小公共契约

---

## 4. 7 个量化后端与量化器数学

| 后端 | 文件 | 激活对称性 | 量化器 | 推荐场景 |
|---|---|---|---|---|
| **lsqplus_v1** | `lsqplus_quantize_V1.py` | **非对称**（学 s+β） | `LSQPlusActivationQuantizer` / `LSQPlusWeightQuantizer` | **部署默认** |
| lsqplus_v2 | `lsqplus_quantize_V2.py` | 非对称 | 同上结构 | 对照 |
| lsq_v1 | `lsqquantize_V1.py` | 对称 | `LSQActivationQuantizer` / `LSQWeightQuantizer` | 轻量、对称硬件 |
| lsq_v2 | `lsqquantize_V2.py` | 对称 | 同上（激活 s 初始化为常数 1） | 对照 |
| minmax | `minmax.py` | 非对称 | `MinMaxActivationQuantizer` / `MinMaxWeightQuantizer` | baseline |
| dorefa | `dorefa.py` | 激活非对称 / 权重对称 | `DorefaActivationQuantizer` / `DorefaWeightQuantizer` | **2026-10 新版**稳定可用（见下方说明） |
| pact | `pact.py` | 非对称（学 clip α） | `PactActivationQuantizer` + `build_weight_quantizer()` | 对比/部分部署 |

### 4.1 量化器 forward 数学差异

- **dorefa（新版，2026-10）**：激活量化器走标准非对称公式（`scale` + `beta`，zero_point 由 beta 隐式表示），与 lsqplus 的激活量化器公式完全一致；权重量化器为线性对称网格 `q_w = maxvalue × round(w/maxvalue × Qp) / Qp`，步长 `maxvalue/Qp`，`per_channel` 按输出通道统计。**v1 版本**使用 `tanh` 域非线性量化（权重）和 `clamp(±1)` 硬截断（激活），2026-10 已全部移除，详见 [dorefa.py 实现变更记录](../quantization/dorefa.py)。
- **lsq_v1/v2**：学 scale `s`，`all_positive` 设定 `Qn/Qp`（负/正量化级数），
  前 20 批做尺度初始化/平滑（`batch_init` 状态机），之后固定。LSQ 经典 STE。
- **lsqplus_v1/v2**：同时学 `s` 和零点偏移 `beta`（`beta` 初始化为 `-1e-9`），
  执行 `ALSQPlus`/`WLSQPlus` 非对称量化，权重用均值/标准差初始化 `s`。
- **minmax**：标准非对称 `q = clamp(round(x/scale)+zp, qmin, qmax)`，支持
  `percent/cluster` 两种离群点收集，`all_positive` 把 `r_min` 截断为 0。
  范围内梯度直通、范围外置零。
- **pact**：学可学习截断阈值 `alpha`（`PactActivationQuantizer`），`all_positive`
  决定 unsigned/signed。权重量级化器由 `build_weight_quantizer(method, ...)`
  工厂分发，可选 `dorefa/minmax/lsqplus_v1/lsqplus_v2`（默认 `lsqplus_v1`）。

### 4.2 统一接口（`__init__.py`）

`load_quant_backend(method)` 查 `_BACKEND_MODULES` 字典返回模块；
`_WEIGHT_OP_NAMES`（带权重算子类名）与 `_ELTWISE_OP_NAMES`（激活算子类名）枚举后，
`weight_quant_ops(backend)` / `all_quant_ops(backend)` 统一收集算子类元组，供
freeze/reset/导出遍历。

---

## 5. 激活量化算子的 forward 精确行为

`quant_inference` 开关的通用语义（每个算子类都遵守）：
- **`quant_inference=False`（训练）**：执行伪量化/反量化（产生梯度）后运算
- **`quant_inference=True`（推理）**：跳过量化，直接走原始浮点运算

> 注意：这个开关的语义是"是否激活量化噪声"。训练时 `False`= 有量化；
> 推理导出浮点图时置 `True` = 去掉量化（配合 §7 clip 折叠，激活量化器会被
> 整体替换）。个别算子如 `QuantCat` 的 `quant_inference` 硬编码为 `True`
> （多输入拼接总是量化各分支）。

各算子在量化路径下（`False`）的具体量化点见 §2.2 表：两输入算子各自量化、
单输入激活只量化输入、`QuantDiv` 分母 clamp 防 0/0。PACT 后端额外把
`QuantSiLU/Sigmoid/ReLU/Softmax/MatMul` 的量化器做成 PACT 风格（学 α）。

---

## 6. PTQ 与 QAT 的生命周期

### 6.1 三阶段

| 阶段 | 做什么 | 量化器状态 |
|---|---|---|
| float | 全精度训练，产出基线 best.pth | 不激活 |
| PTQ | 加载 float 权重，少量校准图前向，**只校准量化器参数**，不训权重 | 校准→freeze |
| QAT | 加载 PTQ 结果，**量化噪声下继续训练权重**（STE 反传） | 权重可训，量化器视后端而定 |

### 6.2 freeze / reset 状态机（`quantization/__init__.py`）

- `freeze_batch_init(model)`：遍历所有量化器，把当前 scale/zp 标记为冻结
  （`quantization/constants.py` 的 `INIT_STATE_FROZEN` 哨兵），PTQ 校准完成后锁定，
  防止 QAT 初期漂移。lsq 系的 `batch_init` 状态机也靠它停止统计
- `reset_quantizer_states(model)`：解冻/重置为第 0 批重新统计，让可学习量化器
  （lsqplus/pact）在 QAT 中继续微调

### 6.3 `quant_inference` 与部署推理路径

训练（`False`）卷积前向 `weight_quantizer(self.weight)` 实时量化权重产生梯度；
推理导出（`True`）权重已烘焙为量化值存入 `.weight`，权重量级化器短路。这个开关
是 `build_float_model` 能安全构造干净浮点图的关键（见 §7.3）。

---

## 7. 部署产物：三件套与 clip 折叠

### 7.1 三件套

| 产物 | 消费者 | 内容 |
|---|---|---|
| `*_quant_params.json/.pth` | 工具链编译器 | 每层激活/权重的 scale + zero_point |
| `*_float.onnx` | 编译输入 | 干净浮点图（无 Q/DQ 节点），权重为量化后值 |
| quant `.pth` | 训练/补发 | fake-quant 训练模型 checkpoint |

> **重要提醒**：JSON 中的 scale / zero_point **仅供交叉验证参考**。嵌入式板端（地平线 NPU、
> TPU 等）的 PTQ 工具会用自身校准数据**重新计算** scale 与 zero_point，最终部署以板端
> 重新算出的值为准。JSON 方便你在编译前比对训练侧与板侧是否一致，不等于部署时直接套用。

导出顺序：`collect_quant_params` 先存 JSON，再 `build_float_model` 出 ONNX，
两者同一次保存、源自同一量化器状态，天然自洽。

### 7.2 `collect_quant_params`（`__init__.py`）

遍历量化模型，对每个量化器用 `activation_scale_zp` / `weight_scale_zp` 提取
scale/zero_point（内部按后端差异处理 lsqplus 的 β、minmax 的 zp、pact 的 α、
**新版 dorefa 激活走 lsqplus scale/beta 路径 + 权重线性对称（amax/Qp）**），打包成 `{layer: {scale, zero_point}}`。`required_quant_keys` 校验
导出完整性（硬断言）。

### 7.3 `build_float_model` 与 clip 折叠

**问题背景**：`act_unsigned` 时激活量化器本质是"先 clip 到 `[0, 255·s]` 再取整"。
QAT 让权重**依赖这个硬截断**。早期导出浮点图把整个激活量化器连根拔掉（连 clip
一起去掉），导致 unsigned+对称后端浮点 ONNX 输出逐层爆炸（±1e5，正常应 ≤640）。

**修复（clip 折叠）**：导出浮点图时激活量化器不删，替换为 `_ClipOnlyQuantizer(lower, upper)`
——只保留截断（clamp）、去掉取整。7 个后端激活量化器各自实现 `clip_bounds()`
返回激活空间边界：

- unsigned 对称后端：`(0, 255·s)`
- lsqplus（带 β）：`(β, β+255·s)`
- signed 对称：`(-128·s, 127·s)`
- pact：`(0, α)` / `(-α, α)`

实现要点（`__init__.py` L224-306）：
1. `deepcopy(quant_model)` → `model.cpu()` → `freeze_batch_init` / `eval`
2. 全模块置 `quant_inference=True`（短路边界）
3. 带权层 `.weight.copy_(weight_quantizer(weight))` 烘焙量化权重，
   权重量级化器换 `_IdentityQuantizer`
4. 递归函数 `_replace_activation_quantizers` 按鸭子类型 `hasattr(child,'clip_bounds')`
   把所有激活量化器换 `_ClipOnlyQuantizer`。**必须递归**——`QuantCat` 把量化器装在
   `self.quantizers = nn.ModuleList`（属性名不是 `activation_quantizer*`）
5. `_ClipOnlyQuantizer` 用 `register_buffer` 存边界（forward 里**不能用 `.to()`**，
   tracing 会产生 `aten::copy_` 导致 ONNX 导出失败），边界 `.detach()`

**默认 `use_clip=False`**：实际部署时板端 PTQ 工具会自行算 scale/zp，浮点 ONNX
通常**不需要** clip（板端硬件 int 域自带截断）。`use_clip=True` 仅用于生成与 QAT
完全等价的参考图做对照验证。

### 7.4 compare 阶段的正确口径

QAT vs Float 的精度对比**只用两个 checkpoint 在 val 集上的指标**，不需要"导出
ONNX vs float 输出"——后者无意义，因为干净浮点 ONNX 没有 scale/zp，NPU 板端 PTQ
工具会自己算。本项目 compare 只比 Float 模型 mAP vs QAT 模型 mAP（`[Compare]
Float/QAT/delta`）。

---

## 8. unsigned/signed 与 per_channel/per_tensor 设计

### 8.1 激活 unsigned（`all_positive=True`，默认）

YOLO 主干大量用 **SiLU/ReLU**，激活值天然 ≥ 0。unsigned（[0,255]）比 signed
（[-128,127]）能用满正半轴分辨率——int8 精度几乎无损的关键。

- **非对称后端**（lsqplus_v1/v2、pact、minmax）：原生支持，稳定无损
- **对称后端**（lsq_v1/v2、minmax）：零点被迫在边界外，unsigned 激活严重截断 →
  **掉点 0.18~0.37，PTQ 崩塌**。对称后端必须配 `act_signed`
  （**2026-10**：dorefa 已改为非对称激活，不再属于对称后端；minmax 激活可配置
  zero_point，技术上能跑 unsigned，但量化损失因无动态缩放而偏大，实测 PTQ 接近 0）

### 8.2 权重 signed（`w_all_positive=False`，默认）

卷积权重有正有负，必须 signed。**weight_unsigned 是错误配置**（负权重在 [0,1]
量化域无法表示），实测 8/8 全部崩塌为 0，作为反例保留。

### 8.3 per_channel（默认 True）

权重按输出通道各一套 scale，精度高于 per_tensor；激活通常 per_tensor。
`mixed_quant`（权重 per_channel + 激活 per_tensor）是常见板端折中，已验证非对称
后端下稳定。

---

## 9. 网络脚本接入机制

### 9.1 后端注入（`set_quant_method`）

`networks_yolo26-detect.py` L367-383：`set_quant_method(method)` 调
`quant_pkg.load_quant_backend(method)`，把 `QuantAdd/QuantCat/QuantConcat/
QuantConv2d/QuantMaxPool/QuantSiLU/...` 绑定为该后端导出的算子类（模块级全局名）。
seg.py（L108）与 pose.py（L119）有**独立**的 `set_quant_method`（不可混用 det 的，
否则 head 量化器 key 缺失）。

### 9.2 搭建时替换（非运行时遍历）

YOLO26 **不是**训练时遍历替换 nn 层，而是构造量化网络时直接用后端算子类建层：
`Conv`（L418-439）在 `quant=True` 时直接 `QuantConv2d(...)`，激活换 `QuantSiLU(...)`，
残差用 `QuantAdd(..., quant_inference=True)`，拼接用 `QuantCat/QuantConcat`。
`_quant_layer_kwargs()`（L406-415）从 `QUANT_CFG` 统一构造 `a_bits/w_bits/
per_channel/all_positive/w_all_positive`（PACT 后端额外 `w_quant`）传给每个量化层。

> 注：`quantization` 包里虽有 `add_quant_op` 式遍历替换的工具，但 YOLO26 主流程走
> "搭建时直接建量化层"，遍历替换只用于部分辅助场景；depthwise/group conv 若走
> 遍历路径需确认被识别为 `nn.Conv2d`。

### 9.3 配置唯一来源（`QUANT_CFG`）

`networks_yolo26-detect.py` L222-232：`DEFAULT_QUANT_CFG` 字段
`a_bits=8 / w_bits=8 / per_channel=True / all_positive=True / w_all_positive=False /
mixed_quant=False / pact_w_quant='lsqplus_v1'`。CLI `--a-bits/--w-bits/--per-channel/
--all-positive/--w-all-positive/--mixed-quant/--pact-w-quant/--seed/--run-tag`
覆盖；`_quant_cfg_tag()` 生成完整单词标签（禁用缩写）。

### 9.4 各 stage 与 checkpoint

`load_checkpoint()`/`save_checkpoint()` 保存/加载 `.pth`；best.pth 用
`ck['state_dict']`，PTQ 的 `.pth` 可能是裸 state_dict 或 `ck['model']`（需兼容）。
`save_quant_outputs()`（L1323-1331）调 `freeze_batch_init()` 后保存量化模型并导出
三件套；`build_float_model()`（L1216-1230）与 `collect_quant_params()`（L1233-1235）
都委托 `quant_pkg`；`verify()`（L1297-1320）硬断言导出 ONNX 无量化节点 +
quant_params 完整。meta 通过 `**(meta or {})` 透传记录量化方法/配置/seed。

---

## 10. 关键工程决策与踩坑记录

| 决策/坑 | 根因 | 解法 |
|---|---|---|
| `_anchors` 设备不匹配 | 检测头懒创建的 `_anchors`/`_strides_tensor` 是普通属性，`model.cpu()` 不迁移，deepcopy 后 GPU/CPU 混用 | 注册为**非持久 buffer**（`persistent=False`，随设备迁移、不进 state_dict、不被 `freeze_batch_init` 当量化器） |
| ONNX 导出 `aten::copy_` 失败 | `_ClipOnlyQuantizer.forward` 里用了 `.to()` | 边界用 `register_buffer`，forward 不做 device 转换 |
| QuantCat 量化器漏替换 | 量化器装在 `ModuleList`，属性名非常规 | 递归 + `hasattr(child,'clip_bounds')` 鸭子类型遍历 |
| bash 断点续跑失效 | `already_ok` 匹配 `[OK] task=` 但日志行是 `[OK] 日期 task=` | 正则锚定带日期的 `[OK]` 行 |
| seg/dorefa OOM（**2026-10 已修复**） | 640 分辨率 + tanh 量化器显存大 | 旧版 seg 与全任务 dorefa 的 QAT 一律 batch=4；新版移除 tanh 后全任务统一 batch=8 |
| LSQ v1/pact dummy 全零 | 全零致 scale=0 → NaN | dummy 输入 `randn*0.1` |
| seg/pose head 量化器缺 key | 用 det 的 set_quant_method 跨任务 | seg/pose 各自独立的 set_quant_method |
| dorefa unsigned 激活坍缩（**2026-10 已修复**） | v1 tanh 域强制对称，无 zero_point | 旧版加可学习尺度 s 临时修复；新版激活改用 LSQ+ 非对称公式，彻底解决 |

**为什么用鸭子类型而非注册表/继承**：7 个后端的量化器内部结构差异大（pact 学 α、
lsqplus 学 β、minmax 无参数、dorefa 用线性对称网格权重 + 非对称激活），强行统一基类束缚实现。`hasattr`
契约（`weight_quantizer`/`clip_bounds`/`quant_inference`）提供跨后端最小公共面，
各后端在各自文件里独立实现全套算子、互不依赖。

---

## 11. 实验验证结论

完整 79 单元横评（4 主配置 × 3 任务 × 7 后端 + weight_unsigned 反例 + mixed_quant
+ 多种子 + 校准量），数据见 [README_CN.md](../README_CN.md) / [README.md](../README.md)
结果章节。核心结论：

- **推荐配置（lsqplus_v1 + per_channel + act_unsigned）三任务无损或近无损**：
  detect −0.0007 / seg −0.0021 / pose +0.0037
- 非对称后端 + unsigned 全部稳定（Δ ≤ 0.039）
- 对称后端 + unsigned 严重掉点（必须 act_signed）
- weight_unsigned 全崩塌（理论预期的错误配置）
- 多种子波动 ≤ 0.012，结论方向稳定
- QAT 能追回 PTQ 大部分损失；unsigned+非对称下 PTQ 已接近 float

---

## 12. 局限与后续方向

**当前局限**：
- 若某自定义 head 出现未在 `_ELTWISE_OP_NAMES` 覆盖的算子，板端仍需确认其量化口径
- LUT 类非线性（silu/sigmoid/softmax）训练侧用 fake-quant 近似，与板端具体 LUT
  位宽可能有细微差异，建议部署后用真实编译器回灌精度复核
- 验证基于 1/100 COCO mini 小样本，绝对精度不代表全量 COCO

**后续可扩展**：
- 新增后端：实现同一套 ~15 个算子类 + quantizer + `clip_bounds()`，注册进
  `_BACKEND_MODULES` 即可，无需改网络脚本
- 混合精度（敏感层保 int16/FP16）可在 `collect_quant_params` 导出时按层标注
- 板端 int 域逐元素 requant 的精确建模（目前用 fake-quant 近似）

---

*文档对应代码版本：见 git 当前提交。算子实现见 `quantization/{backend}.py`，
公共机制见 `quantization/__init__.py`，网络接入见 `networks_yolo26-*.py`。*
