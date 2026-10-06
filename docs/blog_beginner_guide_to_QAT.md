# YOLO26从零开始搞懂神经网络量化PTQ和QAT：一篇详细实战指南

>本文撰写的部分内容来自人工智能大模型Kimi-K3

> 本文基于 PyTorch 开源项目[YOLO26-PTQ-QAT](https://github.com/ZouJiu1/YOLO26-PTQ-QAT)和[aLSQplus/QAT_training](https://github.com/ZouJiu1/LSQplus) 的完整源码、设计文档与实验数据写成。目标读者：**刚入门的深度学习新手**（不需要你有量化背景，老人小孩看了都能懂）。
>
> 阅读时间：约 40 分钟。

---

## 目录

1. [先讲一个故事：为什么要把神经网络“变小”？](#1-先讲一个故事为什么要把神经网络变小)
2. [什么是量化？用“四舍五入”来理解](#2-什么是量化用四舍五入来理解)
3. [整个项目在做一件什么事？](#3-整个项目在做一件什么事)
4. [准备数据：没有数据，一切都是空谈](#4-准备数据没有数据一切都是空谈)
5. [第一步：浮点训练（Float Training）](#5-第一步浮点训练float-training)
6. [第二步：训练后量化校准（PTQ）](#6-第二步训练后量化校准ptq)
7. [第三步：量化感知训练（QAT）](#7-第三步量化感知训练qat)
8. [第四步：精度对比（Compare）](#8-第四步精度对比compare)
9. [第五步：部署导出（ONNX + JSON）](#9-第五步部署导出onnx--json)
10. [七种量化“配方”后端详解](#10-七种量化配方后端详解)
11. [核心源码走读：量化器长什么样](#11-核心源码走读量化器长什么样)
12. [全算子量化：为什么连“加法”都要量化](#12-全算子量化为什么连加法都要量化)
13. [实验结果与推荐配置](#13-实验结果与推荐配置)
14. [踩坑记录：我们踩过的坑，你不用再踩](#14-踩坑记录我们踩过的坑你不用再踩)
15. [总结与下一步](#15-总结与下一步)

---

## 1. 先讲一个故事：为什么要把神经网络“变小”？

### 1.1 一个生活化的场景

想象你训练了一个超级厉害的“看图片识物体”的 AI 模型。它在你的游戏电脑上跑得很欢快，每秒能看 100 张图，识别准确率 95%。

现在老板让你把这个模型装到一辆**自动驾驶汽车**里，或者一部**手机**里，甚至一个**摄像头**里。

问题来了：

- 你的游戏电脑有一块昂贵的 NVIDIA 显卡（GPU），功耗 300 瓦，体积像一块砖头。
- 汽车里的芯片（NPU/TPU）只有指甲盖大小，功耗 5 瓦，内存可能只有 2GB。
- 你的模型有 257 万个参数，每个参数是 32 位浮点数，占 4 个字节。光权重就超过 10MB。

```
┌─────────────────┐         ┌─────────────────┐
│   游戏电脑 GPU   │   vs    │   车载 NPU 芯片   │
│                 │         │                 │
│  体积: 砖头大小   │         │  体积: 指甲盖    │
│  功耗: 300 瓦    │   →→→   │  功耗: 5 瓦     │
│  内存: 16 GB    │         │  内存: 2 GB     │
│  算力: 超强      │         │  算力: 精打细算   │
└─────────────────┘         └─────────────────┘
        你的模型要从左边搬到右边 → 必须"减肥"
```

如果直接把游戏电脑上的模型塞进去，要么：

1. **跑不动**：芯片算力不够，识别一张图要 10 秒，车都撞上了还没识别出来。
2. **装不下**：内存太小，模型加载不进去。
3. **耗不起**：功耗太高，手机电池半小时就耗光。

### 1.2 解决思路：模型压缩

科学家想出了很多办法给模型“减肥”。最常见的就是**量化（Quantization）**。

量化的核心思想非常简单：**把模型里的数字，从“高精度”换成“低精度”**。

就像你把一张 4K 超高清照片，压缩成 1080P 普通照片。细节少了一点，但文件小了 4 倍，加载速度也快了。

在神经网络里：

- **浮点数（FP32）**：每个数字占 32 位，精度高，像 4K 照片。
- **int8 整数**：每个数字只占 8 位，精度低，像 1080P 照片，但体积缩小到 1/4。

### 1.3 再来几个生活类比

| 类比 | FP32 浮点 | int8 量化 | 结果 |
|------|----------|----------|------|
| **酒店星级** | 精确评分 4.73 星 | 四舍五入到 5 星 | 有点误差，但够用了 |
| **人民币面额** | 精确到分（1.2345 元） | 只保留到角（1.2 元） | 日常交易没影响 |
| **音乐音质** | 无损 FLAC（50MB） | 高清 MP3（5MB） | 耳朵几乎听不出差别 |
| **体重秤** | 精确到克（65.437 kg） | 精确到公斤（65 kg） | 体检报告照样用 |

量化后的模型，权重和中间计算结果都用 int8 表示。计算时芯片可以直接用整数运算单元，速度飞快，功耗极低。

**代价**：精度会有一点点下降。但好的量化技术能让下降小到 0.1% 甚至反超原模型。

这就是本项目要做的事情：**把 YOLO26（一个流行的目标检测网络）从 FP32 变成 int8，同时尽量保持精度**。

---

## 2. 什么是量化？用“四舍五入”来理解

### 2.1 最直观的类比：用“停车场车位”来理解

量化本质上就是**把连续的数字映射到离散的格子里**。

想象你把一辆车停进一个停车场。这个停车场有固定数量的车位，每个车位大小一样。

```
浮点数世界（无限精度）          量化后世界（只有 256 个格子）

   12.3456                       12
      ↓                           ↓
   连续路沿                       格子 0   格子 1   格子 2  ...  格子 255
   ──────────→                   [0]    [1]    [2]       [255]
                                    │      │      │         │
                                   0.0    0.1    0.2  ...  25.5
                                    ←──── scale = 0.1 ────→

汽车实际长度 12.3456 → 找到最近的格子 → 停在 123 号格子 → 实际占用 12.3
```

**比喻对应表**：

| 概念 | 停车场比喻 | 含义 |
|------|-----------|------|
| **scale（步长）** | 每个车位的宽度 | 一格代表多少数值 |
| **zero_point（零点）** | 0 号车位对应的真实位置 | 数值 0 落在哪个格子 |
| **Qn / Qp** | 最小/最大车位编号 | 格子编号的边界（如 0~255） |
| **量化误差** | 车长不是车位宽度的整数倍 | 四舍五入损失的精度 |

### 2.2 再看一个比喻：用“温度计”理解 scale 和 zero_point

想象你有一支老式水银温度计，上面只有刻度线，没有数字。

```
    有符号温度计（signed）            无符号温度计（unsigned）

    │ -128  -64   0   64  127        │    0   64  128  192  255
    │   │    │    │    │    │        │    │    │    │    │    │
    └──┴────┴────┴────┴────┘        └───┴────┴────┴────┴────┘
         ←──冰点在中间──→                  ←──冰点在最左边──→
    zero_point = 128                     zero_point = 0
    （-1°C 对应 127 号刻度）              （0°C 对应 0 号刻度）
```

- **有符号量化**：像摄氏温度计，有正有负，0°C 在中间。
- **无符号量化**：像华氏温度计（但只取正半轴），从 0 开始往上。

### 2.3 量化公式

用数学语言说，量化分两步：

```
# 1. 把浮点数映射到整数格子（找车位）
q = round(x / scale) + zero_point

# 2. 把整数限制在有效范围内（不能停到停车场外面）
q = clamp(q, 0, 255)

# 反量化：把整数变回浮点数（告诉别人你停在哪一格）
x_approx = (q - zero_point) * scale
```

`x_approx` 就是对原始 `x` 的近似。因为格子数量有限，这个近似一定有误差。

**图示：量化和反量化的往返过程**

```
  原始浮点数 x = 12.3456
       │
       ▼
  ┌─────────────┐
  │  量化:       │  round(12.3456 / 0.1) + 0 = 123
  │  x → q      │
  └─────────────┘
       │
       ▼
  整数 q = 123 (存到内存，只占 1 字节)
       │
       ▼
  ┌─────────────┐
  │  反量化:     │  (123 - 0) * 0.1 = 12.3
  │  q → x̂      │
  └─────────────┘
       │
       ▼
  近似浮点数 x̂ = 12.3  ←── 误差 = 0.0456，很小！
```

### 2.4 为什么需要两个数：scale 和 zero_point？

- **scale** 决定“一格代表多少”。比如 scale=0.1，整数 123 就代表浮点数 12.3。
- **zero_point** 决定“0 在哪里”。如果数据有正有负（比如权重），需要 zero_point 来平移，让负数也能用 0~255 表示。

**无符号量化（unsigned）**：数据都是非负的（比如图片像素 0~255），zero_point=0，直接映射到 [0, 255]。

**有符号量化（signed）**：数据有正有负（比如权重），映射到 [-128, 127]。

```
  unsigned 映射（只处理正数）          signed 映射（处理正负数）

  浮点  0.0 ─────→ 整数 0            浮点 -1.0 ─────→ 整数 127
  浮点 12.3 ─────→ 整数 123           浮点  0.0 ─────→ 整数 128
  浮点 25.5 ─────→ 整数 255           浮点 +1.0 ─────→ 整数 129

       [0, 255] 全部用来表示正数            [-128, 127] 正负各一半
```

### 2.5 伪量化（Fake Quantization）：训练时的“模拟考”

在训练时，我们不能真的把权重变成整数（那样梯度就断了，没法训练了）。

所以 PyTorch 里用的是**伪量化**：

```
  真实训练路径（正常情况）          伪量化训练路径（QAT）

  输入 x ──→ [卷积计算] ──→ 输出      输入 x ──→ [量化] ──→ [反量化] ──→ [卷积] ──→ 输出
                                              │        │
                                              ▼        ▼
                                            整数 8bit  浮点数（但被"污染"过）
```

1. 把权重先量化到 int8，再反量化回浮点数。
2. 用这个“被污染过”的浮点数继续前向计算。
3. 反向传播时，用一个近似梯度（叫 STE，Straight-Through Estimator）传回去。

**比喻**：就像高考前的模拟考。模拟考的卷子（量化噪声）比高考难一点，你练着练着，真正高考（部署推理）就不怕了。

---

## 3. 整个项目在做一件什么事？

本项目给 **YOLO26**（一个目标检测网络，能同时在图片里画框标出物体位置和类别）提供了完整的量化流水线。

流水线就像工厂的生产线，一共四站。我们用**做菜**来打个比方：

```
            ┌─────────────────────────────────────────────────────────┐
            │                    量化部署流水线                          │
            └─────────────────────────────────────────────────────────┘

  第 1 站          第 2 站           第 3 站            第 4 站
┌──────────┐   ┌──────────┐    ┌──────────┐     ┌────────────────┐
│  浮点训练  │ → │ PTQ 校准  │ →  │ QAT 微调  │  → │ 精度对比 + 部署  │
│  (备菜)   │   │ (调味)   │    │ (试味)   │     │ (装盘上桌)      │
└──────────┘   └──────────┘    └──────────┘     └────────────────┘
     ↑                                                │
     └────────── 如果味道不对（精度不够），回炉重造 ────────┘

  比喻说明：
  - 备菜    = 正常训练一个浮点模型（准备食材）
  - 调味    = 用少量图片统计量化参数（尝味道）
  - 试味    = 带着量化噪声继续训练（反复调味）
  - 装盘上桌 = 生成 ONNX + JSON（给板端芯片吃）
```

| 阶段 | 人话解释 | 做了什么 | 用时占比 |
|------|----------|----------|----------|
| **Float 训练** | 先按正常方式训练一个浮点模型 | 和平时训练神经网络一模一样，产出基线精度 | 大头 |
| **PTQ 校准** | 用少量图片“标定”量化参数 | 统计激活值的范围，算出每一层用多少 scale/zp | 几分钟 |
| **QAT 训练** | 带着量化噪声继续训练 | 模型学会适应 int8 计算的误差，精度回升 | 中等 |
| **对比 + 导出** | 看看掉了多少精度，打包部署文件 | 生成 ONNX 模型 + JSON 量化参数表 | 几秒 |

### 3.1 支持的任务

不只是目标检测，本项目一共支持 **6 个视觉任务**：

| 任务 | 做什么 | 输出 |
|------|--------|------|
| detect | 目标检测 | 画框 + 类别标签 |
| seg | 实例分割 | 框 + 像素级轮廓（mask） |
| pose | 姿态估计 | 框 + 人体 17 个关键点 |
| cls | 图像分类 | 整张图属于哪一类 |
| obb | 旋转框检测 | 带角度的斜框（用于卫星图） |
| depth | 深度估计 | 每个像素离相机多远 |

除了 YOLO26 系列，还额外支持 MobileNetV3 和 CIFAR-CNN 两个小网络，方便你在小数据集上快速实验。

### 3.2 支持的量化“配方”：7 个后端

量化不是只有一种做法。本项目实现了 **7 种不同的量化算法**（后端），就像同一个菜可以用 7 种不同的调料来做：

- `lsqplus_v1`（默认推荐）
- `lsqplus_v2`
- `lsq_v1`
- `lsq_v2`
- `minmax`
- `dorefa`
- `pact`

后面我们会详细讲它们的区别。

---

## 4. 准备数据：没有数据，一切都是空谈

### 4.1 数据集从哪里来？

深度学习模型的训练需要两样东西：**图片** 和 **标注**。

本项目的数据集分两类：

**自动下载（新手友好）**：

运行脚本时，如果本地没有数据，程序会自动从 ultralytics 官方仓库下载：

| 任务 | 数据集 | 大小 | 说明 |
|------|--------|------|------|
| detect / seg / pose | coco8 / coco8-seg / coco8-pose | 几十张 | 官方提供的极小数据集，能跑通流程 |
| obb | dota8-multispectral | 几十张 | 多光谱卫星图（10 个通道） |
| depth | depth8 | 几十张 | RGB 图 + 16-bit PNG 深度图 |
| cls / mobileNet / cifarCNN | CIFAR-10 / ImageNet10 | 自动下载 | 分类标准数据集 |

**自己做数据（真实场景）**：

如果你要检测自己工厂的零件、自己田里的害虫，就需要自己收集数据和标注。

### 4.2 数据采集

数据采集就是用摄像头或从网上收集图片。采集时需要注意：

1. **覆盖场景**：白天、晚上、晴天、阴天、近景、远景。模型没见过的场景会瞎猜。
2. **多样性**：同一类物体要有不同角度、不同大小、不同遮挡程度。
3. **分辨率一致**：尽量保持和目标设备一致的输入尺寸（比如 640×640）。

### 4.3 数据标注

光有图片不够，你需要告诉模型“这张图里有一只猫，猫在左下角”。这个过程叫**标注**。

**标注的比喻**：就像老师批改作业。学生（模型）做完题（预测），老师（标注）告诉他对不对、错在哪。没有老师的批改，学生永远学不到正确的答案。

最常用的标注格式是 **YOLO 格式**：

```
# 每个 .txt 文件对应一张图，每行一个物体
class_id  center_x  center_y  width  height
```

所有数字都是 0~1 之间的相对坐标。例如：

```
0 0.5 0.5 0.3 0.4
```

表示：类别 0 的物体，中心在图的正中间，宽占图的 30%，高占 40%。

**图示：一张图片对应的标注**

```
  图片 (640×480 像素)
  ┌──────────────────────────────┐
  │                              │
  │       ┌────────────┐         │
  │       │            │         │   ← 标注框: 0 0.5 0.5 0.3 0.4
  │       │   一只猫    │         │      (类别0, 中心(0.5, 0.5), 宽30%, 高40%)
  │       │            │         │
  │       └────────────┘         │
  │                              │
  └──────────────────────────────┘

  对应的标注文件 cat.txt 内容:
  0 0.5 0.5 0.3 0.4
```

**不同任务的标注格式**：

| 任务 | 标注内容 | 示例 |
|------|---------|------|
| detect | 矩形框 | `0 0.5 0.5 0.3 0.4` |
| seg | 多边形顶点 | `0 0.1 0.2 0.3 0.2 0.3 0.4 ...` |
| pose | 框 + 17 个关键点 | `0 0.5 0.5 0.3 0.4 0.1 0.2 2 0.3 0.4 2 ...` |

**标注工具推荐**：

- [LabelImg](https://github.com/tzutalin/labelImg)（老牌，简单易用）
- [CVAT](https://cvat.org/)（在线，支持团队标注）
- [Label Studio](https://labelstud.io/)（功能最全）

### 4.4 COCO mini：用 1/100 数据快速验证

如果你不想长时间等待 118000 张 COCO 训练完成（约 20GB），本项目提供了一个 `coco_mini_prepare.py` 脚本：

```bash
python3 script/coco_mini_prepare.py
```

它会从完整 COCO 里每隔 100 张取 1 张，生成一个只有 **1183 张训练图 + 50 张验证图** 的小数据集。

用途：
- 快速验证你的代码能不能跑通
- 快速对比 7 个量化后端的优劣
- 调试超参数

**注意**：因为数据量小，测出来的绝对精度（比如 mAP）不代表真实水平，只能用来**横向对比**不同配置。

### 4.5 数据加载器的配置

本项目的数据加载有几个硬性约定（从源码中总结）：

- 数据拼接用 `torch.concat`，**不能用**量化版本的 `QuantConcat`（数据加载不是模型计算，不该量化）。
- `evaluate` 函数接收的是原始 `data` 字典，不是内部转换后的对象。
- 验证时每次评估结束后会自动可视化前 30 张图，保存到 `fvisualize/` 目录，方便肉眼检查。

---

## 5. 第一步：浮点训练（Float Training）

### 5.1 什么是浮点训练？

这就是你最熟悉的“正常训练”。

模型里的所有数字都是 32 位浮点数（FP32）。权重、偏置、中间特征图，全都是高精度的。

本项目中的 `float_train()` 函数负责这一阶段。

### 5.2 模型从哪里开始？

**最好的起点是预训练权重**。

YOLO26 官方在 GitHub 上发布了预训练好的 `.pt` 文件（`yolo26n.pt`、`yolo26s.pt` 等）。本项目会在你第一次运行时自动下载：

```python
# 自动下载逻辑（networks_yolo26-detect.py）
# 如果本地没有这个文件，会调用 attempt_download_asset 从 GitHub Releases 下载
```

预训练权重已经学会了识别常见物体（人、车、狗、猫……）。在你的数据集上继续训练，可以省大量时间。

### 5.3 模型结构

YOLO26 的网络结构分三大块：

```
输入图片 (3×640×640)
    ↓
Backbone（主干网络）← 负责“看”图片，提取特征
    ↓
Neck（脖子）      ← 负责融合不同尺度的特征
    ↓
Head（检测头）    ← 负责“说”出框的位置和类别
    ↓
输出：框坐标 + 类别概率
```

主干网络由 `Conv`、`C3k2`、`SPPF`、`C2PSA`（Attention）等模块组成。本项目中的实现和 ultralytics 官方完全一致，可以直接加载官方权重。

### 5.4 训练技巧

- **EMA（指数移动平均）**：不只保存当前权重，还保存一个“平滑版”的权重。验证时用平滑版，更稳定。
- **best 后缀**：训练过程中验证精度最高的模型会保存为 `*_best.pth`，同时维护 `*_last.pth`（最新）。
- **同步导出 ONNX**：每次保存 best 权重时，会尝试同时导出 ONNX。失败也不影响训练。
- **resume 接续训练**：如果训练中断，加 `--resume` 可以从 `*_last.pth` 恢复，继续上次的位置。

### 5.5 训练命令示例

```bash
# 完整跑 detect 任务的 4 个阶段
python3 networks_yolo26-detect.py --model yolo26n --stage all --quant lsqplus_v1

# 只跑浮点训练
python3 networks_yolo26-detect.py --stage float

# 只训 1 个 epoch、每个 epoch 只跑 1 个 batch（冒烟测试）
python3 networks_yolo26-detect.py --float-epochs 1 --max-train-batches 1 --max-eval-batches 1
```

---

## 6. 第二步：训练后量化校准（PTQ）

### 6.1 PTQ 在做什么？

浮点模型训练好了。现在我们要把它变成 int8。

最简单的办法是：**直接把权重四舍五入到 int8**。

但这样做通常效果不好，因为模型里的**激活值**（每层输出的特征图）范围很广，如果 scale 选得不好，大量数字会被挤到同一个格子里，信息严重丢失。

**比喻**：PTQ 就像**给新买的体重秤调零**。

```
  体重秤（量化器）刚买回来，指针不归零：
  
  空秤状态显示: 2 kg  ← 这是偏差！
  
  校准过程:
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ 站上空秤  │ →   │ 记录偏差  │ →   │ 调整旋钮  │
  │ 显示 2kg  │     │ 偏差=2kg │     │ 显示=0kg │
  └──────────┘     └──────────┘     └──────────┘
  
  在神经网络里：
  - 空秤 = 不输入图片时的激活值
  - 偏差 = 实际激活值范围与量化范围的偏差
  - 调整旋钮 = 计算 scale 和 zero_point
```

PTQ（Post-Training Quantization，训练后量化）的思想是：

> 用少量真实图片（叫**校准集**）跑一遍模型，统计每一层激活值的实际范围，然后算出最优的 scale 和 zero_point。

**图示：校准前后对比**

```
  校准前（scale 乱选）                校准后（scale 精准）
  
  激活值分布: 0 ~ 10                  激活值分布: 0 ~ 10
  scale = 0.5 (乱选)                  scale = 0.039 (精准)
  
  格子使用率:                         格子使用率:
  [0][1][2][3]...[255]                [0][1][2][3]...[255]
   ↑  ↑  ↑  ↑        ↑                 ↑  ↑  ↑  ↑        ↑
   │  │  │  │        │                 │  │  │  │        │
  0  5  10  15     127.5              0  1  2  3       10
  
  只用了 20 个格子表示 0~10            用满 256 个格子表示 0~10
  → 精度浪费！                        → 精度最大化！
```

### 6.2 校准数据怎么选？

校准数据不需要很多。通常 **20~100 张**就够了。

选法：
- 从训练集里随机抽（最常见）。
- 必须覆盖模型会见的各种场景。
- 不能和测试集重合（否则等于作弊）。

本项目默认用 **20 个 batch** 做校准（命令行参数 `--calibration-batches` 可调）。

### 6.3 校准的具体过程

```python
# 伪代码流程
def PTQ_calibration():
    # 1. 加载浮点模型权重
    float_model = FloatYOLO26().cuda()
    load_checkpoint(float_model, "yolo26n_best.pth")

    # 2. 创建量化版模型，把浮点权重拷贝过去
    quant_model = QuantYOLO26().cuda()
    copy_float_to_quant(float_model, quant_model)

    # 3. 用校准集跑前向，统计激活范围
    for batch in calibration_loader:
        quant_model(batch)  # 内部自动更新 scale/zp

    # 4. 冻结量化参数，保存结果
    freeze_batch_init(quant_model)
```

### 6.4 为什么 PTQ 后精度可能掉很多？

因为 PTQ **只校准量化参数，不更新模型权重**。

模型权重原本是为浮点计算优化的。现在突然换成 int8 计算，有些权重可能对量化特别敏感，一下就从“好权重”变成“坏权重”。

如果 PTQ 后精度掉得不多（比如 mAP 掉 1% 以内），可以直接用。

如果掉得多，就需要进入下一步：**QAT**。

---

## 7. 第三步：量化感知训练（QAT）

### 7.1 QAT 的核心思想

QAT（Quantization-Aware Training）就是**在训练时假装自己在用 int8 计算**。

具体做法：

1. 前向传播时，权重和激活都经过伪量化（量化后再反量化）。
2. 模型看到的不是干净的浮点数，而是“被量化污染过”的数。
3. 反向传播时，梯度通过 STE（直通估计器）传回去，模型学会调整权重来对抗量化噪声。

**比喻**：就像你打靶时，靶子（浮点模型）是静止的。现在让靶子随机抖动（量化噪声），你练着练着，就学会了在抖动的靶子上也能打中十环。

**再一个比喻**：PTQ 是“考试前突击背答案”，QAT 是“平时就按考试标准做题”。后者当然更扎实。

### 7.2 STE（Straight-Through Estimator）：梯度怎么传？

量化函数里有一个 `round()`（四舍五入）。这个函数在数学上是不可导的（阶梯状）。

```
  round() 函数的形状（阶梯状，不可导）:
  
  输出
    │     ┌──── 3
    │     │
    │   ┌─┘     2
    │   │
    │ ┌─┘       1
    │ │
    └─┴──────────────→ 输入
      0   1   2   3
  
  在 x=1.5 处，左边导数=0，右边导数=0，无法训练！
```

怎么办？PyTorch 用了一个“作弊”技巧：

> **前向用 `round()`，反向把 `round()` 当成恒等函数（直接传梯度）。**

```
  STE 前向（假装量化）:              STE 反向（假装没量化）:
  
  x=1.7 ──→ round() ──→ y=2         梯度 ──→ 恒等函数 ──→ 梯度直通
           （真量化）                        （假装没量化）
```

这就是 STE。虽然不完全正确，但在实践中非常有效。

**比喻**：就像玩“传话游戏”。第一个人（前向）把话传错了（量化误差），但后面的人（反向）假装没听到错误，继续按原话传（梯度直通）。这样整条链路能继续运转。

### 7.3 QAT 的训练流程

```python
# 伪代码
def QAT_training():
    # 1. 加载 PTQ 校准好的模型
    quant_model = QuantYOLO26().cuda()
    load_checkpoint(quant_model, "ptq_lsqplus_v1.pth")

    # 2. 解冻量化参数（让某些后端可以继续微调 scale）
    reset_quantizer_states(quant_model)

    # 3. 正常训练，但学习率通常比浮点训练低 10 倍
    for epoch in range(qat_epochs):
        for batch in train_loader:
            output = quant_model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

QAT 的 epoch 数通常是浮点训练的 1/5。比如浮点训了 100 轮，QAT 训 20 轮就够了。

### 7.4 为什么 PTQ 参数必须用来初始化 QAT？

这是一个硬性约束（从 project_memory 中学到的重要经验）：

> 如果 QAT 从零开始随机初始化量化参数，前几轮会特别不稳定，甚至梯度爆炸。

用 PTQ 校准好的 scale/zp 作为起点，QAT 只需要“微调”，而不是“从零学习”。

---

## 8. 第四步：精度对比（Compare）

训练完了，总得知道量化后的模型还灵不灵吧？

`compare_precision()` 函数会同时加载浮点模型和 QAT 模型，在验证集上各跑一遍，输出：

```
[Compare] Float mAP50: 0.5814  |  QAT mAP50: 0.5807  |  Delta: -0.0007
```

Delta（差值）就是量化带来的精度损失。理想情况下希望 Delta 接近 0 甚至为正。

**什么时候算成功？**

- 目标检测：Delta < 0.01（mAP50 掉不到 1%）就算很好。
- 分类任务：Delta < 0.5% 就算很好。

---

## 9. 第五步：部署导出（ONNX + JSON）

### 9.1 部署需要什么文件？

板端芯片（NPU/TPU）需要两个东西：

| 文件 | 格式 | 用途 |
|------|------|------|
| 模型结构 | ONNX | 描述网络的拓扑结构（哪层连哪层） |
| 量化参数 | JSON / .pth | 每层的 scale 和 zero_point |

ONNX 是一个通用的神经网络交换格式，几乎所有推理框架都支持。

### 9.2 导出流程

```python
# 伪代码
def save_quant_outputs(model, prefix):
    # 1. 收集量化参数
    quant_params = collect_quant_params(model)
    save_json(f"{prefix}_quant_params.json", quant_params)

    # 2. 构建干净的浮点参考模型
    float_model = build_float_model(model)

    # 3. 导出 ONNX
    torch.onnx.export(float_model, dummy_input, f"{prefix}_float.onnx")

    # 4. 可选：用 onnxsim 简化 ONNX
    onnxsim.simplify("model.onnx")
```

### 9.3 为什么 ONNX 里不含量化节点？

注意：本项目导出的 ONNX 是**纯浮点图**，不含 `QuantizeLinear` / `DequantizeLinear` 节点。

为什么？

因为板端 NPU 有自己的 PTQ 工具链。你给它一个浮点 ONNX，它自己会用校准数据重新算 scale/zp。**你导出的 JSON 量化参数只是"参考值"**——板端最终部署时会**重新计算**，以板端 PTQ 工具算出的为准；JSON 只是方便你在编译前把"训练侧算出的"和"板端算出的"做一次交叉对比，看看差异是否在可接受范围。

> **记住**：JSON 里的 scale / zero_point ≠ 部署最终用的参数；**嵌入式板子的 PTQ 工具会自动重新算 scale 和 zero_point**，导出的 JSON 仅供你交叉验证。

如果你需要保留 QAT 训练时的截断行为（比如某些组合下权重依赖 clip），可以用 `build_float_model(..., use_clip=True)`，导出带 `Clip` 节点的 ONNX。

### 9.4 输入形状固定

本项目导出的 ONNX 输入形状是固定的 `[1, C, H, W]`：

- detect/seg/pose/obb/depth: `[1, 3, 640, 640]`（obb 是 `[1, 10, 640, 640]`，因为多光谱有 10 个通道）
- cls: `[1, 3, 224, 224]`

没有动态维度（dynamic_axes），这样在可视化工具（如 Netron）里打开，每一层的尺寸都是确定的数字，而不是 `?`。

---

## 10. 七种量化“配方”后端详解

现在我们来逐个介绍 7 个量化后端。它们的核心区别在于：**scale 和 zero_point 是怎么算出来的**。

我们用**量体温**来打比方。不同后端就像不同的量体温方式：

```
  7 种量化后端 = 7 种量体温的方式

  ┌─────────────────────────────────────────────────────────┐
  │  MinMax: 直接用体温计读数（看实际范围）                    │
  │  LSQ: 学一种"快速读数法"（学 scale）                       │
  │  LSQ+: 不仅学读数法，还学"怎么调整零点"（学 scale + beta） │
  │  PACT: 先用手摸额头判断发烧，再用体温计（先截断再量化）     │
  │  DoReFa: 新版改用线性对称刻度尺（权重） + 非对称标准公式（激活）│
  └─────────────────────────────────────────────────────────┘
```

| 后端 | 名称 | 对称性 | 核心机制 | 推荐度 |
|------|------|--------|----------|--------|
| `lsqplus_v1` | LSQ+ | 非对称 | 学 scale + beta（零点偏移） | ⭐⭐⭐ 默认推荐 |
| `lsqplus_v2` | LSQ+ V2 | 非对称 | 同上，初始化方式略有不同 | ⭐⭐ 对照实验 |
| `lsq_v1` | LSQ | 对称 | 只学 scale | ⭐⭐ 轻量选择 |
| `lsq_v2` | LSQ V2 | 对称 | 同上，s 初始化为 1 | ⭐⭐ 对照实验 |
| `minmax` | MinMax | 非对称 | 直接统计 min/max | ⭐⭐⭐ baseline |
| `dorefa` | DoReFa（新版） | 激活非对称 / 权重对称 | 激活走标准 scale+beta 公式，权重线性对称网格（maxvalue/Qp 步长） | ⭐⭐⭐ 稳定可用 |
| `pact` | PACT | 非对称 | 学 clip 阈值 alpha | ⭐⭐ 部分场景好 |

### 10.1 LSQ / LSQ+：可学习的 Scale

**LSQ（Learned Step Size Quantization）** 的核心思想是：

> 把 scale 当成一个可学习的参数，和权重一起用梯度下降优化。

传统 PTQ 的 scale 是统计出来的固定值。LSQ 认为，这个固定值不一定最优，应该让模型自己学。

**LSQ+** 更进一步：不只学 scale，还学 **beta**（零点偏移）。

对于非负激活（SiLU 输出 ≥ 0），LSQ+ 可以学到一个合适的 beta，让量化网格更好地贴合数据分布。

```python
# LSQPlus 伪代码
s = Parameter(1.0)      # 可学习步长
beta = Parameter(-1e-9) # 可学习零点偏移

q = round((x - beta) / s).clamp(Qn, Qp)
x_quant = q * s + beta
```

### 10.2 MinMax：最简单直接的统计

MinMax 不看梯度，只看统计：

```python
r_min = x.min()
r_max = x.max()
scale = (r_max - r_min) / (Qp - Qn)
zero_point = round(Qn - r_min / scale)
```

优点：简单、稳定、没有额外可学习参数。

缺点：对离群值（outlier）敏感。如果某个激活值突然飙到 1000，会把 scale 拉得很大，导致大部分正常值被挤到几个格子里。

### 10.3 PACT：学习截断阈值

PACT（Parameterized Clipping Activation）的思路是：

> 激活值里偶尔有几个极大的离群值，会搞坏量化。不如先截断（clip）到一个阈值 alpha，再量化。

alpha 是可学习的参数。模型会自己决定“多大的值应该被截断”。

```python
alpha = Parameter(10.0)
x_clipped = x.clamp(0, alpha)  # 先截断
q = round(x_clipped / scale)   # 再量化
```

### 10.4 DoReFa（新版）：线性权重网格 + 标准非对称激活

**v1 版本 DoReFa 的问题**：原始论文把权重先通过 `tanh` 非线性压缩到 [-1,1] 再量化，反量化时用 `arctanh` 变换回来。这导致两个严重后果：
1. **大权重被 tanh 压扁**：|w| 越大，tanh(w) 越接近 ±1，量化分辨率被严重压缩——对主导特征方向的大权重特别不利；
2. **激活量化强制对称**：原始实现里激活只有一个对称网格，没有 zero_point，必须配合 act_signed 才能工作，act_unsigned 一上就 PTQ=0。
3. **显存代价翻倍**：tanh/arctanh 是逐元素非线性运算，反向要保留中间张量，8GB 显存上 seg 任务被迫降 batch=4。

**新版 DoReFa 修复**（2026-10）：
- **权重量级化器**：去掉 tanh/arctanh，改为线性对称网格 `q_w = maxvalue × round(w/maxvalue × Qp) / Qp`。步长为 `maxvalue/Qp`，与 LSQ 对称权重公式一致。大权重不再被压扁。
- **激活量化器**：去掉 `/s` 归一化 + `clamp(±1)` 硬截断，改为和 LSQ+ 完全相同的标准非对称公式——`scale` 可学习，`beta` 可学习（zero_point = round(-beta/scale) 隐式表示）。act_unsigned 现在可以正常工作了。

**实验结果**（yolo26n，COCO mini）：
| 任务 | 配置 | 旧版 QAT Δ | 新版 QAT Δ |
|------|------|-----------|-----------|
| detect | per_channel + act_unsigned | −0.2185（PTQ=0） | **−0.0091** ✅ |
| seg | per_tensor + act_unsigned | −0.2906（PTQ=0） | **+0.0072** ✅ |
| pose | per_tensor + act_unsigned | −0.3118（PTQ=0） | **+0.0061** ✅ |
| detect | per_channel + act_signed | −0.0164 | **−0.0058** ✅ |

三任务全部稳定在 Δ ±0.01 以内，act_unsigned 不再是必崩配置。8GB 显存 QAT batch=8 全任务正常，不再需要单独降级。

**设计取舍说明**：新版把 DoReFa 的"创新点"（tanh 域非线性）全部移除，保留了"可学习激活尺度"这一核心思想，但用了 LSQ+ 已经验证过更稳定的非对称公式实现。严格意义上这不再是"经典 DoReFa"，而是"用 DoReFa 名字的线性后端"。保留这个后端纯粹为了让 README 里原有的 7 后端对照矩阵仍然完整；如果只追求部署精度，**lsqplus_v1 仍然是首选推荐**。

### 10.5 后端切换有多简单？

只需要改一个命令行参数：

```bash
python3 networks_yolo26-detect.py --quant lsqplus_v1   # 推荐
python3 networks_yolo26-detect.py --quant minmax       # baseline
python3 networks_yolo26-detect.py --quant pact         # 研究对比
```

所有后端的算子接口完全一致。内部通过 `importlib` 动态加载对应模块。

---

## 11. 核心源码走读：量化器长什么样

现在我们深入到代码里，看看量化器到底是怎么实现的。

### 11.1 LSQPlus 激活量化器

```python
class LSQPlusActivationQuantizer(nn.Module):
    def __init__(self, a_bits=8, all_positive=False, batch_init=20):
        super().__init__()
        self.a_bits = a_bits          # 量化位宽，默认 8
        self.all_positive = all_positive  # 是否无符号
        self.batch_init = batch_init  # 前 20 batch 做初始化

        # 确定量化范围
        if self.all_positive:
            self.Qn = 0
            self.Qp = 255            # 2^8 - 1
        else:
            self.Qn = -128           # -2^(8-1)
            self.Qp = 127            # 2^(8-1) - 1

        # 可学习参数
        self.s = nn.Parameter(torch.ones(1))       # scale
        self.beta = nn.Parameter(torch.tensor([-1e-9]))  # 零点偏移
        self.init_state = INIT_STATE_UNINIT  # 0 = 未初始化
```

**前向传播（核心逻辑）**：

```python
    def forward(self, activation):
        # 前 20 个 batch：用 EMA 统计 scale 和 beta
        if self.init_state < self.batch_init:
            with torch.no_grad():
                cur_min = activation.min()
                cur_max = activation.max()
                cur_s = (cur_max - cur_min) / (self.Qp - self.Qn)
                cur_beta = cur_min - cur_s * self.Qn

                # EMA 更新
                self.s = 0.9 * self.s + 0.1 * cur_s
                self.beta = 0.9 * self.beta + 0.1 * cur_beta
            self.init_state += 1

        # 真正的量化（用自定义 Function，支持 STE）
        q_a = ALSQPlus.apply(activation, self.s, self.g, self.Qn, self.Qp, self.beta)
        return q_a
```

### 11.2 ALSQPlus Function（自定义梯度）

```python
class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        # 1. 量化
        q = Round.apply((weight - beta) / alpha).clamp(Qn, Qp)
        # 2. 反量化
        w_q = q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        # STE：在 [Qn, Qp] 区间内梯度直通，区间外梯度为 0
        # 同时计算 alpha 和 beta 的梯度
        ...
```

### 11.3 量化卷积层

```python
class QuantConv2d(nn.Conv2d):
    def __init__(self, ..., a_bits=8, w_bits=8, all_positive=False, per_channel=False):
        super().__init__(...)
        # 两个量化器：一个管激活输入，一个管权重
        self.activation_quantizer = LSQPlusActivationQuantizer(a_bits, all_positive)
        self.weight_quantizer = LSQPlusWeightQuantizer(w_bits, per_channel=per_channel)

    def forward(self, input):
        # 激活伪量化
        quant_input = self.activation_quantizer(input)
        # 权重量化
        quant_weight = self.weight_quantizer(self.weight)
        # 用量化后的值做卷积
        return F.conv2d(quant_input, quant_weight, self.bias, ...)
```

### 11.4 后端加载机制

```python
# quantization/__init__.py
_BACKEND_MODULES = {
    "lsqplus_v1": "quantization.lsqplus_quantize_V1",
    "minmax": "quantization.minmax",
    ...
}

def load_quant_backend(method):
    return importlib.import_module(_BACKEND_MODULES[method])
```

网络脚本里这样用：

```python
backend = quant_pkg.load_quant_backend("lsqplus_v1")
QuantConv2d = backend.QuantConv2d
QuantAdd = backend.QuantAdd
# ... 然后用这些类搭建网络
```

---

## 12. 全算子量化：为什么连“加法”都要量化

### 12.1 教科书 vs 真实板端

很多入门教程只教你量化 `Conv + ReLU`。这在学术研究里够用了，但**真实部署时完全不够**。

**比喻**：教科书做法就像"只给汽车发动机做保养"，真实板端是"整辆车的每个零件都要适配"。

```
  教科书做法（只量化 Conv+ReLU）:         真实板端（全算子量化）:
  
  [输入FP32] → Conv量化 → [FP32] → ReLU量化 → [FP32] → Add(FP32!) → ...
                          ↑                      ↑
                       只有这里量化            Add 也是 FP32!
  
  [输入int8] → Conv量化 → [int8] → ReLU量化 → [int8] → Add(int8!) → Concat(int8!) → ...
                          ↑                      ↑                ↑
                       全量化                    Add 也量化!      Concat 也量化!
```

真实 NPU 的每一层输入输出都是 int8。包括：

- `Add`（残差连接）
- `Concat`（特征拼接）
- `MaxPool`（下采样）
- `SiLU / Sigmoid / Softmax`（激活函数）
- `Mul / Div`（逐元素乘除）

如果你训练时只量化 Conv，其他都算 FP32，那么部署时这些算子的量化误差是**模型从来没见过**的——结果就是训练时精度很高，上板后直接崩溃。

### 12.2 本项目的全算子覆盖

7 个后端每个都实现了约 15 个量化算子：

| 类别 | 算子 | 量化点 | 生活比喻 |
|------|------|--------|----------|
| 带权重 | QuantConv2d, QuantConvTranspose2d, QuantLinear, QuantMatMul | 输入激活 + 权重 | 主厨炒菜（锅+食材都要标准化） |
| 逐元素 | QuantAdd, QuantSub, QuantMultiply, QuantDiv | 两输入各自量化 | 两道菜混合（各自标准化后再混合） |
| 拼接 | QuantConcat, QuantCat | 每输入分支独立量化 | 把两盘菜拼成一盘（各自装盘后再拼） |
| 池化/激活 | QuantMaxPool, QuantSiLU, QuantSigmoid, QuantReLU, QuantSoftmax | 输入量化 | 最后调味（标准化后再调味） |

**为什么 Concat 要每分支独立量化？**

因为板端拼接时，两个输入张量本来就有各自的 scale。硬件会先把它们 requant（重新量化）到同一个 scale，再拼接。训练时独立量化，才能让模型学会适应这个 requant 噪声。

**比喻**：就像把两杯不同温度的水倒进一个杯子。直接倒（FP32 concat）没问题，但如果两杯水都是按各自温度计的刻度（不同 scale）量的，倒之前必须先换算到同一个刻度（requant），否则混合后的温度就错了。

**为什么 Add 要两个输入各自量化？**

两个输入的 scale 可能完全不同。硬件做 int8 加法前必须先对齐 scale。训练时各自量化，模型才会对对齐误差鲁棒。

**比喻**：就像两个人用不同单位的尺子量东西。一个用厘米，一个用英寸。要把两个长度加起来，必须先换算到同一个单位（对齐 scale），否则加出来的结果是错的。

---

## 13. 实验结果与推荐配置

### 13.1 实验设置

- 数据集：COCO mini（1/100 抽样，detect 1183 训练图 / 50 验证图）
- 模型：YOLO26n（最小尺度）
- 浮点训练：50 epoch
- PTQ 校准：20 batch
- QAT 训练：10 epoch
- 量化位宽：int8（a_bits=8, w_bits=8）

### 13.2 核心结论

**推荐配置：lsqplus_v1 + per_channel + act_unsigned（激活无符号，权重有符号）**

| 任务 | Float 基线 | QAT mAP | Delta | 人话 |
|------|-----------|---------|-------|------|
| detect | 0.5814 | 0.5807 | **-0.0007** | 几乎无损 |
| seg | 0.5001 | 0.4980 | **-0.0021** | 几乎无损 |
| pose | 0.4839 | 0.4876 | **+0.0037** | 反而更好！ |

三任务全部无损或近无损！pose 任务甚至反超了浮点模型（+0.0037），这在量化领域是很少见的好结果。

**可视化对比**：

```
  detect 任务精度对比:
  
  mAP50
   0.60 ┤
   0.58 ┤  ████ Float    ████ QAT(lsqplus_v1)
   0.56 ┤  ████ 0.5814   ████ 0.5807
   0.54 ┤  ████          ████
   0.52 ┤  ████          ████
   0.50 ┤  ████          ████
        └─────────────────────────────
           两条柱子几乎一样高 → 量化基本无损！
```

### 13.3 重要发现：unsigned 激活必须配非对称后端

| 后端类型 | 配 unsigned 激活 | 配 signed 激活 |
|----------|------------------|----------------|
| 非对称（lsqplus_v1/v2, pact, **新版 dorefa**） | ✅ 稳定，无损 | ✅ 也稳定 |
| 对称（lsq_v1/v2, minmax） | ❌ 严重掉点（-0.18 ~ -0.37） | ✅ 稳定 |

**原因**：对称后端的 zero_point 被迫在边界外，unsigned 激活会被严重截断。这是一个系统性的问题，换数据集也救不了。**2026-10 修复**：DoReFa 激活量化器已改为非对称公式，配 unsigned 激活现在稳定（三任务 QAT Δ ±0.01 以内）。

### 13.4 权重必须保持有符号

`--w-all-positive`（权重无符号）在所有后端上都会导致 **mAP = 0.0000**，完全崩塌。

因为卷积权重是零均值的（有正有负），强制无符号会把所有负权重截断为 0，破坏符号平衡。

### 13.5 不同场景的配置推荐

| 场景 | 推荐配置 | 命令 |
|------|----------|------|
| 精度优先（默认） | lsqplus_v1 + per_channel + act_unsigned | `python3 networks_yolo26-detect.py --quant lsqplus_v1` |
| 部署友好（硬件只支持 per-tensor） | lsqplus_v1 + per_tensor + act_unsigned | `--no-per-channel` |
| 快速 baseline | minmax + signed | `--quant minmax --no-all-positive` |
| 敏感层保浮点 | lsqplus_v1 + mixed | `--mixed-quant` |

---

## 14. 踩坑记录：我们踩过的坑，你不用再踩

### 14.1 PTQ mAP = 0

**现象**：PTQ 后精度直接变 0。

**根因**：激活校准范围不准 + 权重量化 3σ 饱和。激活统计少了某些算子（Add、Cat、MaxPool），导致 scale 严重偏差。

**修复**：扩展 hook 范围，把所有产生量化误差的算子都纳入统计。

### 14.2 对称后端 + unsigned 激活 = 输出爆炸

**现象**：QAT 训练时精度正常，但导出 ONNX 后 box 坐标变成 ±1e5（正常应该 ≤ 640）。

**根因**：unsigned 激活每层有硬截断 `clip(0, r_max)`。QAT 权重学会了依赖这个钳位。去掉钳位后，误差逐层复合放大。

**修复**：`build_float_model(..., use_clip=True)` 保留截断节点。

### 14.3 检测头 `_anchors` 设备不匹配

**现象**：`model.cpu()` 后前向报错，说 tensor 在 GPU 上。

**根因**：检测头懒创建的 `_anchors` 是普通 Python 属性，不是 `nn.Buffer`，所以 `.cpu()` 不会迁移它。

**修复**：用 `register_buffer(..., persistent=False)` 注册。

### 14.4 全零 dummy 输入导致 NaN

**现象**：LSQ / PACT 初始化时 scale 变成 0，后续所有输出都是 NaN。

**根因**：dummy forward 用了 `torch.zeros`，所有输出一样，标准差为 0。

**修复**：dummy 输入改用 `torch.randn() * 0.1`。

### 14.5 seg + dorefa 在 8GB 显存上 OOM（**2026-10 已修复**）

**现象**：segmentation 任务用旧版 dorefa 后端时显存溢出。

**根因**：dorefa 的 tanh/arctanh 非线性量化器 + single_mask_loss 的 einsum 操作，显存峰值太高。

**旧修复**：seg 和 dorefa 的 QAT batch size 降为 4（其他任务保持 8）。

**新修复**（2026-10）：新版 dorefa 完全移除了 tanh/arctanh 非线性，权重改为线性对称网格，激活改为标准非对称公式。显存峰值与 lsqplus_v1 一致，**所有任务统一 batch=8，不再需要单独降级**。

### 14.6 路径里有空格导致日志写错

**现象**：`log /mini_sweep.log`（路径里有空格）变成了在当前目录创建 `log` 文件和 `/mini_sweep.log`。

**修复**：路径统一用 `log/mini_sweep.log`（没空格）。

---

## 15. 总结与下一步

### 15.1 你学到了什么？

1. **量化是什么**：把 FP32 换成 int8，让模型能在嵌入式设备上跑。
2. **完整流水线**：Float → PTQ → QAT → Compare → Deploy。
3. **数据准备**：采集、标注、校准集的选择。
4. **7 个后端**：LSQ+（推荐）、MinMax（baseline）、PACT（部分场景好）、DoReFa（新版线性网格，稳定可用，三任务 QAT Δ ±0.01 以内）。
5. **全算子量化**：不只是 Conv，Add、Concat、SiLU 都要量化。
6. **关键配置**：非对称后端配 unsigned 激活、权重必须有符号、per_channel 精度更高。

### 15.2 如果你想动手跑一遍

```bash
# 1. 安装依赖
pip3 install -r requirements.txt

# 2. 跑一个最小 detect 全流程（自动下载 coco8）
python3 networks_yolo26-detect.py --stage all --quant lsqplus_v1

# 3. 看结果
ls model/yolo26-detect/n/
```

### 15.3 下一步可以学什么？

- **混合精度量化**： sensitive 层用 int16/FP32，其他用 int8。
- **剪枝（Pruning）**：把不重要的权重直接删掉，进一步压缩模型。
- **知识蒸馏（Knowledge Distillation）**：让小模型学大模型的输出。
- **真实板端部署**：把 ONNX 和 JSON 参数喂给地平线 / 高通 / 瑞芯微的工具链。

---

## 附录 A：新手术语速查表

读文章时遇到不懂的术语，回来查这张表：

| 术语 | 一句话解释 | 生活类比 |
|------|-----------|---------|
| **量化 (Quantization)** | 把高精度数字变成低精度数字 | 4K 照片压缩成 1080P |
| **FP32** | 32 位浮点数，占 4 字节 | 无损音质的 FLAC |
| **int8** | 8 位整数，占 1 字节 | 高清 MP3 |
| **权重 (Weight)** | 模型学到的参数 | 菜谱里每种调料的用量 |
| **激活 (Activation)** | 每层网络的输出值 | 炒菜过程中锅里的温度 |
| **scale** | 量化时"一格代表多少" | 尺子上每个刻度的宽度 |
| **zero_point** | 数值 0 对应哪个格子 | 温度计的冰点位置 |
| **PTQ** | 训练后量化，只校准不训练 | 体检前称体重调秤 |
| **QAT** | 量化感知训练，带着噪声训练 | 平时就按考试标准做题 |
| **STE** | 直通估计器，让梯度穿过 round | 传话游戏假装没传错 |
| **校准 (Calibration)** | 用少量数据统计量化参数 | 新秤买回家先调零 |
| **mAP** | 目标检测精度指标，越高越好 | 考试得分 |
| **epoch** | 把全部训练数据过一遍 | 把一本习题册做完一遍 |
| **batch** | 一次喂给模型几张图 | 一次炒几盘菜 |
| **ONNX** | 通用模型交换格式 | 不同品牌手机都能打开的 PDF |
| **伪量化 (Fake Quant)** | 训练时模拟量化噪声 | 高考前的模拟考 |
| **per_channel** | 每个输出通道一套 scale | 每道菜单独配一把盐勺 |
| **per_tensor** | 整个张量共用一套 scale | 所有菜共用一把盐勺 |
| **对称量化** | 量化范围以 0 为中心 | 摄氏温度计（有正有负） |
| **非对称量化** | 量化范围可以偏移 | 可以平移刻度的尺子 |
| **EMA** | 指数移动平均，平滑权重 | 股票软件的"5 日均线" |
| **NMS** | 非极大值抑制，去掉重复框 | 同一个猫只保留一个最准的框 |

## 附录 B：文件速查表

| 文件 | 作用 |
|------|------|
| `networks_yolo26-detect.py` | 检测主脚本（seg/pose/cls/obb/depth 类似） |
| `quantization/__init__.py` | 后端加载、参数提取、浮点图构建 |
| `quantization/lsqplus_quantize_V1.py` | LSQ+ V1 后端实现 |
| `script/coco_mini_prepare.py` | 构造 COCO mini 数据集 |
| `script/export_qat_outputs.py` | 从 checkpoint 补发部署产物 |
| `script/visualize.py` | 批量可视化验证集结果 |

---

*本文基于 aLSQplus/QAT_training 项目的源码、README、设计文档及 79 组实验结果写成。如有错误，欢迎指出。*

>本文撰写的部分内容来自人工智能大模型Kimi-K3
