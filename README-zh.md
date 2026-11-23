# 更新日志

2026.10.24：创建图像分割项目

# 一、项目背景
1. **目标**：面向桌面物体整理场景，识别并分割杯子、不透明手串、笔记本电脑，重点分析遮挡、相邻物体、小目标和背景变化对分割效果的影响，在分割质量与推理速度之间选择合适的模型配置。
2. **数据集**：使用自定义数据集训练一个实例检测与分割模型，用于将 3 类物体（杯子、不透明手串、笔记本电脑）从其背景及周围环境中分离出来。
3. **模型**：该模型通过 ultralytics 提供的基于 COCO 数据集预训练的 YOLOv11-seg 模型在自定义数据集上进行微调得到。
4. **数据获取方式**：使用 iPhone 15 Pro Max 固定拍摄配置进行数据采集。该数据集共包含 400 张高质量的不透明手串图像，使用 LabelMe 进行实例分割标注，分为 300 张用于训练和 100 张用于测试。这些图像最初使用 LabelMe JSON 格式进行标注，但 YOLO 采用不同的标注格式，因此使用名为 labelme2yolo 的工具将该数据集转换为 YOLO 可训练的格式和文件结构。不透明手串采用切割法进行标注。
5. **开发方式**：本项目采用分阶段模块化 AI Coding 方式实现，ChatGPT、Codex、Cluade Code 三者协作分工开发。
6. **特别注意**：不透明手串的标注策略有三种，分别是内外轮廓标注法（需自定义训练和后处理代码来合并掩码）、切割法（无需自定义代码，有微小缺口）、多类别标注法（需额外定义类别，自定义后处理代码）

# 二、项目内容
## 2.1 数据集
<table>
  <thead>
    <tr>
      <th rowspan="2">名称</th>
      <th rowspan="2">类别</th>
      <th colspan="3">划分</th>
      <th rowspan="2">链接</th>
    </tr>
    <tr>
      <th>训练集</th>
      <th>验证集</th>
      <th>测试集</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Grapes</td>
      <td>巨峰（Kyoho）</td>
      <td rowspan="3">100</td>
      <td rowspan="3">25</td>
      <td rowspan="3">25</td>
      <td rowspan="3"><a href="https://tianchi.aliyun.com/dataset/230037">点击下载</a></td>
    </tr>
    <tr>
      <td>阳光玫瑰（ShineMuscat）</td>
    </tr>
    <tr>
      <td>夏黑（SummerBlack）</td>
    </tr>
    <tr>
      <td rowspan="3">--</td>
      <td>--</td>
      <td rowspan="3">--</td>
      <td rowspan="3">--</td>
      <td rowspan="3">--</td>
      <td rowspan="3"><a href="#">点击下载</a></td>
    </tr>
    <tr>
      <td>--</td>
    </tr>
  </tbody>
</table>

### 图表

<details>
<summary>Grapes 数据集图表（点击展开，共 4 张）</summary>

![Grapes 总体分布](images/grapes/overview.png)

*图 1：Grapes 总体样本分布。*

![巨峰分布](images/grapes/kyoho.png)

*图 2：巨峰（Kyoho）样本分布。*

![阳光玫瑰分布](images/grapes/shine_muscat.png)

*图 3：阳光玫瑰（ShineMuscat）样本分布。*

![夏黑分布](images/grapes/summer_black.png)

*图 4：夏黑（SummerBlack）样本分布。*

</details>

<details>
<summary>新数据集分布（点击展开）</summary>

![占位图](images/placeholder.png)

*图 5：--*

</details>

## 2.2 数据集

# 三、数据集

## 3.1 数据标注（LabelMe）

### 3.1.1 标注环境

统一标注环境版本

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| Labelme | ≥ 4.5.7 | 支持关键点标注 |
| PyQt5 | 5.15.4 | 图像界面基础 |
| OpenCV | 4.5.4 | 图像处理核心 |

### 3.1.2 标注规范

1. 目标框标注原则：
   - 框体紧贴物体边缘（留白不超过 3 像素）
   - 遮挡物体按可见部分标注
   - 小物体（< 32×32 像素）建议放大标注
2. 关键点标注原则：
   - 关键点必须归属到对应检测框内
   - 可见点标注实际位置（即使被轻微遮挡）
   - 完全不可见点标记为“未标注”状态
   - 对称关键点需保持左右一致性

标注规范示例：【待添加】

### 3.1.3 标注前的统一处理

1. 统一缩放至相同最大边长（保持比例）
2. 生成缩略图用于快速标注（原始图用于最终训练）
3. 自动校正 EXIF 方向信息

### 3.1.4 标注质量验证

1. 关键点超出物体边界框
2. 重叠框体未处理
3. 标签命名不一致
4. 图像尺寸与标注坐标不匹配

## 3.2 格式转换与数据集划分

Labelme 格式转 COCO 格式

数据集划分：

1. 保持各类别在划分集中的分布均衡
2. 同一场景的图像不应分散在不同集合
3. 划分数据集时使用分层抽样

## 3.3 常见训练问题的标注质量诊断

| 现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| 关键点预测偏移 | 标注不一致 | 重新统一标注标准 |
| 漏检小物体 | 标注不完整 | 增强小物体标注 |
| 误检率高 | 负样本不足 | 添加困难样本 |
| 关键点混淆 | 标签错误 | 检查关键点映射 |

做法：

- 随机抽查 10% 的标注图像
- 统计关键点可见性分布
- 验证标注框宽高比分布
- 检查边缘案例的处理一致性

## 3.4 数据增强

# 四、模型训练

## 4.1 预训练模型微调

### 4.1.1 超参数

### 4.1.2 结果分析

经过此训练过程得到的模型（保存为 `yolov8_【目标】_custom.pt`）准确率较高，mAP50-95 为 91.41%，验证分割损失为 0.7888（从初始损失 3.0348 下降），边界框（检测）损失为 0.42515（从 1.2523 下降）。每个训练周期的所有损失、准确率及相关数值可在此处查看，下方还提供了这些数值的图表以及使用该模型进行分割的若干示例图像。

## 4.2 改进模型

### 4.2.1 超参数

### 4.2.2 结果分析

经过此训练过程得到的模型（保存为 `yolov8_【目标】_new.pt`）准确率较高，mAP50-95 为 91.41%，验证分割损失为 0.7888（从初始损失 3.0348 下降），边界框（检测）损失为 0.42515（从 1.2523 下降）。每个训练周期的所有损失、准确率及相关数值可在此处查看，下方还提供了这些数值的图表以及使用该模型进行分割的若干示例图像。

# 五、推理部署

要本地使用此模型，请按照以下步骤操作：

1. 克隆此仓库并进入其目录：
2. 使用 `venv` 或 `Conda` 创建虚拟环境并安装依赖：
3. 如果希望在推理时使用 GPU，请确保安装支持 CUDA 的 PyTorch（安装说明见下方 Windows 系统下 CUDA 12.1 的示例）：
4. 对测试集图像的推理：
5. 对测试集视频的推理：
6. 对自定义图像或视频的推理：

# 六、项目架构

```text
image-seg/
├── checkpoints/                 # 模型权重层
│   ├── downloads/               # 下载预训练权重（别人）
│   └── locals/                  # 本地训练权重（自己）
├── configs/                     # 配置层
│   ├── data-processeds/         # 数据预处理配置
│   ├── trains/                  # 模型训练配置
│   └── inferences/              # 模型推理配置
├── data/                        # 数据层
│   ├── raw/                     # 原始数据
│       └── 数据集名称/          
│           ├── 数据集详情.md
│           ├── train/
│           ├── val/ 
│           └── test/
│   └── processed/               # 预处理数据
│       └── 数据集名称/          
│           ├── 预处理数据集详情.md
│           ├── train/
│           ├── val/ 
│           └── test/
├── src/                         # 逻辑层
│   ├── data/                    # 数据预处理逻辑
│       ├── .py  
│       ├── evals.py  
│       └── models.py
│   ├── evals/                   # 评估指标逻辑
│       └── seg-common_eval.py
│   ├── inferences/              # 模型推理逻辑
│   ├── models/                  # 模型结构逻辑
│       └── 模型名称/          
│           ├── 模型结构详情.md   
│           └── 模型名称.py
│   ├── schemas/                 # 定义数据格式
│       ├── datasets.py  
│       ├── evals.py  
│       └── models.py
│   └── trains/                  # 模型训练逻辑
│       ├── double-stages_train.py
│       ├── few-sample_train.py  
│       └── zero-sample_train.py
├── results/                     # 实验结果层
│   └── 实验名称-时间戳/          
│       ├── 实验详情.md          
│       └── 实验名称-时间戳.log
├── scripts/                     # 执行层
│   ├── train.py
│   ├── inference.py
│   └── data_processed.py
└── requirements.txt             # 依赖库安装清单


# 基于 YOLO11 的葡萄果粒实例分割与模型压缩

个人项目｜2026.10—至今

面向葡萄果粒密集分布、相互遮挡和小目标场景，基于 YOLO11s-seg 完成实例分割微调，并探索结构化剪枝与特征蒸馏对精度、模型规模和推理延迟的影响。

技术栈：Python、PyTorch、Ultralytics、Torch-Pruning、OpenCV；CPU 推理测试使用 ONNX Runtime。

## 数据与评估

- 类别：巨峰（Kyoho）、阳光玫瑰（ShineMuscat）、夏黑（SummerBlack）。
- 现有数据清单：训练集 100 张原图、验证集 25 张、测试集 25 张；使用在线切片，不另建离线切片数据集。
- 训练与验证切片尺寸为 640×640。验证时处理 25 张原图的 1,200 个切片，合并至原图后评估，共 8,598 个标注实例。
- 固定验证条件：FP32、batch=1、conf=0.001、NMS IoU=0.7、max_det=1000。
- 以下精度均来自验证集。测试集尚未用于模型评估；同一葡萄串或采集场景跨集合的相似性仍需核查。

## 方法

1. **Baseline 微调**：以 YOLO11s-seg 预训练权重初始化，训练 50 轮；batch=8、AdamW、初始学习率 0.001、seed=42，启用 AMP。以原生 box 与 mask 综合 fitness 选择 best.pt。
2. **结构化剪枝**：使用 Torch-Pruning 依赖图与分组 L2 权重重要性，完成 10%、20%、30%、40%、50% 五档实验。比例指 18 个安全依赖组内的目标通道删除比例，不等于全模型参数缩减比例；各档普通恢复训练预算为 10 轮。
3. **特征蒸馏**：以 Baseline 为冻结教师、未经恢复的 p50 剪枝模型为学生，训练 20 轮。保留真实标注分割任务损失，加入教师置信度加权 L2 特征损失，蒸馏权重 dis=6.0。推理仅使用学生网络。

## 核心结果

| 模型 | 恢复训练 | Mask mAP50-95 | 参数量（M） | GFLOPs |
| --- | --- | ---: | ---: | ---: |
| Baseline | — | 92.2071% | 10.0834 | 33.1979 |
| p50 剪枝后直接评估 | 无 | 2.5811% | 8.4217 | 30.5277 |
| p50 普通恢复 | 10 轮 | 91.2116% | 8.4217 | 30.5277 |
| p50 特征蒸馏 | 20 轮 | 91.4828% | 8.4217 | 30.5277 |

参数与计算量采用未融合网络口径，输入为 640×640；FLOPs 按 2 FLOPs/MAC 统计。

p50 蒸馏候选相对 Baseline：参数量减少 **16.48%**，理论计算量减少 **8.04%**，Mask mAP50-95 下降 **0.7243 个百分点**。普通恢复与蒸馏训练预算不同，现有比较不能单独证明蒸馏优于等预算普通微调。

历史精度约束为相对 Baseline 最多下降 0.5 个百分点，即不低于 91.7071%；p50 蒸馏候选尚未达到该门槛。五档普通恢复结果中，p30 是满足该约束且参数最少的已测候选：Mask mAP50-95 为 91.7841%，参数量约 9.0183M。

## 已有推理延迟测量

以下引用 2026-11-23 的同批五轮重复测量，不与其他会话的历史测速混合计算加速率。

| 模型 | GPU PyTorch FP32 均值 / P95（ms） | CPU ONNX FP32 均值 / P95（ms） |
| --- | ---: | ---: |
| Baseline | 7.985 / 8.338 | 92.238 / 100.608 |
| p50 特征蒸馏 | 7.701 / 8.170 | 86.556 / 94.422 |

- GPU：RTX 4060 Laptop 8GB，Conv/BN 融合，TF32 关闭，CUDA Events 计时。
- CPU：Intel Core i9-14900HX，ONNX Runtime CPUExecutionProvider，intra-op=4、inter-op=1。
- 输入：同一验证图像的固定 640×640 切片，FP32、batch=1；每模型每设备 5 轮，每轮预热 20 次、测量 100 次，轮换模型顺序。
- 范围：单切片模型前向；不含图像读取、预处理、NMS、掩膜重建、切片循环及原图合并。CPU 与 GPU 使用不同后端，仅分别在各自条件内比较。

本次 p50 蒸馏候选的平均前向延迟相对 Baseline 在 GPU 上减少约 3.56%，CPU 上减少约 6.16%。收益幅度有限，且 CPU 存在轮间漂移；不能据此宣称完整分割流程获得相同比例加速。推理峰值显存、进程内存及整图推理耗时尚待补测。

## 环境与文件索引

已有实验使用 Python 3.10.22、PyTorch 2.11.0+cu128、Ultralytics 8.4.208、ONNX Runtime 1.23.2，Conda 环境为 `image-seg-yolo`。Torch-Pruning 与 OpenCV 版本待从实验环境记录补齐。

下列路径相对于外层项目目录 `D:/agent_items/ultralytics/`：

| 内容 | 路径 |
| --- | --- |
| Baseline 配置 | `configs/train/baseline.yaml` |
| p50 剪枝配置 | `configs/optimize/prune-p50.yaml` |
| 蒸馏配置 | `configs/train/distill-p50.yaml` |
| 剪枝 / 蒸馏入口 | `src/grape_seg/workflows/prune.py`、`distill.py` |
| Baseline 权重 | `runs/segment/baseline-yolo11s-grape-v1-tile640-s42-b8-nondet/weights/best.pt` |
| p50 蒸馏权重 | `runs/segment/distill-baseline-to-p50-raw-s42/weights/best.pt` |
| 五档剪枝报告 | `runs/optimize/prune-baseline-p10-p20-p30-p40-p50-s42-comparison/Comparison-report.md` |
| 蒸馏报告 | `runs/optimize/distill-baseline-to-p50-raw-s42/Distillation-report.md` |
| 重复测速报告与原始样本 | `runs/benchmark/baseline-nano-distill-p50-repeat-640/Comparison-report.md`、`results.json` |

复现时需核对解释器、实际导入包路径、配置和权重哈希。现有实验导入环境中的 Ultralytics 8.4.208；当前 `ultralytics-main/` 源码版本为 8.4.215，不能直接视为同一实验环境。上述配置包含本机绝对路径，迁移前需调整。

## 待补工作

- 补等训练预算的普通微调与蒸馏对照，必要时针对关键比较重复种子实验。
- 在独立进程中分别测量模型推理峰值显存与进程内存，统一测量阶段与输入。
- 补真实原图的完整切片推理耗时，并分解预处理、前向、后处理和合并时间。
- 整理密集、遮挡、小果粒与切片边缘的失败案例，完成数据分组核查后再安排最终测试集评估。

