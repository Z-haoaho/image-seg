# 更新日志

2026.10.24：创建图像分割项目

# 一、项目背景

面向葡萄果粒密集分布、相互遮挡和小目标场景的实例分割任务，基于 YOLO11s-seg 在 Grapes 数据集上完成模型微调，并探索结构化剪枝与特征蒸馏对精度、模型大小和推理延迟的影响，找到符合要求的模型。

指标要求：Mask mAP50-95（%）≥ 90.00%；Params ≤ 9.00M；Speed CPU ONNX ≤ 87.00 ± 5.00ms

技术栈：Python、PyTorch、Ultralytics、Torch-Pruning、OpenCV、ONNX Runtime

计算资源：（CPU）Intel Core i9-14900HX；（GPU）RTX 4060 Laptop GPU


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

**图表（划分分布、主要类别、图像尺寸、图像文件大小、图像格式、每张图像中的对象数、每个实例的点数）**

<details>
<summary>Grapes 数据集图表（点击展开，共 7 张）</summary>

![划分分布](assets/data/images/Grapes/划分分布.png)


![主要类别](assets/data/images/Grapes/主要类别.png)


![图像尺寸](assets/data/images/Grapes/图像尺寸.png)


![图像文件大小](assets/data/images/Grapes/图像文件大小.png)


![图像格式](assets/data/images/Grapes/图像格式.png)


![每张图像中的对象数](assets/data/images/Grapes/每张图像中的对象数.png)


![每个实例的点数](assets/data/images/Grapes/每个实例的点数.png)


</details>

<details>
<summary>-- 数据集图表（点击展开，共 -- 张）</summary>

![--](assets/data/images/--.png)

*图 1：--*

</details>

## 2.2 实验方案
### 2.2.1 模型优化
| 序号 | 实验名称 | 实验目的 | 实验方法 | 配置 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 01 | Baseline | 确立基线 | 基于 yolo11s-seg 预训练模型在 Grapes 数据集上完成全量微调 |  [arg.yaml](assets/exps/Grapes/runs/exp-01/arg.yaml) | --- |
| 02 | yolo11s-seg-HT | 判断调优后是否超越基线 | 在 Baseline 基础上完成超参数调优 | --- | 由于计算资源限制，暂时跳过该实验 |
| 03 | yolo11s-seg-SP10 | 压缩模型大小并测量剪枝造成的即时损伤 | 在 yolo11s-seg 基础上完成模型通道剪枝 | [arg.yaml](assets/exps/Grapes/runs/exp-03/arg.yaml) | --- |
| 04 | yolo11s-seg-SP20 | 压缩模型大小并测量剪枝造成的即时损伤 | 在 yolo11s-seg 基础上完成模型通道剪枝 | [arg.yaml](assets/exps/Grapes/runs/exp-04/arg.yaml) | --- |
| 05 | yolo11s-seg-SP30 | 压缩模型大小并测量剪枝造成的即时损伤 | 在 yolo11s-seg 基础上完成模型通道剪枝 | [arg.yaml](assets/exps/Grapes/runs/exp-05/arg.yaml) | --- |
| 06 | yolo11s-seg-SP40 | 压缩模型大小并测量剪枝造成的即时损伤 | 在 yolo11s-seg 基础上完成模型通道剪枝 | [arg.yaml](assets/exps/Grapes/runs/exp-06/arg.yaml) | --- |
| 07 | yolo11s-seg-SP50 | 压缩模型大小并测量剪枝造成的即时损伤 | 在 yolo11s-seg 基础上完成模型通道剪枝 | [arg.yaml](assets/exps/Grapes/runs/exp-07/arg.yaml) | --- |
| 08 | yolo11s-seg-SP50-FT | 测量继续全量微调训练能够恢复多少 | 在 yolo11s-seg-SP50 基础上完成模型通道剪枝 | [arg.yaml](assets/exps/Grapes/runs/exp-08/arg.yaml) | --- |
| 09 | yolo11s-seg-SP50-KD | 判断教师监督是否带来额外收益 | 以 yolo11s-seg-SP50 为学生，Baseline 为教师，完成模型特征蒸馏 | [arg.yaml](assets/exps/Grapes/runs/exp-09/arg.yaml) | --- |
| 10 | yolo11n-seg-KD | 测量 yolo11n-seg 作为学生模型收益 | 以 yolo11n-seg 为学生，Baseline 为教师，完成模型特征蒸馏 | [arg.yaml](assets/exps/Grapes/runs/exp-10/arg.yaml) | --- |

## 2.3 实验结果
### 2.3.1 性能指标
| 序号 | 实验名称 | Size（pixels） | Box mAP50-95（%） | Mask mAP50-95（%） | Speed CPU ONNX（ms） | Params（M） | FLOPs（B） | 配置 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | Baseline | 640 | 92.41 | 92.21 | 92.24 ± 5.45 | 10.08 | 33.20 |  [arg.yaml](assets/exps/Grapes/runs/exp-01/arg.yaml) | --- |
| 02 | yolo11s-seg-HT | --- | --- | --- | --- | --- | --- | --- | 由于计算资源限制，暂时跳过该实验 |
| 03 | yolo11s-seg-SP10 | 640 | --- | --- | --- | --- | --- | [arg.yaml](assets/exps/Grapes/runs/exp-03/arg.yaml) | --- |
| 04 | yolo11s-seg-SP20 | 640 | --- | --- | --- | --- | --- | [arg.yaml](assets/exps/Grapes/runs/exp-04/arg.yaml) | --- |
| 05 | yolo11s-seg-SP30 | 640 | --- | --- | --- | --- | --- | [arg.yaml](assets/exps/Grapes/runs/exp-05/arg.yaml) | --- |
| 06 | yolo11s-seg-SP40 | 640 | --- | --- | --- | --- | --- | [arg.yaml](assets/exps/Grapes/runs/exp-06/arg.yaml) | --- |
| 07 | yolo11s-seg-SP50 | 640 | 2.47 | 2.58 | 87.03 ± 4.17 | 8.42 | 30.53 | [arg.yaml](assets/exps/Grapes/runs/exp-07/arg.yaml) | --- |
| 08 | yolo11s-seg-SP50-FT | 640 | 91.34 | 91.21 | 85.96 ± 3.84 | 8.42 | 30.53 | [arg.yaml](assets/exps/Grapes/runs/exp-08/arg.yaml) | --- |
| 09 | yolo11s-seg-SP50-KD | 640 | 91.73 | 91.48 | 86.56 ± 4.49 | 8.42 | 30.53 | [arg.yaml](assets/exps/Grapes/runs/exp-09/arg.yaml) | --- |
| 10 | yolo11n-seg-KD | 640 | 89.82 | 89.92 | 33.80 ± 2.27 | 2.86 | 9.86 | [arg.yaml](assets/exps/Grapes/runs/exp-10/arg.yaml) | --- |


### 2.3.2 可视化结果

# 三、项目结构

## 3.1 目录结构
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
```

# 四、文件索引
