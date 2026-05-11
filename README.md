# Zero-Shot Traffic Accident Anticipation — 2nd Place Solution

[中文版](#中文版) | English

> **Competition:** [CVPR @ AutoPilot — Zero-Shot TAA (Kaggle)](https://www.kaggle.com/competitions/zero-shot-taa)
> **Affiliation:** Beijing University of Posts and Telecommunications
> **Final rank:** 2nd Place

A VideoMAE-v2 based per-frame accident-risk predictor for the zero-shot
traffic-accident-anticipation challenge. Given a 5-second dashcam clip
(150 frames @ 30 fps) the model outputs a per-frame risk score
$r_t \in [0,1]$ for $t = 1,\dots,150$.

## Solution Overview

The competition supplies no training set. Our pipeline therefore has four
stages, each implemented in exactly one script:

1. **Dataset construction** — Build a 5 s × 150-frame training set from the
   public **Nexar Crash** dataset. For positive videos we cut off the last
   2 s so the model never sees the collision frames; we then slide a 5 s
   window with 50% overlap and emit DADA-1000-style annotation lines.
2. **Training** — Fine-tune **VideoMAE-v2 (Base)** on 16-frame windows with
   a time-weighted cross-entropy loss (`ExpLoss`).
3. **Inference** — Sliding-window inference on each 150-frame test clip,
   followed by overlap averaging and linear interpolation back to 150
   per-frame risk scores.
4. **Zero-shot domain-adaptive post-processing** — A training-free
   calibration step that anchors each clip on its end-frame value, replaces
   the rest of the sequence with a smooth monotonic curve, and uses the
   empirical median of $\{p_{\text{final}}\}$ as a distribution-moment
   anchor that lifts under-confident positives across the binary decision
   threshold.

## Why VideoMAE-v2

VideoMAE-v2 is the de-facto self-supervised pre-trained model for video
understanding: it captures the spatio-temporal cues that are critical for
collision prediction, generalises well thanks to large-scale pre-training,
and is known to perform strongly on action-recognition tasks structurally
similar to "imminent collision vs. normal driving". In our internal
ablations it consistently outperformed CNN+LSTM and CNN+attention baselines.

## Limitations & Future Work

* **Single-model submission.** No ensembling was attempted.
* **Hand-designed post-processing.** The zero-shot DA step is rule-based;
  replacing the empirical-median anchor with a learnable per-test-set
  calibration head is an obvious next step.
* **Single 16-frame window length.** No experiments on alternative window
  lengths or sampling rates were run due to compute budget.

## Repository Structure

```
.
├── README.md                       # this file
│
├── nexar_config.py                 # central config (paths via env vars)
├── nexar_dataset.py                # train/val Dataset
├── videomae_model.py               # VideoMAE-v2 backbone + per-frame head
│
├── process_5s_clips.py             # Stage 1: dataset construction
├── nexar_train.py                  # Stage 2: training entry point
├── competition_predict.py          # Stage 3: sliding-window inference
├── postprocess_zero_shot_da.py     # Stage 4: zero-shot DA post-processing
│
├── models/                         # trained checkpoint (epoch_14.pth, via HF)
└── .gitignore
```

## Model Weights

The submitted checkpoint is `epoch_14.pth` (~1 GB, 87 M params, VideoMAE-v2
Base). It is hosted on Hugging Face Hub due to GitHub's file-size limits.

* **Hugging Face:** <https://huggingface.co/TimeSouth/zero-shot-taa-2nd-place>
* Download the file and place it under `models/epoch_14.pth` to reproduce
  the inference results, e.g.:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="TimeSouth/zero-shot-taa-2nd-place",
    filename="epoch_14.pth",
    local_dir="models",
)
```

## Dependencies

PyTorch ≥ 2.1, transformers ≥ 4.40, opencv-python, albumentations, numpy,
pandas, scikit-learn, tqdm, safetensors, huggingface_hub. A CUDA-enabled
PyTorch build is required for training (the submitted run took ~17 hours on
a single GPU for 20 epochs).

---

# 中文版

> **比赛：** [CVPR @ AutoPilot 零样本风险预测比赛 (Kaggle)](https://www.kaggle.com/competitions/zero-shot-taa)
> **单位：** 北京邮电大学
> **最终成绩：** 第 2 名

基于 VideoMAE-v2 的逐帧事故风险预测方案。输入 5 s 驾驶员视角视频
（150 帧 @ 30 fps），输出每一帧 $t$ 发生事故的概率 $r_t \in [0,1]$。

## 方法概览

本赛题不提供训练集。整个方案分四个阶段，每个阶段对应一个脚本：

1. **数据集构建** — 用公开 **Nexar Crash** 数据构建与比赛同形态
   （5 s × 150 帧）的训练集；正样本截掉事故发生最后 2 s，避免模型看到爆炸瞬间；
   以 stride=2.5 s 滑窗扩样，生成 DADA-1000 风格的标注。
2. **训练** — 在 16 帧窗口上微调 **VideoMAE-v2 (Base)**，使用时序加权
   交叉熵 (`ExpLoss`)。
3. **推理** — 对每个 150 帧 clip 做滑动窗口推理→窗口级 risk score→
   重叠位置取均值→线性插值还原 150 帧。
4. **零样本域自适应后处理** — 一个无训练的校准步骤：以末帧值锚定整段
   曲线，用一条单调上升的形状先验替换其余 149 帧，并以测试集
   $\{p_{\text{final}}\}$ 的经验中位数作为分布矩锚点，将欠自信的正
   样本抬过二分类判决阈值。

## 为什么选择 VideoMAE-v2

VideoMAE-v2 是当前视频理解领域代表性的自监督预训练模型：擅长捕捉
碰撞预测所需的时空线索，依靠大规模预训练具备出色的泛化能力，在与
"即将发生碰撞 vs. 正常驾驶"结构相似的动作识别任务上表现稳定。我们
内部的消融实验中，它显著优于 CNN+LSTM 与 CNN+Attention 基线。

## 局限与未来工作

* **单模型提交，未做集成。**
* **后处理基于规则**——可以考虑用可学习的 calibration head 替换"末值
  经验中位数"这一锚点。
* **未尝试不同窗口长度。** 受限于算力，仅用了 16 帧窗口。

## 仓库结构

见英文版 *Repository Structure*。

## 模型权重

`epoch_14.pth`（约 1 GB，87 M 参数）由于 GitHub 文件大小限制未直接放在
仓库中，托管在 Hugging Face Hub：

<https://huggingface.co/TimeSouth/zero-shot-taa-2nd-place>

下载后放到 `models/epoch_14.pth` 即可，或者用：

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="TimeSouth/zero-shot-taa-2nd-place",
    filename="epoch_14.pth",
    local_dir="models",
)
```

## 依赖

PyTorch ≥ 2.1, transformers ≥ 4.40, opencv-python, albumentations, numpy,
pandas, scikit-learn, tqdm, safetensors, huggingface_hub。训练需要支持
CUDA 的 PyTorch；本方案在单卡上 20 epoch 约耗时 17 小时。
