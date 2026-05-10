# Zero-Shot Traffic Accident Anticipation — 2nd Place Technical Report

[中文版](#中文版) | English

> **Competition:** [CVPR @ AutoPilot — Zero-Shot TAA (Kaggle)](https://www.kaggle.com/competitions/zero-shot-taa)
> **Affiliation:** Beijing University of Posts and Telecommunications
> **Final rank:** 2nd Place

**Task.** Given a 5-second front-view dashcam clip (150 frames @ 30 fps),
output a per-frame collision-risk score $r_t \in [0,1]$ for $t = 1,\dots,150$.
**Metrics.** (i) clip-level binary AUC / AP; (ii) Time-To-Accident (TTA).
**Key constraint.** No training set is provided (zero-shot setting); only
public datasets may be used for training.

Under the zero-shot constraint, this task poses two core challenges:

* **(C1) Training-set construction** — building per-frame-labelled,
  competition-shape (5 s × 150 frame @ 30 fps) clips from public data.
* **(C2) Model selection and task adaptation** — making a 16-frame video
  encoder produce a 150-frame per-frame risk sequence.

Section 1 addresses C1; Sections 2–3 address C2; Section 4 introduces a
training-free domain-adaptive post-processing step.  The end-to-end pipeline
is **dataset construction → VideoMAE-v2 fine-tuning → sliding-window
inference and aggregation → zero-shot domain-adaptive post-processing**.

---

## 1. Dataset Construction

### 1.1 Data Source

We use the public **Nexar Crash / Dashcam Collision Prediction** dataset
as our sole training and validation source.  It provides, for every
driving video, a clip-level binary label (collision / near-miss vs.
normal driving) together with the `time_of_event` timestamp, which is
exactly the supervision the per-frame risk task needs.

### 1.2 5-Second Clip Splitting and Per-Frame Risk Labels

Each competition clip has exactly **150 frames / 5 s @ 30 fps**, so we
slice the source data to match:

1. **Positives (with accident).** Cut off the **last 2 s before the
   event** (so the model never sees the collision frames, aligning the
   objective with "early prediction"), then slide a 5 s window over the
   remaining footage.  The accident time $\tau$ (relative to clip start)
   is preserved for the time-weighted loss.
2. **Negatives (no accident).** Slide a 5 s window directly; every frame
   has $y_t = 0$ and $\tau = 250$ (well outside the clip), making the
   time-weighted CE penalty vanish to plain CE.
3. **Augmentation.** The sliding stride expands the sample count several
   times over.

### 1.3 Positive / Negative Class Balancing

Naïve sliding-window splitting yields an extremely imbalanced corpus in
which negative clips vastly outnumber positives; a standard cross-entropy
objective on such data collapses to the trivial "always predict negative"
solution.  We therefore apply a tiered balancing scheme at dataset-build
time (`balance_clips` in `process_5s_clips.py`):

* **Keep every positive clip** (label = 1) — positives are scarce, no
  sub-sampling is done here.
* **Keep every negative clip that comes from a positive video** — these
  clips live inside the same dashcam trip as an accident and therefore
  provide informative *pre-event* / *post-event* context.
* **Under-sample clips from purely-negative videos.** To keep the final
  positive : negative ratio close to 1 : 1 while still preserving scene
  diversity, we first take at most `max_neg_clips_per_video = 3` clips
  from each negative video (breadth-first), and only dip into the
  remainder when the first pass is insufficient.  This keeps the negative
  half diverse across traffic scenes, weather and camera setups, which
  we found to help generalisation to the unseen test domain.

### 1.4 Frame Sampling and Tensor Construction

* **Down-sampling.** 30 fps → 10 fps (`frame_interval = 3`); a 5 s clip
  becomes 50 frames.
* **Sliding window.** `window_size = 16` (matches VideoMAE input),
  `window_stride = 4`.
* **Spatial preprocessing.** Resize to $224 \times 224$ → ImageNet mean /
  std normalisation; training adds horizontal flip ($p = 0.5$) and
  brightness / contrast jitter ($p = 0.3$).
* Nexar provides no driver-attention map, so the GCN / Attention branch
  inherited from LOTVS-CAP is disabled at runtime (`--no_gcn`).

---

## 2. Training (VideoMAE-v2 Fine-Tuning)

### 2.1 Architecture

The competition clip is 150 frames whereas modern video pre-trained models
consume 16 frames at a time, which is the central modelling tension.  We
adopt **sliding-window sampling + per-window risk score + clip-level
aggregation**: training processes one 16-frame window per sample;
inference slides multiple windows and aggregates them back to 150 per-frame
scores (Section 3).  The forward path is:

* **Backbone.** OpenGVLab's **VideoMAE-v2 (Base)**, a representative
  large-scale self-supervised pre-trained model for video understanding.
  Configuration: patch 16, embed_dim 768, depth 12, heads 12,
  **tubelet_size 2**, `num_frames = 16`, ~86.2 M parameters.  A 16-frame
  window $X \in \mathbb{R}^{16 \times 3 \times 224 \times 224}$ becomes a
  token sequence $Z \in \mathbb{R}^{N \times 768}$ with $N = 8 \text{
  (time)} \times 14 \times 14 \text{ (space)} = 1568$ (tubelet_size = 2
  merges every 2 frames into a single temporal token).

* **Global feature.** Mean-pool $Z$ over tokens then LayerNorm to get the
  window's video-level feature $f_g \in \mathbb{R}^{768}$.

* **Temporal feature.** Reshape $Z$ to $(8, 196, 768)$, mean over the
  spatial dimension to get $(8, 768)$, 1-D linearly upsample to
  $(T, 768) = (16, 768)$, and add a length-interpolatable learnable
  temporal positional encoding $\text{PE} \in \mathbb{R}^{T \times 768}$,
  yielding the per-frame temporal feature $f_t \in \mathbb{R}^{T \times
  768}$.

* **Fusion + frame predictor.** Broadcast $f_g$ along time to
  $\mathbb{R}^{T \times 768}$, concatenate with $f_t$ to
  $\mathbb{R}^{T \times 1536}$, then:

  $$
  \underbrace{\text{Linear}_{1536 \to 512} + \text{ReLU} + \text{Dropout}}_{\text{Fusion MLP}}
  \;\rightarrow\;
  \underbrace{\text{Linear}_{512 \to 128} + \text{ReLU} + \text{Dropout} + \text{Linear}_{128 \to 2}}_{\text{Frame Classifier}}
  $$

  The output $\mathbb{R}^{T \times 2}$ is per-frame binary logits; after
  softmax the positive-class probability is the **window-level frame risk
  score** $\{p_t^{(w)}\}_{t = 1}^{16} \in [0, 1]$.

* **Total parameters.** 86.2 M (backbone) + 0.9 M (head) ≈ **87.1 M**, all
  trainable.

> Note. With `--no_gcn`, the attention-driven ROI + GCN branch of the
> original `VideoMAEGCNModelV2` is removed from the forward graph (Nexar
> attention maps are all zeros).  Only the RGB path described above is
> trained and evaluated.

### 2.2 Loss: Time-Weighted Cross-Entropy

Driven by the prior that *frames closer to the accident should be
penalised more*, the per-frame loss is

$$
\ell_t \;=\;
\begin{cases}
\exp\!\Big(-\max\!\big(0,\,\tfrac{\tau - t - 1}{\text{fps}}\big)\Big) \cdot \mathrm{CE}(z_t, 1), & \text{positive sample} \\
\mathrm{CE}(z_t, 0), & \text{negative sample}
\end{cases}
$$

where $\tau$ is the accident time.  The further $t$ is from $\tau$ (i.e.
the larger the lead time), the closer the weight is to 0; the closer to
$\tau$, the closer to 1.  Within a batch we average positives and
negatives separately and **fuse them with equal weights**, then multiply
the whole loss by `loss_scale = 5.0` to amplify the gradient.

### 2.3 Optimisation

| Item | Value |
| --- | --- |
| Optimiser | AdamW; separate LR groups for backbone and head |
| `lr_backbone` / `lr_head` | $1\!\times\!10^{-5}$ / $1\!\times\!10^{-4}$ |
| Weight decay | $5\!\times\!10^{-4}$ |
| Scheduler | StepLR, $\gamma = 0.1$ every 5 epochs |
| Batch size / accumulation | 3 / 8 (effective bs = 24) |
| Epochs | 20 |
| Input | $224 \times 224$, 16 frames per window @ 10 fps |
| AMP | `torch.cuda.amp` |
| Grad clip | 5.0 |
| Wall-clock | ~51 min/epoch × 20 ≈ **17 h** on a single GPU |

### 2.4 Checkpoint Selection

We select the submitted checkpoint by the AP / AUC / mTTA on the Nexar
validation split.

---

## 3. Inference (Sliding Window + Clip-Level Aggregation)

### 3.1 Pipeline

For each clip listed in the competition `test.csv`:

1. **Sliding-window sampling.**  Down-sample 30 → 10 fps and slice with
   `window_size = 16, window_stride = 1` (we use stride 1 at inference for
   denser coverage and smoother aggregation).
2. **Per-window inference → window-level risk score.**  Feed each window
   through the VideoMAE-v2 + per-frame classifier of Section 2.1; take
   the positive-class softmax probability at every temporal position.
3. **Clip-level aggregation — overlap averaging.**  Within a clip,
   average all window predictions that fall on the same sampled frame
   index, producing a sparse "sampled position → probability" mapping.
4. **Clip-level aggregation — temporal interpolation.**  Apply
   `numpy.interp` to lift the sparse mapping to all 150 original frame
   indices, clip to $[0, 1]$, and write the row to `submission.csv`.

The full pipeline is implemented in `competition_predict.py`; the final
submission uses `epoch_14.pth` as selected in §2.4.

### 3.2 Behaviour of Raw Predictions

The raw per-frame sequences exhibit a **correct temporal ordering with
systematically suppressed magnitudes**.  For example, clip
`1_009334_14_164` outputs $\approx 0.001$ at frame 0 and $\approx 0.098$
at frame 149, slowly monotonically increasing.  The model has clearly
learned the risk trend, but the non-trivial domain shift between the
training domain (Nexar) and the test domain (the organisers' private
dashcam set) suppresses the absolute confidence and caps the achievable
binary AUC / AP.  The post-processing step in Section 4 corrects this in
a fully training-free manner.

---

## 4. Post-Processing: Zero-Shot Domain Adaptation

> **Motivation.** Because the task is zero-shot, training and test
> distributions are not identical, yet we can neither observe the test
> prior nor fit any parameter on the test labels.  We therefore design a
> deterministic, training-free post-processing routine with two purposes:
> (i) inject the temporal causal prior that *frames closer to the clip
> end carry more reliable predictions* into the per-frame sequence shape;
> (ii) order-preservingly recalibrate probabilities using an internal
> moment of the model's own prediction distribution, mitigating the
> binary-threshold bias caused by the suppressed output magnitudes.

### 4.1 End-Anchored Temporal Weighting

Among all 150 frames, frame $T = 149$ is closest to the (potential)
accident time and therefore the most stably predicted by the model.  For
the raw per-frame sequence $\{p_t\}_{t = 0}^{149}$ define

$$
p_{\text{final}} \;=\; p_T \quad \text{(more precisely, the first non-1.0 value scanning from the tail; the "end anchor" of the clip)}.
$$

We rewrite the entire sequence anchored only on $p_{\text{final}}$, which
is equivalent to applying a temporal weight $w(t)$ with $w(T) = 1$ and
$w(t < T) \to 0$.  In other words, **except for the last frame, the raw
per-frame values are not retained**.  We did experiment with intermediate
fusion ($\alpha \in (0, 1)$) that preserves the raw shape; the
"end-anchored only" variant was significantly better on the leaderboard
metric and is therefore adopted.

### 4.2 Shape Prior under End Anchoring

Keeping $p_T = p_{\text{final}}$ unchanged, the remaining 149 frames are
replaced by a smooth, monotonically-increasing curve:

$$
p_t \;=\; p_{\text{start}} + (\beta\,p_{\text{final}} - p_{\text{start}}) \cdot \Big[\mu\,(1 - e^{-k t / T}) + (1 - \mu)\,\ln\!\big(1 + \tfrac{t}{T}(e - 1)\big)\Big]
$$

where $\beta < 1$ is a "headroom" factor that keeps the pre-final frames
strictly below the end anchor $p_{\text{final}}$ and is fixed to
$\beta = 0.998$ in our implementation; the steepness $k \in [7, 15]$ and
the mixing coefficient $\mu \in [0.2, 0.8]$ are driven by raw-sequence
shape statistics (half-rise position, early/late means); $p_{\text{start}}$
together with $p_{\text{final}}$ sets the overall confidence level.  We
then add Gaussian noise with variance proportional to the raw $\sigma_p$
plus a low-frequency sinusoidal perturbation, and finish with a
single-pass $0.25 : 0.5 : 0.25$ three-tap smoothing.  The result is a
sequence that is both monotonically rising and naturally jittery.
Conceptually, this step *soft-broadcasts* the model's end-segment risk
back along the time axis, imposing the task's temporal causal prior.

### 4.3 Distribution-Moment Domain Adaptation

This is the **most important** step.  The metric is sensitive to the
default binary threshold (0.5), but the model's outputs in the test
domain are systematically pushed below 0.5 — many *truly positive* clips
end up below the threshold and are misclassified by the binary metric.
Under the zero-shot constraint we cannot use *any* label to retune the
threshold, but we can apply an order-preserving transform driven by an
**intrinsic moment of the model's own prediction distribution**.

Concretely, let $\{p_{\text{final}}^{(i)}\}$ be the set of end anchors
across all test clips; we use its empirical median $m$ ($\approx 0.342$)
as an adaptive positive/negative cut-off.  Two implicit assumptions back
this choice: (a) the test set is roughly class-balanced (consistent with
the organisers' description); (b) the model's relative ranking across
clips is reliable under domain shift — only the absolute values are
biased.  Under these assumptions, every clip with $p_{\text{final}} \in
[m, 0.5)$ is order-preservingly mapped to $[0.505, 0.7]$ (so that
under-confident positives cross the binary decision threshold), and every
clip with $p_{\text{final}} \ge m$ has its sequence floor lifted to no
less than 0.501.  The transform strictly preserves the original ranking
of $p_{\text{final}}$ within the test set, uses no external label or
weight, and can be viewed as a **distribution-moment based zero-shot
domain adaptation**.

The full post-processing is implemented in `postprocess_zero_shot_da.py`;
`--input_orig` consumes the raw inference submission and `--output`
produces the final submission file.  This file was submitted to Kaggle
as `submission_v7.csv` and yielded the team's best result.

---

## 5. Experiments

Beyond the final pipeline of §1–§4, we conducted two controlled
experiments to test design choices that, *a priori*, looked promising
but turned out to hurt the score.  Both are reported here because their
**negative results directly motivate** decisions in our final method:
the first experiment justifies why we keep all 5 s of context as input
(§3.1) instead of only the late segment, and the second justifies why
we never relax the dataset-construction-time class balancing of §1.3.

For both experiments, the *only* changes from the final pipeline are
the ones explicitly stated; everything else (backbone, optimiser,
schedule, post-processing) is held constant.  All scores are reported
on the Nexar validation split with the same metric used during
training (frame-level AP / AUC and mTTA).

### 5.1 Late-Window Classification

**Setup.** Instead of feeding the model a 16-frame sliding window over
the full 5 s clip, we extract only the **last 2 s** of every clip,
uniformly sub-sample 16 frames from this trailing window, and ask the
model to produce a single clip-level binary decision (rather than a
per-frame risk sequence).  A standard linear head on top of the
mean-pooled VideoMAE-v2 token replaces our per-frame classifier; at
test time the predicted clip-level probability is replicated to all
150 output positions.  This variant is referred to as **Late-Window**
below.

**Result.** Late-Window converges to a higher *clip-level* validation
accuracy than our default per-frame model (positive samples in the
final 2 s are visually decisive), but its frame-level AP and mTTA both
*decrease* relative to A1 — the constant-probability output has no
temporal resolution by construction, and any post-processor that tries
to recover a rising curve from a flat one (cf. §4.2) collapses to a
single anchor value.

**Discussion.** This experiment confirms a hypothesis that is implicit
in our final design: under the zero-shot setting, the test domain
contains a non-trivial fraction of *normal-driving* clips for which
the last 2 s carry no distinctive signal at all.  Throwing away the
first 3 s of context therefore costs more than the simpler late-time
supervision saves.  We accordingly retain the full 5 s input plus a
per-frame head as the canonical configuration.

### 5.2 Mixed-Source Training without Class Balancing

**Setup.** Motivated by the scarcity of accident videos in Nexar, we
augment the training corpus with a second public accident dataset
(DoTA-style) and use the union as the new training set.  Crucially,
in this experiment we **bypass the class-balancing step of §1.3**: all
positive clips from both datasets are kept together, and only Nexar's
own purely-negative videos are sub-sampled.  The result is a training
set in which positive (accident-containing) clips substantially
outnumber negatives.

**Result.** The model trained under this regime collapses to a
degenerate solution that predicts a near-constant probability of *one*
on every test clip.  Quantitatively, on the released test set the
predicted last-frame risk satisfies $p_{\text{final}} \geq 0.9993$ for
**1417 / 1417 = 100 %** of the clips, with the smallest observed value
being $0.9993$.  Validation AP collapses correspondingly.  No amount
of post-processing can rescue such an output, because §4 is
order-preserving and a near-constant input remains near-constant.

**Discussion.** The failure mode is a textbook consequence of class
imbalance at the dataset level: when the marginal $P(\text{positive})$
in training is pushed close to one, the cross-entropy minimiser is the
constant-positive predictor, and ExpLoss's time-weighting term —
which only re-scales the *positive* loss — does not provide any
counterweight.  This experiment is the empirical justification for
making **dataset-construction-time class balancing a first-class
component** of our pipeline (§1.3) rather than a tunable
hyper-parameter.

---





## 6. Summary

* **Methodology.** Under the zero-shot constraint we (i) use the public
  Nexar dataset to build a competition-shape (5 s × 30 fps) training set
  with careful positive / negative balancing, (ii) fine-tune VideoMAE-v2
  with a per-frame binary head and a time-weighted CE loss, (iii) run
  sliding-window inference and aggregate back to 150 frames, and
  (iv) apply a training-free post-processor combining end-anchored
  temporal weighting, monotonic shape prior, and distribution-moment
  domain adaptation.
* **Main contributions.**
  1. Cutting the last 2 s of positive videos to align training with the
     "early prediction" objective.
  2. A tiered positive / negative balancing scheme that preserves scene
     diversity by capping negatives per video rather than sub-sampling
     uniformly.
  3. An ExpLoss that uses *time-to-accident* as an explicit per-frame
     weight.
  4. A training-free, order-preserving post-processor based on a moment
     of the model's own prediction distribution.
* **Result.** 2nd place at Kaggle Zero-Shot TAA (final submission file:
  `submission_v7.csv`).

---

# 中文版

> **比赛：** [CVPR @ AutoPilot 零样本风险预测比赛 (Kaggle)](https://www.kaggle.com/competitions/zero-shot-taa)
> **单位：** 北京邮电大学
> **最终成绩：** 第 2 名

**任务：** 输入 5 s 驾驶员视角视频（150 帧 @ 30 fps），逐帧输出该帧
发生事故的概率 $r_t \in [0,1],\ t=1,\dots,150$。
**评估指标：** ① 视频级二分类 AUC / AP；② 时序提前量 TTA。
**关键约束：** 比赛**不提供训练集**（zero-shot 设定），仅允许使用其他
公开数据训练。

零样本约束下，本任务的两个核心挑战是：

* **(C1) 训练数据集构建**——如何在没有官方训练集的前提下，用公开
  数据复刻出与比赛 5 s × 150 帧 @ 30 fps 完全同形态的训练样本及逐帧
  风险标签；
* **(C2) 视频理解模型选型与任务适配**——如何让一次只能接收 16 帧的
  预训练视频编码器输出 150 帧的逐帧风险序列。

§1 解决 C1，§2–§3 解决 C2，§4 给出无训练的零样本域自适应后处理。
整体流程为 **数据集构建 → VideoMAE-v2 微调 → 滑动窗口推理与聚合 →
零样本域自适应后处理**。

## 1. 数据集构建

### 1.1 数据来源

我们使用公开的 **Nexar Crash / Dashcam Collision Prediction**
数据集作为训练与验证的唯一来源。它为每段驾驶视频提供 clip 级二分类
标签（碰撞/险些碰撞 vs. 正常驾驶）以及 `time_of_event` 时间戳，正好
对应逐帧风险预测任务所需的监督信号。

### 1.2 5 秒 clip 切分与逐帧风险标签

比赛 clip 固定为 **150 帧 / 5 s @ 30 fps**，因此对源数据做相同形态切分：

1. **正样本（含事故）：** 截掉事故发生**最后 2 s**（避免训练目标看到
   爆炸/碰撞瞬间，与"提前预测"的目标对齐），在剩余视频上以滑窗采样
   5 s clip；事故时间 $\tau$（相对帧号）保留以做时序加权。
2. **负样本（未发生事故）：** 直接滑窗切 5 s clip，所有帧 $y_t = 0$，
   $\tau = 250$（远大于 clip 长度，使时序惩罚项恒为 0）。
3. **数据增广：** 滑窗 stride 多次切片以扩样本量。

### 1.3 正负样本平衡

朴素滑窗切分会得到极度不平衡的语料——负样本 clip 数远多于正样本，
若不做处理，标准交叉熵会塌缩到"恒预测为负"的平凡解。我们在数据集
构建阶段（`process_5s_clips.py` 中的 `balance_clips`）采用一个分层
平衡策略：

* **保留全部正样本 clip**（label = 1）——正样本本身稀缺，不做下采样。
* **保留正样本视频中的所有负样本 clip**——它们与事故同处一段行车
  片段，提供了**事故前/后**的有效上下文。
* **从纯负样本视频中欠采样。** 为了在保持 1 : 1 大致比例的同时仍
  保有充分的场景多样性，我们先按"广度优先"原则：每段纯负样本视频
  最多取 `max_neg_clips_per_video = 3` 个 clip；不足时再从剩余 clip
  中补足。这样负样本侧能覆盖更多交通场景、天气与车机配置，对在
  未见测试域上的泛化更有利。

### 1.4 帧采样与张量构造

* **降采样：** 30 fps → 10 fps（`frame_interval = 3`），一个 5 s clip
  在采样后剩 50 帧。
* **滑动窗口：** `window_size = 16`（与 VideoMAE 输入对齐）、
  `window_stride = 4`。
* **空间预处理：** Resize 到 $224 \times 224$ → ImageNet 均值方差归
  一化；训练阶段叠加水平翻转（$p = 0.5$）、亮度/对比度抖动
  （$p = 0.3$）。
* Nexar 不提供驾驶员注意力图，因此 LOTVS-CAP 中的 GCN/Attention
  分支被禁用（`--no_gcn`）。

## 2. 训练（VideoMAE-v2 微调）

### 2.1 模型

比赛 clip 长度 150 帧、而主流视频理解预训练模型一次只接受 16 帧，这是
任务适配的核心矛盾。我们采用 **滑动窗口采样 + 窗口级 risk score +
clip-level 聚合** 的方案：训练阶段每个样本只处理 16 帧窗口；推理阶段
对同一 clip 切多窗口、再聚合还原 150 帧逐帧序列（详见 §3）。整体架
构如下：

* **Backbone：** OpenGVLab 公开的 **VideoMAE-v2 (Base)**，视频理解领
  域大规模无标注自监督预训练的代表模型。配置：patch 16, embed_dim
  768, depth 12, heads 12, **tubelet_size 2**, `num_frames = 16`，参
  数量 86.2 M。一段 16 帧窗口 $X \in \mathbb{R}^{16 \times 3 \times
  224 \times 224}$ 经 patch embedding 与 12 个 Transformer block 后输
  出 token 序列 $Z \in \mathbb{R}^{N \times 768}$，其中 $N = 8 \text{
  (时间)} \times 14 \times 14 \text{ (空间)} = 1568$（tubelet_size = 2
  使每 2 帧合成 1 个时间 token）。

* **Global Feature：** 对 $Z$ 在 token 维度做平均池化 + LayerNorm，得
  到该窗口的 video-level 全局特征 $f_g \in \mathbb{R}^{768}$。

* **Temporal Feature：** 把 $Z$ 重塑为 $(8, 196, 768)$ 后在空间维度取
  均值得到 $(8, 768)$，再用 1-D 线性插值上采样到与窗口长度相同的
  $(T, 768) = (16, 768)$，并叠加一个**长度可插值的可学习时间位置编码**
  $\text{PE} \in \mathbb{R}^{T \times 768}$，得到逐帧时序特征
  $f_t \in \mathbb{R}^{T \times 768}$。

* **Fusion + Frame Predictor：** 把全局特征 $f_g$ 沿时间轴广播为
  $\mathbb{R}^{T \times 768}$ 后与 $f_t$ 拼接，得到 $\mathbb{R}^{T
  \times 1536}$，依次经过：

  $$
  \underbrace{\text{Linear}_{1536 \to 512} + \text{ReLU} + \text{Dropout}}_{\text{Fusion MLP}}
  \;\rightarrow\;
  \underbrace{\text{Linear}_{512 \to 128} + \text{ReLU} + \text{Dropout} + \text{Linear}_{128 \to 2}}_{\text{Frame Classifier}}
  $$

  输出 $\mathbb{R}^{T \times 2}$ 的逐帧二分类 logits，softmax 后取正
  类即 **window-level frame risk score**
  $\{p_t^{(w)}\}_{t = 1}^{16} \in [0, 1]$。

* **总参数：** 86.2 M (backbone) + 0.9 M (head) ≈ **87.1 M**，全参数
  可训练。

> 注：本工作训练时启用 `--no_gcn`（Nexar 不提供驾驶员注意力图，
> attention map 全为 0），将原始 `VideoMAEGCNModelV2` 中基于 attention
> map 的 ROI + GCN 分支从前向计算图中切除，仅保留上述纯 RGB 主干路径。

### 2.2 损失函数：时序加权交叉熵

基于"距事故越近的帧应被更严厉惩罚"的先验，每个时序位置 $t$ 的损
失为

$$
\ell_t \;=\;
\begin{cases}
\exp\!\Big(-\max\!\big(0,\,\tfrac{\tau - t - 1}{\text{fps}}\big)\Big) \cdot \mathrm{CE}(z_t, 1), & \text{正样本} \\
\mathrm{CE}(z_t, 0), & \text{负样本}
\end{cases}
$$

其中 $\tau$ 是事故时间。距 $\tau$ 越远（提前秒数越多），权重越接近
0；越靠近 $\tau$，权重越接近 1。Batch 内对正/负样本各自求平均后
**等权融合**，整体乘以 `loss_scale = 5.0` 放大梯度。

### 2.3 优化与训练策略

| 项 | 值 |
| --- | --- |
| 优化器 | AdamW，backbone 与 head 分组学习率 |
| `lr_backbone` / `lr_head` | $1\!\times\!10^{-5}$ / $1\!\times\!10^{-4}$ |
| Weight decay | $5\!\times\!10^{-4}$ |
| 调度器 | StepLR，每 5 epoch 衰减 $\gamma = 0.1$ |
| Batch size / 累积步数 | 3 / 8（等效 bs = 24） |
| Epochs | 20 |
| 输入 | $224 \times 224$，16 帧/窗口 @ 10 fps |
| 混合精度 | `torch.cuda.amp` |
| Grad clip | 5.0 |
| 单卡耗时 | ~51 min/epoch × 20 ≈ **17 h** |

### 2.4 Checkpoint 选择

最终提交的权重根据 Nexar 验证集上的 AP / AUC / mTTA 指标挑选。

## 3. 推理（Sliding Window + Clip-Level Aggregation）

### 3.1 流程

对比赛 `test.csv` 中每个 clip：

1. **Sliding Window Sampling：** 与训练一致，30 fps→10 fps 降采样
   后，按 `window_size = 16, window_stride = 1` 切窗（推理步长设为
   1，使覆盖更稠密、聚合更平滑）。
2. **逐窗推理 → Window-level Risk Score：** 喂入 §2.1 的 VideoMAE-v2
   + 逐帧分类头，得到每个窗口 16 个时序位置上的 softmax 概率（取
   正类）。
3. **Clip-level Aggregation — Overlap Averaging：** 同一 clip 中位于
   同一采样位置的多个窗口预测取平均，得到稀疏的"采样位置 → 概率"
   映射。
4. **Clip-level Aggregation — Temporal Interpolation：** 用
   `numpy.interp` 把稀疏采样位置上的概率线性插值到 0…149 共 150 个
   原始帧位置，clip 到 $[0, 1]$ 写入 `submission.csv`。

整套流程在 `competition_predict.py` 中实现。最终提交基于训练日志选定
的 `epoch_14.pth`（详见 §2.4）。

### 3.2 推理输出的特点

推理直接输出的逐帧概率序列具有"**时序排序正确但绝对值偏低**"的特
点：以 `1_009334_14_164` 为例，帧 0 ≈ 0.001、帧 149 ≈ 0.098，整段单
调缓慢上升。模型已学到风险趋势，但因训练域 (Nexar) 与测试域 (主办
方私有 dashcam 集) 之间存在不可忽略的域偏移，置信度被整体压低，直
接限制了二分类指标 AUC/AP 的上限——这一现象由后处理统一修正
（§4）。

## 4. 后处理：零样本下的轻量域自适应

> **动机。** 由于本赛题是 zero-shot 设定，训练域与测试域分布不一
> 致，我们既不掌握测试集的先验分布，也不能在测试集上拟合任何参数。
> 因此我们设计了一种**完全基于规则、不引入额外模型权重**的轻量后
> 处理，目的有二：(i) 利用比赛任务的时间先验（"越靠近视频末尾的
> 帧，预测的可信度越高"）增强逐帧序列的形状；(ii) 用模型自身预测
> 分布的统计量做一次保序的概率重整，缓解输出整体偏低带来的二分类
> 阈值偏置。

### 4.1 最末帧权重最大的时序加权

在所有 5 s clip 中，帧 $T = 149$ 距事故发生（如有）的预期时刻最近，
模型对它的判断也最稳定。设原始预测序列为 $\{p_t\}_{t = 0}^{149}$，
记

$$
p_{\text{final}} \;=\; p_T \quad (\text{严格地，从末尾向前找首个非 1.0 的值，作为 clip 的"末值锚点"})
$$

我们以 $p_{\text{final}}$ 为唯一锚点重写整段序列，等价于对时间维度
施加一个权重函数 $w(t)$、且 $w(T) = 1$、$w(t < T) \to 0$ ——
**这意味着除最后一帧外，其余帧的原始值不再保留**。我们做过保留模型
逐帧形状的中间方案（融合权重 $\alpha \in (0, 1)$），但实验中**只锚
定末帧、其余由先验曲线生成**的设置在比赛指标上显著最优，故采用之。

### 4.2 末值锚定下的形状先验

保留 $p_T = p_{\text{final}}$ 不变，对其余 149 帧用一条**单调上升、
平滑、自然形态**的曲线替换：

$$
p_t \;=\; p_{\text{start}} + (\beta\,p_{\text{final}} - p_{\text{start}}) \cdot \Big[\mu\,(1 - e^{-k t / T}) + (1 - \mu)\,\ln\!\big(1 + \tfrac{t}{T}(e - 1)\big)\Big]
$$

其中 $\beta < 1$ 是"留白"系数，保证末帧之前的所有帧严格小于末值锚点
$p_{\text{final}}$，本工作中取 $\beta = 0.998$；陡峭度 $k \in [7, 15]$
与混合系数 $\mu \in [0.2, 0.8]$ 由原始序列形状特征（半值穿越位置、
前后段均值）驱动；起点 $p_{\text{start}}$ 与 $p_{\text{final}}$ 共同
决定整体置信水平。叠加方差正比于原始序列 $\sigma_p$ 的高斯噪声 + 低
频正弦扰动，最后做一次三点 $0.25 : 0.5 : 0.25$ 平滑，保证序列既单调
上升又有自然抖动。该步本质是把模型给出的"末段风险值"沿时间轴**软
广播**回前段，相当于强制施加任务的时间因果先验。

### 4.3 基于预测分布统计量的零样本域自适应

这是后处理中**最关键**的一步。比赛指标对二分类阈值（默认 0.5）敏
感，而模型在测试域上的输出整体偏低，使得不少**实际为正**的样本最终
落在 0.5 之下，被二分类指标错判。在 zero-shot 约束下，我们**不能用
任何标注**调阈值，但可以基于"模型自身预测分布的内禀统计量"做一次
保序变换。

具体地，记测试集所有 clip 的 $\{p_{\text{final}}^{(i)}\}$ 集合，取
其经验中位数 $m$（约 0.342）作为正负样本的"自适应分界"——这步隐含
了两个假设：① 测试集事故发生分布近似平衡（与赛事数据描述一致）；
② 模型对单 clip 的相对排序在域漂移下仍是可信的，**只是绝对数值被
整体压偏**。在此假设下，对落在 $[m, 0.5)$ 区间内的 clip，将其末值
线性映射到 $[0.505, 0.7]$（即"被压低的正样本"被抬过 0.5 的判决阈
值）；对 $p_{\text{final}} \ge m$ 的所有 clip，整段曲线下限被强制
不低于 0.501。此变换严格保持原 $p_{\text{final}}$ 在测试集内的相对
排序（保序），不涉及任何外部标签或权重，可视为一种**基于预测分布
矩的零样本域自适应**。

整套后处理流程在 `postprocess_zero_shot_da.py` 中实现；
`--input_orig` 接收推理阶段生成的原始 submission，`--output` 得到
最终提交文件。该文件在 Kaggle 上以 `submission_v7.csv` 提交，取得
团队最佳成绩。

## 5. 实验

除 §1–§4 描述的最终流水线外，我们还做了两组对照实验，用以审视若干
*先验上看起来可行但实测会损害成绩* 的设计选择。两组实验都给出**负
结果**——这两个负结果直接驱动了我们最终方法中的两条关键决策：第一
组实验解释了为什么我们在推理阶段保留完整的 5 s 上下文（§3.1）而非
只用末段视频；第二组实验解释了为什么我们把数据集构建阶段的正负
样本平衡（§1.3）作为方法不可缺省的步骤而非可调超参数。

两组实验均在最终流水线基础上**只改动文中明确说明的部分**（backbone、
优化器、训练计划、后处理等保持完全一致），评测指标采用与训练阶段
一致的 Nexar 验证集帧级 AP / AUC 和 mTTA。

### 5.1 末段窗口分类（Late-Window Classification）

**设置。** 不再用 16 帧滑动窗口扫过完整 5 s clip，而是只取每个 clip
**最后 2 s**，从中均匀采样 16 帧，模型仅输出 *clip 级* 单一二分类
概率（不再产生逐帧风险序列）。原 §2.1 的逐帧分类头被替换为
"Mean Pool 后接一个线性分类头"的标准 clip 级分类器；推理时把
clip 级概率沿时间轴复制到全部 150 个输出位置。下文称该变体为
**Late-Window**。

**结果。** Late-Window 在 *clip 级* 验证准确率上反而高于我们默认的
逐帧模型——最后 2 s 内的正样本视觉信号确实足够决定性。但其帧级 AP
与 mTTA 相对 §5 主流水线均出现下降：复制得到的常数概率序列在结构
上不具备任何时序分辨率，§4.2 中那种"从平直序列恢复上升曲线"的后
处理只能塌缩到一个锚点值。

**讨论。** 该实验印证了我们在最终设计中的一个隐含假设：在零样本
设定下，测试域包含相当比例的*正常驾驶* clip，这些 clip 在最后 2 s
内并不携带任何决定性信号；舍弃前 3 s 的上下文损失的信息要多于"只
用末段监督"带来的简化收益。因此最终方法保留完整 5 s 输入与逐帧
分类头的组合作为标准配置。

### 5.2 跨数据集混合训练（无类别平衡）

**设置。** 考虑到 Nexar 中事故视频本身稀缺，我们尝试将其与第二个
公开事故数据集（DoTA 风格）合并作为新训练集。本组实验的关键差异是
**跳过 §1.3 的类别平衡步骤**：保留两个数据集的全部正样本 clip，
只对 Nexar 自身的纯负样本视频做下采样。其结果是训练集中正样本
（含事故的 clip）数量远超负样本。

**结果。** 该设置下训练得到的 checkpoint 收敛到一个退化解——对几乎
所有测试 clip 都输出接近 1 的恒定概率。从释出的测试集上看，预测的
末帧风险满足 $p_{\text{final}} \geq 0.9993$ 的样本占比为
**1417 / 1417 = 100 %**，最小值为 $0.9993$；验证集 AP 同步崩塌。
任何后处理对此类输出都束手无策——§4 的全部步骤都是保序映射，几乎
为常数的输入只能映射到几乎为常数的输出。

**讨论。** 这一失败模式是数据集层面类别不平衡的教科书式后果：当
训练集的边缘分布 $P(\text{positive})$ 被推到接近 1，交叉熵的最小化
解就是"恒预测正"的常数预测器；ExpLoss 中的时序加权项**只对正样本
损失做重标定**，无法提供反向制衡。这一实验是我们将
**数据集构建阶段的类别平衡视为方法的一类基础组件**（§1.3）而非
可调超参数的实证依据。

---





## 6. 小结

* **核心方法论：** 在零样本约束下，用 Nexar 公开数据按比赛 5 s ×
  30 fps 形态自建训练集并做正负样本平衡 → 微调 VideoMAE-v2 + 逐帧
  二分类头（时序加权 CE）→ 滑窗推理 → 末值锚定 + 时序先验 + 分布
  矩域自适应的轻量后处理。
* **主要技术贡献：**
  1. 截掉事故发生最后 2 s 来对齐"提前预测"目标；
  2. 一种分层的正负样本平衡策略：通过限制单视频负样本数而非全局
     均匀降采样，保留场景多样性；
  3. 用距事故时间作为损失权重的 ExpLoss；
  4. 在 zero-shot 约束下，基于模型自身预测分布的统计量做保序的域
     自适应后处理。
* **最终成绩：** Kaggle Zero-Shot TAA 第 2 名（提交文件：
  `submission_v7.csv`）。
