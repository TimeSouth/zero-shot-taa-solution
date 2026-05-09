"""
Stage 2 — Training entry point.

Fine-tunes VideoMAE-v2 + per-frame classifier on the 5 s Nexar clips
produced by `process_5s_clips.py`.  See Section 2 of `TECH_REPORT.md`.
"""

import argparse
import os
import sys
import random
import time
import datetime
import logging
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
try:
    from torch.amp import GradScaler, autocast as _autocast
    def amp_autocast(enabled=True):
        return _autocast('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast as _autocast
    def amp_autocast(enabled=True):
        return _autocast(enabled=enabled)
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------------------------------------------------------------
# Allow running this script directly from the repository root.
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nexar_config import *
from nexar_dataset import NexarVideoMAEDataset, NexarGlobalDataset, NexarLast2sDataset
from videomae_gcn import (
    VideoMAEGCNModel, VideoMAEGCNModelV2, freeze_backbone_layers,
    get_videomae_backbone, get_inner_model, extract_videomae_tokens,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Nexar 5s Clip VideoMAE+GCN 训练")
    parser.add_argument("--model", choices=["v1", "v2"], default="v2",
                        help="模型版本: v1=窗口级预测, v2=逐帧预测")
    parser.add_argument("--backbone", choices=["base", "giant"], default="base",
                        help="VideoMAEv2 backbone 规模: base(768d) 或 giant(1408d)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="实验名称")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr_backbone", type=float, default=LEARNING_RATE_BACKBONE)
    parser.add_argument("--lr_head", type=float, default=LEARNING_RATE_HEAD)
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--window_stride", type=int, default=WINDOW_STRIDE)
    parser.add_argument("--sample_fps", type=int, default=SAMPLE_FPS)
    parser.add_argument("--debug", action="store_true", help="Debug 模式")
    parser.add_argument("--global_mode", action="store_true",
                        help="全局模式: 整个 clip 降采样后一次性输入")
    parser.add_argument("--global_16f", action="store_true",
                        help="16帧全局模式: 整个 clip 降采样到16帧")
    parser.add_argument("--freeze_backbone_layers", type=int, default=FREEZE_BACKBONE_LAYERS,
                        help="冻结 VideoMAE backbone 前 N 层")
    parser.add_argument("--loss_scale", type=float, default=None,
                        help="Loss 缩放因子")
    parser.add_argument("--clip_weight", type=float, default=0.0,
                        help="Clip 级别约束权重")
    parser.add_argument("--no_gcn", action="store_true",
                        help="禁用 GCN 分支（推荐，因为 Nexar 无 attention map）")
    parser.add_argument("--random_roi", action="store_true",
                        help="消融实验: ROI 使用随机采样")
    # ---- 混合数据集训练 ----
    parser.add_argument("--mix_dota", action="store_true",
                        help="混合 DoTA 数据集训练（正样本，只取 anomaly_start 之前的帧）")
    parser.add_argument("--dota_root", type=str,
                        default=os.environ.get(
                            "DOTA_ROOT",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "data", "DoTA", "processed"),
                        ),
                        help="DoTA processed dataset root (only used by "
                             "--mix_dota; defaults to $DOTA_ROOT).")
    parser.add_argument("--nexar_ann_train", type=str, default=None,
                        help="自定义 Nexar 训练标注文件（如未平衡版本），默认用 nexar_config 中的 ANN_TRAIN")
    parser.add_argument("--balance_mix", type=float, default=0,
                        help="混合数据集正负样本平衡: 0=不平衡, 1.0=1:1, 2.0=正样本:负样本=1:2")
    # ---- Last-2s 方案 (2nd-place solution adapted) ----
    parser.add_argument("--last2s", action="store_true",
                        help="启用 last-2s 方案: 取最后2s做二分类(2nd place solution)")
    parser.add_argument("--last_seconds", type=float, default=LAST_NS,
                        help="取最后 N 秒（默认 2.0）")
    parser.add_argument("--last2s_frames", type=int, default=LAST_NS_NUM_FRAMES,
                        help="最后N秒采样帧数（默认 16）")
    parser.add_argument("--temperature", type=float, default=LAST_NS_TEMPERATURE,
                        help="温度缩放因子（默认 2.0）")
    parser.add_argument("--no_balance", action="store_true",
                        help="禁用正负样本平衡（默认启用）")
    return parser.parse_args()


# ============ 工具函数 ============

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def setup_logging(exp_name=None):
    if exp_name:
        log_dir = os.path.join(LOG_DIR, exp_name)
        weights_dir = os.path.join(WEIGHTS_DIR, exp_name)
    else:
        log_dir = LOG_DIR
        weights_dir = WEIGHTS_DIR
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"train_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(), weights_dir


def collate_fn(batch):
    rgb = torch.stack([b[0] for b in batch])
    attn = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    metas = [b[3] for b in batch]
    return rgb, attn, labels, metas


# ============ 损失函数 ============

class ExpLoss(nn.Module):
    """
    指数衰减交叉熵损失（与 LOTVS-CAP 一致）。
    注意: Nexar 的 toa 已经是相对于 clip start 的帧号，
    不需要再减去 start。
    """

    def __init__(self, loss_scale=None, clip_weight=0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.loss_scale = loss_scale if loss_scale is not None else LOSS_SCALE
        self.clip_weight = clip_weight

    def forward(self, logits, target, metas=None):
        B, T, _ = logits.shape
        device = logits.device

        logits_flat = logits.reshape(B * T, 2)
        target_onehot = torch.zeros(B * T, 2, device=device, dtype=torch.float32)
        target_cls = torch.zeros(B * T, device=device, dtype=torch.long)
        toa_vec = torch.zeros(B * T, device=device, dtype=torch.float32)
        time_vec = torch.zeros(B * T, device=device, dtype=torch.float32)

        for b in range(B):
            clip_label = metas[b]['label'] if metas is not None else 0
            # Nexar 的 toa 已经是相对帧号，不需要减 start
            toa_raw = metas[b]['toa'] if metas is not None else 0
            win_indices = metas[b].get('win_indices', None) if metas is not None else None

            # 正样本: toa 直接使用（已是相对帧号）
            # 负样本: toa = 250（远大于 clip 长度 150）
            toa_orig = toa_raw

            for t in range(T):
                idx = b * T + t
                if clip_label == 1:
                    target_onehot[idx, 1] = 1.0
                    target_cls[idx] = 1
                else:
                    target_onehot[idx, 0] = 1.0
                    target_cls[idx] = 0
                toa_vec[idx] = toa_orig

                if win_indices is not None:
                    t_clamped = min(t, len(win_indices) - 1)
                    time_vec[idx] = win_indices[t_clamped]
                else:
                    time_vec[idx] = t

        penalty = -torch.max(
            torch.zeros_like(toa_vec),
            (toa_vec - time_vec - 1) / FPS
        )
        ce = self.ce_loss(logits_flat, target_cls)
        pos_loss = -torch.mul(torch.exp(penalty), -ce)
        neg_loss = ce

        pos_mask = target_onehot[:, 1]
        neg_mask = target_onehot[:, 0]
        n_pos = pos_mask.sum().clamp(min=1)
        n_neg = neg_mask.sum().clamp(min=1)

        pos_loss_mean = (pos_loss * pos_mask).sum() / n_pos
        neg_loss_mean = (neg_loss * neg_mask).sum() / n_neg

        loss = 0.5 * pos_loss_mean + 0.5 * neg_loss_mean
        loss = loss * self.loss_scale

        if self.clip_weight > 0:
            clip_logit_diff = logits[:, :, 1] - logits[:, :, 0]
            clip_max_logits = clip_logit_diff.max(dim=1)[0]
            clip_labels = torch.zeros(B, device=device, dtype=torch.float32)
            for b in range(B):
                if metas is not None and metas[b]['label'] == 1:
                    clip_labels[b] = 1.0
            clip_loss = nn.functional.binary_cross_entropy_with_logits(
                clip_max_logits, clip_labels, reduction='mean'
            )
            loss = loss + self.clip_weight * clip_loss

        return loss


def get_loss_fn(loss_scale=None, clip_weight=0.0):
    return ExpLoss(loss_scale=loss_scale, clip_weight=clip_weight)


# ============ Last-2s 方案: 简单二分类模型 ============

class VideoMAEClassifier(nn.Module):
    """
    简单 VideoMAE + 线性分类头，适配 2nd-place solution 方案。
    将碰撞预测任务简化为二分类: collision(1) vs normal(0)。
    """

    def __init__(
        self,
        model_id=MODEL_ID,
        embed_dim=EMBED_DIM,
        num_classes=2,
        drop_path_rate=DROP_PATH_RATE,
        dropout=0.1,
    ):
        super().__init__()
        self.backbone = get_videomae_backbone(model_id, drop_path_rate)
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, rgb, attn_maps=None):
        """
        rgb: (B, T, 3, H, W)
        attn_maps: 不使用，保持接口兼容

        返回:
            logits: (B, num_classes) 分类 logits
        """
        B, T, C, H, W = rgb.shape
        # VideoMAE 要求帧数为 tubelet_size(2) 的倍数
        T_input = T
        rgb_for_mae = rgb
        if T % 2 != 0:
            rgb_for_mae = torch.cat([rgb, rgb[:, -1:]], dim=1)
            T_input = T + 1

        rgb_input = rgb_for_mae.permute(0, 2, 1, 3, 4).float()  # (B, C, T, H, W)
        tokens = extract_videomae_tokens(self.backbone, rgb_input)  # (B, N, embed_dim)

        # 全局平均池化
        global_feat = self.fc_norm(tokens.mean(dim=1))  # (B, embed_dim)
        global_feat = self.dropout(global_feat)
        logits = self.classifier(global_feat)  # (B, num_classes)

        return logits


def validate_last2s(model, val_loader, criterion, device, temperature=2.0):
    """Last-2s 方案的验证函数"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for rgb, attn, labels, metas in val_loader:
            rgb = rgb.to(device)
            binary_labels = torch.tensor(
                [m['binary_label'] for m in metas], dtype=torch.long, device=device
            )

            with amp_autocast(enabled=USE_AMP):
                logits = model(rgb)
                loss = criterion(logits / temperature, binary_labels)

            total_loss += loss.item() * rgb.size(0)
            pred = logits.argmax(dim=1)
            correct += pred.eq(binary_labels).sum().item()
            total += binary_labels.size(0)

            # 收集 softmax 概率用于计算 AUC
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(binary_labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1) * 100.0

    # 计算 AUC
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except (ValueError, Exception):
        auc = 0.0

    # 计算 AP
    try:
        ap = average_precision_score(all_labels, all_scores)
    except (ValueError, Exception):
        ap = 0.0

    return avg_loss, accuracy, auc, ap


def train_last2s(args):
    """
    Last-2s 方案训练主函数。
    适配 2nd-place solution: 简单二分类 + 温度缩放 + 数据平衡。
    """
    exp_name = args.exp_name or f"last2s_{args.last_seconds}s"
    logger, weights_dir = setup_logging(exp_name)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"Last-{args.last_seconds}s Binary Classification Training")
    logger.info(f"  (adapted from 2nd-place Nexar solution)")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: batch={args.batch_size}, epochs={args.epochs}")
    logger.info(f"  lr={args.lr_backbone} (unified)")
    logger.info(f"  last_seconds={args.last_seconds}, target_frames={args.last2s_frames}")
    logger.info(f"  temperature={args.temperature}")
    logger.info(f"  balance={not args.no_balance}")
    logger.info(f"  freeze_backbone_layers={args.freeze_backbone_layers}")
    logger.info(f"  weights_dir={weights_dir}")
    logger.info("=" * 60)

    # ---- 数据集 ----
    balance = not args.no_balance
    train_ds = NexarLast2sDataset(
        ann_file=ANN_TRAIN, rgb_root=DATA_ROOT_TRAIN,
        last_seconds=args.last_seconds, target_frames=args.last2s_frames,
        original_fps=FPS, img_size=IMG_SIZE,
        is_train=True, balance=balance,
        seed=SEED,
    )
    val_ds = NexarLast2sDataset(
        ann_file=ANN_TEST, rgb_root=DATA_ROOT_TEST,
        last_seconds=args.last_seconds, target_frames=args.last2s_frames,
        original_fps=FPS, img_size=IMG_SIZE,
        is_train=False, balance=False,  # 验证集不做平衡
        seed=SEED,
    )

    # Debug 模式
    if args.debug:
        debug_limit = 100
        if len(train_ds) > debug_limit:
            train_ds.samples = train_ds.samples[:debug_limit]
        if len(val_ds) > debug_limit:
            val_ds.samples = val_ds.samples[:debug_limit]
        logger.info(f"[DEBUG] 截断数据集: train={len(train_ds)}, val={len(val_ds)}")

    n_pos_train = sum(1 for s in train_ds.samples if s[2] == 1)
    n_neg_train = sum(1 for s in train_ds.samples if s[2] == 0)
    n_pos_val = sum(1 for s in val_ds.samples if s[2] == 1)
    n_neg_val = sum(1 for s in val_ds.samples if s[2] == 0)
    logger.info(f"Train: {len(train_ds)} samples (pos={n_pos_train}, neg={n_neg_train})")
    logger.info(f"Val: {len(val_ds)} samples (pos={n_pos_val}, neg={n_neg_val})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn, collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn, collate_fn=collate_fn,
    )

    # ---- 模型 ----
    model = VideoMAEClassifier(
        model_id=MODEL_ID, embed_dim=EMBED_DIM,
        num_classes=2, drop_path_rate=DROP_PATH_RATE,
    ).to(device)

    # 冻结 backbone 层
    if args.freeze_backbone_layers > 0:
        frozen_count = freeze_backbone_layers(model, args.freeze_backbone_layers)
        logger.info(f"  冻结 backbone 前 {args.freeze_backbone_layers} 层"
                    f"（共冻结 {frozen_count} 个参数）")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: total={total_params/1e6:.1f}M, trainable={trainable_params/1e6:.1f}M")

    # ---- 优化器（统一学习率，与 2nd-place 方案一致）----
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_backbone, weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=SCHEDULER_ETA_MIN,
    )

    scaler = GradScaler(enabled=USE_AMP)
    criterion = nn.CrossEntropyLoss()

    # 恢复训练
    start_epoch = 0
    best_auc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("val_auc", 0.0)
        logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    # ---- 训练循环 ----
    temperature = args.temperature
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (rgb, attn, labels, metas) in enumerate(train_loader):
            rgb = rgb.to(device)
            binary_labels = torch.tensor(
                [m['binary_label'] for m in metas], dtype=torch.long, device=device
            )

            with amp_autocast(enabled=USE_AMP):
                logits = model(rgb)
                # 温度缩放
                loss = criterion(logits / temperature, binary_labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * ACCUMULATION_STEPS * rgb.size(0)
            epoch_samples += rgb.size(0)

            # 统计准确率
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                epoch_correct += pred.eq(binary_labels).sum().item()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            if batch_idx % 50 == 0:
                avg_loss = epoch_loss / max(epoch_samples, 1)
                acc = epoch_correct / max(epoch_samples, 1) * 100
                logger.info(
                    f"E{epoch+1} B{batch_idx}/{len(train_loader)} "
                    f"loss={loss.item()*ACCUMULATION_STEPS:.4f} avg={avg_loss:.4f} acc={acc:.1f}%"
                )

        # 处理最后不足 accumulation_steps 的梯度
        if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        # ---- 验证 ----
        val_loss, val_acc, val_auc, val_ap = validate_last2s(
            model, val_loader, criterion, device, temperature=temperature
        )

        elapsed = time.time() - t0
        train_avg_loss = epoch_loss / max(epoch_samples, 1)
        train_acc = epoch_correct / max(epoch_samples, 1) * 100

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} [{elapsed:.0f}s] "
            f"train_loss={train_avg_loss:.4f} train_acc={train_acc:.1f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% "
            f"val_AUC={val_auc:.4f} val_AP={val_ap:.4f}"
        )

        # ---- 保存 checkpoint ----
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_ap": val_ap,
            "config": {
                "mode": "last2s",
                "last_seconds": args.last_seconds,
                "target_frames": args.last2s_frames,
                "temperature": args.temperature,
                "balance": balance,
                "batch_size": args.batch_size,
                "lr": args.lr_backbone,
                "freeze_backbone_layers": args.freeze_backbone_layers,
                "dataset": "nexar_clips_5s",
            },
        }
        torch.save(ckpt, os.path.join(weights_dir, "latest.pth"))

        save_path = os.path.join(weights_dir, f"epoch_{epoch+1}.pth")
        torch.save(ckpt, save_path)
        logger.info(f"  Saved checkpoint to {save_path}")

        if is_best:
            save_path = os.path.join(
                weights_dir,
                f"best_last2s_AUC{val_auc:.4f}_AP{val_ap:.4f}.pth"
            )
            torch.save(ckpt, save_path)
            logger.info(f"  ★ New best! AUC={val_auc:.4f} Saved to {save_path}")

    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")


# ============ Clip 级评估工具 ============

def aggregate_clip_predictions(val_dataset, window_preds, sample_fps, original_fps):
    """将滑动窗口的逐帧预测聚合为 clip 级别的完整帧概率序列。"""
    frame_interval = max(1, original_fps // sample_fps)

    clip_frame_preds = defaultdict(lambda: defaultdict(list))

    for win_idx, probs in window_preds.items():
        if hasattr(val_dataset, 'dataset'):
            real_idx = val_dataset.indices[win_idx]
            ds = val_dataset.dataset
        else:
            real_idx = win_idx
            ds = val_dataset

        clip_idx, win_indices, win_start = ds.windows[real_idx]

        for t, rel_idx in enumerate(win_indices[:len(probs)]):
            sampled_pos = rel_idx // frame_interval
            clip_frame_preds[clip_idx][sampled_pos].append(probs[t])

    all_pred = []
    all_labels = []
    all_toas = []
    all_starts = []

    if hasattr(val_dataset, 'dataset'):
        ds = val_dataset.dataset
    else:
        ds = val_dataset

    for clip_idx in sorted(clip_frame_preds.keys()):
        clip = ds.clips[clip_idx]
        clip_length = clip.end - clip.start + 1
        num_sampled = len(range(0, clip_length, frame_interval))

        sampled_probs = np.zeros(num_sampled, dtype=np.float32)
        for pos, prob_list in clip_frame_preds[clip_idx].items():
            if pos < num_sampled:
                sampled_probs[pos] = np.mean(prob_list)

        frame_probs = np.zeros(clip_length, dtype=np.float32)
        if num_sampled >= 2:
            sampled_positions = np.arange(num_sampled) * frame_interval
            frame_probs = np.interp(
                np.arange(clip_length),
                sampled_positions,
                sampled_probs
            ).astype(np.float32)
        elif num_sampled == 1:
            frame_probs[:] = sampled_probs[0]

        # Nexar 的 toa 已经是相对帧号，直接使用
        toa_out = clip.toa

        all_pred.append(frame_probs)
        all_labels.append(clip.label)
        all_toas.append(toa_out)
        all_starts.append(clip.start)

    return all_pred, all_labels, all_toas, all_starts


def compute_ap_mtta(all_pred, all_labels, all_toas, fps):
    """计算 AP, mTTA, TTA@R80"""
    max_T = max(len(p) for p in all_pred)
    pred_mat = np.zeros((len(all_pred), max_T), dtype=np.float32)
    label_arr = np.array(all_labels, dtype=np.int32)
    toa_arr = np.array(all_toas, dtype=np.int32)
    for i, p in enumerate(all_pred):
        pred_mat[i, :len(p)] = p

    total_seconds = max_T / fps

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(toa_arr):
        if label_arr[idx] > 0:
            pred = pred_mat[idx, :int(toa)]
        else:
            pred = pred_mat[idx, :]
        if len(pred) > 0:
            min_pred = min(min_pred, np.min(pred))
        preds_eval.append(pred)
        n_frames += len(pred)

    if n_frames == 0 or np.sum(label_arr) == 0:
        return 0.0, 0.0, 0.0

    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0

    for Th in np.arange(max(min_pred, 0), 1.0, 0.1):
        Tp = 0.0
        Tp_Fp = 0.0
        time_acc = 0.0
        counter = 0.0
        for i in range(len(preds_eval)):
            tp = np.where(preds_eval[i] * label_arr[i] >= Th)
            Tp += float(len(tp[0]) > 0)
            if float(len(tp[0]) > 0) > 0:
                time_acc += tp[0][0] / float(toa_arr[i])
                counter = counter + 1
            Tp_Fp += float(len(np.where(preds_eval[i] >= Th)[0]) > 0)
        if Tp_Fp == 0:
            continue
        else:
            Precision[cnt] = Tp / Tp_Fp
        if np.sum(label_arr) == 0:
            continue
        else:
            Recall[cnt] = Tp / np.sum(label_arr)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1 - time_acc / counter)
        cnt += 1

    if cnt == 0:
        return 0.0, 0.0, 0.0

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _, rep_index = np.unique(Recall, return_index=True)
    rep_index = rep_index[1:]
    if len(rep_index) == 0:
        return 0.0, 0.0, 0.0

    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]

    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    mTTA = np.mean(new_Time) * total_seconds

    sort_t = new_Time[np.argsort(new_Recall)]
    sort_r = np.sort(new_Recall)
    TTA_R80 = sort_t[np.argmin(np.abs(sort_r - 0.8))] * total_seconds

    return float(AP), float(mTTA), float(TTA_R80)


def compute_tta05(all_pred, all_labels, all_toas, fps, thresh=0.5):
    """计算 TTA@0.5"""
    time_sum, count = 0.0, 0
    for pred, label, toa in zip(all_pred, all_labels, all_toas):
        if label != 1:
            continue
        hits = np.where(np.array(pred) >= thresh)[0]
        if len(hits) > 0:
            time_sum += max((toa - hits[0]) / fps, 0)
            count += 1
    return time_sum / count if count > 0 else 0.0


def compute_auc(all_pred, all_labels, all_toas):
    """Video-level AUC"""
    scores = []
    for pred, label, toa in zip(all_pred, all_labels, all_toas):
        pred = np.array(pred)
        if label > 0 and int(toa) > 0:
            scores.append(float(np.max(pred[:int(toa)])))
        else:
            scores.append(float(np.max(pred)))
    try:
        return roc_auc_score(all_labels, scores)
    except (ValueError, Exception):
        return 0.0


def compute_standard_ap(all_pred, all_labels, all_toas):
    """标准 AP（sklearn）"""
    scores = []
    for pred, label, toa in zip(all_pred, all_labels, all_toas):
        pred = np.array(pred)
        if label > 0 and int(toa) > 0:
            scores.append(float(np.max(pred[:int(toa)])))
        else:
            scores.append(float(np.max(pred)))
    try:
        return average_precision_score(all_labels, scores)
    except (ValueError, Exception):
        return 0.0


# ============ 验证 ============

def validate(model, val_loader, val_dataset, loss_fn, device, logger, sample_fps, original_fps,
             dataset_name="Nexar"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    window_preds = {}
    win_counter = 0

    with torch.no_grad():
        for rgb, attn, labels, metas in val_loader:
            rgb = rgb.to(device)
            attn = attn.to(device)
            labels = labels.to(device)

            with amp_autocast(enabled=USE_AMP):
                pred_logits, pred_frames_logits = model(rgb, attn)

                if pred_frames_logits.dim() == 3:
                    loss = loss_fn(pred_frames_logits, labels, metas=metas)
                else:
                    loss = loss_fn(pred_logits.unsqueeze(1).unsqueeze(-1).expand(-1, -1, 2),
                                   labels[:, -1:], metas=metas)

            total_loss += loss.item() * rgb.size(0)
            total_samples += rgb.size(0)

            if pred_frames_logits.dim() == 3:
                frame_probs = torch.softmax(pred_frames_logits, dim=-1)[:, :, 1].cpu().numpy()
            else:
                frame_probs = torch.sigmoid(pred_logits).cpu().numpy().reshape(-1, 1)

            for i in range(rgb.size(0)):
                window_preds[win_counter] = frame_probs[i]
                win_counter += 1

    avg_loss = total_loss / max(total_samples, 1)

    all_pred, all_labels, all_toas, all_starts = aggregate_clip_predictions(
        val_dataset, window_preds, sample_fps, original_fps
    )

    eval_fps = original_fps
    AP, mTTA, TTA_R80 = compute_ap_mtta(all_pred, all_labels, all_toas, eval_fps)
    tta05 = compute_tta05(all_pred, all_labels, all_toas, eval_fps)
    try:
        auc = compute_auc(all_pred, all_labels, all_toas)
    except Exception:
        auc = 0.0
    std_ap = compute_standard_ap(all_pred, all_labels, all_toas)

    pos_clips = sum(1 for l in all_labels if l == 1)
    logger.info(
        f"  Val[{dataset_name}]: loss={avg_loss:.4f} | clips={len(all_labels)} (pos={pos_clips})"
    )
    logger.info(
        f"  Val[{dataset_name}]: AP={AP:.4f} sAP={std_ap:.4f} AUC={auc:.4f} "
        f"TTA@0.5={tta05:.2f}s mTTA={mTTA:.2f}s TTA@R80={TTA_R80:.2f}s"
    )

    return avg_loss, AP, mTTA


# ============ 训练主循环 ============

def train(args):
    logger, weights_dir = setup_logging(args.exp_name)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"Nexar 5s Clip VideoMAE+GCN Training (model={args.model})")
    if args.exp_name:
        logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: batch={args.batch_size}, epochs={args.epochs}")
    logger.info(f"  lr_backbone={args.lr_backbone}, lr_head={args.lr_head}")
    logger.info(f"  window_stride={args.window_stride}, sample_fps={args.sample_fps}")
    logger.info(f"  global_mode={args.global_mode}, global_16f={args.global_16f}")
    logger.info(f"  freeze_backbone_layers={args.freeze_backbone_layers}")
    logger.info(f"  loss_scale={args.loss_scale if args.loss_scale is not None else LOSS_SCALE}")
    logger.info(f"  no_gcn={args.no_gcn}, random_roi={args.random_roi}")
    logger.info(f"  mix_dota={args.mix_dota}")
    logger.info(f"  weights_dir={weights_dir}")
    logger.info("=" * 60)

    # ---- 数据集 ----
    # 支持自定义 Nexar 标注文件（如未平衡版本，包含更多负样本）
    nexar_ann_train = args.nexar_ann_train if args.nexar_ann_train else ANN_TRAIN
    if nexar_ann_train != ANN_TRAIN:
        logger.info(f"  [自定义标注] Nexar 训练标注: {nexar_ann_train}")

    if args.global_16f:
        train_ds = NexarGlobalDataset(
            ann_file=nexar_ann_train, rgb_root=DATA_ROOT_TRAIN,
            target_frames=GLOBAL_16F_NUM_FRAMES,
            sample_fps=args.sample_fps, original_fps=FPS,
            img_size=IMG_SIZE, is_train=True,
        )
        val_ds = NexarGlobalDataset(
            ann_file=ANN_TEST, rgb_root=DATA_ROOT_TEST,
            target_frames=GLOBAL_16F_NUM_FRAMES,
            sample_fps=args.sample_fps, original_fps=FPS,
            img_size=IMG_SIZE, is_train=False,
        )
        actual_window_size = GLOBAL_16F_NUM_FRAMES
        logger.info(f"  [16帧全局模式] target_frames={GLOBAL_16F_NUM_FRAMES}")
    elif args.global_mode:
        train_ds = NexarGlobalDataset(
            ann_file=nexar_ann_train, rgb_root=DATA_ROOT_TRAIN,
            target_frames=GLOBAL_NUM_FRAMES,
            sample_fps=args.sample_fps, original_fps=FPS,
            img_size=IMG_SIZE, is_train=True,
        )
        val_ds = NexarGlobalDataset(
            ann_file=ANN_TEST, rgb_root=DATA_ROOT_TEST,
            target_frames=GLOBAL_NUM_FRAMES,
            sample_fps=args.sample_fps, original_fps=FPS,
            img_size=IMG_SIZE, is_train=False,
        )
        actual_window_size = GLOBAL_NUM_FRAMES
        logger.info(f"  [全局模式] target_frames={GLOBAL_NUM_FRAMES}")
    else:
        train_ds = NexarVideoMAEDataset(
            ann_file=nexar_ann_train, rgb_root=DATA_ROOT_TRAIN,
            window_size=WINDOW_SIZE, window_stride=args.window_stride,
            sample_fps=args.sample_fps, original_fps=FPS,
            img_size=IMG_SIZE, is_train=True,
        )
        val_ds = NexarVideoMAEDataset(
            ann_file=ANN_TEST, rgb_root=DATA_ROOT_TEST,
            window_size=WINDOW_SIZE, window_stride=args.window_stride,
            sample_fps=args.sample_fps, original_fps=FPS,
            img_size=IMG_SIZE, is_train=False,
        )
        actual_window_size = WINDOW_SIZE

    # ---- 混合额外数据集到训练集（可选）----
    extra_train_datasets = []

    if args.mix_dota:
        dota_ann_train = os.path.join(args.dota_root, "training.txt")
        dota_rgb_train = os.path.join(args.dota_root, "training", "rgb_videos")
        if os.path.exists(dota_ann_train):
            logger.info(f"[混合训练] 加载 DoTA 训练数据...")
            # DoTA 是 10fps，不需要 sample_fps 降采样（frame_interval=1）
            dota_train_ds = NexarVideoMAEDataset(
                ann_file=dota_ann_train, rgb_root=dota_rgb_train,
                window_size=WINDOW_SIZE, window_stride=args.window_stride,
                sample_fps=10, original_fps=10,  # DoTA 已经是 10fps
                img_size=IMG_SIZE, is_train=True,
            )
            extra_train_datasets.append(("DoTA", dota_train_ds))
            logger.info(f"  DoTA train: {len(dota_train_ds)} windows")
        else:
            logger.warning(f"  DoTA 标注文件不存在: {dota_ann_train}")

    if extra_train_datasets:
        from torch.utils.data import ConcatDataset
        all_train_parts = [train_ds] + [ds for _, ds in extra_train_datasets]
        original_len = len(train_ds)
        train_ds = ConcatDataset(all_train_parts)
        logger.info(f"[混合训练] 合并后训练集: {len(train_ds)} windows "
                    f"(Nexar={original_len}" +
                    "".join(f", {name}={len(ds)}" for name, ds in extra_train_datasets) + ")")

    # ---- 混合数据集正负样本平衡（可选）----
    # balance_mix > 0 时启用：欠采样多数类使正负比达到 1:balance_mix
    # 欠采样正样本时按优先级：先减 DoTA → 再减 Nexar
    if args.balance_mix > 0:
        # 支持 ConcatDataset 和普通 Dataset
        candidate_datasets = train_ds.datasets if hasattr(train_ds, 'datasets') else [train_ds]

        # 识别每个子数据集的来源名称
        # ConcatDataset 中的顺序: [Nexar, DoTA(可选)]
        ds_names = []
        if hasattr(train_ds, 'datasets'):
            # 第一个始终是 Nexar
            ds_names.append("Nexar")
            for name, _ in extra_train_datasets:
                ds_names.append(name)
        else:
            ds_names.append("Nexar")

        logger.info(f"[数据平衡] 开始平衡正负样本 (ratio=1:{args.balance_mix})...")
        logger.info(f"  子数据集: {list(zip(ds_names, [len(d) for d in candidate_datasets]))}")

        # 遍历所有子数据集，按 label 和来源分组
        # pos_by_source[source_name] = list of global_indices
        pos_by_source = {}
        neg_indices = []
        pos_indices_all = []
        offset = 0
        for ds_idx, sub_ds in enumerate(candidate_datasets):
            source = ds_names[ds_idx] if ds_idx < len(ds_names) else f"DS{ds_idx}"
            clips = sub_ds.clips if hasattr(sub_ds, 'clips') else []
            windows = sub_ds.windows if hasattr(sub_ds, 'windows') else []
            src_pos = []
            src_neg = 0
            for local_idx, win in enumerate(windows):
                clip_idx = win[0]
                if clip_idx < len(clips):
                    label = clips[clip_idx].label
                else:
                    label = 0
                global_idx = offset + local_idx
                if label == 1:
                    src_pos.append(global_idx)
                    pos_indices_all.append(global_idx)
                else:
                    neg_indices.append(global_idx)
                    src_neg += 1
            pos_by_source[source] = src_pos
            logger.info(f"  {source}: pos={len(src_pos)}, neg={src_neg}")
            offset += len(sub_ds)

        n_pos = len(pos_indices_all)
        n_neg = len(neg_indices)
        logger.info(f"  平衡前总计: pos={n_pos}, neg={n_neg}")

        if n_pos <= n_neg:
            # 负样本更多 → 采样负样本（简单随机）
            minority_count = n_pos
            target_majority = int(minority_count * args.balance_mix)
            if target_majority < n_neg:
                random.shuffle(neg_indices)
                neg_indices = neg_indices[:target_majority]
            pos_indices = pos_indices_all
            logger.info(f"  策略: 欠采样负样本 {n_neg} → {len(neg_indices)}")
        else:
            # 正样本更多 → 按优先级欠采样正样本
            # 优先级: 先减 DoTA → 再减 Nexar
            target_pos = int(n_neg * args.balance_mix)
            need_to_remove = n_pos - target_pos

            logger.info(f"  策略: 欠采样正样本 {n_pos} → {target_pos} (需移除 {need_to_remove})")

            # 按优先级排序: DoTA 先减，Nexar 其次
            remove_priority = ["DoTA", "Nexar"]
            removed_count = {}
            indices_to_remove = set()

            for source in remove_priority:
                if need_to_remove <= 0:
                    break
                src_pos = pos_by_source.get(source, [])
                if not src_pos:
                    continue
                can_remove = min(len(src_pos), need_to_remove)
                random.shuffle(src_pos)
                for idx in src_pos[:can_remove]:
                    indices_to_remove.add(idx)
                removed_count[source] = can_remove
                need_to_remove -= can_remove
                logger.info(f"    从 {source} 移除 {can_remove} 个正样本"
                            f" (剩余 {len(src_pos) - can_remove})")

            if need_to_remove > 0:
                logger.warning(f"    仍需移除 {need_to_remove} 个正样本但已无可移除来源")

            pos_indices = [idx for idx in pos_indices_all if idx not in indices_to_remove]
            logger.info(f"  各数据集保留正样本: " + ", ".join(
                f"{src}={len([i for i in pos_by_source.get(src,[]) if i not in indices_to_remove])}"
                for src in ds_names))

        balanced_indices = pos_indices + neg_indices
        random.shuffle(balanced_indices)
        train_ds = Subset(train_ds, balanced_indices)
        logger.info(f"  平衡后: pos={len(pos_indices)}, neg={len(neg_indices)}, "
                    f"total={len(balanced_indices)}, "
                    f"ratio={len(pos_indices)}:{len(neg_indices)}")

    # ---- Debug 模式 ----
    if args.debug:
        debug_limit = 500
        if len(train_ds) > debug_limit:
            train_ds = Subset(train_ds, list(range(debug_limit)))
        if len(val_ds) > debug_limit:
            val_ds = Subset(val_ds, list(range(debug_limit)))
        logger.info(f"[DEBUG] 截断数据集: train={len(train_ds)}, val={len(val_ds)}")

    def _get_clips(ds):
        if hasattr(ds, 'clips'):
            return ds.clips
        if hasattr(ds, 'dataset') and hasattr(ds.dataset, 'clips'):
            return ds.dataset.clips
        # ConcatDataset: 汇总所有子数据集的 clips
        if hasattr(ds, 'datasets'):
            all_clips = []
            for sub_ds in ds.datasets:
                c = _get_clips(sub_ds)
                if c is not None:
                    all_clips.extend(c)
            return all_clips if all_clips else None
        return None

    train_clips = _get_clips(train_ds)
    val_clips = _get_clips(val_ds)
    logger.info(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")
    if train_clips is not None:
        logger.info(f"Train clips: {len(train_clips)}")
    if val_clips is not None:
        logger.info(f"Val clips: {len(val_clips)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn, collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn, collate_fn=collate_fn,
    )

    # ---- 模型 ----
    if args.backbone == "giant":
        _model_id = "OpenGVLab/VideoMAEv2-giant"
        _embed_dim = 1408
        logger.info(f"  [Giant] model_id={_model_id}, embed_dim={_embed_dim}")
    else:
        _model_id = MODEL_ID
        _embed_dim = EMBED_DIM

    if args.model == "v1":
        model = VideoMAEGCNModel(
            model_id=_model_id, embed_dim=_embed_dim, num_rois=NUM_ROIS,
            roi_size=ROI_SIZE, gcn_hidden=GCN_HIDDEN, gcn_dropout=GCN_DROPOUT,
            fusion_dim=FUSION_DIM, img_size=IMG_SIZE, drop_path_rate=DROP_PATH_RATE,
        ).to(device)
    else:
        model = VideoMAEGCNModelV2(
            model_id=_model_id, embed_dim=_embed_dim, num_rois=NUM_ROIS,
            roi_size=ROI_SIZE, gcn_hidden=GCN_HIDDEN, gcn_dropout=GCN_DROPOUT,
            fusion_dim=FUSION_DIM, img_size=IMG_SIZE, drop_path_rate=DROP_PATH_RATE,
            window_size=actual_window_size,
            disable_gcn=args.no_gcn, random_roi=args.random_roi,
        ).to(device)

    # ---- 冻结 backbone 前 N 层 ----
    if args.freeze_backbone_layers > 0:
        frozen_count = freeze_backbone_layers(model, args.freeze_backbone_layers)
        logger.info(f"  冻结 backbone 前 {args.freeze_backbone_layers} 层"
                    f"（共冻结 {frozen_count} 个参数）")

    # 参数分组
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    logger.info(
        f"Parameters: backbone={sum(p.numel() for p in backbone_params)/1e6:.1f}M, "
        f"head={sum(p.numel() for p in head_params)/1e6:.1f}M"
    )
    logger.info(
        f"  total={total_params/1e6:.1f}M, trainable={trainable_params/1e6:.1f}M, "
        f"frozen={frozen_params/1e6:.1f}M"
    )

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE == "step":
        scheduler = StepLR(optimizer, step_size=1, gamma=SCHEDULER_GAMMA)
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=SCHEDULER_T_MAX, eta_min=SCHEDULER_ETA_MIN
        )

    scaler = GradScaler(enabled=USE_AMP)
    loss_fn = get_loss_fn(loss_scale=args.loss_scale, clip_weight=args.clip_weight)

    # 恢复训练
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    # ---- 训练循环 ----
    global_step = 0
    best_AP = -1.0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (rgb, attn, labels, metas) in enumerate(train_loader):
            rgb = rgb.to(device)
            attn = attn.to(device)
            labels = labels.to(device)

            with amp_autocast(enabled=USE_AMP):
                pred_logits, pred_frames_logits = model(rgb, attn)

                if pred_frames_logits.dim() == 3:
                    loss = loss_fn(pred_frames_logits, labels, metas=metas) / ACCUMULATION_STEPS
                else:
                    loss = loss_fn(pred_logits.unsqueeze(1).unsqueeze(-1).expand(-1, -1, 2),
                                   labels[:, -1:], metas=metas) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * ACCUMULATION_STEPS * rgb.size(0)
            epoch_samples += rgb.size(0)

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            if batch_idx % 50 == 0:
                avg = epoch_loss / max(epoch_samples, 1)
                logger.info(
                    f"E{epoch+1} B{batch_idx}/{len(train_loader)} "
                    f"loss={loss.item()*ACCUMULATION_STEPS:.4f} avg={avg:.4f}"
                )

        # 处理最后不足 accumulation_steps 的梯度
        if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 学习率调度
        if SCHEDULER_TYPE == "step":
            if (epoch + 1) % SCHEDULER_STEP_EPOCH == 0:
                scheduler.step()
        else:
            scheduler.step()

        # ---- 验证（Nexar）----
        val_loss, val_AP, val_mTTA = validate(
            model, val_loader, val_ds, loss_fn, device, logger,
            sample_fps=args.sample_fps, original_fps=FPS,
            dataset_name="Nexar"
        )

        elapsed = time.time() - t0
        train_avg_loss = epoch_loss / max(epoch_samples, 1)
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} [{elapsed:.0f}s] "
            f"train_loss={train_avg_loss:.4f} val_loss={val_loss:.4f} "
            f"Nexar: AP={val_AP:.4f} mTTA={val_mTTA:.2f}s"
        )

        # ---- 保存 checkpoint ----
        is_best = val_AP > best_AP
        if is_best:
            best_AP = val_AP
            best_val_loss = val_loss

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_AP": val_AP,
            "val_mTTA": val_mTTA,
            "config": {
                "model": args.model,
                "batch_size": args.batch_size,
                "lr_backbone": args.lr_backbone,
                "lr_head": args.lr_head,
                "window_stride": args.window_stride,
                "sample_fps": args.sample_fps,
                "global_mode": args.global_mode,
                "global_16f": args.global_16f,
                "freeze_backbone_layers": args.freeze_backbone_layers,
                "no_gcn": args.no_gcn,
                "dataset": "nexar_clips_5s",
            },
        }
        torch.save(ckpt, os.path.join(weights_dir, "latest.pth"))

        save_path = os.path.join(weights_dir, f"epoch_{epoch+1}.pth")
        torch.save(ckpt, save_path)
        logger.info(f"  Saved checkpoint to {save_path}")

        if is_best:
            save_path = os.path.join(
                weights_dir,
                f"best_{args.model}_loss{val_loss:.4f}_AP{val_AP:.4f}.pth"
            )
            torch.save(ckpt, save_path)
            logger.info(f"  ★ New best! Saved to {save_path}")

    logger.info(f"Training complete. Best AP: {best_AP:.4f}, Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    if args.last2s:
        train_last2s(args)
    else:
        train(args)
