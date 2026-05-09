"""
Stage 3 — Inference on the competition test set.

Runs sliding-window inference on every 150-frame test clip, then aggregates
the window-level outputs back to a per-frame risk sequence via overlap
averaging and linear interpolation.  The output CSV matches the submission
format expected by the competition.  See Section 3 of `TECH_REPORT.md`.
"""

import argparse
import os
import sys
import csv
import json
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast as _autocast
    def amp_autocast(enabled=True):
        return _autocast('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast
    def amp_autocast(enabled=True):
        return _autocast(enabled=enabled)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow running the script directly from the repository root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nexar_config import (
    MODEL_ID, EMBED_DIM, NUM_ROIS, ROI_SIZE, GCN_HIDDEN, GCN_DROPOUT,
    FUSION_DIM, IMG_SIZE, WINDOW_SIZE, SAMPLE_FPS, FPS, USE_AMP,
    GLOBAL_16F_NUM_FRAMES, NUM_WORKERS,
)
from videomae_gcn import VideoMAEGCNModel, VideoMAEGCNModelV2


# ============ 数据结构 ============

class CompetitionClip:
    """比赛 test.csv 中的一个 clip"""
    def __init__(self, clip_id, video_id, start_frame, end_frame, caption):
        self.clip_id = clip_id          # 如 "1_009334_14_164"
        self.video_id = video_id        # 如 "1/009334"
        self.start_frame = start_frame  # 如 14
        self.end_frame = end_frame      # 如 164
        self.caption = caption
        self.num_frames = end_frame - start_frame  # 150


def load_test_csv(csv_path):
    """加载比赛 test.csv"""
    clips = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = CompetitionClip(
                clip_id=row['id'].strip(),
                video_id=row['video_id'].strip(),
                start_frame=int(row['start_frame']),
                end_frame=int(row['end_frame']),
                caption=row.get('caption', '').strip(),
            )
            clips.append(clip)
    return clips


# ============ 数据集 ============

class CompetitionSlidingWindowDataset(Dataset):
    """
    比赛测试集的滑动窗口数据集。
    对每个 clip 做滑动窗口采样，用于推理后合并为 150 帧的逐帧概率。

    帧路径: {data_root}/{video_id}/images/{frame_idx:06d}.jpg (1-indexed)
    """

    def __init__(
        self,
        clips,
        data_root,
        window_size=16,
        window_stride=1,
        sample_fps=10,
        original_fps=30,
        img_size=224,
    ):
        self.clips = clips
        self.data_root = data_root
        self.window_size = window_size
        self.window_stride = window_stride
        self.sample_fps = sample_fps
        self.original_fps = original_fps
        self.img_size = img_size
        self.frame_interval = max(1, original_fps // sample_fps)  # 30//10 = 3

        # 构建所有滑动窗口
        self.windows = []
        for clip_idx, clip in enumerate(clips):
            clip_length = clip.num_frames  # 150
            # 按采样帧率获取帧索引（相对于 clip 内的偏移，0-indexed）
            sampled_indices = list(range(0, clip_length, self.frame_interval))
            num_sampled = len(sampled_indices)

            if num_sampled <= window_size:
                self.windows.append((clip_idx, sampled_indices, 0))
            else:
                for win_start in range(0, num_sampled - window_size + 1, window_stride):
                    win_indices = sampled_indices[win_start:win_start + window_size]
                    self.windows.append((clip_idx, win_indices, win_start))

        # 推理用 transform
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.windows)

    def _get_frame_path(self, video_id, abs_frame_idx):
        """
        获取帧路径。
        比赛数据帧命名: {abs_frame_idx:06d}.jpg (1-indexed)
        """
        return os.path.join(
            self.data_root, video_id, "images", f"{abs_frame_idx:06d}.jpg"
        )

    def _load_frame(self, video_id, abs_frame_idx):
        """加载 RGB 帧"""
        path = self._get_frame_path(video_id, abs_frame_idx)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 找不到帧时返回黑色图像
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        clip_idx, win_indices, win_start = self.windows[idx]
        clip = self.clips[clip_idx]

        # 确保窗口恰好 window_size 帧
        while len(win_indices) < self.window_size:
            win_indices.append(win_indices[-1])

        rgb_frames = []
        attn_frames = []

        for rel_idx in win_indices:
            # rel_idx 是相对于 clip 起始的偏移（0-indexed）
            # 比赛帧是 1-indexed，所以绝对帧号 = start_frame + rel_idx
            abs_idx = clip.start_frame + rel_idx

            # 加载 RGB
            rgb = self._load_frame(clip.video_id, abs_idx)
            rgb = self.transform(image=rgb)["image"]  # (3, H, W)
            rgb_frames.append(rgb)

            # 无 attention map，使用全零
            attn = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            attn_frames.append(torch.tensor(attn).unsqueeze(0))  # (1, H, W)

        rgb_tensor = torch.stack(rgb_frames)    # (T, 3, H, W)
        attn_tensor = torch.stack(attn_frames)  # (T, 1, H, W)

        return rgb_tensor, attn_tensor, clip_idx, win_start, win_indices


class CompetitionGlobalDataset(Dataset):
    """
    比赛测试集的全局模式数据集。
    整个 clip 降采样到 target_frames 帧后一次性输入模型。
    """

    def __init__(
        self,
        clips,
        data_root,
        target_frames=16,
        sample_fps=10,
        original_fps=30,
        img_size=224,
    ):
        self.clips = clips
        self.data_root = data_root
        self.target_frames = target_frames
        self.sample_fps = sample_fps
        self.original_fps = original_fps
        self.img_size = img_size
        self.frame_interval = max(1, original_fps // sample_fps)

        # 全局模式: 每个 clip 一个样本
        self.windows = []
        for clip_idx, clip in enumerate(clips):
            clip_length = clip.num_frames
            sampled_indices = list(range(0, clip_length, self.frame_interval))
            num_sampled = len(sampled_indices)

            if num_sampled > target_frames:
                step = num_sampled / target_frames
                selected = [sampled_indices[int(i * step)] for i in range(target_frames)]
                sampled_indices = selected

            self.windows.append((clip_idx, sampled_indices, 0))

        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.windows)

    def _get_frame_path(self, video_id, abs_frame_idx):
        return os.path.join(
            self.data_root, video_id, "images", f"{abs_frame_idx:06d}.jpg"
        )

    def _load_frame(self, video_id, abs_frame_idx):
        path = self._get_frame_path(video_id, abs_frame_idx)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        clip_idx, win_indices, win_start = self.windows[idx]
        clip = self.clips[clip_idx]

        while len(win_indices) < self.target_frames:
            win_indices.append(win_indices[-1])

        rgb_frames = []
        attn_frames = []

        for rel_idx in win_indices:
            abs_idx = clip.start_frame + rel_idx
            rgb = self._load_frame(clip.video_id, abs_idx)
            rgb = self.transform(image=rgb)["image"]
            rgb_frames.append(rgb)

            attn = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            attn_frames.append(torch.tensor(attn).unsqueeze(0))

        rgb_tensor = torch.stack(rgb_frames)
        attn_tensor = torch.stack(attn_frames)

        return rgb_tensor, attn_tensor, clip_idx, win_indices


# ============ Collate 函数 ============

def collate_fn_sliding(batch):
    rgb = torch.stack([b[0] for b in batch])
    attn = torch.stack([b[1] for b in batch])
    clip_indices = [b[2] for b in batch]
    win_starts = [b[3] for b in batch]
    win_indices_list = [b[4] for b in batch]
    return rgb, attn, clip_indices, win_starts, win_indices_list


def collate_fn_global(batch):
    rgb = torch.stack([b[0] for b in batch])
    attn = torch.stack([b[1] for b in batch])
    clip_indices = [b[2] for b in batch]
    win_indices_list = [b[3] for b in batch]
    return rgb, attn, clip_indices, win_indices_list


# ============ 模型加载 ============

def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model_type = config.get("model", "v2")
    sample_fps = config.get("sample_fps", SAMPLE_FPS)
    no_gcn = config.get("no_gcn", True)
    global_16f = config.get("global_16f", False)

    print(f"Checkpoint config: model={model_type}, sample_fps={sample_fps}, "
          f"no_gcn={no_gcn}, global_16f={global_16f}")

    if global_16f:
        window_size = GLOBAL_16F_NUM_FRAMES
    else:
        window_size = WINDOW_SIZE

    if model_type == "v1":
        model = VideoMAEGCNModel(
            model_id=MODEL_ID, embed_dim=EMBED_DIM, num_rois=NUM_ROIS,
            roi_size=ROI_SIZE, gcn_hidden=GCN_HIDDEN, gcn_dropout=GCN_DROPOUT,
            fusion_dim=FUSION_DIM, img_size=IMG_SIZE, drop_path_rate=0.0,
        ).to(device)
    else:
        model = VideoMAEGCNModelV2(
            model_id=MODEL_ID, embed_dim=EMBED_DIM, num_rois=NUM_ROIS,
            roi_size=ROI_SIZE, gcn_hidden=GCN_HIDDEN, gcn_dropout=GCN_DROPOUT,
            fusion_dim=FUSION_DIM, img_size=IMG_SIZE, drop_path_rate=0.0,
            window_size=window_size, disable_gcn=no_gcn,
        ).to(device)

    # 兼容旧版 checkpoint (output_dim=1)
    use_sigmoid = False
    if "model" in ckpt:
        fp_weight_key = "frame_predictor.3.weight"
        pred_weight_key = "predictor.3.weight"
        ckpt_out_dim = None
        if fp_weight_key in ckpt["model"]:
            ckpt_out_dim = ckpt["model"][fp_weight_key].shape[0]
        elif pred_weight_key in ckpt["model"]:
            ckpt_out_dim = ckpt["model"][pred_weight_key].shape[0]

        if ckpt_out_dim == 1:
            print(f"[兼容模式] 旧版 checkpoint (output_dim=1)")
            if hasattr(model, 'frame_predictor'):
                model.frame_predictor[-1] = nn.Linear(128, 1)
            elif hasattr(model, 'predictor'):
                model.predictor[-1] = nn.Linear(128, 1)
            model = model.to(device)
            use_sigmoid = True

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")

    return model, config, use_sigmoid


# ============ 推理函数 ============

def predict_sliding_window(args):
    """滑动窗口模式推理"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: 滑动窗口")

    # 加载模型
    model, config, use_sigmoid = load_model(args.checkpoint, device)
    sample_fps = config.get("sample_fps", args.sample_fps)

    # 加载测试数据
    clips = load_test_csv(args.test_csv)
    print(f"Loaded {len(clips)} clips from {args.test_csv}")

    if args.debug:
        clips = clips[:args.debug_limit]
        print(f"[DEBUG] 只处理前 {len(clips)} 个样本")

    # 构建数据集
    ds = CompetitionSlidingWindowDataset(
        clips=clips,
        data_root=args.data_root,
        window_size=WINDOW_SIZE,
        window_stride=args.window_stride,
        sample_fps=sample_fps,
        original_fps=FPS,
        img_size=IMG_SIZE,
    )

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_fn_sliding,
    )
    print(f"Clips: {len(clips)}, Windows: {len(ds)}")

    # 推理
    clip_preds = {}
    with torch.no_grad():
        for bi, (rgb, attn, ci_list, ws_list, wi_list) in enumerate(loader):
            rgb, attn = rgb.to(device), attn.to(device)
            with amp_autocast(enabled=USE_AMP):
                _, pred_frames_logits = model(rgb, attn)
                if use_sigmoid:
                    pred_frames = torch.sigmoid(pred_frames_logits.squeeze(-1))
                else:
                    pred_frames = torch.softmax(pred_frames_logits, dim=-1)[:, :, 1]

            for i in range(rgb.size(0)):
                ci = ci_list[i]
                if ci not in clip_preds:
                    clip_preds[ci] = {}
                pf = pred_frames[i].cpu().numpy()
                for t, ri in enumerate(wi_list[i]):
                    ri = ri.item() if isinstance(ri, torch.Tensor) else ri
                    clip_preds[ci].setdefault(ri, []).append(float(pf[t]))

            if (bi + 1) % 50 == 0 or (bi + 1) == len(loader):
                print(f"  推理进度: {bi+1}/{len(loader)}")

    # 合并为逐帧概率并生成 submission
    print("合并预测结果...")
    frame_interval = max(1, FPS // sample_fps)
    submission_rows = []

    for clip_idx, clip in enumerate(clips):
        clip_length = clip.num_frames  # 150
        pred_arr = np.zeros(clip_length, dtype=np.float32)

        if clip_idx in clip_preds:
            fp = clip_preds[clip_idx]
            # 对每个采样位置取平均
            sampled = {k: np.mean(v) for k, v in fp.items()}
            if sampled:
                keys = sorted(sampled.keys())
                vals = [sampled[k] for k in keys]
                if len(keys) >= 2:
                    pred_arr = np.interp(
                        np.arange(clip_length), keys, vals
                    ).astype(np.float32)
                elif len(keys) == 1:
                    pred_arr[:] = vals[0]

        # 裁剪到 [0, 1]
        pred_arr = np.clip(pred_arr, 0.0, 1.0)

        # 格式化为 JSON 列表
        risk_list = [round(float(v), 6) for v in pred_arr]
        submission_rows.append((clip.clip_id, risk_list))

    # 写入 submission.csv
    _write_submission(submission_rows, args.output)


def predict_global_16f(args):
    """全局 16 帧模式推理"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: 全局 16 帧")

    # 加载模型
    model, config, use_sigmoid = load_model(args.checkpoint, device)
    sample_fps = config.get("sample_fps", args.sample_fps)

    # 加载测试数据
    clips = load_test_csv(args.test_csv)
    print(f"Loaded {len(clips)} clips from {args.test_csv}")

    if args.debug:
        clips = clips[:args.debug_limit]
        print(f"[DEBUG] 只处理前 {len(clips)} 个样本")

    # 构建数据集
    ds = CompetitionGlobalDataset(
        clips=clips,
        data_root=args.data_root,
        target_frames=GLOBAL_16F_NUM_FRAMES,
        sample_fps=sample_fps,
        original_fps=FPS,
        img_size=IMG_SIZE,
    )

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate_fn_global,
    )
    print(f"Clips: {len(clips)}, Samples: {len(ds)}")

    # 推理
    clip_preds = {}
    with torch.no_grad():
        for bi, (rgb, attn, ci_list, wi_list) in enumerate(loader):
            rgb, attn = rgb.to(device), attn.to(device)
            with amp_autocast(enabled=USE_AMP):
                _, pred_frames_logits = model(rgb, attn)
                if use_sigmoid:
                    pred_frames = torch.sigmoid(pred_frames_logits.squeeze(-1))
                else:
                    pred_frames = torch.softmax(pred_frames_logits, dim=-1)[:, :, 1]

            for i in range(rgb.size(0)):
                ci = ci_list[i]
                win_indices = wi_list[i]
                pf = pred_frames[i].cpu().numpy()

                if ci not in clip_preds:
                    clip_preds[ci] = {}
                for t, rel_idx in enumerate(win_indices[:len(pf)]):
                    clip_preds[ci][rel_idx] = float(pf[t])

            if (bi + 1) % 50 == 0 or (bi + 1) == len(loader):
                print(f"  推理进度: {bi+1}/{len(loader)}")

    # 合并为逐帧概率并生成 submission
    print("合并预测结果...")
    submission_rows = []

    for clip_idx, clip in enumerate(clips):
        clip_length = clip.num_frames  # 150
        pred_arr = np.zeros(clip_length, dtype=np.float32)

        if clip_idx in clip_preds:
            sampled = clip_preds[clip_idx]
            if sampled:
                keys = sorted(sampled.keys())
                vals = [sampled[k] for k in keys]
                if len(keys) >= 2:
                    pred_arr = np.interp(
                        np.arange(clip_length), keys, vals
                    ).astype(np.float32)
                elif len(keys) == 1:
                    pred_arr[:] = vals[0]

        pred_arr = np.clip(pred_arr, 0.0, 1.0)
        risk_list = [round(float(v), 6) for v in pred_arr]
        submission_rows.append((clip.clip_id, risk_list))

    _write_submission(submission_rows, args.output)


# ============ 输出 ============

def _write_submission(submission_rows, output_path):
    """写入比赛格式的 submission.csv"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "risk"])
        for clip_id, risk_list in submission_rows:
            risk_str = json.dumps(risk_list)
            writer.writerow([clip_id, risk_str])

    print(f"\n{'='*60}")
    print(f"  Submission 已保存: {output_path}")
    print(f"  总样本数: {len(submission_rows)}")
    print(f"{'='*60}")

    # 验证格式
    _validate_submission(output_path)


def _validate_submission(csv_path):
    """验证 submission.csv 格式"""
    errors = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row_count = 0
        for row in reader:
            row_count += 1
            try:
                risk = json.loads(row['risk'])
                if len(risk) != 150:
                    errors.append(f"  行 {row_count} ({row['id']}): risk 长度 {len(risk)} != 150")
                for v in risk:
                    if not (0.0 <= v <= 1.0):
                        errors.append(f"  行 {row_count} ({row['id']}): 值 {v} 超出 [0,1]")
                        break
            except json.JSONDecodeError:
                errors.append(f"  行 {row_count} ({row['id']}): risk 不是有效 JSON")

    if errors:
        print(f"\n[WARNING] 格式验证发现 {len(errors)} 个问题:")
        for e in errors[:10]:
            print(e)
    else:
        print(f"  格式验证通过 ✓ ({row_count} 行)")


# ============ 主函数 ============

def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot TAA 比赛推理")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the trained checkpoint (e.g. epoch_14.pth).")
    p.add_argument("--data_root", type=str,
                   default=os.environ.get(
                       "COMP_DATA_ROOT",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "data", "zero_shot_taa", "Out"),
                   ),
                   help="Root of the competition test frames.  Defaults to "
                        "$COMP_DATA_ROOT.")
    p.add_argument("--test_csv", type=str,
                   default=os.path.join(os.path.dirname(__file__), "competition", "test.csv"),
                   help="Path to the competition test.csv.")
    p.add_argument("--output", type=str, default="submission.csv",
                   help="输出 submission.csv 路径")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--window_stride", type=int, default=1,
                   help="滑动窗口步长（推理时建议用 1）")
    p.add_argument("--sample_fps", type=int, default=SAMPLE_FPS,
                   help="降采样帧率（默认从 checkpoint 读取）")
    p.add_argument("--global_16f", action="store_true",
                   help="使用全局 16 帧模式推理")
    p.add_argument("--debug", action="store_true",
                   help="Debug 模式")
    p.add_argument("--debug_limit", type=int, default=10,
                   help="Debug 模式下处理的样本数")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  Zero-shot Accident Anticipation 比赛推理")
    print("=" * 60)
    print(f"  checkpoint  = {args.checkpoint}")
    print(f"  data_root   = {args.data_root}")
    print(f"  test_csv    = {args.test_csv}")
    print(f"  output      = {args.output}")
    print(f"  batch_size  = {args.batch_size}")
    print(f"  global_16f  = {args.global_16f}")
    print(f"  debug       = {args.debug}")
    print("=" * 60)

    if args.global_16f:
        predict_global_16f(args)
    else:
        predict_sliding_window(args)
