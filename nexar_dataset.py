"""
Nexar 5s Clip 数据集：加载 process_5s_clips.py 输出的帧图片 + 标注文件。
标注格式与 DADA-1000 完全一致：
  {video_id}/{clip_idx} {label} {start} {end} {toa},{description}
  例如: 00822/006 1 0 149 135,nexar_00822_clip006

帧图片命名格式: {clip_idx:06d}.jpg（如 000000.jpg, 000001.jpg, ...）
目录结构: rgb_videos/{video_id}/{clip_idx}/000000.jpg

支持两种模式:
  1. 滑动窗口模式: 对每个 clip 做滑动窗口采样
  2. 全局模式: 整个 clip 降采样后一次性输入
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List

import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class NexarClipSample:
    """一个标注行对应的 clip 信息"""
    video_dir: str       # RGB 帧目录
    vid_key: str         # 如 "00822/006"
    label: int           # 0 或 1
    start: int           # clip 起始帧号（通常为 0）
    end: int             # clip 结束帧号（通常为 149）
    toa: int             # 事故帧号（正样本: clip 内相对帧号; 负样本: 250 即 clip 外）
    text: str            # 描述文本


def load_nexar_annotations(ann_file: str, rgb_root: str) -> List[NexarClipSample]:
    """
    加载标注文件，返回 NexarClipSample 列表。
    标注格式: {video_id}/{clip_idx} {label} {start} {end} {toa},{text}
    """
    samples = []
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            meta, text = line.split(",", 1)
            parts = meta.strip().split()
            vid_key = parts[0]  # 如 "00822/006"
            label = int(parts[1])
            start = int(parts[2])
            end = int(parts[3])
            toa = int(parts[4])

            # 构建帧目录路径
            video_dir = os.path.join(rgb_root, vid_key.replace("/", os.sep))

            samples.append(NexarClipSample(
                video_dir=video_dir,
                vid_key=vid_key,
                label=label,
                start=start,
                end=end,
                toa=toa,
                text=text.strip(),
            ))
    return samples


def get_nexar_frame_path(video_dir: str, frame_idx: int) -> str:
    """
    获取帧路径。帧命名格式: {frame_idx:06d}.jpg
    """
    return os.path.join(video_dir, f"{frame_idx:06d}.jpg")


class NexarVideoMAEDataset(Dataset):
    """
    Nexar 5s Clip 滑动窗口数据集。
    对每个 clip 进行滑动窗口采样，每个窗口输出:
    - rgb: (T, 3, H, W) 归一化后的 RGB 帧
    - attn: (T, 1, H, W) 全零 attention map（Nexar 无 attention map）
    - frame_label: (T,) 窗口内每帧的软标签
    - meta: dict 包含 clip 级别信息
    """

    def __init__(
        self,
        ann_file: str,
        rgb_root: str,
        window_size: int = 16,
        window_stride: int = 4,
        sample_fps: int = 10,
        original_fps: int = 30,
        img_size: int = 224,
        is_train: bool = True,
    ):
        self.window_size = window_size
        self.window_stride = window_stride
        self.sample_fps = sample_fps
        self.original_fps = original_fps
        self.img_size = img_size
        self.is_train = is_train

        # 帧采样间隔
        self.frame_interval = max(1, original_fps // sample_fps)  # 30//10 = 3

        # 加载标注
        self.clips = load_nexar_annotations(ann_file, rgb_root)

        # 构建所有滑动窗口样本
        self.windows = []
        for clip_idx, clip in enumerate(self.clips):
            clip_length = clip.end - clip.start + 1
            # 按采样帧率获取帧索引（相对于 clip start 的原始帧号）
            sampled_indices = list(range(0, clip_length, self.frame_interval))
            num_sampled = len(sampled_indices)

            if num_sampled <= window_size:
                # clip 太短，只生成一个窗口（padding）
                self.windows.append((clip_idx, sampled_indices, 0))
            else:
                # 滑动窗口
                for win_start in range(0, num_sampled - window_size + 1, window_stride):
                    win_indices = sampled_indices[win_start:win_start + window_size]
                    self.windows.append((clip_idx, win_indices, win_start))

        # 数据增强
        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.windows)

    def _load_rgb_frame(self, video_dir: str, abs_frame_idx: int) -> np.ndarray:
        """加载并返回 RGB 帧 (H, W, 3)"""
        path = get_nexar_frame_path(video_dir, abs_frame_idx)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        clip_idx, win_indices, win_start = self.windows[idx]
        clip = self.clips[clip_idx]

        # 确保窗口恰好 window_size 帧
        while len(win_indices) < self.window_size:
            win_indices.append(win_indices[-1])

        rgb_frames = []
        attn_frames = []
        frame_labels = []

        for rel_idx in win_indices:
            abs_idx = clip.start + rel_idx  # 绝对帧号

            # 加载 RGB
            rgb = self._load_rgb_frame(clip.video_dir, abs_idx)
            rgb = self.transform(image=rgb)["image"]  # (3, H, W)
            rgb_frames.append(rgb)

            # Nexar 无 attention map，使用全零
            attn = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            attn_frames.append(torch.tensor(attn).unsqueeze(0))  # (1, H, W)

            # 逐帧标签（与 LOTVS-CAP 一致）
            if clip.label == 1:
                toa_rel = clip.toa  # 已经是相对于 clip start 的帧号
                frame_prob = np.exp(-max(0, (toa_rel - rel_idx - 1) / self.original_fps))
            else:
                frame_prob = 0.0
            frame_labels.append(frame_prob)

        rgb_tensor = torch.stack(rgb_frames)       # (T, 3, H, W)
        attn_tensor = torch.stack(attn_frames)     # (T, 1, H, W)
        label_tensor = torch.tensor(frame_labels, dtype=torch.float32)  # (T,)

        meta = {
            "clip_idx": clip_idx,
            "vid_key": clip.vid_key,
            "label": clip.label,
            "toa": clip.toa,
            "start": clip.start,
            "end": clip.end,
            "win_start": win_start,
            "win_indices": win_indices,
        }

        return rgb_tensor, attn_tensor, label_tensor, meta


class NexarGlobalDataset(Dataset):
    """
    Nexar 5s Clip 全局模式数据集。
    整个 clip 降采样后一次性输入模型，不做滑动窗口。
    每个样本 = 一个完整的 clip（降采样到 target_frames 帧）。
    """

    def __init__(
        self,
        ann_file: str,
        rgb_root: str,
        target_frames: int = 50,
        sample_fps: int = 10,
        original_fps: int = 30,
        img_size: int = 224,
        is_train: bool = True,
    ):
        self.target_frames = target_frames
        self.sample_fps = sample_fps
        self.original_fps = original_fps
        self.img_size = img_size
        self.is_train = is_train

        # 帧采样间隔
        self.frame_interval = max(1, original_fps // sample_fps)  # 30//10 = 3

        # 加载标注
        self.clips = load_nexar_annotations(ann_file, rgb_root)

        # 全局模式: 每个 clip 生成一个样本
        self.windows = []
        for clip_idx, clip in enumerate(self.clips):
            clip_length = clip.end - clip.start + 1
            sampled_indices = list(range(0, clip_length, self.frame_interval))
            num_sampled = len(sampled_indices)

            if num_sampled > target_frames:
                step = num_sampled / target_frames
                selected = [sampled_indices[int(i * step)] for i in range(target_frames)]
                sampled_indices = selected

            self.windows.append((clip_idx, sampled_indices, 0))

        # 数据增强
        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.windows)

    def _load_rgb_frame(self, video_dir: str, abs_frame_idx: int) -> np.ndarray:
        """加载并返回 RGB 帧 (H, W, 3)"""
        path = get_nexar_frame_path(video_dir, abs_frame_idx)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        clip_idx, win_indices, win_start = self.windows[idx]
        clip = self.clips[clip_idx]

        # 确保帧数 = target_frames（不足则 padding 最后一帧）
        while len(win_indices) < self.target_frames:
            win_indices.append(win_indices[-1])

        rgb_frames = []
        attn_frames = []
        frame_labels = []

        for t_idx, rel_idx in enumerate(win_indices):
            abs_idx = clip.start + rel_idx

            # 加载 RGB
            rgb = self._load_rgb_frame(clip.video_dir, abs_idx)
            rgb = self.transform(image=rgb)["image"]  # (3, H, W)
            rgb_frames.append(rgb)

            # Nexar 无 attention map，使用全零
            attn = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            attn_frames.append(torch.tensor(attn).unsqueeze(0))  # (1, H, W)

            # 逐帧标签
            if clip.label == 1:
                toa_rel = clip.toa
                frame_prob = np.exp(-max(0, (toa_rel - rel_idx - 1) / self.original_fps))
            else:
                frame_prob = 0.0
            frame_labels.append(frame_prob)

        rgb_tensor = torch.stack(rgb_frames)       # (T, 3, H, W)
        attn_tensor = torch.stack(attn_frames)     # (T, 1, H, W)
        label_tensor = torch.tensor(frame_labels, dtype=torch.float32)  # (T,)

        meta = {
            "clip_idx": clip_idx,
            "vid_key": clip.vid_key,
            "label": clip.label,
            "toa": clip.toa,
            "start": clip.start,
            "end": clip.end,
            "win_start": win_start,
            "win_indices": win_indices,
        }

        return rgb_tensor, attn_tensor, label_tensor, meta


class NexarInferenceDataset(Dataset):
    """
    Nexar 推理用数据集：对每个 clip 生成所有滑动窗口。
    推理时需要将所有窗口的预测合并为逐帧概率序列。
    """

    def __init__(
        self,
        ann_file: str,
        rgb_root: str,
        window_size: int = 16,
        window_stride: int = 1,
        sample_fps: int = 10,
        original_fps: int = 30,
        img_size: int = 224,
    ):
        self.window_size = window_size
        self.window_stride = window_stride
        self.sample_fps = sample_fps
        self.original_fps = original_fps
        self.img_size = img_size
        self.frame_interval = max(1, original_fps // sample_fps)

        self.clips = load_nexar_annotations(ann_file, rgb_root)

        # 为每个 clip 构建滑动窗口
        self.windows = []
        self.clip_window_ranges = []

        for clip_idx, clip in enumerate(self.clips):
            clip_length = clip.end - clip.start + 1
            sampled_indices = list(range(0, clip_length, self.frame_interval))
            num_sampled = len(sampled_indices)

            start_win_idx = len(self.windows)

            if num_sampled <= window_size:
                self.windows.append((clip_idx, sampled_indices, 0))
            else:
                for win_start in range(0, num_sampled - window_size + 1, window_stride):
                    win_indices = sampled_indices[win_start:win_start + window_size]
                    self.windows.append((clip_idx, win_indices, win_start))

            end_win_idx = len(self.windows)
            self.clip_window_ranges.append((start_win_idx, end_win_idx))

        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        clip_idx, win_indices, win_start = self.windows[idx]
        clip = self.clips[clip_idx]

        while len(win_indices) < self.window_size:
            win_indices.append(win_indices[-1])

        rgb_frames = []
        attn_frames = []

        for rel_idx in win_indices:
            abs_idx = clip.start + rel_idx
            # RGB
            path = get_nexar_frame_path(clip.video_dir, abs_idx)
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            rgb = self.transform(image=img)["image"]
            rgb_frames.append(rgb)

            # 全零 attention
            attn = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            attn_frames.append(torch.tensor(attn).unsqueeze(0))

        rgb_tensor = torch.stack(rgb_frames)
        attn_tensor = torch.stack(attn_frames)

        return rgb_tensor, attn_tensor, clip_idx, win_start, win_indices


class NexarLast2sDataset(Dataset):
    """
    2nd-place solution 适配数据集：取每个 clip 最后 N 秒的帧，均匀采样到 target_frames 帧。
    做简单二分类 (collision=1 vs normal=0)。

    与原始 2nd-place 方案的对应关系：
    - 原方案从原始视频滑动窗口提取 2s 片段 → 这里从已切好的 5s clip 中截取最后 2s
    - 原方案 target_fps=8, window=16帧 → 这里 last_seconds=2.0, target_frames=16
    - 正样本：toa 落在最后 N 秒内（即事故即将发生）
    - 负样本：toa 在最后 N 秒之外（或 label=0）
    - 支持正负样本平衡（欠采样多数类）

    用法:
        ds = NexarLast2sDataset(
            ann_file=ANN_TRAIN, rgb_root=DATA_ROOT_TRAIN,
            last_seconds=2.0, target_frames=16, original_fps=30,
            img_size=224, is_train=True, balance=True,
        )
    """

    def __init__(
        self,
        ann_file: str,
        rgb_root: str,
        last_seconds: float = 2.0,
        target_frames: int = 16,
        original_fps: int = 30,
        img_size: int = 224,
        is_train: bool = True,
        balance: bool = True,
        pos_threshold: float = 1.5,
        seed: int = 42,
    ):
        self.last_seconds = last_seconds
        self.target_frames = target_frames
        self.original_fps = original_fps
        self.img_size = img_size
        self.is_train = is_train
        self.balance = balance
        self.pos_threshold = pos_threshold  # toa 在 clip 结束前 pos_threshold 秒内则为正

        # 加载标注
        self.clips = load_nexar_annotations(ann_file, rgb_root)

        # 计算最后 N 秒对应的帧数
        self.last_n_frames = int(last_seconds * original_fps)  # 2.0 * 30 = 60 帧

        # 为每个 clip 构建样本（每个 clip 生成一个样本）
        self._build_samples()

        # 正负样本平衡
        if balance and is_train:
            self._balance_samples(seed)

        # 数据增强
        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def _build_samples(self):
        """
        为每个 clip 构建一个样本：截取最后 last_seconds 秒，均匀采样 target_frames 帧。
        标签逻辑：
        - 原始 label=0 → 二分类 label=0（negative）
        - 原始 label=1 且 toa 在最后 N 秒内 → 二分类 label=1（positive）
        - 原始 label=1 但 toa 不在最后 N 秒内 → 二分类 label=0（因为看最后2s无法预见事故）
        """
        self.samples = []  # list of (clip_idx, frame_indices, binary_label)

        for clip_idx, clip in enumerate(self.clips):
            clip_length = clip.end - clip.start + 1  # 通常 150 帧

            # 取最后 N 秒的帧范围
            last_start = max(0, clip_length - self.last_n_frames)
            last_end = clip_length  # 不含

            # 最后 N 秒内的所有帧索引（相对于 clip start）
            available_indices = list(range(last_start, last_end))

            # 均匀采样到 target_frames 帧
            if len(available_indices) >= self.target_frames:
                step = len(available_indices) / self.target_frames
                sampled_indices = [available_indices[int(i * step)] for i in range(self.target_frames)]
            else:
                # 不足则重复最后一帧填充
                sampled_indices = available_indices[:]
                while len(sampled_indices) < self.target_frames:
                    sampled_indices.append(sampled_indices[-1])

            # 确定二分类标签
            if clip.label == 0:
                binary_label = 0
            else:
                # toa 是相对于 clip start 的帧号
                # 判断 toa 是否在最后 N 秒内
                # 即 toa >= last_start（事故在最后N秒的范围内）
                toa_rel = clip.toa
                if toa_rel >= last_start:
                    binary_label = 1
                else:
                    # 事故已在更早的时刻发生，最后2s看不到事故前的变化
                    binary_label = 0

            self.samples.append((clip_idx, sampled_indices, binary_label))

    def _balance_samples(self, seed: int = 42):
        """欠采样多数类，使正负样本数量一致"""
        pos_samples = [s for s in self.samples if s[2] == 1]
        neg_samples = [s for s in self.samples if s[2] == 0]

        n_pos = len(pos_samples)
        n_neg = len(neg_samples)

        if n_pos == 0 or n_neg == 0:
            return

        rng = random.Random(seed)

        if n_neg > n_pos:
            neg_samples = rng.sample(neg_samples, n_pos)
        elif n_pos > n_neg:
            pos_samples = rng.sample(pos_samples, n_neg)

        self.samples = pos_samples + neg_samples
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _load_rgb_frame(self, video_dir: str, abs_frame_idx: int) -> np.ndarray:
        """加载并返回 RGB 帧 (H, W, 3)"""
        path = get_nexar_frame_path(video_dir, abs_frame_idx)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        clip_idx, sampled_indices, binary_label = self.samples[idx]
        clip = self.clips[clip_idx]

        rgb_frames = []
        for rel_idx in sampled_indices:
            abs_idx = clip.start + rel_idx
            rgb = self._load_rgb_frame(clip.video_dir, abs_idx)
            rgb = self.transform(image=rgb)["image"]  # (3, H, W)
            rgb_frames.append(rgb)

        rgb_tensor = torch.stack(rgb_frames)  # (T, 3, H, W)
        label_tensor = torch.tensor(binary_label, dtype=torch.long)

        # 兼容现有 collate_fn: 返回 (rgb, attn, frame_label, meta)
        # attn 全零，frame_label 用 clip-level label 填充
        attn_tensor = torch.zeros(self.target_frames, 1, self.img_size, self.img_size)
        frame_label_tensor = torch.full((self.target_frames,), float(binary_label))

        meta = {
            "clip_idx": clip_idx,
            "vid_key": clip.vid_key,
            "label": binary_label,
            "orig_label": clip.label,
            "toa": clip.toa,
            "start": clip.start,
            "end": clip.end,
            "win_start": 0,
            "win_indices": sampled_indices,
            "binary_label": binary_label,
        }

        return rgb_tensor, attn_tensor, frame_label_tensor, meta
