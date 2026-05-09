"""
Stage 1 — Dataset construction.

Splits each ~40 s Nexar dashcam video into a sequence of 5 s clips that
share the exact temporal layout of the competition test set
(150 frames @ 30 fps), and writes DADA-1000-style annotation lines so the
same Dataset class can read both Nexar and DADA-1000.

For positive videos we cut off the last 2 s before slicing, so the model
never sees the collision frames.  See Section 1 of `TECH_REPORT.md`.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # Raw Nexar dataset (override via env vars).
    video_base = os.environ.get(
        "NEXAR_VIDEO_BASE",
        os.path.join(_THIS_DIR, "data", "nexar_raw", "train"),
    )
    csv_path = os.environ.get(
        "NEXAR_LABEL_CSV",
        os.path.join(_THIS_DIR, "data", "nexar_raw", "train.csv"),
    )
    # Where to dump the 5s clips.  The same env var is consumed by
    # `nexar_config.py`.
    output_dir = os.environ.get(
        "CLIPS_ROOT",
        os.path.join(_THIS_DIR, "data", "nexar_clips_5s"),
    )

    # ---- Clip layout ----
    clip_duration = 5.0
    clip_stride   = 2.5
    target_fps    = 30

    # ---- Train / val split ----
    test_size = 0.2
    seed      = 42

    # ---- Parallelism ----
    num_threads = min((os.cpu_count() or 4) * 2, 16)

    # ---- Class balance ----
    balance = True
    max_neg_clips_per_video = 3



def parse_args():
    parser = argparse.ArgumentParser(description="Nexar 视频拆分为 5s clip")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印拆分信息，不实际写文件")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="只处理前 N 个视频（调试用）")
    parser.add_argument("--clip_duration", type=float, default=Config.clip_duration)
    parser.add_argument("--clip_stride", type=float, default=Config.clip_stride)
    parser.add_argument("--target_fps", type=int, default=Config.target_fps)
    parser.add_argument("--no_balance", action="store_true",
                        help="不做正负样本平衡，保留所有 clip")
    parser.add_argument("--max_neg_clips_per_video", type=int,
                        default=Config.max_neg_clips_per_video,
                        help="负样本视频最多采样的 clip 数（默认 3）")
    return parser.parse_args()


def get_video_path(video_id):
    """在 positive/ 或 negative/ 子目录中查找视频"""
    vid_file = f"{int(video_id):05d}.mp4"
    for subdir in ["positive", "negative"]:
        p = os.path.join(Config.video_base, subdir, vid_file)
        if os.path.exists(p):
            return p, subdir
    return None, None


def process_single_video(video_id, target, time_of_event, split, args):
    """
    处理单个视频：拆分为多个 5s clip，保存帧图片，返回标注列表。
    如果 args._selected_clips 不为 None，则只处理被选中的 clip（阶段2）。
    如果 args._plan_only 为 True，则只计算 clip 元信息，不读帧（阶段1）。

    Returns:
        list of dict: 每个 clip 的标注信息
        dict: 调试统计信息
    """
    video_path, subdir = get_video_path(video_id)
    if video_path is None:
        return [], {"status": "not_found", "video_id": video_id}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], {"status": "open_failed", "video_id": video_id}

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if orig_fps > 0 else 0

    if duration < args.clip_duration:
        cap.release()
        return [], {"status": "too_short", "video_id": video_id, "duration": duration}

    # 计算采样间隔：从原始 fps 采样到 target_fps
    sample_interval = orig_fps / args.target_fps  # 例如 28.9/30 ≈ 0.963

    # 计算 clip 的起止时间
    clip_annotations = []
    stats = {
        "status": "ok",
        "video_id": video_id,
        "target": target,
        "time_of_event": time_of_event,
        "orig_fps": orig_fps,
        "total_frames": total_frames,
        "duration": duration,
        "clips": []
    }

    # 滑动窗口拆分
    clip_start_times = []
    t = 0.0
    while t + args.clip_duration <= duration + 0.01:  # 允许微小浮点误差
        clip_start_times.append(t)
        t += args.clip_stride

    # 如果最后一个 clip 没有覆盖到视频末尾，补一个末尾 clip
    last_possible_start = duration - args.clip_duration
    if last_possible_start > 0 and (not clip_start_times or clip_start_times[-1] < last_possible_start - 0.1):
        clip_start_times.append(last_possible_start)

    # 判断是否只做规划（不读帧）
    plan_only = getattr(args, '_plan_only', False) or args.dry_run
    # 判断哪些 clip 需要写帧
    selected_clips = getattr(args, '_selected_clips', None)  # set of clip_key

    vid_key_base = f"{int(video_id):05d}"

    # 只在需要写帧时才读取视频帧
    all_frames = None
    if not plan_only:
        # 检查是否有任何 clip 需要处理
        need_any_clip = (selected_clips is None)  # 无过滤 → 全部需要
        if not need_any_clip and selected_clips is not None:
            for clip_idx in range(len(clip_start_times)):
                clip_key = f"{vid_key_base}/{clip_idx:03d}"
                if clip_key in selected_clips:
                    need_any_clip = True
                    break

        if need_any_clip:
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)

    cap.release()

    for clip_idx, clip_start_t in enumerate(clip_start_times):
        clip_end_t = clip_start_t + args.clip_duration
        expected_num_frames = int(args.clip_duration * args.target_fps)  # 5s * 30fps = 150
        actual_num_frames = expected_num_frames

        # 确定标签和 toa
        is_positive = (target == 1)

        if is_positive and time_of_event is not None and not np.isnan(time_of_event):
            if clip_start_t <= time_of_event <= clip_end_t:
                label = 1
                toa_in_clip = int(round((time_of_event - clip_start_t) * args.target_fps))
                toa_in_clip = min(toa_in_clip, actual_num_frames - 1)
                toa_abs = toa_in_clip
            else:
                label = 0
                toa_abs = actual_num_frames + 100
        else:
            label = 0
            toa_abs = actual_num_frames + 100

        clip_key = f"{vid_key_base}/{clip_idx:03d}"

        # 写帧图片（非 plan_only 且 clip 被选中时）
        frame_indices = []
        if not plan_only and all_frames is not None:
            should_write = (selected_clips is None) or (clip_key in selected_clips)
            if should_write:
                clip_frame_dir = os.path.join(
                    Config.output_dir, split, "rgb_videos", vid_key_base, f"{clip_idx:03d}"
                )
                os.makedirs(clip_frame_dir, exist_ok=True)
                for fi in range(expected_num_frames):
                    t_frame = clip_start_t + fi / args.target_fps
                    orig_frame_idx = int(round(t_frame * orig_fps))
                    orig_frame_idx = min(orig_frame_idx, len(all_frames) - 1)
                    frame_indices.append(orig_frame_idx)
                    frame = all_frames[orig_frame_idx]
                    frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
                    frame_path = os.path.join(clip_frame_dir, f"{fi:06d}.jpg")
                    cv2.imwrite(frame_path, frame_resized)
        else:
            # plan_only 模式下也计算 frame_indices 用于统计
            for fi in range(expected_num_frames):
                t_frame = clip_start_t + fi / args.target_fps
                orig_frame_idx = int(round(t_frame * orig_fps))
                orig_frame_idx = min(orig_frame_idx, total_frames - 1)
                frame_indices.append(orig_frame_idx)

        # DADA 格式标注
        start_frame = 0
        end_frame = actual_num_frames - 1
        description = f"nexar_{vid_key_base}_clip{clip_idx:03d}"

        ann_line = f"{clip_key} {label} {start_frame} {end_frame} {toa_abs},{description}"
        clip_annotations.append({
            "ann_line": ann_line,
            "clip_key": clip_key,
            "label": label,
            "start": start_frame,
            "end": end_frame,
            "toa": toa_abs,
            "num_frames": actual_num_frames,
            "clip_start_t": clip_start_t,
            "clip_end_t": clip_end_t,
        })

        stats["clips"].append({
            "clip_idx": clip_idx,
            "clip_start_t": round(clip_start_t, 3),
            "clip_end_t": round(clip_end_t, 3),
            "label": label,
            "toa": toa_abs,
            "num_frames": actual_num_frames,
            "frame_indices_range": f"[{frame_indices[0] if frame_indices else 0}, {frame_indices[-1] if frame_indices else 0}]",
        })

    return clip_annotations, stats


def balance_clips(all_annotations, ok_stats, args):
    """
    平衡正负样本 clip 数量。
    策略：
      1. 保留所有正样本 clip（label=1）
      2. 正样本视频中的负样本 clip 全部保留（它们提供了事故前后的上下文）
      3. 从纯负样本视频中随机采样 clip，使总负样本数 ≈ 总正样本数
         - 优先保证丰富度：从尽可能多的负样本视频中各取少量
    """
    balanced = {}

    for split in ["training", "testing"]:
        anns = all_annotations[split]

        # 分类：正样本 clip / 正样本视频的负样本 clip / 纯负样本视频的负样本 clip
        pos_clips = []           # label=1 的 clip
        pos_vid_neg_clips = []   # 正样本视频中 label=0 的 clip
        neg_vid_neg_clips = defaultdict(list)  # 纯负样本视频的 clip，按视频分组

        # 构建正样本视频 ID 集合
        pos_video_ids = set()
        for s in ok_stats:
            if s["target"] == 1:
                pos_video_ids.add(f"{int(s['video_id']):05d}")

        for a in anns:
            vid_id = a["clip_key"].split("/")[0]  # e.g. "00822"
            if a["label"] == 1:
                pos_clips.append(a)
            elif vid_id in pos_video_ids:
                pos_vid_neg_clips.append(a)
            else:
                neg_vid_neg_clips[vid_id].append(a)

        num_pos = len(pos_clips)
        num_pos_vid_neg = len(pos_vid_neg_clips)

        # 需要从纯负样本视频中采样的数量
        need_from_neg_vid = max(0, num_pos - num_pos_vid_neg)

        # 从纯负样本视频中采样，优先保证丰富度
        sampled_neg_clips = []
        neg_vid_ids = list(neg_vid_neg_clips.keys())
        random.shuffle(neg_vid_ids)

        if need_from_neg_vid > 0 and neg_vid_ids:
            # 第一轮：每个负样本视频取 min(max_neg_clips_per_video, 可用数) 个
            max_per_vid = args.max_neg_clips_per_video
            for vid_id in neg_vid_ids:
                clips = neg_vid_neg_clips[vid_id]
                random.shuffle(clips)
                take = min(max_per_vid, len(clips))
                sampled_neg_clips.extend(clips[:take])
                if len(sampled_neg_clips) >= need_from_neg_vid:
                    break

            # 如果还不够，继续从剩余视频中补充
            if len(sampled_neg_clips) < need_from_neg_vid:
                remaining = []
                for vid_id in neg_vid_ids:
                    clips = neg_vid_neg_clips[vid_id]
                    if len(clips) > max_per_vid:
                        remaining.extend(clips[max_per_vid:])
                random.shuffle(remaining)
                sampled_neg_clips.extend(remaining[:need_from_neg_vid - len(sampled_neg_clips)])

            # 截断到需要的数量
            sampled_neg_clips = sampled_neg_clips[:need_from_neg_vid]

        # 如果正样本视频的负样本 clip 已经超过正样本数，也需要采样
        if num_pos_vid_neg > num_pos:
            random.shuffle(pos_vid_neg_clips)
            pos_vid_neg_clips = pos_vid_neg_clips[:num_pos]
            sampled_neg_clips = []  # 不需要额外负样本

        total_neg = len(pos_vid_neg_clips) + len(sampled_neg_clips)

        # 合并
        balanced_anns = pos_clips + pos_vid_neg_clips + sampled_neg_clips

        # 按 clip_key 排序，保持稳定输出
        balanced_anns.sort(key=lambda a: a["clip_key"])

        balanced[split] = balanced_anns

        # 打印平衡信息
        neg_vid_count = len(set(a["clip_key"].split("/")[0] for a in sampled_neg_clips))
        print(f"\n  [{split}] 平衡采样:")
        print(f"    正样本 clip: {num_pos}")
        print(f"    正样本视频的负样本 clip: {min(num_pos_vid_neg, num_pos)}"
              f" (原始 {num_pos_vid_neg})")
        print(f"    纯负样本视频采样 clip: {len(sampled_neg_clips)}"
              f" (来自 {neg_vid_count} 个视频)")
        print(f"    最终负样本 clip: {total_neg}")
        print(f"    最终总 clip: {len(balanced_anns)}")
        print(f"    正负比: 1:{total_neg/max(num_pos,1):.2f}")

    return balanced


def main():
    args = parse_args()
    Config.clip_duration = args.clip_duration
    Config.clip_stride = args.clip_stride
    Config.target_fps = args.target_fps
    Config.balance = not args.no_balance

    random.seed(Config.seed)
    np.random.seed(Config.seed)

    print("=" * 70)
    print("  Nexar 视频拆分为 5s Clip")
    print("=" * 70)
    print(f"  clip_duration = {args.clip_duration}s")
    print(f"  clip_stride   = {args.clip_stride}s")
    print(f"  target_fps    = {args.target_fps}")
    print(f"  output_dir    = {Config.output_dir}")
    print(f"  dry_run       = {args.dry_run}")
    print(f"  num_samples   = {args.num_samples or 'all'}")
    print(f"  balance       = {Config.balance}")
    print(f"  max_neg_clips_per_video = {args.max_neg_clips_per_video}")
    print("=" * 70)

    # 读取 CSV
    df = pd.read_csv(Config.csv_path)
    print(f"\n[INFO] 读取 train.csv: {len(df)} 行")
    print(f"  正样本: {len(df[df['target']==1])}, 负样本: {len(df[df['target']==0])}")

    if args.num_samples:
        df = df.head(args.num_samples)
        print(f"  [DEBUG] 只处理前 {args.num_samples} 个视频")

    # 划分 train/test
    ids = df['id'].unique()
    train_ids, test_ids = train_test_split(ids, test_size=Config.test_size, random_state=Config.seed)
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    print(f"\n[INFO] 数据集划分: train={len(train_ids)}, test={len(test_ids)}")

    # 创建输出目录
    if not args.dry_run:
        for split in ["training", "testing"]:
            os.makedirs(os.path.join(Config.output_dir, split, "rgb_videos"), exist_ok=True)

    # ==================== 阶段1：规划（只读 metadata，不读帧） ====================
    print(f"\n[INFO] 阶段1: 规划 clip 拆分（{len(df)} 个视频）...")
    args._plan_only = True
    args._selected_clips = None

    all_annotations = {"training": [], "testing": []}
    all_stats = []
    error_count = 0
    # 保存每行的信息，阶段2需要重新处理
    video_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="阶段1-规划"):
        video_id = int(row['id'])
        target = int(row['target'])
        time_of_event = row.get('time_of_event', None)

        split = "training" if video_id in train_ids else "testing"
        video_rows.append((video_id, target, time_of_event, split))

        annotations, stats = process_single_video(video_id, target, time_of_event, split, args)
        all_stats.append(stats)

        if stats["status"] != "ok":
            error_count += 1
            if error_count <= 10:
                print(f"  [WARN] video {video_id}: {stats['status']}")
            continue

        all_annotations[split].extend(annotations)

    # ==================== 输出详细统计信息 ====================
    print("\n" + "=" * 70)
    print("  拆分统计")
    print("=" * 70)

    ok_stats = [s for s in all_stats if s["status"] == "ok"]
    fail_stats = [s for s in all_stats if s["status"] != "ok"]

    print(f"\n成功处理: {len(ok_stats)} 个视频, 失败: {len(fail_stats)} 个")
    if fail_stats:
        for s in fail_stats[:5]:
            print(f"  失败: video_id={s['video_id']}, reason={s['status']}")

    # 视频时长统计
    durations = [s["duration"] for s in ok_stats]
    if durations:
        print(f"\n视频时长统计:")
        print(f"  min={min(durations):.2f}s, max={max(durations):.2f}s, "
              f"mean={np.mean(durations):.2f}s, median={np.median(durations):.2f}s")

    # FPS 统计
    fps_list = [s["orig_fps"] for s in ok_stats]
    if fps_list:
        print(f"\n原始 FPS 统计:")
        print(f"  min={min(fps_list):.1f}, max={max(fps_list):.1f}, "
              f"mean={np.mean(fps_list):.1f}, median={np.median(fps_list):.1f}")

    # Clip 数量统计
    clips_per_video = [len(s["clips"]) for s in ok_stats]
    if clips_per_video:
        print(f"\n每个视频的 clip 数:")
        print(f"  min={min(clips_per_video)}, max={max(clips_per_video)}, "
              f"mean={np.mean(clips_per_video):.1f}, total={sum(clips_per_video)}")

    # 按 split 统计
    for split in ["training", "testing"]:
        anns = all_annotations[split]
        pos = [a for a in anns if a["label"] == 1]
        neg = [a for a in anns if a["label"] == 0]
        print(f"\n{split} (平衡前):")
        print(f"  总 clip 数: {len(anns)}")
        print(f"  正样本 clip: {len(pos)}")
        print(f"  负样本 clip: {len(neg)}")
        if pos:
            toa_list = [a["toa"] for a in pos]
            print(f"  正样本 toa 统计: min={min(toa_list)}, max={max(toa_list)}, "
                  f"mean={np.mean(toa_list):.1f}")
            nf_list = [a["num_frames"] for a in pos]
            print(f"  正样本帧数统计: min={min(nf_list)}, max={max(nf_list)}, "
                  f"mean={np.mean(nf_list):.1f}")

    # ==================== 示例 clip 详情 ====================
    print("\n" + "=" * 70)
    print("  示例 clip 详情（前 3 个正样本视频）")
    print("=" * 70)

    pos_stats = [s for s in ok_stats if s["target"] == 1][:3]
    for s in pos_stats:
        print(f"\n视频 {s['video_id']} (target={s['target']}, "
              f"time_of_event={s['time_of_event']:.3f}s, "
              f"duration={s['duration']:.2f}s, fps={s['orig_fps']:.1f}):")
        for c in s["clips"]:
            marker = " ★" if c["label"] == 1 else ""
            print(f"  clip {c['clip_idx']:3d}: "
                  f"t=[{c['clip_start_t']:.1f}s, {c['clip_end_t']:.1f}s], "
                  f"frames={c['num_frames']}, "
                  f"label={c['label']}, toa={c['toa']}{marker}")

    # 打印前 3 个负样本视频
    neg_stats = [s for s in ok_stats if s["target"] == 0][:3]
    print(f"\n--- 前 3 个负样本视频 ---")
    for s in neg_stats:
        print(f"\n视频 {s['video_id']} (target={s['target']}, "
              f"duration={s['duration']:.2f}s, fps={s['orig_fps']:.1f}):")
        for c in s["clips"][:5]:  # 只打印前5个clip
            print(f"  clip {c['clip_idx']:3d}: "
                  f"t=[{c['clip_start_t']:.1f}s, {c['clip_end_t']:.1f}s], "
                  f"frames={c['num_frames']}, "
                  f"label={c['label']}, toa={c['toa']}")
        if len(s["clips"]) > 5:
            print(f"  ... 共 {len(s['clips'])} 个 clip")

    # ==================== 正负样本平衡 ====================
    if Config.balance:
        print("\n" + "=" * 70)
        print("  正负样本平衡采样")
        print("=" * 70)
        balanced_annotations = balance_clips(all_annotations, ok_stats, args)
    else:
        balanced_annotations = all_annotations

    # 平衡后统计
    if Config.balance:
        print("\n" + "-" * 40)
        print("  平衡后统计:")
        print("-" * 40)
        for split in ["training", "testing"]:
            anns = balanced_annotations[split]
            pos = [a for a in anns if a["label"] == 1]
            neg = [a for a in anns if a["label"] == 0]
            print(f"\n  {split}:")
            print(f"    总 clip 数: {len(anns)}")
            print(f"    正样本 clip: {len(pos)}")
            print(f"    负样本 clip: {len(neg)}")
            # 统计涉及的视频数
            vid_ids = set(a["clip_key"].split("/")[0] for a in anns)
            print(f"    涉及视频数: {len(vid_ids)}")

    # ==================== 阶段2：只写被选中的 clip 帧图片 ====================
    if not args.dry_run:
        # 收集所有被选中的 clip_key
        selected_clips = set()
        for split in ["training", "testing"]:
            for a in balanced_annotations[split]:
                selected_clips.add(a["clip_key"])

        # 计算需要处理的视频（只处理包含被选中 clip 的视频）
        selected_vid_ids = set(ck.split("/")[0] for ck in selected_clips)
        videos_to_process = [(vid_id, target, toe, split)
                             for vid_id, target, toe, split in video_rows
                             if f"{vid_id:05d}" in selected_vid_ids]

        total_selected = len(selected_clips)
        total_before = sum(len(all_annotations[s]) for s in ["training", "testing"])
        skip_ratio = 1.0 - total_selected / max(total_before, 1)

        print(f"\n" + "=" * 70)
        print(f"  阶段2: 写帧图片")
        print(f"=" * 70)
        print(f"  被选中的 clip: {total_selected} / {total_before} "
              f"(跳过 {skip_ratio*100:.1f}%)")
        print(f"  需要处理的视频: {len(videos_to_process)} / {len(video_rows)}")

        args._plan_only = False
        args._selected_clips = selected_clips

        for vid_id, target, toe, split in tqdm(videos_to_process, desc="阶段2-写帧"):
            process_single_video(vid_id, target, toe, split, args)

        print(f"  [INFO] 帧图片写入完成!")
    else:
        args._plan_only = True

    # ==================== 验证逻辑正确性 ====================
    print("\n" + "=" * 70)
    print("  逻辑验证")
    print("=" * 70)

    # 验证1: 正样本视频至少有一个 label=1 的 clip
    pos_videos_with_pos_clip = 0
    pos_videos_without_pos_clip = 0
    for s in ok_stats:
        if s["target"] == 1:
            has_pos_clip = any(c["label"] == 1 for c in s["clips"])
            if has_pos_clip:
                pos_videos_with_pos_clip += 1
            else:
                pos_videos_without_pos_clip += 1
                if pos_videos_without_pos_clip <= 3:
                    print(f"  [WARN] 正样本视频 {s['video_id']} 没有 label=1 的 clip! "
                          f"time_of_event={s['time_of_event']:.3f}s, duration={s['duration']:.2f}s")

    print(f"\n验证1 - 正样本视频覆盖:")
    print(f"  有正样本 clip 的正样本视频: {pos_videos_with_pos_clip}")
    print(f"  无正样本 clip 的正样本视频: {pos_videos_without_pos_clip}")

    # 验证2: 负样本视频不应有 label=1 的 clip
    neg_videos_with_pos_clip = 0
    for s in ok_stats:
        if s["target"] == 0:
            has_pos_clip = any(c["label"] == 1 for c in s["clips"])
            if has_pos_clip:
                neg_videos_with_pos_clip += 1
    print(f"\n验证2 - 负样本视频不应有正样本 clip:")
    print(f"  有正样本 clip 的负样本视频: {neg_videos_with_pos_clip} (应为 0)")

    # 验证3: 正样本 clip 的 toa 应在 [0, num_frames-1] 范围内
    toa_out_of_range = 0
    for split_anns in balanced_annotations.values():
        for a in split_anns:
            if a["label"] == 1:
                if a["toa"] < 0 or a["toa"] >= a["num_frames"]:
                    toa_out_of_range += 1
    print(f"\n验证3 - 正样本 toa 范围:")
    print(f"  toa 超出 [0, num_frames-1] 的正样本 clip: {toa_out_of_range} (应为 0)")

    # 验证4: 每个 clip 的帧数应为 clip_duration * target_fps
    expected_frames = int(args.clip_duration * args.target_fps)
    wrong_frame_count = 0
    for split_anns in balanced_annotations.values():
        for a in split_anns:
            if a["num_frames"] != expected_frames:
                wrong_frame_count += 1
    print(f"\n验证4 - clip 帧数一致性:")
    print(f"  期望帧数: {expected_frames}")
    print(f"  帧数不一致的 clip: {wrong_frame_count}")

    # 验证5: 正样本 clip 的 toa 对应的时间应接近 time_of_event
    print(f"\n验证5 - 正样本 toa 时间对齐（抽样检查）:")
    checked = 0
    for s in ok_stats:
        if s["target"] == 1 and checked < 5:
            for c in s["clips"]:
                if c["label"] == 1 and checked < 5:
                    toa_time = c["clip_start_t"] + c["toa"] / args.target_fps
                    diff = abs(toa_time - s["time_of_event"])
                    status = "✓" if diff < 0.1 else "✗"
                    print(f"  {status} video {s['video_id']} clip {c['clip_idx']}: "
                          f"toa_time={toa_time:.3f}s, event_time={s['time_of_event']:.3f}s, "
                          f"diff={diff:.4f}s")
                    checked += 1

    # ==================== 写标注文件 ====================
    if not args.dry_run:
        for split in ["training", "testing"]:
            ann_path = os.path.join(Config.output_dir, f"{split}.txt")
            with open(ann_path, "w") as f:
                for a in balanced_annotations[split]:
                    f.write(a["ann_line"] + "\n")
            print(f"\n[INFO] 标注文件已保存: {ann_path} ({len(balanced_annotations[split])} 行)")

        # 保存统计信息
        import json
        stats_path = os.path.join(Config.output_dir, "split_stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "config": {
                    "clip_duration": args.clip_duration,
                    "clip_stride": args.clip_stride,
                    "target_fps": args.target_fps,
                    "balance": Config.balance,
                    "max_neg_clips_per_video": args.max_neg_clips_per_video,
                },
                "summary_before_balance": {
                    "total_videos": len(ok_stats),
                    "training_clips": len(all_annotations["training"]),
                    "testing_clips": len(all_annotations["testing"]),
                    "training_pos": sum(1 for a in all_annotations["training"] if a["label"] == 1),
                    "training_neg": sum(1 for a in all_annotations["training"] if a["label"] == 0),
                    "testing_pos": sum(1 for a in all_annotations["testing"] if a["label"] == 1),
                    "testing_neg": sum(1 for a in all_annotations["testing"] if a["label"] == 0),
                },
                "summary_after_balance": {
                    "training_clips": len(balanced_annotations["training"]),
                    "testing_clips": len(balanced_annotations["testing"]),
                    "training_pos": sum(1 for a in balanced_annotations["training"] if a["label"] == 1),
                    "training_neg": sum(1 for a in balanced_annotations["training"] if a["label"] == 0),
                    "testing_pos": sum(1 for a in balanced_annotations["testing"] if a["label"] == 1),
                    "testing_neg": sum(1 for a in balanced_annotations["testing"] if a["label"] == 0),
                }
            }, f, indent=2)
        print(f"[INFO] 统计信息已保存: {stats_path}")
    else:
        print("\n[DRY RUN] 未写入任何文件")

    # ==================== 打印标注文件示例 ====================
    print("\n" + "=" * 70)
    print("  标注文件示例（前 10 行）")
    print("=" * 70)
    for split in ["training", "testing"]:
        print(f"\n{split}.txt:")
        for a in balanced_annotations[split][:10]:
            print(f"  {a['ann_line']}")
        print(f"  ... 共 {len(balanced_annotations[split])} 行")

    print("\n" + "=" * 70)
    print("  完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
