"""
VideoMAE + Attention-ROI-GCN 融合模型

架构:
  1. VideoMAE 分支: VideoMAEv2-base → 时空 token → 全局池化 → 768维全局特征
  2. GCN 分支: Attention Map → Top-K ROI 选择 → RoIAlign 提取局部特征 → GCN 图推理 → 空间推理特征
  3. 门控融合: gate * videomae_feat + (1-gate) * gcn_feat → MLP → 逐帧事故概率
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.ops import roi_align
from transformers import AutoModel, AutoConfig


# ============ GCN 模块（移植自 downstream/models/gcn.py）============

class GraphConvolution(nn.Module):
    """简单 GCN 层"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: (B, N, in_features)
        adj: (B, N, N)
        """
        support = torch.matmul(x, self.weight)   # (B, N, out)
        output = torch.matmul(adj, support)       # (B, N, out)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    """两层 GCN"""

    def __init__(self, nin, nhid, nout, dropout=0.1):
        super().__init__()
        self.gc1 = GraphConvolution(nin, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        return x


# ============ Attention ROI 选择模块 ============

class AttentionROISelector(nn.Module):
    """
    从 attention map 中选择 Top-K 高注意力区域作为 ROI。
    使用连通域分析找到注意力热点区域，输出 ROI 坐标。
    """

    def __init__(self, num_rois=6, roi_size=7, feature_dim=768, img_size=224, random_roi=False):
        super().__init__()
        self.num_rois = num_rois
        self.roi_size = roi_size
        self.img_size = img_size
        self.random_roi = random_roi  # 消融实验: 随机选择 ROI 而非 attention 引导

        # 将 attention map 编码为特征（用于 RoIAlign 的特征图）
        # 输入: (B*T, 1, H, W) → (B*T, feature_dim, H/16, W/16)
        self.attn_feature_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        # 输出尺寸: 224/16 = 14x14

    def _select_rois_from_attn(self, attn_maps):
        """
        从 attention map 中选择 Top-K ROI 区域。
        attn_maps: (B*T, 1, H, W) 值域 [0, 1]
        返回: rois_list - list of (B*T, num_rois, 4) 格式为 (x1, y1, x2, y2) 归一化坐标
        """
        BT, _, H, W = attn_maps.shape
        device = attn_maps.device

        # 使用自适应阈值 + 网格划分来选择 ROI
        # 将图像划分为 grid，选择注意力最高的 grid cell
        grid_h, grid_w = 4, 4  # 4x4 网格 = 16 个候选区域
        cell_h, cell_w = H // grid_h, W // grid_w

        attn_np = attn_maps.squeeze(1)  # (BT, H, W)

        # 计算每个 grid cell 的平均注意力
        # (BT, grid_h, grid_w)
        cell_scores = torch.zeros(BT, grid_h, grid_w, device=device)
        for i in range(grid_h):
            for j in range(grid_w):
                cell = attn_np[:, i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_scores[:, i, j] = cell.mean(dim=(-2, -1))

        # 选择 Top-K cell
        cell_scores_flat = cell_scores.view(BT, -1)  # (BT, 16)
        k = min(self.num_rois, cell_scores_flat.shape[1])
        if self.random_roi:
            # 消融实验: 随机选择同样数量的 grid cell，不依赖 attention 分数
            top_indices = torch.stack([
                torch.randperm(cell_scores_flat.shape[1], device=device)[:k]
                for _ in range(BT)
            ])  # (BT, k)
        else:
            _, top_indices = cell_scores_flat.topk(k, dim=1)  # (BT, k)

        # 转换为 ROI 坐标 (x1, y1, x2, y2)，使用原始像素坐标
        rois = torch.zeros(BT, k, 4, device=device)
        for n in range(k):
            idx = top_indices[:, n]  # (BT,)
            row = idx // grid_w      # (BT,)
            col = idx % grid_w       # (BT,)
            rois[:, n, 0] = col.float() * cell_w        # x1
            rois[:, n, 1] = row.float() * cell_h        # y1
            rois[:, n, 2] = (col.float() + 1) * cell_w  # x2
            rois[:, n, 3] = (row.float() + 1) * cell_h  # y2

        # padding 到 num_rois
        if k < self.num_rois:
            pad = rois[:, -1:, :].expand(-1, self.num_rois - k, -1)
            rois = torch.cat([rois, pad], dim=1)

        return rois  # (BT, num_rois, 4)

    def forward(self, attn_maps, rgb_features=None):
        """
        attn_maps: (B, T, 1, H, W)
        rgb_features: (B, T, C, Hf, Wf) 可选，如果提供则从 RGB 特征图上做 RoIAlign

        返回:
            roi_features: (B, T, num_rois, feature_dim)  每个 ROI 的特征
            rois: (B*T, num_rois, 4)  ROI 坐标
        """
        B, T, C_attn, H, W = attn_maps.shape
        device = attn_maps.device

        # 展平 batch 和 time
        attn_flat = attn_maps.view(B * T, C_attn, H, W)  # (BT, 1, H, W)

        # 编码 attention map 为特征图
        feat_map = self.attn_feature_encoder(attn_flat)  # (BT, feat_dim, Hf, Wf)
        feat_dim = feat_map.shape[1]
        Hf, Wf = feat_map.shape[2], feat_map.shape[3]

        # 选择 ROI
        rois = self._select_rois_from_attn(attn_flat)  # (BT, num_rois, 4)

        # 缩放 ROI 坐标到特征图尺度
        scale_x = Wf / W
        scale_y = Hf / H
        rois_scaled = rois.clone()
        rois_scaled[:, :, 0] *= scale_x
        rois_scaled[:, :, 1] *= scale_y
        rois_scaled[:, :, 2] *= scale_x
        rois_scaled[:, :, 3] *= scale_y

        # 构建 torchvision roi_align 需要的格式: list of (num_rois, 5) [batch_idx, x1, y1, x2, y2]
        batch_indices = torch.arange(B * T, device=device).unsqueeze(1).expand(-1, self.num_rois)
        rois_for_align = torch.cat([
            batch_indices.reshape(-1, 1).float(),
            rois_scaled.reshape(-1, 4)
        ], dim=1)  # (BT*num_rois, 5)

        # RoIAlign
        roi_features = roi_align(
            feat_map, rois_for_align,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            aligned=True
        )  # (BT*num_rois, feat_dim, roi_size, roi_size)

        # 全局平均池化
        roi_features = roi_features.mean(dim=(-2, -1))  # (BT*num_rois, feat_dim)
        roi_features = roi_features.view(B, T, self.num_rois, feat_dim)

        return roi_features, rois.view(B, T, self.num_rois, 4)


# ============ 构建邻接矩阵 ============

def build_adjacency(num_rois, device):
    """
    构建全连接 + 自环的邻接矩阵（归一化）。
    返回: (num_rois+1, num_rois+1) 包含一个额外的 state 节点
    """
    N = num_rois + 1  # +1 for state node
    adj = torch.ones(N, N, device=device)
    # 度归一化 D^{-1/2} A D^{-1/2}
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    return adj


# ============ VideoMAE Backbone 工具函数 ============

def get_videomae_backbone(model_id, drop_path_rate=0.1):
    """
    加载 VideoMAEv2 backbone。

    transformers >= 5.x 默认使用 meta device 加速初始化，
    但 VideoMAEv2 的 VisionTransformer.__init__ 中
    torch.linspace(...).item() 在 meta tensor 上会报错。

    解决方案: 手动加载 config，然后用 AutoModel.from_config 在 CPU 上
    实例化模型，再手动加载预训练权重。
    """
    import logging
    logger = logging.getLogger(__name__)

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # 设置 drop_path_rate
    if hasattr(cfg, "drop_path_rate"):
        cfg.drop_path_rate = drop_path_rate
    if hasattr(cfg, "model_config") and isinstance(cfg.model_config, dict):
        cfg.model_config["drop_path_rate"] = drop_path_rate

    logger.info(f"Model config: {cfg.model_config if hasattr(cfg, 'model_config') else cfg}")

    # 方案: 先在 CPU 上用 from_config 实例化空模型（绕过 meta device），
    # 再手动下载并加载预训练权重
    try:
        # 尝试直接加载，设置 low_cpu_mem_usage=False 和 device_map=None
        backbone = AutoModel.from_pretrained(
            model_id, config=cfg, trust_remote_code=True,
            low_cpu_mem_usage=False, device_map=None,
        )
        logger.info("成功通过 from_pretrained(low_cpu_mem_usage=False) 加载模型")
    except RuntimeError as e:
        if "meta tensor" in str(e) or "meta" in str(e).lower():
            logger.warning(f"from_pretrained 遇到 meta tensor 错误: {e}")
            logger.info("使用备用方案: from_config + 手动加载权重...")

            # Step 1: 在 CPU 上实例化空模型
            backbone = AutoModel.from_config(cfg, trust_remote_code=True)

            # Step 2: 下载并加载预训练权重
            from huggingface_hub import hf_hub_download
            import safetensors.torch
            import os, glob

            # 尝试下载 safetensors 或 pytorch_model.bin
            try:
                weight_file = hf_hub_download(model_id, "model.safetensors")
                state_dict = safetensors.torch.load_file(weight_file)
            except Exception:
                try:
                    weight_file = hf_hub_download(model_id, "pytorch_model.bin")
                    state_dict = torch.load(weight_file, map_location="cpu")
                except Exception:
                    # 尝试分片权重
                    snapshot_dir = os.path.dirname(
                        hf_hub_download(model_id, "config.json")
                    )
                    shard_files = sorted(glob.glob(
                        os.path.join(snapshot_dir, "model-*.safetensors")
                    ))
                    if shard_files:
                        state_dict = {}
                        for sf in shard_files:
                            state_dict.update(safetensors.torch.load_file(sf))
                    else:
                        raise RuntimeError(
                            f"无法找到 {model_id} 的权重文件"
                        )

            # 加载权重（允许部分不匹配）
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"缺失的权重键: {missing[:5]}...")
            if unexpected:
                logger.warning(f"多余的权重键: {unexpected[:5]}...")
            logger.info("备用方案加载成功")
        else:
            raise

    return backbone


def get_inner_model(backbone):
    """找到 VideoMAE 内部包含 patch_embed 的模型"""
    if hasattr(backbone, "patch_embed"):
        return backbone
    for name, child in backbone.named_children():
        if hasattr(child, "patch_embed"):
            return child
    raise RuntimeError(
        f"Cannot find patch_embed. Submodules: {[n for n, _ in backbone.named_children()]}"
    )


def freeze_backbone_layers(model, num_layers):
    """
    冻结 VideoMAE backbone 的前 num_layers 个 transformer block。
    同时冻结 patch_embed 和 pos_embed（这些是底层特征提取，通常不需要微调）。

    VideoMAEv2-base 共 12 个 transformer block:
      - num_layers=0: 不冻结任何层（默认行为）
      - num_layers=8: 冻结前 8 层，只微调后 4 层 + head
      - num_layers=12: 冻结全部 backbone（只训练 head）

    Args:
        model: VideoMAEGCNModel 或 VideoMAEGCNModelV2 实例
        num_layers: 要冻结的 transformer block 数量

    Returns:
        frozen_count: 被冻结的参数数量
    """
    import logging
    logger = logging.getLogger(__name__)

    inner = get_inner_model(model.backbone)
    frozen_count = 0

    # 1. 冻结 patch_embed（底层卷积，提取 patch token）
    if hasattr(inner, 'patch_embed'):
        for param in inner.patch_embed.parameters():
            param.requires_grad = False
            frozen_count += 1

    # 2. 冻结 pos_embed（位置编码）
    if hasattr(inner, 'pos_embed') and inner.pos_embed is not None:
        inner.pos_embed.requires_grad = False
        frozen_count += 1

    # 3. 冻结 cls_token（如果存在）
    if hasattr(inner, 'cls_token') and inner.cls_token is not None:
        inner.cls_token.requires_grad = False
        frozen_count += 1

    # 4. 冻结前 num_layers 个 transformer block
    total_blocks = len(inner.blocks)
    num_to_freeze = min(num_layers, total_blocks)
    for i in range(num_to_freeze):
        for param in inner.blocks[i].parameters():
            param.requires_grad = False
            frozen_count += 1

    logger.info(f"冻结 backbone: patch_embed + pos_embed + 前 {num_to_freeze}/{total_blocks} 个 block "
                f"(共 {frozen_count} 个参数张量)")

    return frozen_count


def extract_videomae_tokens(backbone, x):
    """
    提取 VideoMAE token 级别特征（不做最终池化）。
    支持任意帧数输入：当输入帧数 != 预训练帧数(16)时，
    自动对位置编码进行 3D 插值（时间+空间）。

    x: (B, C, T, H, W)  T 可以是 16（原始）或其他值（如 50）
    返回: (B, num_tokens, embed_dim)
    """
    model = get_inner_model(backbone)
    B = x.size(0)
    x = model.patch_embed(x)  # (B, N_new, embed_dim)
    if model.pos_embed is not None:
        pos_embed = model.pos_embed  # (1, N_orig, embed_dim)
        N_new = x.shape[1]
        N_orig = pos_embed.shape[1]
        if N_new != N_orig:
            # 需要插值位置编码
            # VideoMAEv2-base: patch_size=16, tubelet_size=2
            # 原始 16 帧 → 8 时间 token, 224/16=14 空间 → 14x14=196 空间 token
            # N_orig = 8 * 196 = 1568
            # 推断原始时空网格尺寸
            orig_t = 8  # 16 帧 / tubelet_size=2
            orig_hw = int(math.sqrt(N_orig // orig_t))  # 14
            assert orig_t * orig_hw * orig_hw == N_orig, \
                f"位置编码维度不匹配: {N_orig} != {orig_t}*{orig_hw}*{orig_hw}"

            # 推断新的时空网格尺寸
            T_input = x.shape[1] // (orig_hw * orig_hw)  # 新的时间 token 数
            new_hw = orig_hw  # 空间分辨率不变
            assert T_input * new_hw * new_hw == N_new, \
                f"新 token 数不匹配: {N_new} != {T_input}*{new_hw}*{new_hw}"

            # 3D 插值: (1, N_orig, D) → (1, orig_t, orig_hw, orig_hw, D)
            embed_dim = pos_embed.shape[2]
            pos_embed_3d = pos_embed.reshape(1, orig_t, orig_hw, orig_hw, embed_dim)
            # 转为 (1, D, orig_t, orig_hw, orig_hw) 用于 F.interpolate
            pos_embed_3d = pos_embed_3d.permute(0, 4, 1, 2, 3)
            pos_embed_3d = F.interpolate(
                pos_embed_3d, size=(T_input, new_hw, new_hw),
                mode='trilinear', align_corners=False
            )
            # 转回 (1, N_new, D)
            pos_embed = pos_embed_3d.permute(0, 2, 3, 4, 1).reshape(1, N_new, embed_dim)

        x = x + pos_embed.expand(B, -1, -1).type_as(x).clone().detach()
    x = model.pos_drop(x)
    for blk in model.blocks:
        x = blk(x)
    return x


# ============ 主模型 ============

class VideoMAEGCNModel(nn.Module):
    """
    VideoMAE + Attention-ROI-GCN 双分支融合模型。

    前向流程:
    1. VideoMAE 分支: RGB → VideoMAEv2 → token features → 池化 → 全局特征 (768)
    2. GCN 分支: Attention Map → ROI 选择 → RoIAlign → GCN 推理 → 空间特征
    3. 门控融合 → MLP → 逐帧事故概率
    """

    def __init__(
        self,
        model_id="OpenGVLab/VideoMAEv2-base",
        embed_dim=768,
        num_rois=6,
        roi_size=7,
        gcn_hidden=256,
        gcn_dropout=0.1,
        fusion_dim=512,
        img_size=224,
        drop_path_rate=0.1,
    ):
        super().__init__()

        # ---- VideoMAE Backbone ----
        self.backbone = get_videomae_backbone(model_id, drop_path_rate)
        self.fc_norm = nn.LayerNorm(embed_dim)

        # ---- GCN 分支 ----
        self.num_rois = num_rois
        self.roi_selector = AttentionROISelector(
            num_rois=num_rois, roi_size=roi_size,
            feature_dim=embed_dim, img_size=img_size
        )

        # GCN: 输入 = roi_features (embed_dim), 输出 = gcn_hidden
        # 节点数 = num_rois + 1 (state node)
        self.gcn = GCN(
            nin=embed_dim, nhid=gcn_hidden, nout=gcn_hidden,
            dropout=gcn_dropout
        )

        # State 节点初始化投影（从全局 VideoMAE 特征投影）
        self.state_proj = nn.Linear(embed_dim, embed_dim)

        # ---- 融合层 ----
        # VideoMAE 全局特征: embed_dim
        # GCN state 节点特征: gcn_hidden
        self.gate_fc = nn.Sequential(
            nn.Linear(embed_dim + gcn_hidden, 1),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim + gcn_hidden, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # ---- 预测头: 输出每帧事故 logits ----
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, rgb, attn_maps):
        """
        rgb: (B, T, 3, H, W)
        attn_maps: (B, T, 1, H, W)

        返回:
            pred: (B,) 当前窗口的事故概率（取窗口内最大值）
            pred_frames: (B, T) 逐帧事故概率
        """
        B, T, C, H, W = rgb.shape
        device = rgb.device

        # ---- 1. VideoMAE 分支 ----
        # (B, T, C, H, W) → (B, C, T, H, W)
        rgb_input = rgb.permute(0, 2, 1, 3, 4).float()
        tokens = extract_videomae_tokens(self.backbone, rgb_input)  # (B, N_tokens, embed_dim)
        # 全局池化
        global_feat = self.fc_norm(tokens.mean(dim=1))  # (B, embed_dim)

        # ---- 2. GCN 分支 ----
        # 提取 ROI 特征
        roi_features, rois = self.roi_selector(attn_maps)  # (B, T, num_rois, embed_dim)

        # 对时间维度做平均池化，得到每个 ROI 的时间聚合特征
        roi_feat_avg = roi_features.mean(dim=1)  # (B, num_rois, embed_dim)

        # 构建图节点: [roi_1, ..., roi_K, state_node]
        state_feat = self.state_proj(global_feat).unsqueeze(1)  # (B, 1, embed_dim)
        graph_nodes = torch.cat([roi_feat_avg, state_feat], dim=1)  # (B, num_rois+1, embed_dim)

        # 邻接矩阵
        adj = build_adjacency(self.num_rois, device)  # (num_rois+1, num_rois+1)
        adj = adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # GCN 推理
        gcn_out = self.gcn(graph_nodes, adj)  # (B, num_rois+1, gcn_hidden)

        # 取 state 节点特征
        state_out = gcn_out[:, -1, :]  # (B, gcn_hidden)

        # ---- 3. 门控融合 ----
        combined = torch.cat([global_feat, state_out], dim=1)  # (B, embed_dim + gcn_hidden)
        fused = self.fusion_proj(combined)  # (B, fusion_dim)

        # ---- 4. 逐帧预测 ----
        # 将融合特征扩展到每帧（窗口内共享同一个融合特征，但通过时间位置编码区分）
        # 简单方案: 每帧使用相同的融合特征预测
        pred_frames = self.predictor(fused).squeeze(-1)  # (B, 1) → (B,)

        # 扩展为逐帧预测（窗口内所有帧共享同一预测值）
        pred_frames_expanded = pred_frames.unsqueeze(1).expand(-1, T)  # (B, T)

        return pred_frames, pred_frames_expanded


class VideoMAEGCNModelV2(nn.Module):
    """
    V2 版本: 逐帧预测，使用时间感知的融合。
    每帧的 ROI 特征 + 全局特征 → 逐帧独立预测。

    支持两种模式:
    - 滑动窗口模式 (window_size=16): 与原始行为一致
    - 全局模式 (window_size=50): 整个 clip 降采样后一次性输入，
      VideoMAE 位置编码自动插值
    """

    def __init__(
        self,
        model_id="OpenGVLab/VideoMAEv2-base",
        embed_dim=768,
        num_rois=6,
        roi_size=7,
        gcn_hidden=256,
        gcn_dropout=0.1,
        fusion_dim=512,
        img_size=224,
        drop_path_rate=0.1,
        window_size=16,
        disable_gcn=False,
        random_roi=False,
    ):
        super().__init__()

        self.window_size = window_size
        self.embed_dim = embed_dim
        self.disable_gcn = disable_gcn  # 消融实验: 不使用 GCN 分支

        # ---- VideoMAE Backbone ----
        self.backbone = get_videomae_backbone(model_id, drop_path_rate)
        self.fc_norm = nn.LayerNorm(embed_dim)

        # ---- GCN 分支（消融实验可禁用）----
        if not disable_gcn:
            self.num_rois = num_rois
            self.roi_selector = AttentionROISelector(
                num_rois=num_rois, roi_size=roi_size,
                feature_dim=embed_dim, img_size=img_size,
                random_roi=random_roi,
            )

            self.gcn = GCN(
                nin=embed_dim, nhid=gcn_hidden, nout=gcn_hidden,
                dropout=gcn_dropout
            )

            self.state_proj = nn.Linear(embed_dim, embed_dim)

        # ---- 时间位置编码（支持插值到任意长度）----
        # 初始化为 window_size 长度，forward 时如果 T != window_size 则插值
        self.temporal_pe = nn.Parameter(torch.randn(1, window_size, embed_dim) * 0.02)

        # ---- 逐帧融合 + 预测 ----
        if disable_gcn:
            # 消融: 不用 GCN，融合输入 = global_feat(embed_dim) + temporal_feat(embed_dim)
            fusion_input_dim = embed_dim + embed_dim
        else:
            # 完整模型: global_feat(embed_dim) + gcn_state(gcn_hidden) + temporal_feat(embed_dim)
            fusion_input_dim = embed_dim + gcn_hidden + embed_dim

        self.frame_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # 输出 2 类 logits，与 LOTVS-CAP GRUNet(output_dim=2) 完全一致
        self.frame_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, rgb, attn_maps):
        """
        rgb: (B, T, 3, H, W)
        attn_maps: (B, T, 1, H, W)

        返回:
            pred: (B,) 窗口级预测（取 max）
            pred_frames: (B, T) 逐帧概率
        """
        B, T, C, H, W = rgb.shape
        device = rgb.device

        # ---- 1. VideoMAE 分支 ----
        # VideoMAE 要求输入帧数为 tubelet_size(2) 的倍数
        # 如果 T 不是 2 的倍数，padding 一帧
        T_input = T
        rgb_for_mae = rgb
        if T % 2 != 0:
            rgb_for_mae = torch.cat([rgb, rgb[:, -1:]], dim=1)  # 复制最后一帧
            T_input = T + 1

        rgb_input = rgb_for_mae.permute(0, 2, 1, 3, 4).float()  # (B, C, T_input, H, W)
        tokens = extract_videomae_tokens(self.backbone, rgb_input)  # (B, N, embed_dim)
        global_feat = self.fc_norm(tokens.mean(dim=1))  # (B, embed_dim)

        # 将 tokens 重塑为时空网格，提取每帧的特征
        # VideoMAEv2-base: tubelet_size=2, patch_size=16
        # T_input 帧 → T_input//2 个时间 token, 14x14=196 空间 token
        num_temporal = T_input // 2
        num_spatial = tokens.shape[1] // num_temporal
        tokens_reshaped = tokens.view(B, num_temporal, num_spatial, self.embed_dim)
        # 每个时间步的特征: 对空间维度池化
        temporal_feats = tokens_reshaped.mean(dim=2)  # (B, num_temporal, embed_dim)

        # 上采样到 T 帧（原始输入帧数，不含 padding）
        temporal_feats = F.interpolate(
            temporal_feats.permute(0, 2, 1),  # (B, embed_dim, num_temporal)
            size=T, mode="linear", align_corners=False
        ).permute(0, 2, 1)  # (B, T, embed_dim)

        # 加时间位置编码（支持插值到任意长度）
        if T == self.temporal_pe.shape[1]:
            pe = self.temporal_pe
        else:
            # 插值时间位置编码: (1, window_size, D) → (1, T, D)
            pe = F.interpolate(
                self.temporal_pe.permute(0, 2, 1),  # (1, D, window_size)
                size=T, mode='linear', align_corners=False
            ).permute(0, 2, 1)  # (1, T, D)
        temporal_feats = temporal_feats + pe

        # ---- 2. GCN 分支（逐帧）（消融实验可跳过）----
        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, T, -1)  # (B, T, embed_dim)

        if not self.disable_gcn:
            roi_features, rois = self.roi_selector(attn_maps)  # (B, T, num_rois, embed_dim)

            # 逐帧 GCN
            gcn_states = []
            for t in range(T):
                roi_t = roi_features[:, t, :, :]  # (B, num_rois, embed_dim)
                state_t = self.state_proj(temporal_feats[:, t, :]).unsqueeze(1)  # (B, 1, embed_dim)
                nodes_t = torch.cat([roi_t, state_t], dim=1)  # (B, num_rois+1, embed_dim)

                adj = build_adjacency(self.num_rois, device).unsqueeze(0).expand(B, -1, -1)
                gcn_out_t = self.gcn(nodes_t, adj)  # (B, num_rois+1, gcn_hidden)
                gcn_states.append(gcn_out_t[:, -1, :])  # (B, gcn_hidden)

            gcn_states = torch.stack(gcn_states, dim=1)  # (B, T, gcn_hidden)

            # 融合: global_feat + gcn_states + temporal_feats
            frame_input = torch.cat([
                global_feat_expanded, gcn_states, temporal_feats
            ], dim=2)  # (B, T, embed_dim + gcn_hidden + embed_dim)
        else:
            # 消融: 不用 GCN，只用 VideoMAE 特征
            frame_input = torch.cat([
                global_feat_expanded, temporal_feats
            ], dim=2)  # (B, T, embed_dim + embed_dim)

        # ---- 3. 逐帧融合 + 预测 ----

        frame_fused = self.frame_fusion(frame_input)  # (B, T, fusion_dim)
        pred_logits = self.frame_predictor(frame_fused)  # (B, T, out_dim)

        # 窗口级预测: 根据输出维度自动选择处理方式
        out_dim = pred_logits.shape[-1]
        if out_dim == 1:
            # 旧版: 输出 (B, T, 1)，用 sigmoid 获取概率后取 max
            pred = pred_logits.squeeze(-1).max(dim=1)[0]  # (B,)
        else:
            # 新版: 输出 (B, T, 2)，取正类(class 1) logit 的最大值
            pred = pred_logits[:, :, 1].max(dim=1)[0]  # (B,)

        return pred, pred_logits
