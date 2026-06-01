import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torchvision.models as models

class ClipAdapter(nn.Module):
    def __init__(self, c_in, c_out=None, bottleneck=None):
        super(ClipAdapter, self).__init__()
        self.c_out = c_out if c_out is not None else c_in

        self.net = nn.Sequential(
            nn.Linear(c_in, self.c_out),
        )

    def forward(self, x):
        return self.net(x)

class TransformerAdapter(nn.Module):
    def __init__(self, c_in, nhead=4, dropout=0.1, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=c_in, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return x + self.transformer(x)

class CrossModalFusionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1, enable_attention=True, enable_mlp=True):
        super().__init__()
        self.enable_attention = enable_attention
        self.enable_mlp = enable_mlp
        # 1. 交叉注意力层
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        
        # 2. 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # 3. 前馈网络 (FFN) - 提供非线性变换能力
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        """
        x: 视觉特征 (Patch tokens) -> 作为 Query
        context: 文本特征 (Text tokens) -> 作为 Key 和 Value
        """
        # 第一阶段：注意力 + 残差
        # x 为 Query, context 为 Key 和 Value
        if self.enable_attention:
            attn_out, _ = self.attn(self.norm1(x), context, context)
            x = x + self.dropout(attn_out)
        
        # 第二阶段：FFN + 残差
        if self.enable_mlp:
            x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x

class CLIP_Inplanted(nn.Module):
    def __init__(self, c_in, device, prompt_c_in=768, num_layers=1, enable_vision_adapter=True, enable_cross_attention=True, enable_cross_attention_mlp=True):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.device = device
        self.feature_dim = c_in
        self.prompt_dim = prompt_c_in
        self.num_layers = num_layers
        self.enable_vision_adapter = enable_vision_adapter
        self.enable_cross_attention = enable_cross_attention
        self.enable_cross_attention_mlp = enable_cross_attention_mlp
        self.vision_adapter = TransformerAdapter(c_in=self.feature_dim)
        self.prompt_adapter_patch = ClipAdapter(c_in=self.prompt_dim, c_out=self.feature_dim)
        self.prompt_adapter_cls = ClipAdapter(c_in=self.prompt_dim, c_out=self.feature_dim)
        self.prompt_adapter_cls.load_state_dict(self.prompt_adapter_patch.state_dict())
        self.cross_attention_patch = CrossModalFusionBlock(
            feature_dim=self.feature_dim,
            enable_attention=self.enable_cross_attention,
            enable_mlp=self.enable_cross_attention_mlp,
        )
        self.cross_attention_cls = CrossModalFusionBlock(
            feature_dim=self.feature_dim,
            enable_attention=self.enable_cross_attention,
            enable_mlp=self.enable_cross_attention_mlp,
        )

    def encode_visual(self, full_seq, layer_idx: int):
        if not self.enable_vision_adapter:
            return full_seq
        return self.vision_adapter(full_seq)

    def adapt_text_patch(self, text_features):
        return self.prompt_adapter_patch(text_features)

    def adapt_text_cls(self, text_features):
        return self.prompt_adapter_cls(text_features)

    def fuse_cls(self, cls_features, text_features, layer_idx: int):
        if not (self.enable_cross_attention or self.enable_cross_attention_mlp):
            return cls_features
        return self.cross_attention_cls(cls_features, text_features)

    def fuse_patch(self, patch_features, text_features, layer_idx: int):
        if not (self.enable_cross_attention or self.enable_cross_attention_mlp):
            return patch_features
        return self.cross_attention_patch(patch_features, text_features)
        
    def resize_vit_adapters(self, num_layers: int):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_layers != self.num_layers:
            self.num_layers = num_layers
        self.vision_adapter.to(self.device)
        self.prompt_adapter_patch.to(self.device)
        self.prompt_adapter_cls.to(self.device)
        self.cross_attention_patch.to(self.device)
        self.cross_attention_cls.to(self.device)

    def forward(self,):
        return