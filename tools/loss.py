import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: class weights. When a list/tuple/ndarray is provided, values are
                  used as direct per-class multipliers without normalization.
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target, valid_mask=None):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        valid_weight = None
        if valid_mask is not None:
            valid_weight = valid_mask.float()
            if valid_weight.dim() == target.dim() - 1:
                valid_weight = valid_weight.unsqueeze(1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        if valid_weight is not None:
            valid_weight = torch.squeeze(valid_weight, 1)
            valid_weight = valid_weight.contiguous().view(-1)
        target = torch.squeeze(target, 1)
        target = target.contiguous().view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # print(one_hot_key.shape)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if valid_weight is not None:
            if valid_weight.device != loss.device:
                valid_weight = valid_weight.to(loss.device)
            valid_weight = valid_weight.clamp(0.0, 1.0)
            valid_count = valid_weight.sum()
            if valid_count.item() <= 0:
                return loss.sum() * 0.0
            loss = loss * valid_weight
            return loss.sum() / valid_count

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss


class TriRegionalPrototypeAlignmentLoss(nn.Module):
    """
    Region-balanced prototype alignment for normal/anomaly regions, with an
    optional background region when a third mask/prototype is available.

    The feature bundle is produced by get_anomaly_map(..., return_feature_bundle=True):
      patch_features: list of [B, P, D] patch tokens
      prompt_vectors: [B, C, D] normal/anomaly(/background) text prototypes
      token_shapes: list of (H, W) patch-grid shapes
    """

    def __init__(self, logit_scale=20.0, min_region_weight=1e-6):
        super().__init__()
        self.logit_scale = logit_scale
        self.min_region_weight = min_region_weight

    def forward(self, feature_bundle, tri_mask):
        if feature_bundle is None:
            raise ValueError("feature_bundle is required for TriRegionalPrototypeAlignmentLoss")
        if tri_mask.dim() != 4 or tri_mask.shape[1] < 2:
            raise ValueError("tri_mask must have shape [B, C, H, W] with C >= 2")

        prompt_vectors = feature_bundle.get("prompt_vectors")
        patch_features_list = feature_bundle.get("patch_features", [])
        token_shapes = feature_bundle.get("token_shapes", [])

        if prompt_vectors is None or prompt_vectors.shape[1] < 2:
            raise ValueError("feature_bundle['prompt_vectors'] must have shape [B, C, D] with C >= 2")
        if len(patch_features_list) != len(token_shapes):
            raise ValueError("feature_bundle patch_features and token_shapes must have the same length")

        num_regions = min(tri_mask.shape[1], prompt_vectors.shape[1])
        tri_mask = tri_mask[:, :num_regions].float().clamp(0.0, 1.0)
        prompt_vectors = F.normalize(prompt_vectors[:, :num_regions], dim=-1)

        total_loss = tri_mask.new_zeros(())
        valid_layers = 0

        for patch_features, token_shape in zip(patch_features_list, token_shapes):
            token_h, token_w = token_shape
            patch_features = F.normalize(patch_features, dim=-1)
            logits = self.logit_scale * torch.bmm(
                patch_features,
                prompt_vectors.transpose(1, 2),
            )

            B, P, C = logits.shape
            logits_flat = logits.reshape(B * P, C)
            layer_losses = []

            for class_idx in range(num_regions):
                region_weight = F.adaptive_avg_pool2d(
                    tri_mask[:, class_idx:class_idx + 1],
                    (token_h, token_w),
                ).flatten(1)
                valid_weight = region_weight.sum()
                if valid_weight.item() <= self.min_region_weight:
                    continue

                target = torch.full(
                    (B * P,),
                    class_idx,
                    dtype=torch.long,
                    device=logits.device,
                )
                ce = F.cross_entropy(logits_flat, target, reduction="none").view(B, P)
                class_loss = (ce * region_weight).sum() / valid_weight.clamp_min(self.min_region_weight)
                layer_losses.append(class_loss)

            if not layer_losses:
                continue

            total_loss = total_loss + torch.stack(layer_losses).mean()
            valid_layers += 1

        if valid_layers == 0:
            return tri_mask.new_zeros(())
        return total_loss / valid_layers
