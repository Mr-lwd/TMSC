import os
import math
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from utils import encode_text_with_prompt_ensemble
from Datasets import DATASET_CLASSES

_SIM_MATRIX_PRINTED = False


def normalize_layers(layers):
    return [int(layer) for layer in layers]


def build_layers_tag(layers):
    return "layers_" + "-".join(str(layer) for layer in normalize_layers(layers))


def build_anomaly_dice_tag(weight):
    return f"anomalydice{weight:.6g}"


def build_prompt_mode_tag():
    return "dualprompt"


def build_fusion_tag(enable_cross_attention, enable_cross_attention_mlp):
    if enable_cross_attention and enable_cross_attention_mlp:
        return "xattn"
    if enable_cross_attention:
        return "onlyattn"
    if enable_cross_attention_mlp:
        return "onlyMLP"
    return "noxattn"


def build_ablation_tag(enable_vision_adapter, enable_cross_attention, enable_cross_attention_mlp):
    vision_tag = "vadapt" if enable_vision_adapter else "novadapt"
    fusion_tag = build_fusion_tag(enable_cross_attention, enable_cross_attention_mlp)
    return f"{vision_tag}_{fusion_tag}"


def build_focal_weight_tag(normal_weight, anomaly_weight, background_weight=None):
    weights = [float(normal_weight), float(anomaly_weight)]
    if background_weight is not None:
        weights.append(float(background_weight))
    if all(abs(weight - 1.0) < 1e-8 for weight in weights):
        return ""
    return "focalw" + "-".join(f"{weight:.6g}" for weight in weights)


def build_tri_mask_calib_tag(weight):
    if weight <= 0:
        return ""
    return f"trpa{weight:.6g}"


def build_logit_scale_tag(cls_logit_scale, patch_logit_scale):
    cls_logit_scale = float(cls_logit_scale)
    patch_logit_scale = float(patch_logit_scale)
    if abs(cls_logit_scale - 100.0) < 1e-8 and abs(patch_logit_scale - 100.0) < 1e-8:
        return ""
    return f"cscale{cls_logit_scale:.6g}_pscale{patch_logit_scale:.6g}"


def build_fg_bg_consistency_tag(weight, warmup_epochs, blend_alpha, blur_kernel):
    if weight <= 0:
        return ""
    return (
        f"fbcons{weight:.6g}"
        f"_warm{int(warmup_epochs)}"
        f"_alpha{blend_alpha:.6g}"
        f"_blur{int(blur_kernel)}"
    )


def build_guided_visual_cache_tag(blend_alpha, blur_kernel):
    return f"guided_alpha{blend_alpha:.6g}_blur{int(blur_kernel)}"


def load_torch_file(file_path):
    try:
        return torch.load(file_path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(file_path, map_location='cpu')


def cached_layers_match(cached_data, layers):
    cached_layers = cached_data.get('layers')
    if cached_layers is None:
        return False
    return normalize_layers(cached_layers) == normalize_layers(layers)

def extract_category(path: str):
    p = path.replace("\\", "/")
    for cats in DATASET_CLASSES.values():
        for cat in cats:
            if f"/{cat}/" in p:
                return cat
    parts = p.split('/')
    return parts[-4] if len(parts) >= 4 else parts[-1]


def _resolve_visual_feature_path(cache_dir, visual_cache_subdir, image_path, forced_category=None):
    pth = image_path.replace('\\', '/')
    category = forced_category if forced_category is not None else extract_category(pth)
    subdir = pth.split('/')[-2]
    filename = os.path.basename(pth)
    out_dir = os.path.join(cache_dir, visual_cache_subdir, category, subdir)
    fp = os.path.join(out_dir, filename + '.pt')
    return fp, out_dir, category


def _slice_image_info(image_info, indices):
    subset = {}
    index_list = list(indices)
    for key, value in image_info.items():
        if torch.is_tensor(value):
            subset[key] = value[index_list]
        elif isinstance(value, (list, tuple)):
            subset[key] = [value[i] for i in index_list]
        else:
            subset[key] = value
    return subset


def prepare_multiclass_mask(image_info, device, forced_category=None, dataset_name=None, is_training=True):
    image_path = image_info["image_path"]
    mask = (image_info["mask"].to(device) > 0.5).float()

    B, _, H, W = mask.shape
    new_mask = torch.zeros((B, 3, H, W), device=device)

    for i in range(len(image_path)):
        current_anomaly_mask = mask[i, 0].float()
        new_mask[i, 1] = current_anomaly_mask

        if not is_training:
            continue

        pth = image_path[i].replace('\\', '/')
        category = forced_category if forced_category is not None else extract_category(pth)
        filename = os.path.basename(pth)

        try:
            if 'Data/Images/' in pth:
                rel_path = pth.split('Data/Images/')[-1]
                if rel_path.startswith('/'):
                    rel_path = rel_path[1:]
            else:
                parts = pth.split(f"/{category}/")
                if len(parts) > 1:
                    rel_path = parts[-1]
                else:
                    p_parts = pth.split('/')
                    rel_path = "/".join(p_parts[-3:])
        except Exception:
            rel_path = filename

        base_name = os.path.splitext(filename)[0]
        subdir = os.path.dirname(rel_path)
        fore_mask_name = f"{base_name}_normal_fore_mask.png"
        back_mask_name = f"{base_name}_normal_back_mask.png"
        dset = dataset_name

        fore_mask_path = os.path.join('/root/autodl-tmp/ADDINOv3_lwd/visualizations_normal_masks', dset, 'normal_fore_masks', category, subdir, fore_mask_name)
        back_mask_path = os.path.join('/root/autodl-tmp/ADDINOv3_lwd/visualizations_normal_masks', dset, 'normal_back_masks', category, subdir, back_mask_name)

        if os.path.exists(fore_mask_path):
            f_mask = cv2.imread(fore_mask_path, cv2.IMREAD_GRAYSCALE)
            f_mask = cv2.resize(f_mask, (W, H))
            f_mask = torch.from_numpy(f_mask).float().to(device) / 255.0
            f_mask = (f_mask > 0.5).float()
        else:
            f_mask = torch.zeros((H, W)).to(device)
            print(fore_mask_path)
            print(f"Foreground mask not found for {pth}. Using zeros.")
            exit(0)

        if os.path.exists(back_mask_path):
            b_mask = cv2.imread(back_mask_path, cv2.IMREAD_GRAYSCALE)
            b_mask = cv2.resize(b_mask, (W, H))
            b_mask = torch.from_numpy(b_mask).float().to(device) / 255.0
            b_mask = (b_mask > 0.5).float()
        else:
            b_mask = 1.0 - f_mask
            print(back_mask_path)
            print(f"Background mask not found for {pth}. Using 1 - foreground.")
            exit(0)

        new_mask[i, 2] = b_mask * (1 - current_anomaly_mask)
        new_mask[i, 0] = f_mask * (1 - current_anomaly_mask)

    return new_mask

def get_feature_dinov3(image_path, batch_img, device, Dino_model, layers=None):
    with torch.inference_mode():
        layers = [5, 11, 17, 23] if layers is None else layers
        patch_tokens_dict = {i: [] for i in layers}
        cls_tokens_dict = {i: [] for i in layers}

        for j in range(len(image_path)):
            patch_dict, tokens_dict, cls_dict = {}, {}, {}
            handles = []

            image = batch_img[j].unsqueeze(0).to(device)

            anchor = getattr(Dino_model, "norm", None) or getattr(Dino_model, "fc_norm", None)
            assert anchor is not None, "There is no norm/fc_norm module, please print(Dino_model) to confirm the name"

            for i in layers:
                def _mk_hook(idx):
                    def _hook(module, inp, out):
                        tokens_dict[idx] = anchor(out[0]).detach().cpu()
                    return _hook
                handles.append(Dino_model.blocks[i].register_forward_hook(_mk_hook(i)))

            with torch.inference_mode():
                _ = Dino_model(image)

            for h in handles:
                h.remove()

            for i, toks in tokens_dict.items():
                tokens = toks[:, 5:, :]
                # tokens = (tokens - tokens.mean(dim=1, keepdim=True)) / (
                #     tokens.std(dim=1, keepdim=True) + 1e-6
                # )
                patch_dict[i] = tokens
                cls_dict[i] = toks[:, 0, :].unsqueeze(1)

            for i in layers:
                patch_tokens_dict[i].append(patch_dict[i].to(device))
                cls_tokens_dict[i].append(cls_dict[i].to(device))

        patch_tokens = [torch.cat(patch_tokens_dict[i], dim=0) for i in layers]
        cls_token = [torch.cat(cls_tokens_dict[i], dim=0) for i in layers]

        return cls_token, patch_tokens


def build_guided_background_view(image, background_mask, blend_alpha=0.2, blur_kernel=15):
    if blur_kernel < 1:
        raise ValueError("blur_kernel must be >= 1")
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    if background_mask.dim() != 4 or background_mask.shape[1] != 1:
        raise ValueError("background_mask must have shape [B, 1, H, W]")

    background_mask = background_mask.float().clamp(0.0, 1.0)
    if blur_kernel == 1:
        blurred_image = image
    else:
        padding = blur_kernel // 2
        blurred_image = F.avg_pool2d(image, kernel_size=blur_kernel, stride=1, padding=padding)
    background_view = blend_alpha * image + (1.0 - blend_alpha) * blurred_image
    return image * (1.0 - background_mask) + background_view * background_mask


def compute_masked_patch_consistency_loss(source_feature_bundle, target_feature_bundle, anomaly_mask, detach_source=True):
    if source_feature_bundle is None or target_feature_bundle is None:
        raise ValueError("feature_bundle is required for consistency loss")
    if anomaly_mask.dim() != 4 or anomaly_mask.shape[1] != 1:
        raise ValueError("anomaly_mask must have shape [B, 1, H, W]")

    total_loss = anomaly_mask.new_zeros(())
    valid_layers = 0

    for src_patch, tgt_patch, token_shape in zip(
        source_feature_bundle["patch_features"],
        target_feature_bundle["patch_features"],
        source_feature_bundle["token_shapes"],
    ):
        token_h, token_w = token_shape
        pooled_mask = F.adaptive_max_pool2d(anomaly_mask.float(), (token_h, token_w))
        pooled_mask = (pooled_mask > 0).float().flatten(2).squeeze(1)
        valid_mask = pooled_mask.sum()
        if valid_mask.item() <= 0:
            continue

        if detach_source:
            src_patch = src_patch.detach()
        src_patch = F.normalize(src_patch, dim=-1)
        tgt_patch = F.normalize(tgt_patch, dim=-1)

        cosine_similarity = (src_patch * tgt_patch).sum(dim=-1).clamp(-1.0, 1.0)
        layer_loss = ((1.0 - cosine_similarity) * pooled_mask).sum() / valid_mask.clamp_min(1.0)
        total_loss = total_loss + layer_loss
        valid_layers += 1

    if valid_layers == 0:
        return anomaly_mask.new_zeros(())
    return total_loss / valid_layers

def get_anomaly_map(clip_model, image_info, device, model, Dino_model, layers=None, precomputed_dir=None, forced_category=None, dataset_name=None, is_training=True, return_feature_bundle=False, image_override=None, prepared_mask=None, visual_cache_subdir='train', cls_logit_scale=100.0, patch_logit_scale=100.0):
    image = image_override.to(device) if image_override is not None else image_info["image"].to(device)
    image_path = image_info["image_path"]
    if prepared_mask is None:
        mask = prepare_multiclass_mask(
            image_info,
            device,
            forced_category=forced_category,
            dataset_name=dataset_name,
            is_training=is_training,
        )
    else:
        mask = prepared_mask.to(device)
    
    y = image_info["is_anomaly"]

    # textual branch
    if precomputed_dir is None:
        with torch.no_grad():
            text_feats_list = [
                encode_text_with_prompt_ensemble(
                    clip_model,
                    (forced_category if forced_category is not None else extract_category(image_path[i])),
                    device,
                    '',
                    y,
                )
                for i in range(len(image_path))
            ]
        text_feature = torch.stack(text_feats_list, dim=0).to(device)
    else:
        feats_list = []
        for i in range(len(image_path)):
            pth = image_path[i].replace('\\', '/')
            category = forced_category if forced_category is not None else extract_category(pth)
            p = os.path.join(precomputed_dir, 'text', category + '.pt')
            try:
                feats = torch.load(p, map_location='cpu', weights_only=True)
            except TypeError:
                feats = torch.load(p, map_location='cpu')
            feats_list.append(feats)
        text_feature = torch.stack(feats_list, dim=0).to(device)
    adjusted_feats_map = []
    adjusted_feats_cls = []
    for i in range(len(image_path)):
        patch_prompt_features = [
            model.adapt_text_patch(text_feature[i, :, 0]),
            model.adapt_text_patch(text_feature[i, :, 1]),
            model.adapt_text_patch(text_feature[i, :, 2]),
        ]
        cls_prompt_features = [
            model.adapt_text_cls(text_feature[i, :, 0]),
            model.adapt_text_cls(text_feature[i, :, 1]),
            model.adapt_text_cls(text_feature[i, :, 2]),
        ]
        adjusted_feats_map.append(torch.stack(patch_prompt_features, dim=1))
        adjusted_feats_cls.append(torch.stack(cls_prompt_features, dim=1))
    
    adjusted_text_feature_map = torch.stack(adjusted_feats_map, dim=0)
    adjusted_text_feature_cls = torch.stack(adjusted_feats_cls, dim=0)

    global _SIM_MATRIX_PRINTED
    if (not is_training) and (not _SIM_MATRIX_PRINTED):
        with torch.no_grad():
            tf_sample = adjusted_text_feature_map[0].clone().detach()
            tf_sample = tf_sample / tf_sample.norm(dim=0, keepdim=True)
            sim_matrix = tf_sample.T @ tf_sample
            channel_names = ["Normal", "Anomaly", "Background"]
            print(f"\nText Feature Similarity Matrix (Map Adapter) ({', '.join(channel_names)}):")
            print(sim_matrix.cpu().numpy())
        _SIM_MATRIX_PRINTED = True


    # visual branch
    use_cached_visual = precomputed_dir is not None and visual_cache_subdir is not None
    if not use_cached_visual:
        cls_token, patch_tokens = get_feature_dinov3(image_path, image, device, Dino_model, layers=layers)
    else:
        patch_tokens_dict = {i: [] for i in layers}
        cls_tokens_dict = {i: [] for i in layers}
        
        # First try to load all precomputed features to avoid partial loading
        try:
            all_loaded = True
            loaded_data = []
            for j in range(len(image_path)):
                fp, _, _ = _resolve_visual_feature_path(precomputed_dir, visual_cache_subdir, image_path[j], forced_category=forced_category)
                if not os.path.exists(fp):
                    all_loaded = False
                    break
                data = load_torch_file(fp)
                if not cached_layers_match(data, layers):
                    all_loaded = False
                    break
                loaded_data.append(data)
            
            if not all_loaded:
                print(f"Warning: Some precomputed features missing or layer metadata mismatched, falling back to online extraction for batch.")
                cls_token, patch_tokens = get_feature_dinov3(image_path, image, device, Dino_model, layers=layers)
            else:
                for data in loaded_data:
                    for i in layers:
                        pt = data['patch'][i]
                        ct = data['cls'][i]
                        if pt.dim() == 2:
                            pt = pt.unsqueeze(0)
                        if ct.dim() == 2:
                            ct = ct.unsqueeze(0)
                        patch_tokens_dict[i].append(pt.to(device))
                        cls_tokens_dict[i].append(ct.to(device))
                patch_tokens = [torch.cat(patch_tokens_dict[i], dim=0) for i in layers]
                cls_token = [torch.cat(cls_tokens_dict[i], dim=0) for i in layers]
        except Exception as e:
            print(f"Error loading precomputed features: {e}, falling back to online extraction.")
            cls_token, patch_tokens = get_feature_dinov3(image_path, image, device, Dino_model, layers=layers)
    L = len(cls_token)
    # assert len(model.cls_token_adapter) == L and len(model.patch_token_adapter) == L
    anomaly_maps_cross_modal = []
    global_anomaly_scores = []
    feature_bundle = None
    if return_feature_bundle:
        feature_bundle = {
            'patch_features': [],
            'cls_features': [],
            'token_shapes': [],
            'prompt_vectors': None,
        }
    for i in range(L):
        # Concatenate CLS and Patch tokens
        cls_t = cls_token[i].to(device) # [B, 1, D]
        patch_t = patch_tokens[i].to(device) # [B, P, D]
        full_seq = torch.cat([cls_t, patch_t], dim=1) # [B, 1+P, D]
        
        # Shared vision adapter with layer-specific conditioning.
        full_out = model.encode_visual(full_seq, i) # [B, 1+P, D]
        
        # Split
        cls_features = full_out[:, 0:1, :] # [B, 1, D]
        patch_features = full_out[:, 1:, :] # [B, P, D]
        
        # cls_features = model.vision_adapter[i](cls_t)
        # patch_features = model.vision_adapter[i](patch_t)
        
        cls_features = cls_features / cls_features.norm(dim=-1, keepdim=True)
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        
        # User request:
        # Visual feature: Concatenate CLS + Patch feature -> 2D dimension
        # CLS: [B, 1, D], Patch: [B, P, D]
        # Expand CLS to [B, P, D] and concat with Patch -> [B, P, 2D]
        # B_dim, P_dim, D_dim = patch_features.shape
        # cls_expanded = cls_features.expand(-1, P_dim, -1) # [B, P, D]
        # visual_features_map = torch.cat([cls_expanded, patch_features], dim=-1) # [B, P, 2D]
        # visual_features_map = visual_features_map / visual_features_map.norm(dim=-1, keepdim=True)
        
        # Global Visual Feature: CLS + Average(Patch) -> [B, 2D]
        # CLS: [B, 1, D], Mean(Patch): [B, 1, D]
        # patch_mean = patch_features.mean(dim=1, keepdim=True) # [B, 1, D]
        # visual_features_global = torch.cat([cls_features, patch_mean], dim=-1) # [B, 1, 2D]
        # visual_features_global = visual_features_global / visual_features_global.norm(dim=-1, keepdim=True)

        # patch uses 2-way or 3-way prompts; cls can also use the background prompt as
        # a context sink when background masks are enabled, but image-level scoring keeps
        # only the normal/anomaly pair.
        map_text_features = adjusted_text_feature_map.permute(0, 2, 1)
        map_text_features = map_text_features / map_text_features.norm(dim=-1, keepdim=True)
        cls_text_features = adjusted_text_feature_cls.permute(0, 2, 1)
        cls_text_features = cls_text_features / cls_text_features.norm(dim=-1, keepdim=True)
        if feature_bundle is not None and feature_bundle['prompt_vectors'] is None:
            feature_bundle['prompt_vectors'] = map_text_features

        # Calculate Global Anomaly Score (After Cross Attention for CLS)
        # CLS: [B, 1, D]
        # Text: [B, 2, D] or [B, 3, D]
        # Score: [B, 1, 2] after keeping the normal/anomaly pair for image-level output.
        cls_attn_out = model.fuse_cls(cls_features, cls_text_features, i)
        cls_attn_out = cls_attn_out / cls_attn_out.norm(dim=-1, keepdim=True)
        cls_similarity = cls_logit_scale * (cls_attn_out @ cls_text_features.permute(0, 2, 1))
        anomaly_score = cls_similarity[:, :, :2]
        global_anomaly_scores.append(anomaly_score)

        # Cross Attention
        # We use Patch as Query to obtain spatial maps.
        # Query: Patch [B, P, D]
        # Key/Value: Text [B, C, D]
        patch_attn_out = model.fuse_patch(patch_features, map_text_features, i)
        # patch_attn_out: [B, P, D] (Text-enhanced Patch Features)
        patch_attn_out = patch_attn_out / patch_attn_out.norm(dim=-1, keepdim=True)

        anomaly_map_cross_modal = patch_logit_scale * (patch_attn_out @ map_text_features.permute(0, 2, 1))

        # Processing for anomaly_map_cross_modal
        B, P, num_text_channels = anomaly_map_cross_modal.shape
        S1 = int(math.sqrt(P))
        while S1 > 1 and (P % S1) != 0:
            S1 -= 1
        S2 = P // S1
        if feature_bundle is not None:
            feature_bundle['patch_features'].append(patch_attn_out)
            feature_bundle['cls_features'].append(cls_attn_out)
            feature_bundle['token_shapes'].append((S1, S2))
        am = anomaly_map_cross_modal.permute(0, 2, 1).reshape(B, num_text_channels, S1, S2)
        anomaly_map_cross_modal = F.interpolate(am, size=(512, 512), mode='bilinear', align_corners=False)
        # 对每个样本的每个通道分别归一化
        anomaly_map_cross_modal = torch.softmax(anomaly_map_cross_modal, dim=1)
        # (L,)
        anomaly_maps_cross_modal.append(anomaly_map_cross_modal)
    
    # (L, B, 3, 512, 512) -> (B, 3, 512, 512)
    # User Request: Take only the last layer for anomaly_map_cross_modal
    # if is_training:
    anomaly_map_cross_modal = torch.mean(torch.stack(anomaly_maps_cross_modal, dim=0), dim=0)
    # (L, B, 1, 2) -> (B, 2)
    global_anomaly_score = torch.mean(torch.stack(global_anomaly_scores, dim=0), dim=0).squeeze(1)

    return None, mask, anomaly_map_cross_modal, global_anomaly_score, feature_bundle

def save_backbone_features_for_dataset(data_loader, device, Dino_model, clip_model, layers, cache_dir, text_device=None, forced_category=None, force_extraction=False):
    return save_backbone_features_for_dataset_variant(
        data_loader,
        device,
        Dino_model,
        clip_model,
        layers,
        cache_dir,
        text_device=text_device,
        forced_category=forced_category,
        force_extraction=force_extraction,
        visual_cache_subdir='train',
    )


def save_backbone_features_for_dataset_variant(data_loader, device, Dino_model, clip_model, layers, cache_dir, text_device=None, forced_category=None, force_extraction=False, visual_cache_subdir='train', guided_blend_alpha=0.2, guided_blur_kernel=15, use_guided_background_view=False, dataset_name=None):
    text_dir = os.path.join(cache_dir, 'text')
    visual_dir = os.path.join(cache_dir, visual_cache_subdir)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)
    saved_cats = set()
    if text_device is None:
        text_device = device
    if use_guided_background_view and dataset_name is None:
        raise ValueError("guided background pre-extraction requires dataset_name")
    with torch.inference_mode():
        for image_info in data_loader:
            image_path = image_info['image_path']
            missing_idx = []
            out_paths = []
            for j in range(len(image_path)):
                fp, out_dir, category = _resolve_visual_feature_path(cache_dir, visual_cache_subdir, image_path[j], forced_category=forced_category)
                os.makedirs(out_dir, exist_ok=True)
                out_paths.append((fp, category))
                needs_extraction = force_extraction or not os.path.isfile(fp)
                if not needs_extraction:
                    try:
                        cached_data = load_torch_file(fp)
                        needs_extraction = not cached_layers_match(cached_data, layers)
                    except Exception:
                        needs_extraction = True
                if needs_extraction:
                    missing_idx.append(j)
            if len(missing_idx) > 0:
                subset_info = _slice_image_info(image_info, missing_idx)
                image = subset_info['image'].to(device)
                path_subset = [image_path[k] for k in missing_idx]
                if use_guided_background_view:
                    subset_mask = prepare_multiclass_mask(
                        subset_info,
                        device,
                        forced_category=forced_category,
                        dataset_name=dataset_name,
                        is_training=True,
                    )
                    image = build_guided_background_view(
                        image,
                        subset_mask[:, 2:3],
                        blend_alpha=guided_blend_alpha,
                        blur_kernel=guided_blur_kernel,
                    )
                cls_token, patch_tokens = get_feature_dinov3(path_subset, image, device, Dino_model, layers=layers)
                for jj, k in enumerate(missing_idx):
                    fp, category = out_paths[k]
                    data = {'layers': layers, 'cls': {}, 'patch': {}}
                    for li, l in enumerate(layers):
                        data['cls'][l] = cls_token[li][jj].cpu().unsqueeze(0)
                        data['patch'][l] = patch_tokens[li][jj].cpu().unsqueeze(0)
                    torch.save(data, fp)
                    
                    # Release memory
                    del data
            
            # Release memory
            if len(missing_idx) > 0:
                del image, cls_token, patch_tokens
                if use_guided_background_view:
                    del subset_mask

            for _, category in out_paths:
                cat_fp = os.path.join(text_dir, category + '.pt')
                if force_extraction or (not os.path.isfile(cat_fp) and category not in saved_cats):
                    feats = encode_text_with_prompt_ensemble(clip_model, category, text_device, '', '')
                    torch.save(feats.detach().cpu(), cat_fp)
                    saved_cats.add(category)
                    
                    # Release memory
                    del feats
