import torch
import numpy as np
import random
from tqdm import tqdm
import os
from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted as model_adapter
import Datasets.visa as visa
import Datasets.mvtec as mvtec
import time
import argparse
from tools.loss import FocalLoss, BinaryDiceLoss, TriRegionalPrototypeAlignmentLoss
from tools.utils import build_ablation_tag, build_anomaly_dice_tag, build_focal_weight_tag, build_layers_tag, build_logit_scale_tag, build_prompt_mode_tag, build_tri_mask_calib_tag, get_anomaly_map, save_backbone_features_for_dataset
from torch.optim.lr_scheduler import LambdaLR
from Datasets import DATASET_REGISTRY, DATASET_CLASSES

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



visa_ALL = {
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
}

use_cuda = torch.cuda.is_available()
kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}


def prepare_data(dataset_name, category, args, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"❌ Unsupported dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]

    test_dataset = dataset_cls(
        source=root_path,
        split=split_cls.TEST,
        classname=category,
        resize=512,
        imagesize=512,
    )

    print(
        f"✅ Loaded [{dataset_name}] ({category}) train set, size: {len(test_dataset)}"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    return test_loader

def train_epoch(optimizer, loss_focal, loss_dice, loss_trpa, epoch, seg_loss_list, global_anomaly_loss_list, trpa_loss_list, loss_list, clip_model, start_time, train_data, selected_layers, precomputed_dir, dataset_name, use_background_mask, anomaly_dice_weight, tri_mask_calib_weight, cls_logit_scale, patch_logit_scale):
    loss_fn = torch.nn.CrossEntropyLoss()
    enable_trpa = tri_mask_calib_weight > 0
    for idx, image_info in enumerate(train_data):
        _, mask, anomaly_map_cross_modal, global_anomaly_score, feature_bundle = (
            get_anomaly_map(
                clip_model,
                image_info,
                device,
                model,
                Dino_model,
                layers=selected_layers,
                precomputed_dir=precomputed_dir,
                dataset_name=dataset_name,
                use_background_mask=use_background_mask,
                return_feature_bundle=enable_trpa,
                cls_logit_scale=cls_logit_scale,
                patch_logit_scale=patch_logit_scale,
            )
        )
        seg_loss_anomaly = loss_dice(anomaly_map_cross_modal[:, 1, :, :], mask[:, 1, :, :])
        if use_background_mask:
            # seg_loss_normal = loss_dice(anomaly_map_cross_modal[:, 0, :, :], mask[:, 0, :, :])
            seg_loss_avg = anomaly_dice_weight * seg_loss_anomaly
        else:
            seg_loss_avg = anomaly_dice_weight * seg_loss_anomaly
        
        mask_indices = torch.argmax(mask, dim=1).unsqueeze(1).float()
        
        focal_valid_mask = None
        if use_background_mask:
            focal_valid_mask = (mask[:, 0:1, :, :] + mask[:, 1:2, :, :]).clamp(0.0, 1.0)

        focal_loss_all = loss_focal(
            anomaly_map_cross_modal,
            mask_indices,
            valid_mask=focal_valid_mask,
        )
        
        trpa_loss = anomaly_map_cross_modal.new_zeros(())
        if enable_trpa:
            trpa_loss = loss_trpa(feature_bundle, mask)

        seg_loss = focal_loss_all + seg_loss_avg + tri_mask_calib_weight * trpa_loss
        
        labels = image_info["is_anomaly"].to(device).long().view(-1)  # (B)
        logits = global_anomaly_score  # (B, 2), cls global only uses normal/anomaly
        global_anomaly_loss = loss_fn(logits, labels)

        loss = 0.5 * seg_loss + 0.5 * global_anomaly_loss

        print(
            f"Epoch {epoch+1}/{10} | Batch {idx+1}/{len(train_data)} "
            f"| loss: {loss.item():.4f} | seg_loss: {seg_loss.item():.4f} | trpa_loss: {trpa_loss.item():.4f} | global_anomaly_loss: {global_anomaly_loss.item():.4f} | Time: {time.time()-start_time:.2f}s",
            end="\r",
            flush=True,
        )
        seg_loss_list.append(seg_loss.item())
        global_anomaly_loss_list.append(global_anomaly_loss.item())
        trpa_loss_list.append(trpa_loss.item())
        loss_list.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Release memory
        del mask, anomaly_map_cross_modal, global_anomaly_score
        del seg_loss, trpa_loss, global_anomaly_loss, loss, feature_bundle

    return

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path", type=str, default="./Result", help="path to result"
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--dataset", type=str, default="mvtec", help="dataset")
    parser.add_argument("--epoch", type=int, default=6, help="epoch")
    parser.add_argument("--dino_arch", type=str, default="dinov3_vitl16")
    # parser.add_argument("--dino_arch", type=str, default="dinov3_vits16")
    parser.add_argument(
        "--dino_weights",
        type=str,
        default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        # default="./dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
    parser.add_argument("--clip_name", type=str, default="ViT-L-14-336px")
    parser.add_argument("--selected_layers", type=str, default="14,17,20,23")
    parser.add_argument("--lr", type=float, default=0.00001, help="lr")
    parser.add_argument("--force_extraction", type=bool, default=False, help="Force extraction of features")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--branch", type=str, default="430", help="branch")
    parser.add_argument("--use_background_mask", action="store_true", default=True, help="Enable 3-class supervision with normal/anomaly/background masks. Enabled by default.")
    parser.add_argument("--anomaly_dice_weight", type=float, default=1.0, help="Anomaly-region dice loss weight.")
    parser.add_argument("--focal_normal_weight", type=float, default=1.0, help="Focal loss weight for the normal class.")
    parser.add_argument("--focal_anomaly_weight", type=float, default=1.0, help="Focal loss weight for the anomaly class.")
    parser.add_argument("--focal_background_weight", type=float, default=1.0, help="Compatibility weight for the background channel; background pixels are ignored by focal loss when --use_background_mask is enabled.")
    parser.add_argument("--cls_logit_scale", type=float, default=100.0, help="Temperature/logit scale for CLS image-level normal/anomaly similarity.")
    parser.add_argument("--patch_logit_scale", type=float, default=100.0, help="Temperature/logit scale for patch-level normal/anomaly/background similarity maps.")
    parser.add_argument("--tri_mask_calib_weight", type=float, default=0.2, help="Prototype alignment loss weight. Uses normal/anomaly regions without --use_background_mask and normal/anomaly/background regions with it.")
    args = parser.parse_args()
    setup_seed(args.seed)
    selected_layers = [int(x.strip()) for x in args.selected_layers.split(",") if x.strip()]
    if not selected_layers:
        raise ValueError("selected_layers cannot be empty")
    layers_tag = build_layers_tag(selected_layers)
    enable_vision_adapter = True
    enable_cross_attention = True
    enable_cross_attention_mlp = True
    branch=args.branch
    mode_tag = "bgmask" if args.use_background_mask else "nobgmask"
    feature_run_branch = f"{branch}_{mode_tag}_{layers_tag}"
    run_branch = f"{feature_run_branch}_{build_anomaly_dice_tag(args.anomaly_dice_weight)}"
    run_branch = f"{run_branch}_{build_prompt_mode_tag()}"
    run_branch = f"{run_branch}_{build_ablation_tag(enable_vision_adapter, enable_cross_attention, enable_cross_attention_mlp)}"
    logit_scale_tag = build_logit_scale_tag(args.cls_logit_scale, args.patch_logit_scale)
    if logit_scale_tag:
        run_branch = f"{run_branch}_{logit_scale_tag}"
    focal_weight_tag = build_focal_weight_tag(
        args.focal_normal_weight,
        args.focal_anomaly_weight,
        args.focal_background_weight if args.use_background_mask else None,
    )
    if focal_weight_tag:
        run_branch = f"{run_branch}_{focal_weight_tag}"
    if args.use_background_mask:
        run_branch = f"{run_branch}_fgfocal"
    tri_mask_calib_tag = build_tri_mask_calib_tag(args.tri_mask_calib_weight)
    if tri_mask_calib_tag:
        run_branch = f"{run_branch}_{tri_mask_calib_tag}"
    seed_prefix=f"SEED_{args.seed}"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.result_path}", exist_ok=True)

    focal_class_weights = [args.focal_normal_weight, args.focal_anomaly_weight]
    if args.use_background_mask:
        focal_class_weights.append(args.focal_background_weight)
    if any(weight <= 0 for weight in focal_class_weights):
        raise ValueError("Focal class weights must be positive.")
    if args.cls_logit_scale <= 0 or args.patch_logit_scale <= 0:
        raise ValueError("cls_logit_scale and patch_logit_scale must be positive.")
    if args.tri_mask_calib_weight < 0:
        raise ValueError("tri_mask_calib_weight must be non-negative.")

    repo_dir = "./dinov3"
    dino_arch = args.dino_arch
    Dinov3_model_path = args.dino_weights
    Dino_model = torch.hub.load(
        repo_dir, dino_arch, source="local", weights=Dinov3_model_path
    )
    Dino_model.to(device)
    Dino_model.eval()

    clip_name = args.clip_name
    clip_model = create_model(
        model_name=clip_name,
        img_size=512,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.to(device)
    clip_model.eval()

    # AD-DINOv3
    model = model_adapter(
        c_in=Dino_model.embed_dim,
        device=device,
        prompt_c_in=clip_model.transformer.width,
        num_layers=len(selected_layers),
        enable_vision_adapter=enable_vision_adapter,
        enable_cross_attention=enable_cross_attention,
        enable_cross_attention_mlp=enable_cross_attention_mlp,
    )
    model.to(device)
    model.train()

    train_data = prepare_data(args.dataset, 'ALL', args, **kwargs)
    precomputed_dir = os.path.join(args.result_path, 'features', args.dataset, dino_arch, clip_name, feature_run_branch)
    prepare_device = torch.device('cuda')
    save_backbone_features_for_dataset(train_data, device, Dino_model, clip_model, selected_layers, precomputed_dir, text_device=prepare_device, force_extraction=args.force_extraction)

    def should_update_cross_attention_param(name: str) -> bool:
        if "cross_attention_" not in name:
            return False
        if enable_cross_attention and (".attn." in name or ".norm1." in name):
            return True
        if enable_cross_attention_mlp and (".mlp." in name or ".norm2." in name):
            return True
        return False

    params_to_update = []
    for name, param in model.named_parameters():
        should_update = False
        if "prompt_adapter" in name:
            should_update = True
        elif enable_vision_adapter and "vision_adapter" in name:
            should_update = True
        elif should_update_cross_attention_param(name):
            should_update = True

        if should_update:
            print(f"Learnable parameter: {name}")
            params_to_update.append(param)

    # train_data already prepared above

    optimizer = torch.optim.AdamW(
        params_to_update, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2
    )

    total_steps = args.epoch * len(train_data)
    warmup_steps = int(0.03 * total_steps)

    scheduler = LambdaLR(optimizer, lr_lambda)

    # If you want to speed up the convergence, please use the following line of code.
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch/10 + 1))

    loss_focal = FocalLoss(alpha=focal_class_weights)
    loss_dice = BinaryDiceLoss()
    loss_trpa = TriRegionalPrototypeAlignmentLoss()
    
    
    for epoch in range(args.epoch):
        start_time = time.time()
        seg_loss_list, loss_list, global_anomaly_loss_list, trpa_loss_list = (
            [],
            [],
            [],
            [],
        )

        train_epoch(optimizer, loss_focal, loss_dice, loss_trpa, epoch, seg_loss_list, global_anomaly_loss_list, trpa_loss_list, loss_list, clip_model, start_time, train_data, selected_layers, precomputed_dir, args.dataset, args.use_background_mask, args.anomaly_dice_weight, args.tri_mask_calib_weight, args.cls_logit_scale, args.patch_logit_scale)
        print()
        # scheduler.step()

        os.makedirs(f"{args.result_path}/ckpt/{args.dataset}/{dino_arch}/{clip_name}/{run_branch}/{seed_prefix}", exist_ok=True)
        torch.save({
            'vision_adapter': model.vision_adapter.state_dict(), 
            'prompt_adapter_patch': model.prompt_adapter_patch.state_dict(),
            'prompt_adapter_cls': model.prompt_adapter_cls.state_dict(),
            'cross_attention_patch': model.cross_attention_patch.state_dict(),
            'cross_attention_cls': model.cross_attention_cls.state_dict()
        }, f"{args.result_path}/ckpt/{args.dataset}/{dino_arch}/{clip_name}/{run_branch}/{seed_prefix}/{epoch}.pth")
        
        with open(f"{args.result_path}/ckpt/{args.dataset}/{dino_arch}/{clip_name}/{run_branch}/{seed_prefix}/loss.txt", "a") as f:
            f.write(
                f"epoch_{epoch}: "
                f"seg_loss={np.mean(seg_loss_list):.6f}\t"
                f"trpa_loss={np.mean(trpa_loss_list):.6f}\t"
                f"global_anomaly_loss={np.mean(global_anomaly_loss_list):.6f}\t"
                f"total_loss={np.mean(loss_list):.6f}\n"
            )
        print(
            f"epoch_{epoch}: "
            f"seg_loss={np.mean(seg_loss_list):.6f}, "
            f"trpa_loss={np.mean(trpa_loss_list):.6f}, "
            f"global_anomaly_loss={np.mean(global_anomaly_loss_list):.6f}, "
            f"total_loss={np.mean(loss_list):.6f}"
        )
