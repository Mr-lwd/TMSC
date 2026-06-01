import torch
import numpy as np
import random
from tqdm import tqdm
import os
from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted as model_adapter
import argparse
from tools.metrics import compute_binary_metrics, format_metric, min_max_normalize
from tools.utils import build_ablation_tag, build_anomaly_dice_tag, build_focal_weight_tag, build_layers_tag, build_logit_scale_tag, build_prompt_mode_tag, build_tri_mask_calib_tag, get_anomaly_map, save_backbone_features_for_dataset
from tools.visualization import visualization
from Datasets import DATASET_REGISTRY, DATASET_CLASSES

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
        f"✅ Loaded [{dataset_name}] ({category}) test set, size: {len(test_dataset)}"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    return test_loader

def test(clip_model, result_path, epoch, selected_layers, precomputed_dir, dataset_name, use_background_mask, cls_logit_scale, patch_logit_scale):
    category_metrics = []
    print(
        f"--------------------------------------Testing epoch {epoch}--------------------------------------"
    )
    for category in sorted(DATASET_CLASSES[args.dataset]):
        os.makedirs(result_path + f"/{category}", exist_ok=True)
        pixel_pred = []
        pixel_gt = []
        image_pred = []
        image_gt = []
        img_list = []
        test_data = prepare_data(args.dataset, category, args, **kwargs)
        save_backbone_features_for_dataset(
            test_data,
            device,
            Dino_model,
            clip_model,
            selected_layers,
            precomputed_dir,
            text_device=device,
            forced_category=category,
            force_extraction=args.force_extraction,
        )

        for image_info in tqdm(test_data):
            with torch.no_grad():
                _, mask, anomaly_map_cross_modal, global_anomaly_score, _ = get_anomaly_map(
                    clip_model,
                    image_info,
                    device,
                    model,
                    Dino_model,
                    layers=selected_layers,
                    precomputed_dir=precomputed_dir,
                    forced_category=category,
                    dataset_name=dataset_name,
                    is_training=False,
                    use_background_mask=use_background_mask,
                    cls_logit_scale=cls_logit_scale,
                    patch_logit_scale=patch_logit_scale,
                )

                # We need ground truth mask for anomaly evaluation.
                # mask in get_anomaly_map is now 3 channels: [B, 3, H, W]
                # Channel 1 is anomaly mask.
                # For evaluation, we compare predicted anomaly map (channel 1) with GT anomaly mask (channel 1).
                
                # pixel_gt.extend(mask.squeeze(1).cpu().detach().numpy()) 
                # Original mask was [B, 1, H, W], squeeze(1) -> [B, H, W]
                # New mask is [B, 3, H, W]. We want anomaly channel.
                pixel_gt.extend(mask[:, 1, :, :].cpu().detach().numpy())
                
                img_list.extend(image_info["image_path"])
                
                # pixel_pred.extend(
                #     anomaly_map_cross_modal[:, 1, :, :].cpu().detach().numpy()
                # )
                # We use the anomaly channel (index 1) for anomaly detection evaluation.
                pixel_pred.extend(
                    anomaly_map_cross_modal[:, 1, :, :].cpu().detach().numpy()
                )
                
                # Image-level evaluation
                image_gt.extend(image_info["is_anomaly"].numpy())
                
                # Use global_anomaly_score (CLS token similarity) for image-level detection.
                # cls-global only uses normal/anomaly logits.
                logits = global_anomaly_score  # (B, 2)
                probs = torch.softmax(logits, dim=1)
                # Channel 1 is Anomaly probability
                image_probs = probs[:, 1]
                
                # Optional: fuse with Pixel Max?
                # For now, let's use CLS score as requested.
                image_pred.extend(image_probs.cpu().detach().numpy())
            
            # Release memory
            # del mask, anomaly_map_cross_modal, batch_sum, global_anomaly_score
            # torch.cuda.empty_cache()

        gt_mask_list = np.array(pixel_gt)
        pred_mask_list = min_max_normalize(np.array(pixel_pred))
        save_flag = os.environ.get("SAVE_TRIPTYCH", "1")
        # print(f"save_flag: {save_flag}")
        if save_flag == "1" or save_flag.lower() == "true":
            visualization(img_list, pred_mask_list, gt_mask_list, category, result_path)

        image_gt_list = np.array(image_gt)
        image_pred_list = np.array(image_pred)

        pixel_metrics = compute_binary_metrics(gt_mask_list, pred_mask_list)
        image_metrics = compute_binary_metrics(image_gt_list, image_pred_list)

        metrics = {
            "category": category,
            "i_auroc": image_metrics["auroc"],
            "i_aupr": image_metrics["aupr"],
            "i_f1max": image_metrics["f1max"],
            "p_auroc": pixel_metrics["auroc"],
            "p_aupr": pixel_metrics["aupr"],
            "p_f1max": pixel_metrics["f1max"],
        }
        category_metrics.append(metrics)

        print(
            f"{category}: "
            f"I-AUROC={format_metric(metrics['i_auroc'])}\t"
            f"I-AUPR={format_metric(metrics['i_aupr'])}\t"
            f"I-F1max={format_metric(metrics['i_f1max'])}\t"
            f"P-AUROC={format_metric(metrics['p_auroc'])}\t"
            f"P-AUPR={format_metric(metrics['p_aupr'])}\t"
            f"P-F1max={format_metric(metrics['p_f1max'])}"
        )

    summary = {
        "i_auroc": float(np.nanmean([m["i_auroc"] for m in category_metrics])),
        "i_aupr": float(np.nanmean([m["i_aupr"] for m in category_metrics])),
        "i_f1max": float(np.nanmean([m["i_f1max"] for m in category_metrics])),
        "p_auroc": float(np.nanmean([m["p_auroc"] for m in category_metrics])),
        "p_aupr": float(np.nanmean([m["p_aupr"] for m in category_metrics])),
        "p_f1max": float(np.nanmean([m["p_f1max"] for m in category_metrics])),
    }

    print(f"mean_I-AUROC: {format_metric(summary['i_auroc'])}")
    print(f"mean_I-AUPR: {format_metric(summary['i_aupr'])}")
    print(f"mean_I-F1max: {format_metric(summary['i_f1max'])}")
    print(f"mean_P-AUROC: {format_metric(summary['p_auroc'])}")
    print(f"mean_P-AUPR: {format_metric(summary['p_aupr'])}")
    print(f"mean_P-F1max: {format_metric(summary['p_f1max'])}")

    metric_file = f"{result_path}/metric.txt"

    with open(metric_file, "a") as f:
        f.write(f"----------Dataset: {args.dataset}----------\n")
        f.write(
            "Classname      I-AUROC  I-AUPR   I-F1max  P-AUROC  P-AUPR   P-F1max"
            f"  epoch_{epoch}\n"
        )
        for metrics in category_metrics:
            f.write(
                f"{metrics['category']:<14s}"
                f"{format_metric(metrics['i_auroc']):>8s}  "
                f"{format_metric(metrics['i_aupr']):>8s}  "
                f"{format_metric(metrics['i_f1max']):>8s}  "
                f"{format_metric(metrics['p_auroc']):>8s}  "
                f"{format_metric(metrics['p_aupr']):>8s}  "
                f"{format_metric(metrics['p_f1max']):>8s}\n"
            )
        f.write(
            f"{'mean':<14s}"
            f"{format_metric(summary['i_auroc']):>8s}  "
            f"{format_metric(summary['i_aupr']):>8s}  "
            f"{format_metric(summary['i_f1max']):>8s}  "
            f"{format_metric(summary['p_auroc']):>8s}  "
            f"{format_metric(summary['p_aupr']):>8s}  "
            f"{format_metric(summary['p_f1max']):>8s}\n\n"
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path", type=str, default="./TESTING_ALL", help="path to result"
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--dataset", type=str, default="visa", help="dataset")
    parser.add_argument("--dino_arch", type=str, default="dinov3_vitl16")
    parser.add_argument(
        "--dino_weights",
        type=str,
        default="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    parser.add_argument("--clip_name", type=str, default="ViT-L-14-336px")
    parser.add_argument("--selected_layers", type=str, default="14,17,20,23")
    parser.add_argument("--force_extraction", type=bool, default=False, help="Force extraction of features")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--epoch", type=int, default=5, help="epoch to test")
    parser.add_argument("--branch", type=str, default="430", help="branch")
    parser.add_argument("--train_dataset", type=str, default="mvtec", help="train dataset")
    parser.add_argument("--use_background_mask", action="store_true", default=True, help="Enable 3-class supervision with normal/anomaly/background masks. Enabled by default.")
    parser.add_argument("--anomaly_dice_weight", type=float, default=1.0, help="Anomaly-region dice loss weight.")
    parser.add_argument("--focal_normal_weight", type=float, default=1.0, help="Focal loss weight for the normal class.")
    parser.add_argument("--focal_anomaly_weight", type=float, default=1.0, help="Focal loss weight for the anomaly class.")
    parser.add_argument("--focal_background_weight", type=float, default=1.0, help="Compatibility weight for the background channel; background pixels are ignored by focal loss when --use_background_mask is enabled.")
    parser.add_argument("--cls_logit_scale", type=float, default=100.0, help="Temperature/logit scale for CLS image-level normal/anomaly similarity.")
    parser.add_argument("--patch_logit_scale", type=float, default=100.0, help="Temperature/logit scale for patch-level normal/anomaly/background similarity maps.")
    parser.add_argument("--tri_mask_calib_weight", type=float, default=0.2, help="Prototype alignment loss weight used when constructing checkpoint paths.")
    args = parser.parse_args()
    selected_layers = [int(x.strip()) for x in args.selected_layers.split(",") if x.strip()]
    if not selected_layers:
        raise ValueError("selected_layers cannot be empty")
    if args.cls_logit_scale <= 0 or args.patch_logit_scale <= 0:
        raise ValueError("cls_logit_scale and patch_logit_scale must be positive.")
    layers_tag = build_layers_tag(selected_layers)
    enable_vision_adapter = True
    enable_cross_attention = True
    enable_cross_attention_mlp = True

    def parse_device(dev_str: str) -> torch.device:
        if dev_str in ("auto", "cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        if dev_str.startswith("cuda:"):
            try:
                idx = int(dev_str.split(":", 1)[1])
            except Exception:
                idx = 0
            count = torch.cuda.device_count()
            if torch.cuda.is_available() and 0 <= idx < count:
                return torch.device(f"cuda:{idx}")
            return (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        return torch.device(dev_str)

    device = parse_device(args.device)
    if device.type == "cuda":
        print(
            f"Selected device: {device}, name: {torch.cuda.get_device_name(device.index)}"
        )
    else:
        print(f"Selected device: {device}")
    os.makedirs(f"{args.result_path}/{args.dataset}", exist_ok=True)

    repo_dir = "./dinov3"
    dino_arch = args.dino_arch
    try:
        Dinov3_model_path = args.dino_weights
        Dino_model = torch.hub.load(
            repo_dir, dino_arch, source="local", weights=Dinov3_model_path
        )
    except Exception:
        Dino_model = torch.hub.load(
            repo_dir, dino_arch, source="local", pretrained=True
        )
    Dino_model.to(device)
    Dino_model.eval()

    # loading clip
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

    # loading AD-DINOv3
    model = model_adapter(
        c_in=Dino_model.embed_dim,
        device=device,
        prompt_c_in=clip_model.transformer.width,
        num_layers=len(selected_layers),
        enable_vision_adapter=enable_vision_adapter,
        enable_cross_attention=enable_cross_attention,
        enable_cross_attention_mlp=enable_cross_attention_mlp,
    )
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
    train_dataset=args.train_dataset
    # train_dataset="visa"
    seed_prefix=f"SEED_{args.seed}"
    # for i in range(40, 60, 2):
    i=args.epoch
    ckpt = f"./Result/ckpt/{train_dataset}/{dino_arch}/{clip_name}/{run_branch}/{seed_prefix}/{i}.pth"  
    state = torch.load(ckpt, map_location=device)
    required_keys = [
        "vision_adapter",
        "cross_attention_patch",
        "cross_attention_cls",
    ]
    missing_keys = [key for key in required_keys if key not in state]
    has_dual_prompt_adapters = "prompt_adapter_patch" in state and "prompt_adapter_cls" in state
    has_shared_prompt_adapter = "prompt_adapter" in state
    if not has_dual_prompt_adapters and not has_shared_prompt_adapter:
        missing_keys.append("prompt_adapter_patch/prompt_adapter_cls")
    if missing_keys:
        raise KeyError(
            f"Checkpoint {ckpt} is missing keys required by the current architecture: {missing_keys}"
        )

    model.vision_adapter.load_state_dict(state["vision_adapter"])
    if has_dual_prompt_adapters:
        model.prompt_adapter_patch.load_state_dict(state["prompt_adapter_patch"])
        model.prompt_adapter_cls.load_state_dict(state["prompt_adapter_cls"])
    else:
        model.prompt_adapter_patch.load_state_dict(state["prompt_adapter"])
        model.prompt_adapter_cls.load_state_dict(state["prompt_adapter"])
    model.cross_attention_patch.load_state_dict(state["cross_attention_patch"])
    model.cross_attention_cls.load_state_dict(state["cross_attention_cls"])

    model.to(device)
    model.eval()
    out_dir = f"{args.result_path}/{args.dataset}/{dino_arch}/{clip_name}/{run_branch}/{seed_prefix}/{i}"
    os.makedirs(out_dir, exist_ok=True)

    precomputed_dir = os.path.join(
        args.result_path, "features", args.dataset, dino_arch, clip_name, feature_run_branch
    )
    test(clip_model, out_dir, i, selected_layers, precomputed_dir, args.dataset, args.use_background_mask, args.cls_logit_scale, args.patch_logit_scale)
