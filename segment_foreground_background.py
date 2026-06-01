import argparse
import importlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MVTec_TEXTURE_CLASSES = {"carpet", "leather", "tile", "wood", "grid", "mesh"}
DEFAULT_HF_MODELS = {
    "rmbg": "briaai/RMBG-2.0",
    "birefnet": "ZhengPeng7/BiRefNet",
}
DEFAULT_INPUT_CANDIDATES = [
    Path("/root/autodl-tmp/Data/Industrial_Dataset/VisA_20220922"),
    Path("/root/autodl-tmp/Data/Industrial_Dataset/MVTecAD"),
    # Path("/root/autodl-tmp/Data/Industrial_Dataset/MPDD"),
    # Path("/root/autodl-tmp/Data/Industrial_Dataset/BTAD"),
]
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/ADDINOv3_lwd/foreground_masks")


def parse_args():
    parser = argparse.ArgumentParser(description="Foreground/background separation with a dedicated segmentation/matting model.")
    parser.add_argument("--input", "--data_root", dest="input", type=str, default=None, help="Input image path, image directory, or dataset root.")
    parser.add_argument("--output_dir", "--output_root", dest="output_dir", type=str, default=None, help="Directory to save generated masks.")
    parser.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "generic", "mvtec", "visa"], help="Dataset traversal mode.")
    parser.add_argument("--backend", type=str, default="rmbg", choices=["rmbg", "birefnet"], help="Foreground segmentation backend.")
    parser.add_argument("--model_name", type=str, default=None, help="HF model name or local model directory.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated/private repos. If omitted, read from HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold applied to the alpha map.")
    parser.add_argument("--close_kernel", type=int, default=3, help="Morphological close kernel size. Use 0 to disable.")
    parser.add_argument("--erode_iterations", type=int, default=0, help="Optional 1-pixel style tightening after closing.")
    parser.add_argument("--max_images", type=int, default=0, help="Process at most N images. 0 means all.")
    parser.add_argument("--save_alpha", action="store_true", help="Also save the soft alpha map.")
    parser.add_argument("--save_overlay", action="store_true", help="Also save RGB overlay visualization.")
    parser.add_argument("--save_rgba", action="store_true", help="Also save foreground/background RGBA cutouts.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples whose output mask already exists.")
    return parser.parse_args()


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def resolve_default_input() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No default dataset root found. Please pass --input/--data_root explicitly."
    )


def limit_records(records, max_images: int):
    if max_images > 0:
        return records[:max_images]
    return records


def detect_dataset_format(input_path: Path) -> str:
    if input_path.is_file():
        return "generic"

    children = [path for path in input_path.iterdir() if path.is_dir()]
    if any((child / "Data" / "Images").is_dir() for child in children):
        return "visa"
    if any((child / "train").is_dir() or (child / "test").is_dir() for child in children):
        return "mvtec"
    return "generic"


def list_generic_images(input_path: Path):
    if input_path.is_file():
        return [(input_path, Path(input_path.name), "single")]

    records = []
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            records.append((path, path.relative_to(input_path), "generic"))
    return records


def list_mvtec_images(data_root: Path):
    records = []
    categories = sorted(path for path in data_root.iterdir() if path.is_dir())
    for cat_path in categories:
        for phase in ("train", "test"):
            phase_path = cat_path / phase
            if not phase_path.is_dir():
                continue

            subcats = sorted(path for path in phase_path.iterdir() if path.is_dir())
            for subcat_path in subcats:
                for image_path in sorted(subcat_path.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        relative_path = Path(cat_path.name) / phase / subcat_path.name / image_path.name
                        group_name = f"{cat_path.name} {phase}/{subcat_path.name}"
                        records.append((image_path, relative_path, group_name))
    return records


def list_visa_images(data_root: Path):
    records = []
    categories = sorted(path for path in data_root.iterdir() if (path / "Data" / "Images").is_dir())
    for cat_path in categories:
        images_root = cat_path / "Data" / "Images"
        sub_types = sorted(path for path in images_root.iterdir() if path.is_dir())
        for sub_type_path in sub_types:
            for image_path in sorted(sub_type_path.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    relative_path = Path(cat_path.name) / sub_type_path.name / image_path.name
                    group_name = f"{cat_path.name} {sub_type_path.name}"
                    records.append((image_path, relative_path, group_name))
    return records


def list_image_records(input_path: Path, dataset_format: str, max_images: int):
    resolved_format = detect_dataset_format(input_path) if dataset_format == "auto" else dataset_format

    if resolved_format == "mvtec":
        records = list_mvtec_images(input_path)
    elif resolved_format == "visa":
        records = list_visa_images(input_path)
    else:
        records = list_generic_images(input_path)

    return resolved_format, limit_records(records, max_images)


def resolve_output_dir(output_dir_arg: str | None, resolved_format: str) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    return DEFAULT_OUTPUT_ROOT / resolved_format


def is_mvtec_dense_texture(resolved_format: str, relative_path: Path) -> bool:
    if resolved_format != "mvtec" or not relative_path.parts:
        return False
    return relative_path.parts[0].lower() in MVTec_TEXTURE_CLASSES


def build_full_foreground_alpha(image: Image.Image) -> np.ndarray:
    return np.ones((image.height, image.width), dtype=np.float32)


def extract_mask_tensor(model_output):
    for attr_name in ("logits", "predictions", "masks", "alpha", "pred_alpha"):
        tensor = getattr(model_output, attr_name, None)
        if torch.is_tensor(tensor):
            return tensor

    if isinstance(model_output, dict):
        for tensor in model_output.values():
            if torch.is_tensor(tensor):
                return tensor

    if isinstance(model_output, (tuple, list)):
        for item in model_output:
            if torch.is_tensor(item):
                return item
            if hasattr(item, "logits") and torch.is_tensor(item.logits):
                return item.logits

    raise RuntimeError("Could not find a segmentation tensor in the model output.")


class HFForegroundSeparator:
    def __init__(self, model_name: str, device: str, hf_token: str | None = None):
        try:
            transformers_module = importlib.import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "HF backend requires transformers. Install it with `pip install transformers timm`."
            ) from exc

        AutoImageProcessor = transformers_module.AutoImageProcessor
        AutoModelForImageSegmentation = transformers_module.AutoModelForImageSegmentation

        self.device = torch.device(device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token,
            )
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token,
            )
        except OSError as exc:
            message = str(exc).lower()
            if "gated repo" in message or "401 client error" in message or "access to model" in message:
                raise OSError(
                    f"Cannot access model {model_name}. Pass --hf_token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN before running."
                ) from exc
            raise

        self.model.to(self.device)
        self.model.eval()

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        mask_tensor = extract_mask_tensor(outputs)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)

        mask_tensor = F.interpolate(
            mask_tensor,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )

        if mask_tensor.shape[1] == 1:
            alpha = torch.sigmoid(mask_tensor[:, 0])
        else:
            probabilities = torch.softmax(mask_tensor, dim=1)
            alpha = 1.0 - probabilities[:, 0]

        return np.clip(alpha[0].detach().cpu().numpy(), 0.0, 1.0)


class RMBGForegroundSeparator:
    def __init__(self, model_name: str, device: str, hf_token: str | None = None):
        try:
            transformers_module = importlib.import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "RMBG backend requires transformers. Install it with `pip install transformers timm`."
            ) from exc

        AutoModelForImageSegmentation = transformers_module.AutoModelForImageSegmentation
        self.device = torch.device(device)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token,
            )
        except OSError as exc:
            message = str(exc).lower()
            if "gated repo" in message or "401 client error" in message or "access to model" in message:
                raise OSError(
                    f"Cannot access model {model_name}. It is gated. Pass --hf_token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN before running."
                ) from exc
            raise

        torch.set_float32_matmul_precision("high")
        self.model.to(self.device)
        self.model.eval()

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        input_tensor = self.transform_image(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        if isinstance(outputs, (tuple, list)):
            mask_tensor = outputs[-1]
        else:
            mask_tensor = extract_mask_tensor(outputs)

        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)

        mask_tensor = torch.sigmoid(mask_tensor)
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        return np.clip(mask_tensor[0, 0].detach().cpu().numpy(), 0.0, 1.0)


def build_separator(args, device: str):
    model_name = args.model_name or DEFAULT_HF_MODELS[args.backend]
    hf_token = resolve_hf_token(args.hf_token)

    if args.backend == "rmbg":
        return RMBGForegroundSeparator(model_name=model_name, device=device, hf_token=hf_token)

    return HFForegroundSeparator(model_name=model_name, device=device, hf_token=hf_token)


def postprocess_alpha(alpha: np.ndarray, threshold: float, close_kernel: int, erode_iterations: int) -> np.ndarray:
    mask = (np.clip(alpha, 0.0, 1.0) >= threshold).astype(np.uint8) * 255

    if close_kernel and close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if erode_iterations > 0:
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=erode_iterations)

    return mask


def build_overlay(image_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.copy()
    color = np.zeros_like(overlay)
    color[:, :, 0] = 255
    mask_bool = binary_mask > 127
    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float32) * (1.0 - alpha) + color[mask_bool].astype(np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def build_output_base(output_dir: Path, relative_path: Path) -> Path:
    output_base = output_dir / relative_path.parent / relative_path.stem
    output_base.parent.mkdir(parents=True, exist_ok=True)
    return output_base


def save_outputs(output_base: Path, image_rgb: np.ndarray, alpha_map: np.ndarray, binary_mask: np.ndarray, save_alpha: bool, save_overlay: bool, save_rgba: bool):
    alpha_u8 = (np.clip(alpha_map, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(binary_mask).save(output_base.with_name(output_base.name + "_mask.png"))

    if save_alpha:
        Image.fromarray(alpha_u8).save(output_base.with_name(output_base.name + "_alpha.png"))

    if save_rgba:
        foreground_rgba = np.dstack([image_rgb, alpha_u8])
        background_rgba = np.dstack([image_rgb, 255 - alpha_u8])
        Image.fromarray(foreground_rgba).save(output_base.with_name(output_base.name + "_fg_rgba.png"))
        Image.fromarray(background_rgba).save(output_base.with_name(output_base.name + "_bg_rgba.png"))

    if save_overlay:
        overlay = build_overlay(image_rgb, binary_mask)
        Image.fromarray(overlay).save(output_base.with_name(output_base.name + "_overlay.png"))


def main():
    args = parse_args()
    input_path = Path(args.input) if args.input else resolve_default_input()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    device = resolve_device(args.device)
    resolved_format, image_records = list_image_records(input_path, args.dataset_format, args.max_images)
    output_dir = resolve_output_dir(args.output_dir, resolved_format)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_records:
        raise RuntimeError(f"No images found under: {input_path}")

    print(f"Input root: {input_path}")
    print(f"Output root: {output_dir}")
    print(f"Dataset format: {resolved_format}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_name or DEFAULT_HF_MODELS[args.backend]}")
    print(f"Device: {device}")
    print(f"Images: {len(image_records)}")

    separator = None
    current_group = None
    progress = tqdm(image_records, total=len(image_records), dynamic_ncols=True)
    for image_path, relative_path, group_name in progress:
        if group_name != current_group:
            current_group = group_name
            print(f"Processing {group_name}...")

        output_base = build_output_base(output_dir, relative_path)
        output_mask_path = output_base.with_name(output_base.name + "_mask.png")
        if args.skip_existing and output_mask_path.exists():
            continue

        progress.set_description(group_name)
        image = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image)
        if is_mvtec_dense_texture(resolved_format, relative_path):
            alpha_map = build_full_foreground_alpha(image)
        else:
            if separator is None:
                separator = build_separator(args, device)
            alpha_map = separator.predict_alpha(image)

        binary_mask = postprocess_alpha(
            alpha_map,
            threshold=args.threshold,
            close_kernel=args.close_kernel,
            erode_iterations=args.erode_iterations,
        )
        save_outputs(
            output_base,
            image_rgb,
            alpha_map,
            binary_mask,
            save_alpha=args.save_alpha,
            save_overlay=args.save_overlay,
            save_rgba=args.save_rgba,
        )


if __name__ == "__main__":
    main()