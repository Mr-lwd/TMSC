# TMSC

TMSC is a DINOv3 and CLIP based zero-shot anomaly detection implementation with tri-mask supervision and cross-modal prompt adaptation. This repository is an implementation based on the ICASSP 2026 paper AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration, and includes training, cross-dataset evaluation, and qualitative visualization.

The pipeline extracts multi-layer patch tokens and CLS tokens from DINOv3, aligns them with CLIP text prompts through learnable adapters, and produces anomaly maps with cross-modal similarity. When background-mask supervision is enabled, the model jointly predicts normal, anomaly, and background regions.

## Quick Start

### 1. Installation

```bash
cd TMSC
conda create -n tmsc python=3.10
conda activate tmsc
pip install -r requirements.txt
```

### 2. Dataset Layout

Dataset roots are defined in `Datasets/__init__.py` and resolved relative to this directory as `../Data`.

This project currently uses 4 industrial datasets: MVTec AD, VisA, MPDD, and BTAD.

```text
../Data/
└── Industrial_Dataset/
  ├── MVTecAD/
  │   ├── bottle/
  │   ├── cable/
  │   └── ...
  ├── VisA_20220922/
  │   ├── candle/
  │   ├── capsules/
  │   └── ...
  ├── MPDD/
  │   ├── bracket_black/
  │   ├── connector/
  │   └── ...
  └── BTAD/
    └── BTech_Dataset_transformed/
      ├── 01/
      ├── 02/
      └── 03/
```

Supported dataset keys:

- `mvtec`
- `visa`
- `mpdd`
- `btad`

`train.sh` currently trains on `mvtec` and `visa`, while `test.sh` runs cross-dataset evaluation from `mvtec -> visa` and `visa -> mpdd, mvtec, btad`.

### 3. Foreground and Background Masks

Background-mask supervision is always enabled. The loader in `tools/utils.py` currently expects foreground and background masks at the following hard-coded location:

```text
/root/autodl-tmp/ADDINOv3_lwd/visualizations_normal_masks/
├── <dataset>/
│   ├── normal_fore_masks/
│   │   └── <category>/<subdir>/<image_name>_normal_fore_mask.png
│   └── normal_back_masks/
│       └── <category>/<subdir>/<image_name>_normal_back_mask.png
```

If your masks are stored elsewhere, either move or symlink them into the path above, or update the hard-coded paths in `tools/utils.py` before training.

The repository also includes `segment_foreground_background.py` for foreground extraction, but its default output directory is different from the path consumed by the training loader, so you may need to adapt the output location or file naming.

### 4. Pretrained Weights

The default model configuration is defined in `config.sh`:

```bash
DINO_ARCH="dinov3_vitl16"
DINO_WEIGHTS="./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
CLIP_NAME="ViT-L-14-336px"
```

Please prepare the pretrained weights as follows:

| Pretrained Model | Source Link | Expected Path |
| --- | --- | --- |
| DINOv3 `dinov3_vitl16` | [DINOv3 official repository](https://github.com/facebookresearch/dinov3) | `./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| CLIP `ViT-L-14-336px` | [OpenAI CLIP checkpoint](https://openaipublic.blob.core.windows.net/clip/models/ViT-L-14-336px.pt) | `./CLIP/ckpt/ViT-L-14-336px.pt` |

Notes:

- `train.py` and `test.py` load DINOv3 weights from the path passed by `--dino_weights`.
- `CLIP/clip.py` will try to auto-download the CLIP checkpoint into `./CLIP/ckpt/` if it is missing.
- If you switch to another backbone or CLIP variant, update `config.sh` accordingly.

### 5. Training

The recommended entry point is `config.sh` plus `train.sh`.

```bash
bash train.sh
```

By default, `train.sh` uses:

- datasets: `mvtec`, `visa`
- selected layers: `14,17,20,23`
- DINO backbone: `dinov3_vitl16`
- CLIP model: `ViT-L-14-336px`
- background-mask supervision: enabled
- tri-mask calibration weight: `0.2`
- seeds: `1`, `2`, `4`

You can also launch a single run directly:

```bash
python train.py \
  --dataset mvtec \
  --dino_arch dinov3_vitl16 \
  --dino_weights ./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --clip_name ViT-L-14-336px \
  --selected_layers 14,17,20,23 \
  --seed 1 \
  --branch 430 \
  --tri_mask_calib_weight 0.2
```

### 6. Evaluation

To run the default evaluation workflow:

```bash
bash test.sh
```

To evaluate a single checkpoint manually:

```bash
python test.py \
  --result_path ./TESTING_ALL \
  --dataset visa \
  --train_dataset mvtec \
  --dino_arch dinov3_vitl16 \
  --dino_weights ./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --clip_name ViT-L-14-336px \
  --selected_layers 14,17,20,23 \
  --epoch 5 \
  --seed 4 \
  --tri_mask_calib_weight 0.2
```

## Outputs

Training and evaluation create the following outputs:

- Backbone feature cache: `./Result/features/<dataset>/<dino_arch>/<clip_name>/<feature_branch>/`
- Training checkpoints: `./Result/ckpt/<dataset>/<dino_arch>/<clip_name>/<run_branch>/SEED_<seed>/`
- Evaluation metrics: `./TESTING_ALL/metric.txt`
- Category visualizations: `./TESTING_ALL/<category>/`

## Repository Highlights

- `train.py`: training entry with multi-layer DINOv3 features
- `test.py`: evaluation, metric logging, and triptych visualization
- `config.sh`: default model, dataset, and ablation settings used by the shell scripts
- `segment_foreground_background.py`: helper script for foreground extraction

## Citation

This repository is based on the following paper:

```bibtex
@inproceedings{yuan2026ad,
  title={Ad-dinov3: Enhancing dinov3 for zero-shot anomaly detection with anomaly-aware calibration},
  author={Yuan, Jingyi and Ye, Jianxiong and Chen, Wenkang and Gao, Chenqiang},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={11202--11206},
  year={2026},
  organization={IEEE}
}
```

## Acknowledgement

This implementation builds on top of DINOv3 and CLIP. Please also refer to the upstream projects for backbone definitions and pretrained checkpoints.
