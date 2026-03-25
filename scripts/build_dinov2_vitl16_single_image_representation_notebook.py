#!/usr/bin/env python3

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "notebooks" / "dinov2_vitl16_single_image_representation.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"].update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        }
    )

    nb.cells = [
        md(
            """
            # DINOv2 ViT-L/16 Single-Image Representation

            This notebook loads one of the shared SAYCAM `DINOv2 ViT-L/16` checkpoints from
            `/scratch/gpfs/BRENDEN/dinov2_vitl16`, preprocesses a single image, and returns its representation.

            Notes:
            - `MODEL_VARIANT` can be `S`, `A`, `Y`, or `SAY`.
            - The primary representation exposed here is the normalized class-token embedding:
              `cls_representation`.
            - Patch-token outputs are also included in case you want spatial features later.
            - Run this notebook in the environment that has `torch`, `torchvision`, `PIL`, and this repo on disk.
            """
        ),
        code(
            """
            from __future__ import annotations

            import os
            import sys
            from pathlib import Path
            from types import SimpleNamespace

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import display
            from PIL import Image


            def find_repo_root(start: Path | None = None) -> Path:
                current = (start or Path.cwd()).resolve()
                for candidate in [current, *current.parents]:
                    if (candidate / "dinov2").is_dir() and (candidate / "scripts").is_dir():
                        return candidate
                raise FileNotFoundError("Could not locate the dinov2 repo root from the current working directory.")


            REPO_ROOT = find_repo_root()
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))


            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 160)
            print(f"Repo root: {REPO_ROOT}")
            print(f"Torch version: {torch.__version__}")
            """
        ),
        code(
            """
            SHARED_ROOT = Path("/scratch/gpfs/BRENDEN/dinov2_vitl16")
            MODEL_VARIANT = "S"  # One of: S, A, Y, SAY
            IMAGE_PATH = REPO_ROOT / "spatial_relation_bias" / "complex-relational-reasoning" / "BeachBall_0004_support.png"
            PREFERRED_DEVICE = "auto"  # One of: auto, cuda, cpu
            SAVE_CLS_REPRESENTATION = False
            SAVE_PATH = REPO_ROOT / "tmp" / f"dinov2_vitl16_{MODEL_VARIANT.lower()}_cls_representation.npy"

            MODEL_VARIANT = MODEL_VARIANT.upper()
            VALID_VARIANTS = {"S", "A", "Y", "SAY"}
            if MODEL_VARIANT not in VALID_VARIANTS:
                raise ValueError(f"MODEL_VARIANT must be one of {sorted(VALID_VARIANTS)}, got {MODEL_VARIANT!r}")

            checkpoint_path = SHARED_ROOT / f"dinov2_vitl16_saycam_{MODEL_VARIANT}_125k_teacher_checkpoint.pth"
            config_path = SHARED_ROOT / f"dinov2_vitl16_saycam_{MODEL_VARIANT}_config.yaml"

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config: {config_path}")
            if not Path(IMAGE_PATH).exists():
                raise FileNotFoundError(f"Missing image: {IMAGE_PATH}")

            print(f"Checkpoint: {checkpoint_path}")
            print(f"Config:     {config_path}")
            print(f"Image:      {IMAGE_PATH}")
            """
        ),
        code(
            """
            def choose_device(preferred: str = "auto") -> torch.device:
                preferred = preferred.lower()
                if preferred == "cuda":
                    if not torch.cuda.is_available():
                        raise RuntimeError("PREFERRED_DEVICE='cuda' but CUDA is not available.")
                    return torch.device("cuda")
                if preferred == "cpu":
                    return torch.device("cpu")
                if preferred != "auto":
                    raise ValueError(f"Unsupported device preference: {preferred!r}")
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")


            device = choose_device(PREFERRED_DEVICE)

            if any(name.startswith("dinov2") for name in sys.modules):
                print("dinov2 is already imported in this kernel. If you changed PREFERRED_DEVICE, restart the kernel and rerun from the top.")

            if device.type == "cpu":
                os.environ["XFORMERS_DISABLED"] = "1"
                print("CPU mode selected; set XFORMERS_DISABLED=1 so attention falls back to the non-xFormers path.")
            else:
                print("CUDA mode selected; leaving xFormers enabled if available.")

            from dinov2.data.transforms import make_classification_eval_transform
            from dinov2.models import build_model_from_cfg
            from dinov2.utils.config import setup
            from dinov2.utils.utils import load_pretrained_weights


            def build_dinov2_backbone(config_file: Path, pretrained_weights: Path, device: torch.device):
                args = SimpleNamespace(config_file=str(config_file), opts=[], output_dir="")
                config = setup(args)
                model, _ = build_model_from_cfg(config, only_teacher=True)
                load_pretrained_weights(model, str(pretrained_weights), "teacher")
                model.eval()
                model.to(device)
                return model, config


            def preprocess_image(image_path: Path, crop_size: int):
                resize_size = int(round(crop_size * 256 / 224))
                transform = make_classification_eval_transform(
                    resize_size=resize_size,
                    crop_size=crop_size,
                )
                pil_image = Image.open(image_path).convert("RGB")
                image_tensor = transform(pil_image).unsqueeze(0)
                return pil_image, image_tensor


            def extract_representations(model, image_tensor: torch.Tensor, device: torch.device):
                with torch.inference_mode():
                    features = model.forward_features(image_tensor.to(device))

                cls_token = features["x_norm_clstoken"][0].detach().cpu()
                patch_tokens = features["x_norm_patchtokens"][0].detach().cpu()
                mean_patch_token = patch_tokens.mean(dim=0)

                return {
                    "cls_token": cls_token,
                    "patch_tokens": patch_tokens,
                    "mean_patch_token": mean_patch_token,
                }
            """
        ),
        code(
            """
            model, config = build_dinov2_backbone(config_path, checkpoint_path, device=device)
            crop_size = int(config.crops.global_crops_size)
            pil_image, image_tensor = preprocess_image(Path(IMAGE_PATH), crop_size=crop_size)

            print(f"Device:     {device}")
            print(f"Crop size:  {crop_size}")
            print(f"Input size: {tuple(image_tensor.shape)}")

            plt.figure(figsize=(5, 5))
            plt.imshow(pil_image)
            plt.axis("off")
            plt.title(f"{MODEL_VARIANT} input image")
            plt.show()
            """
        ),
        code(
            """
            representations = extract_representations(model, image_tensor, device=device)

            cls_representation = representations["cls_token"].numpy()
            patch_representations = representations["patch_tokens"].numpy()
            mean_patch_representation = representations["mean_patch_token"].numpy()

            summary = pd.DataFrame(
                [
                    {"name": "cls_representation", "shape": cls_representation.shape},
                    {"name": "patch_representations", "shape": patch_representations.shape},
                    {"name": "mean_patch_representation", "shape": mean_patch_representation.shape},
                ]
            )
            display(summary)

            preview = pd.DataFrame(
                {
                    "dimension": np.arange(16),
                    "cls_representation": cls_representation[:16],
                    "mean_patch_representation": mean_patch_representation[:16],
                }
            )
            display(preview)
            """
        ),
        code(
            """
            if SAVE_CLS_REPRESENTATION:
                SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.save(SAVE_PATH, cls_representation)
                print(f"Saved cls_representation to {SAVE_PATH}")
            else:
                print("SAVE_CLS_REPRESENTATION is False; not writing a .npy file.")
            """
        ),
        code(
            """
            cls_representation
            """
        ),
    ]
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a notebook for extracting a DINOv2 ViT-L/16 representation from one image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Notebook output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote notebook to {args.output}")


if __name__ == "__main__":
    main()
