# DiffSynth-Studio Release

[Chinese version](./README_zh.md)

Code for the paper: [Controllable Video Object Insertion
via Multi-View Priors](https://arxiv.org/abs/2604.14556)

Abstract: 
Video object insertion is a critical task for dynamically inserting new objects into existing environments. Previous video generation methods focus primarily on synthesizing entire scenes while struggle with ensuring consistent object appearance, spatial alignment, and temporal coherence when inserting objects into existing videos. In this paper, we propose a novel solution for Video Object Insertion, which integrates multi-view object priors to address the common challenges of appearance inconsistency and occlusion handling in dynamic environments. By lifting 2D reference images into multi-view representations and leveraging a dual-path view-consistent conditioning mechanism, our framework ensures stable identity guidance and robust integration across diverse viewpoints. A quality-aware weighting mechanism is also conducted to adaptively handle noisy or imperfect inputs. Additionally, we introduce an Integration-Aware Consistency Module that guarantee spatial realism, effectively resolving occlusion and boundary artifacts while maintaining temporal continuity across frames. Experimental results show that our solution significantly improves the quality of video object insertion, providing stable and realistic integration.

## TODO

- [ ] Release the project's fine-tuned checkpoints.
- [ ] Release the training and evaluation datasets.

## Installation

Recommended environment: Linux, Python 3.10, CUDA 12.1, and PyTorch 2.4.0.

### 1. Base installation (inference)

This is sufficient for single-video inference and demos. First install the CUDA-compatible version of PyTorch from the [PyTorch website](https://pytorch.org/get-started/locally/), then run:

```bash
conda create -n diffsynth-release python=3.10 -y
conda activate diffsynth-release
cd /path/to/DiffSynth-Studio-Realease

pip install -r requirements.txt
pip install -e .
```

### 2. Download and place the models

The release package does not include model weights. Download them after confirming their licenses, and place them as shown below:

| Component | Download source | Destination |
| --- | --- | --- |
| Wan2.1-VACE-1.3B base model | [Hugging Face: Wan-AI/Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) | `models/Wan-AI/Wan2.1-VACE-1.3B/` |
| UMT5-XXL text encoder and tokenizer | Included with Wan2.1-VACE-1.3B | `models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth` and `models/Wan-AI/Wan2.1-VACE-1.3B/google/umt5-xxl/` |
| Wan VAE | Included with Wan2.1-VACE-1.3B | `models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth` |
| LongCLIP (multi-view consistency check) | [Hugging Face: zer0int/LongCLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14) | `models/LongCLIP-GmP-ViT-L-14/` |
| This project's fine-tuned full/LoRA checkpoint | TODO | `models/checkpoints/diffsynth_multiview_full.safetensors` or `models/checkpoints/diffsynth_multiview_lora.safetensors` |

Afterward, you can run the [demo](#demo) below.

### 3. Optional installation (training)

The training data pipeline uses `diffsynth/trainers/mesh_util.py` for loading `.glb` files, multi-view rendering, and lighting. It requires additional Python packages and a buildable installation of **nvdiffrast**.

```bash
# Use the same environment created in step 1.
pip install -r requirements-train.txt
```

Install nvdiffrast from source:

```bash
mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast
```

## Demo

Run from the release root after completing installation steps 1 and 2. Results are written to `demo/out/`.

### Prepare data quickly

Place inputs according to the directory convention below. The script derives paths from `--video_name` automatically; this example uses `car-roundabout`:

```text
demo/
├── src/car-roundabout.mp4          # Input video; a directory with an image sequence of the same name also works.
├── ref/car-roundabout/             # Multi-view reference images: a directory of images or a single image file.
│   ├── *_view00_*.png
│   └── ...
├── mask/car-roundabout/            # Mask aligned with the video: an image-sequence directory or mask/car-roundabout.mp4.
│   ├── 00000.png
│   └── ...
└── prompt/car-roundabout.txt       # Text prompt.
```

After placing the files, run this command from the repository root:

```bash
mkdir -p demo/out
python demo/infer_single_video_with_reference42.py \
  --video_name car-roundabout
```

The default paths resolve as follows:

| Argument | Default |
| --- | --- |
| `--video_path` | `demo/src/<video_name>.mp4` (falls back to a directory with the same name if absent) |
| `--reference_image_path` | `demo/ref/<video_name>/` |
| `--mask_video_path` | `demo/mask/<video_name>.mp4` (falls back to a directory with the same name if absent) |
| `--prompt` | `demo/prompt/<video_name>.txt` (a text prompt can also be passed directly) |
| `--base_model_path` | `models/Wan-AI/Wan2.1-VACE-1.3B` (or the `WAN2_1_VACE_1_3B_MODEL_PATH` environment variable) |
| `--checkpoint_type` | `lora`, corresponding to `models/checkpoints/diffsynth_multiview_lora.safetensors` |

### Custom paths (optional)

If the data is not stored in the convention above, pass explicit paths. The remaining arguments may still be omitted:

```bash
python demo/infer_single_video_with_reference42.py \
  --video_name my_clip \
  --video_path /path/to/input.mp4 \
  --reference_image_path /path/to/reference_images \
  --mask_video_path /path/to/mask.mp4 \
  --prompt "a detailed description of the desired video"
```

For all arguments, run:

```bash
python demo/infer_single_video_with_reference42.py --help
```
