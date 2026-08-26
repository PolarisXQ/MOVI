# DiffSynth-Studio Release



## TODO

- [ ] 发布项目微调 checkpoint。
- [ ] 发布训练与评测数据集。

## 安装

建议环境：Linux、Python 3.10、CUDA 12.1、PyTorch 2.4.0。

### 1. 基础安装（推理）

仅满足单视频推理 / demo。请先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装与 CUDA 匹配的 PyTorch，再执行：

```bash
conda create -n diffsynth-release python=3.10 -y
conda activate diffsynth-release
cd /path/to/DiffSynth-Studio-Realease

pip install -r requirements.txt
pip install -e .
```

### 2. 下载并放置模型

发布包不携带任何权重。请确认模型许可证后下载，并按下表放置：

| 内容 | 下载来源 | 目标位置 |
| --- | --- | --- |
| Wan2.1-VACE-1.3B 基础模型 | [Hugging Face: Wan-AI/Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) | `models/Wan-AI/Wan2.1-VACE-1.3B/` |
| UMT5-XXL 文本编码器与 tokenizer | 随 Wan2.1-VACE-1.3B 获取 | `models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth`、`models/Wan-AI/Wan2.1-VACE-1.3B/google/umt5-xxl/` |
| Wan VAE | 随 Wan2.1-VACE-1.3B 获取 | `models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth` |
| LongCLIP（多视角一致性检查） | [Hugging Face: zer0int/LongCLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14) | `models/LongCLIP-GmP-ViT-L-14/` |
| 本项目微调 full/LoRA checkpoint | TODO | `models/checkpoints/diffsynth_multiview_full.safetensors` 或 `models/checkpoints/diffsynth_multiview_lora.safetensors` |

配置一致性模型路径（任选其一；demo 默认已指向本地目录）：

```bash
# 推荐：环境变量
export CLIP_CONSISTENCY_MODEL_PATH="$PWD/models/LongCLIP-GmP-ViT-L-14"

# 或命令行
python demo/infer_single_video_with_reference42.py \
  --clip_consistency_model_path "$PWD/models/LongCLIP-GmP-ViT-L-14" \
  ...
```

完成后即可直接跑下方 [Demo](#demo)。

### 3. 可选安装（训练）

训练数据管线会用到 `diffsynth/trainers/mesh_util.py`（加载 `.glb`、多视角渲染与光照）。需要额外 Python 包与可编译的 **nvdiffrast**。

```bash
# 在已完成「1. 基础安装」的同一环境中：
pip install -r requirements-train.txt
```

从源码安装 nvdiffrast。

```bash
mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast
```

## Demo

在发布版根目录运行（需完成安装步骤 1–2）。结果写入 `demo/out/`。

### 快速准备数据

按下面目录约定放置输入，脚本会根据 `--video_name` 自动拼路径（以 `car-roundabout` 为例）：

```text
demo/
├── src/car-roundabout.mp4          # 输入视频（也可为同名图片序列目录）
├── ref/car-roundabout/             # 多视角参考图（目录内多张图，或单张图片文件）
│   ├── *_view00_*.png
│   └── ...
├── mask/car-roundabout/            # 与视频对齐的 mask（图片序列目录，或 mask/car-roundabout.mp4）
│   ├── 00000.png
│   └── ...
└── prompt/car-roundabout.txt       # 文本 prompt
```

放置好后，在仓库根目录只需指定名字即可：

```bash
mkdir -p demo/out
python demo/infer_single_video_with_reference42.py \
  --video_name car-roundabout \
  --num_reference_views 4
```

也可用 `bash demo/run.sh`（默认跑 `car-roundabout`）。

默认会解析为：

| 参数 | 默认值 |
| --- | --- |
| `--video_path` | `demo/src/<video_name>.mp4`（若不存在可回退为同名目录） |
| `--reference_image_path` | `demo/ref/<video_name>/` |
| `--mask_video_path` | `demo/mask/<video_name>.mp4`（若不存在可回退为同名目录） |
| `--prompt` | `demo/prompt/<video_name>.txt`（也可直接传一段文字） |
| `--base_model_path` | `models/Wan-AI/Wan2.1-VACE-1.3B`（或环境变量 `WAN2_1_VACE_1_3B_MODEL_PATH`） |
| `--checkpoint_type` | `lora`，对应 `models/checkpoints/diffsynth_multiview_lora.safetensors` |

### 自定义路径（可选）

数据不在上述约定位置时，再显式传入路径；其余参数仍可省略：

```bash
python demo/infer_single_video_with_reference42.py \
  --video_name my_clip \
  --video_path /path/to/input.mp4 \
  --reference_image_path /path/to/reference_images \
  --mask_video_path /path/to/mask.mp4 \
  --prompt "a detailed description of the desired video"
```

<!-- #使用 full checkpoint 时加 `--checkpoint_type full` 与 `--full_checkpoint_path`。输入帧数会调整为 `4n+1`（至少 17）；默认 832×480、81 帧。 -->
全部参数见 `python demo/infer_single_video_with_reference42.py --help`。