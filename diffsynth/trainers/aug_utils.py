from PIL import Image
import numpy as np
import torch
import cv2
import random
from typing import Tuple, Optional


def apply_horizontal_flip(image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped_image, flipped_mask


def apply_vertical_flip(image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
    flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return flipped_image, flipped_mask


def apply_crop(image: Image.Image, mask: Image.Image, crop_ratio: float = 0.9) -> Tuple[Image.Image, Image.Image]:
    width, height = image.size
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)

    x = (width - new_width) // 2
    y = (height - new_height) // 2

    cropped_image = image.crop((x, y, x + new_width, y + new_height))
    cropped_image = cropped_image.resize((width, height), Image.Resampling.BILINEAR)

    cropped_mask = mask.crop((x, y, x + new_width, y + new_height))
    cropped_mask = cropped_mask.resize((width, height), Image.Resampling.NEAREST)

    return cropped_image, cropped_mask


def apply_gaussian_blur(image: Image.Image, mask: Image.Image,
                       kernel_size_range: Tuple[int, int] = (3, 7),
                       sigma_range: Tuple[float, float] = (0.5, 2.0)) -> Tuple[Image.Image, Image.Image]:
    img_array = np.array(image)

    kernel_size = random.choice(range(kernel_size_range[0], kernel_size_range[1] + 1, 2))
    sigma = random.uniform(sigma_range[0], sigma_range[1])

    blurred_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
    blurred_image = Image.fromarray(blurred_array)

    return blurred_image, mask


def apply_fft_high_freq(image: Image.Image, mask: Image.Image,
                        freq_ratio: float = 0.3, device: str = 'cuda') -> Tuple[Image.Image, Image.Image]:
    img_array = np.array(image).astype(np.float32)

    if len(img_array.shape) == 3:
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(device)
        channels = []

        for c in range(img_tensor.shape[0]):
            channel = img_tensor[c]

            fft = torch.fft.fft2(channel)
            fft_shift = torch.fft.fftshift(fft)

            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2

            y = torch.arange(rows, device=device).float()
            x = torch.arange(cols, device=device).float()
            Y, X = torch.meshgrid(y, x, indexing='ij')

            radius = int(min(rows, cols) * freq_ratio / 2)
            mask_fft = ((X - ccol)**2 + (Y - crow)**2 <= radius**2).float()

            fft_shift_filtered = fft_shift * mask_fft

            fft_ishift = torch.fft.ifftshift(fft_shift_filtered)
            img_back = torch.fft.ifft2(fft_ishift)
            img_back = img_back.real

            img_back = torch.clamp(img_back, 0, 255)
            channels.append(img_back)

        filtered_tensor = torch.stack(channels, dim=0).permute(1, 2, 0)
        filtered_array = filtered_tensor.cpu().numpy().astype(np.uint8)
    else:
        img_tensor = torch.from_numpy(img_array).to(device)

        fft = torch.fft.fft2(img_tensor)
        fft_shift = torch.fft.fftshift(fft)

        rows, cols = img_tensor.shape
        crow, ccol = rows // 2, cols // 2

        y = torch.arange(rows, device=device).float()
        x = torch.arange(cols, device=device).float()
        Y, X = torch.meshgrid(y, x, indexing='ij')

        radius = int(min(rows, cols) * freq_ratio / 2)
        mask_fft = ((X - ccol)**2 + (Y - crow)**2 <= radius**2).float()

        fft_shift_filtered = fft_shift * mask_fft
        fft_ishift = torch.fft.ifftshift(fft_shift_filtered)
        img_back = torch.fft.ifft2(fft_ishift)
        img_back = img_back.real

        filtered_array = torch.clamp(img_back, 0, 255).cpu().numpy().astype(np.uint8)

    filtered_image = Image.fromarray(filtered_array)

    return filtered_image, mask


def vace_reference_unmasked_center_scale(
    ref_pil: Image.Image,
    mask_hw: np.ndarray,
    target_wh: Tuple[int, int],
    random_scale: bool,
    scale_min: float,
    scale_max: float,
) -> Image.Image:
    tw, th = int(target_wh[0]), int(target_wh[1])
    ref = np.asarray(ref_pil.convert("RGB"), dtype=np.float32)
    fg = np.clip(mask_hw.astype(np.float32), 0.0, 1.0)
    fg = 1 - fg
    rh, rw = ref.shape[0], ref.shape[1]
    if fg.shape[0] != rh or fg.shape[1] != rw:
        mpil = Image.fromarray((fg * 255.0).astype(np.uint8), mode="L").resize((rw, rh), Image.Resampling.NEAREST)
        fg = (np.array(mpil) > 127).astype(np.float32)

    work = ref * (1.0 - fg[..., None])
    um = (1.0 - fg) > 0.5
    ys, xs = np.where(um)
    if len(ys) == 0:
        return Image.fromarray(np.clip(work, 0.0, 255.0).astype(np.uint8), mode="RGB")

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    patch = work[y0:y1, x0:x1]
    ph, pw = patch.shape[0], patch.shape[1]
    if ph < 1 or pw < 1:
        return Image.fromarray(np.clip(work, 0.0, 255.0).astype(np.uint8), mode="RGB")

    lo = float(min(scale_min, scale_max))
    hi = float(max(scale_min, scale_max))
    if random_scale and hi > 0:
        s = float(random.uniform(lo, hi))
    else:
        s = 1.0

    nh = max(1, int(round(ph * s)))
    nw = max(1, int(round(pw * s)))
    if nh > th or nw > tw:
        fit = float(min(th / nh, tw / nw))
        nh = max(1, int(round(nh * fit)))
        nw = max(1, int(round(nw * fit)))

    patch_u8 = np.clip(patch, 0.0, 255.0).astype(np.uint8)
    pil_p = Image.fromarray(patch_u8, mode="RGB").resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    ox = (tw - nw) // 2
    oy = (th - nh) // 2
    canvas.paste(pil_p, (ox, oy))
    return canvas


def vace_reference_replace_black_with_white(pil: Image.Image) -> Image.Image:
    arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    black = (arr[..., 0] == 0) & (arr[..., 1] == 0) & (arr[..., 2] == 0)
    if not np.any(black):
        return pil.convert("RGB") if pil.mode != "RGB" else pil
    out = arr.copy()
    out[black] = 255
    return Image.fromarray(out, mode="RGB")
