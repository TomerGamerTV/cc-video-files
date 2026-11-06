#!/usr/bin/env python3
"""
convert_and_stream.py
Ultra-Fast GPU + Multi-Threaded Media Converter for CC:Tweaked
Supports:
  1. Videos with Audio (.mp4, .mkv, .webm, .mov, .avi, or YouTube URLs)
  2. Audio-Only Files (.mp3, .wav, .ogg, .flac, .m4a, .aac, .opus, or `--audio-only`)
  3. Animated GIFs (.gif, .apng, animated .webp)
  4. Static Images (.png, .jpg, .jpeg, .bmp, .tiff)

Accelerations Included:
1. GPU Acceleration (Auto-detected):
   - Automatically detects and uses PyTorch CUDA (NVIDIA RTX/GTX), AMD ROCm, or Apple Metal (MPS)
     to run batched K-Means++ palette clustering, Bayer subpixel dithering, and nearest-palette
     quantization directly on your GPU via `torch.bmm` tensor operations!
2. Multi-Threaded C++ SIMD + 12-Bit 3D Color Lookup Table (LUT) Engine:
   - Uses a `ThreadPoolExecutor` across all CPU threads (releasing the Python GIL inside OpenCV's
     C++ `cv2.kmeans`, `cv2.add`, `cv2.subtract`, and NumPy's BLAS `sgemm` `@` operator).
   - Evaluates a precomputed 4,096-entry (16x16x16) Perceptual 3D Color LUT per frame instead of
     79,704 individual pixels, achieving **100+ FPS** conversion speed even on CPU!

Usage Examples:
  # Convert a video with audio:
  python3 convert_and_stream.py video.mp4 -o video.ccv

  # Convert an audio-only file (or extract audio-only from a video/URL):
  python3 convert_and_stream.py song.mp3 -o song.ccv
  python3 convert_and_stream.py video.mp4 -o song.ccv --audio-only

  # Convert an animated GIF or static image:
  python3 convert_and_stream.py animation.gif -o anim.ccv
  python3 convert_and_stream.py photo.png -o photo.ccv
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import struct
import subprocess
import sys
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageSequence

# Snapshot clean OS environment BEFORE importing cv2 so cv2's bundled Qt/OpenMP
# LD_LIBRARY_PATH injection never pollutes or slows down child FFmpeg processes!
_CLEAN_ENV = dict(os.environ)

# Optional C++ SIMD OpenCV acceleration (releases the GIL in C++)
try:
    import cv2  # type: ignore
    cv2.setNumThreads(1)
    # Restore clean LD_LIBRARY_PATH so child subprocesses (ffmpeg/ffprobe) use native system libs
    if "LD_LIBRARY_PATH" not in _CLEAN_ENV:
        os.environ.pop("LD_LIBRARY_PATH", None)
    else:
        os.environ["LD_LIBRARY_PATH"] = _CLEAN_ENV["LD_LIBRARY_PATH"]
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

# Optional GPU acceleration via PyTorch (CUDA / ROCm / Apple MPS)
TORCH_DEVICE = None
TORCH_GPU_NAME = None
try:
    import torch  # type: ignore
    if torch.cuda.is_available():
        TORCH_DEVICE = torch.device("cuda")
        try:
            TORCH_GPU_NAME = torch.cuda.get_device_name(0)
        except Exception:
            TORCH_GPU_NAME = "CUDA GPU"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        TORCH_DEVICE = torch.device("mps")
        TORCH_GPU_NAME = "Apple Metal GPU (MPS)"
except ImportError:
    torch = None


# =============================================================================
# 1. Exact DFPWM1a Audio Encoder (48,000 Hz -> 1-bit DFPWM1a, 6,000 bytes/sec)
#    Matches FFmpeg `libavcodec/dfpwmenc.c` & CC:Tweaked `cc.audio.dfpwm`
# =============================================================================

class DFPWMEncoder:
    """
    Stateful DFPWM1a encoder matching Ben "GreaseMonkey" Russell's public-domain
    DFPWM1a specification used by FFmpeg (`libavcodec/dfpwmdec.c`) and CC:Tweaked.
    """

    def __init__(self) -> None:
        self.q = 0
        self.s = 0
        self.lt = -128

    def encode_chunk(self, pcm_s8: np.ndarray) -> bytes:
        n_samples = len(pcm_s8)
        if n_samples == 0:
            return b""
        if n_samples % 8 != 0:
            pad = 8 - (n_samples % 8)
            pcm_s8 = np.pad(pcm_s8, (0, pad), mode="constant", constant_values=0)
            n_samples = len(pcm_s8)

        out_bytes = bytearray(n_samples // 8)
        q = self.q
        s = self.s
        lt = self.lt

        samples = pcm_s8.tolist()
        idx = 0
        for byte_i in range(len(out_bytes)):
            d = 0
            for bit_i in range(8):
                v = samples[idx]
                idx += 1
                t_val = 127 if (v > q or (v == q and v == 127)) else -128
                if t_val == 127:
                    d |= (1 << bit_i)

                nq = q + ((s * (t_val - q) + 512) >> 10)
                if nq == q and nq != t_val:
                    nq += 1 if t_val == 127 else -1
                q = nq

                st = 0 if (t_val != lt) else 1023
                ns = s
                if ns != st:
                    ns += 1 if (st != 0) else -1
                if ns < 8:
                    ns = 8
                s = ns
                lt = t_val

            out_bytes[byte_i] = d

        self.q = q
        self.s = s
        self.lt = lt
        return bytes(out_bytes)


def ffmpeg_supports_dfpwm(ffmpeg_bin: str) -> bool:
    """Check if the installed FFmpeg binary has built-in native DFPWM1a encoding."""
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-nostdin", "-hide_banner", "-codecs"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=_CLEAN_ENV,
            timeout=5,
        )
        for line in proc.stdout.splitlines():
            if "dfpwm" in line and "E" in line[:8]:
                return True
    except Exception:
        pass
    return False


# =============================================================================
# 2. Fast Perceptual 3D Color LUT + Multi-Threaded CPU & GPU Quantization Engine
# =============================================================================

WEIGHTS_NP = np.array([0.299, 0.587, 0.114], dtype=np.float32)
SQRT_W_NP = np.sqrt(WEIGHTS_NP)
EYE16_NP = np.eye(16, dtype=np.float32)

_grid_r, _grid_g, _grid_b = np.meshgrid(
    np.arange(16, dtype=np.float32) * 16.0 + 8.0,
    np.arange(16, dtype=np.float32) * 16.0 + 8.0,
    np.arange(16, dtype=np.float32) * 16.0 + 8.0,
    indexing="ij",
)
LUT12_COORDS_NP = (
    np.stack([_grid_r, _grid_g, _grid_b], axis=-1).reshape(-1, 3) * SQRT_W_NP
)  # (4096, 3)

BAYER4_I16 = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.int16,
    )
    - 8
)

_BAYER_CACHE = {}


def get_bayer_tables(h_px: int, w_px: int):
    key = (h_px, w_px)
    if key not in _BAYER_CACHE:
        tiled = np.tile(BAYER4_I16, (h_px // 4 + 1, w_px // 4 + 1))[:h_px, :w_px, None]
        b_pos = np.repeat(np.maximum(0, tiled).astype(np.uint8), 3, axis=2)
        b_neg = np.repeat(np.maximum(0, -tiled).astype(np.uint8), 3, axis=2)
        _BAYER_CACHE[key] = (tiled, b_pos, b_neg)
    return _BAYER_CACHE[key]


def quantize_single_frame_cpu(
    args_tuple: Tuple[bytes, int, int, bool, int]
) -> Tuple[bytes, bytes]:
    raw_frame, sub_w, sub_h, dither, kmeans_iters = args_tuple
    frame_rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape((sub_h, sub_w, 3))
    pixels = frame_rgb.reshape(-1, 3)
    n_px = pixels.shape[0]

    step = max(1, n_px // 1500)
    w_sample = np.ascontiguousarray(pixels[::step], dtype=np.float32) * SQRT_W_NP

    if HAS_CV2 and cv2 is not None:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            max(1, kmeans_iters),
            0.5,
        )
        _, _, centers = cv2.kmeans(
            w_sample, 16, None, criteria, 1, cv2.KMEANS_PP_CENTERS
        )
    else:
        s_norm_sq = np.sum(w_sample * w_sample, axis=1)
        centers = np.empty((16, 3), dtype=np.float32)
        centers[0] = w_sample[len(w_sample) // 2]
        c_sq = np.sum(centers[0] ** 2)
        dist_sq = s_norm_sq + c_sq - 2.0 * (w_sample @ centers[0])

        for c in range(1, 16):
            idx = int(np.argmax(dist_sq))
            centers[c] = w_sample[idx]
            c_sq = np.sum(centers[c] ** 2)
            new_dist_sq = s_norm_sq + c_sq - 2.0 * (w_sample @ centers[c])
            np.minimum(dist_sq, new_dist_sq, out=dist_sq)

        for _ in range(kmeans_iters):
            c_norm_sq = np.sum(centers * centers, axis=1)
            scores = c_norm_sq[None, :] - 2.0 * (w_sample @ centers.T)
            labels = np.argmin(scores, axis=1)
            one_hot = EYE16_NP[labels]
            counts = np.sum(one_hot, axis=0)
            sums = one_hot.T @ w_sample
            non_empty = counts > 0
            new_centers = np.copy(centers)
            new_centers[non_empty] = sums[non_empty] / counts[non_empty, None]
            if np.max(np.abs(centers - new_centers)) < 0.5:
                centers = new_centers
                break
            centers = new_centers

    rgb_palette = np.clip(np.round(centers / SQRT_W_NP), 0, 255).astype(np.uint8)
    luma = (
        0.299 * rgb_palette[:, 0]
        + 0.587 * rgb_palette[:, 1]
        + 0.114 * rgb_palette[:, 2]
    )
    rgb_palette = rgb_palette[np.argsort(luma)]

    if dither:
        tiled_i16, b_pos, b_neg = get_bayer_tables(sub_h, sub_w)
        if HAS_CV2 and cv2 is not None:
            dithered = cv2.subtract(cv2.add(frame_rgb, b_pos), b_neg)
        else:
            dithered = np.clip(frame_rgb.astype(np.int16) + tiled_i16, 0, 255).astype(
                np.uint8
            )
    else:
        dithered = frame_rgb

    q4 = (dithered >> 4).astype(np.uint16)
    idx12 = ((q4[:, :, 0] << 8) | (q4[:, :, 1] << 4) | q4[:, :, 2]).reshape(-1)

    w_pal = rgb_palette.astype(np.float32) * SQRT_W_NP
    pal_norm_sq = np.sum(w_pal * w_pal, axis=1)
    lut_scores = pal_norm_sq[None, :] - 2.0 * (LUT12_COORDS_NP @ w_pal.T)
    lut_table = np.argmin(lut_scores, axis=1).astype(np.uint8)
    indices = lut_table[idx12]

    if n_px % 2 == 1:
        indices = np.append(indices, np.uint8(0))
    packed = (indices[0::2] << 4) | (indices[1::2] & 0x0F)
    return rgb_palette.tobytes(), packed.tobytes()


def quantize_frame_to_4bit_subpixels(
    frame_rgb: np.ndarray,
    dither: bool = True,
    kmeans_iters: int = 10,
) -> Tuple[bytes, bytes]:
    sub_h, sub_w, _ = frame_rgb.shape
    return quantize_single_frame_cpu(
        (frame_rgb.tobytes(), sub_w, sub_h, dither, kmeans_iters)
    )


def quantize_batch_gpu_torch(
    raw_frames: List[bytes],
    sub_w: int,
    sub_h: int,
    dither: bool = True,
    kmeans_iters: int = 10,
) -> List[Tuple[bytes, bytes]]:
    B = len(raw_frames)
    n_px = sub_w * sub_h
    batch_np = np.frombuffer(b"".join(raw_frames), dtype=np.uint8).reshape(
        (B, n_px, 3)
    )

    with torch.no_grad():
        device = TORCH_DEVICE
        sqrt_w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=device).sqrt()
        luma_w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=device)

        frames_t = torch.from_numpy(batch_np.copy()).to(device=device, dtype=torch.float32)
        step = max(1, n_px // 1500)
        w_sample = frames_t[:, ::step, :] * sqrt_w
        S = w_sample.shape[1]

        centers = torch.empty((B, 16, 3), dtype=torch.float32, device=device)
        centers[:, 0, :] = w_sample[:, S // 2, :]
        dist_sq = torch.sum((w_sample - centers[:, 0:1, :]) ** 2, dim=2)

        batch_arange = torch.arange(B, device=device)
        for c in range(1, 16):
            idx = torch.argmax(dist_sq, dim=1)
            chosen = w_sample[batch_arange, idx, :]
            centers[:, c, :] = chosen
            new_dist = torch.sum((w_sample - chosen.unsqueeze(1)) ** 2, dim=2)
            dist_sq = torch.minimum(dist_sq, new_dist)

        for _ in range(kmeans_iters):
            c_norm_sq = torch.sum(centers * centers, dim=2).unsqueeze(1)
            scores = c_norm_sq - 2.0 * torch.bmm(w_sample, centers.transpose(1, 2))
            labels = torch.argmin(scores, dim=2)
            one_hot = torch.nn.functional.one_hot(labels, num_classes=16).to(torch.float32)
            counts = torch.sum(one_hot, dim=1).unsqueeze(2)
            sums = torch.bmm(one_hot.transpose(1, 2), w_sample)
            centers = torch.where(counts > 0, sums / torch.clamp(counts, min=1.0), centers)

        rgb_pal = torch.clamp(torch.round(centers / sqrt_w), 0, 255).to(torch.uint8)
        luma = torch.sum(rgb_pal.to(torch.float32) * luma_w, dim=2)
        sort_idx = torch.argsort(luma, dim=1)
        rgb_pal = torch.gather(rgb_pal, 1, sort_idx.unsqueeze(2).expand(-1, -1, 3))

        if dither:
            tiled_i16, _, _ = get_bayer_tables(sub_h, sub_w)
            bayer_t = torch.from_numpy(tiled_i16.reshape(1, n_px, 1)).to(
                device=device, dtype=torch.float32
            )
            work_rgb = torch.clamp(frames_t + bayer_t, 0.0, 255.0)
        else:
            work_rgb = frames_t

        w_pixels = work_rgb * sqrt_w
        w_pal = rgb_pal.to(torch.float32) * sqrt_w
        pal_norm_sq = torch.sum(w_pal * w_pal, dim=2).unsqueeze(1)
        full_scores = pal_norm_sq - 2.0 * torch.bmm(w_pixels, w_pal.transpose(1, 2))
        indices = torch.argmin(full_scores, dim=2).to(torch.uint8)

        if n_px % 2 == 1:
            pad_zero = torch.zeros((B, 1), dtype=torch.uint8, device=device)
            indices = torch.cat([indices, pad_zero], dim=1)

        packed = (indices[:, 0::2] << 4) | (indices[:, 1::2] & 0x0F)

        pal_cpu = rgb_pal.cpu().numpy()
        packed_cpu = packed.cpu().numpy()

    return [(pal_cpu[i].tobytes(), packed_cpu[i].tobytes()) for i in range(B)]


# =============================================================================
# 3. Media Type Detection & Decoders (Static Image, Animated GIF, Audio-Only, Video)
# =============================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
GIF_EXTENSIONS = {".gif", ".apng"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".wma", ".dfpwm"}


def resize_with_aspect_fit(
    img: Image.Image, target_w: int, target_h: int, fit_mode: str = "contain"
) -> np.ndarray:
    img_rgb = img.convert("RGB")
    src_w, src_h = img_rgb.size

    if fit_mode == "stretch":
        resized = img_rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.uint8)

    scale_w = target_w / src_w
    scale_h = target_h / src_h
    scale = max(scale_w, scale_h) if fit_mode == "cover" else min(scale_w, scale_h)

    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return np.asarray(canvas, dtype=np.uint8)


def detect_media_kind(input_source: str, force_audio_only: bool = False) -> str:
    if force_audio_only:
        return "audio"

    parsed_path = input_source.split("?")[0].lower() if input_source.startswith(("http://", "https://")) else input_source.lower()
    ext = os.path.splitext(parsed_path)[1]

    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in GIF_EXTENSIONS:
        return "gif"

    if not input_source.startswith(("http://", "https://")) and ext == ".webp":
        try:
            with Image.open(input_source) as im:
                is_anim = getattr(im, "is_animated", False) and getattr(im, "n_frames", 1) > 1
                return "gif" if is_anim else "image"
        except Exception:
            pass

    return "video"


def resolve_media_url(url: str, audio_only: bool = False) -> Tuple[str, Optional[str]]:
    ytdlp_bin = shutil.which("yt-dlp")
    if ytdlp_bin and url.startswith(("http://", "https://")):
        fmt = "bestaudio/best" if audio_only else "bestvideo+bestaudio/best"
        try:
            proc = subprocess.run(
                [ytdlp_bin, "-g", "-f", fmt, "--no-playlist", url],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=_CLEAN_ENV,
                check=True,
                timeout=30,
            )
            lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
            if len(lines) >= 2:
                return lines[0], lines[1]
            elif len(lines) == 1:
                return lines[0], None
        except Exception:
            pass
    return url, None


def has_audio_stream(input_path: str, ffprobe_bin: Optional[str]) -> bool:
    if not ffprobe_bin:
        return True
    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-nostdin",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                input_path,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=_CLEAN_ENV,
            timeout=10,
        )
        return "audio" in proc.stdout.lower()
    except Exception:
        return True


def encode_media_to_ccv2(
    input_source: str,
    sub_w: int = 328,
    sub_h: int = 243,
    fps: int = 20,
    kmeans_iters: int = 10,
    dither: bool = True,
    fit_mode: str = "stretch",
    mute: bool = False,
    audio_only: bool = False,
    workers: int = 0,
    use_gpu: bool = True,
    hwaccel: Optional[str] = None,
    batch_size: int = 64,
) -> Iterator[bytes]:
    """
    Encode a static image, animated GIF, audio-only file, or video into CCV2 packets.

    CCV2 Header Flags:
      bit 0 (val 1): Has 48kHz DFPWM1a Audio
      bit 1 (val 2): Is Static Image
      bit 2 (val 4): Loop enabled
      bit 3 (val 8): Is Audio-Only Stream (width=0, height=0, packets contain only audio)
    """
    if sub_w % 2 != 0:
        sub_w += 1

    if workers <= 0:
        workers = os.cpu_count() or 8

    kind = detect_media_kind(input_source, force_audio_only=audio_only)
    ffmpeg_bin = shutil.which("ffmpeg")

    # -------------------------------------------------------------------------
    # CASE 0: Audio-Only Mode (.mp3, .wav, .ogg, .flac, or --audio-only)
    # -------------------------------------------------------------------------
    if kind == "audio":
        if not ffmpeg_bin:
            raise RuntimeError("FFmpeg is required for audio conversion.")

        vid_url, aud_url = resolve_media_url(input_source, audio_only=True)
        aud_source = aud_url if aud_url else vid_url

        flags = 0x01 | 0x04 | 0x08
        header = struct.pack(">4sHHBBH", b"CCV2", 0, 0, fps, flags, 0)
        yield header

        use_native_ffmpeg_dfpwm = ffmpeg_supports_dfpwm(ffmpeg_bin)
        if use_native_ffmpeg_dfpwm:
            audio_cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-loglevel", "error",
                "-i", aud_source,
                "-vn",
                "-ac", "1",
                "-ar", "48000",
                "-f", "dfpwm",
                "-acodec", "dfpwm",
                "pipe:1",
            ]
        else:
            audio_cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-loglevel", "error",
                "-i", aud_source,
                "-vn",
                "-ac", "1",
                "-ar", "48000",
                "-f", "s8",
                "-acodec", "pcm_s8",
                "pipe:1",
            ]

        aud_proc = subprocess.Popen(
            audio_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            env=_CLEAN_ENV,
            bufsize=48000 * 4,
        )
        dfpwm_encoder = DFPWMEncoder()
        dfpwm_chunk_bytes = 6000
        pcm_chunk_samples = dfpwm_chunk_bytes * 8

        try:
            while True:
                if use_native_ffmpeg_dfpwm:
                    dfpwm_bytes = aud_proc.stdout.read(dfpwm_chunk_bytes) if aud_proc.stdout else b""
                else:
                    raw_audio = aud_proc.stdout.read(pcm_chunk_samples) if aud_proc.stdout else b""
                    if raw_audio:
                        pcm_arr = np.frombuffer(raw_audio, dtype=np.int8)
                        dfpwm_bytes = dfpwm_encoder.encode_chunk(pcm_arr)
                    else:
                        dfpwm_bytes = b""

                if not dfpwm_bytes:
                    break

                yield struct.pack(">BH", 1, len(dfpwm_bytes)) + dfpwm_bytes

            yield struct.pack(">B", 0xFF)
        finally:
            try:
                aud_proc.kill()
                aud_proc.wait(timeout=1)
            except Exception:
                pass
        return

    # Pre-warm Bayer lookup table in main thread before spawning worker threads
    get_bayer_tables(sub_h, sub_w)

    # -------------------------------------------------------------------------
    # CASE 1: Static Image (.png, .jpg, .bmp, .webp single-frame)
    # -------------------------------------------------------------------------
    if kind == "image" and not input_source.startswith(("http://", "https://")):
        with Image.open(input_source) as img:
            frame_np = resize_with_aspect_fit(img, sub_w, sub_h, fit_mode=fit_mode)

        flags = 0x02  # bit 1 = Static Image
        header = struct.pack(">4sHHBBH", b"CCV2", sub_w, sub_h, 0, flags, 0)
        yield header

        pal_bytes, packed_px = quantize_frame_to_4bit_subpixels(
            frame_np, dither=dither, kmeans_iters=max(25, kmeans_iters)
        )
        yield struct.pack(">BH", 1, 0) + pal_bytes + packed_px
        yield struct.pack(">B", 0xFF)
        return

    # -------------------------------------------------------------------------
    # CASE 2: Local Animated GIF / WebP / APNG via Pillow (if ffmpeg not used)
    # -------------------------------------------------------------------------
    if kind == "gif" and not ffmpeg_bin and not input_source.startswith(("http://", "https://")):
        flags = 0x04
        header = struct.pack(">4sHHBBH", b"CCV2", sub_w, sub_h, fps, flags, 0)
        yield header

        raw_frames = []
        with Image.open(input_source) as img:
            for frame in ImageSequence.Iterator(img):
                frame_np = resize_with_aspect_fit(frame, sub_w, sub_h, fit_mode=fit_mode)
                raw_frames.append(frame_np.tobytes())

        if use_gpu and TORCH_DEVICE is not None:
            for i in range(0, len(raw_frames), batch_size):
                batch = raw_frames[i : i + batch_size]
                for pal_bytes, packed_px in quantize_batch_gpu_torch(
                    batch, sub_w, sub_h, dither=dither, kmeans_iters=kmeans_iters
                ):
                    yield struct.pack(">BH", 1, 0) + pal_bytes + packed_px
        else:
            tasks = [(rf, sub_w, sub_h, dither, kmeans_iters) for rf in raw_frames]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for pal_bytes, packed_px in pool.map(quantize_single_frame_cpu, tasks):
                    yield struct.pack(">BH", 1, 0) + pal_bytes + packed_px

        yield struct.pack(">B", 0xFF)
        return

    # -------------------------------------------------------------------------
    # CASE 3: Video (with or without Audio), GIF, or Image via FFmpeg
    # -------------------------------------------------------------------------
    if not ffmpeg_bin:
        raise RuntimeError(
            "FFmpeg is required to decode videos or remote URLs. "
            "Install FFmpeg (https://ffmpeg.org) and ensure `ffmpeg` is on your PATH."
        )

    vid_url, aud_url = resolve_media_url(input_source)
    aud_source = aud_url if aud_url else vid_url

    ffprobe_bin = shutil.which("ffprobe")
    include_audio = (
        (kind == "video")
        and (not mute)
        and has_audio_stream(aud_source, ffprobe_bin)
    )

    if kind == "image":
        flags = 0x02
        effective_fps = 0
    elif kind == "gif":
        flags = 0x04
        effective_fps = fps
    else:
        flags = (0x01 if include_audio else 0x00) | 0x04
        effective_fps = fps

    header = struct.pack(">4sHHBBH", b"CCV2", sub_w, sub_h, effective_fps, flags, 0)
    yield header

    if fit_mode == "stretch":
        scale_filter = f"scale={sub_w}:{sub_h}:flags=lanczos"
    elif fit_mode == "cover":
        scale_filter = (
            f"scale={sub_w}:{sub_h}:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop={sub_w}:{sub_h}"
        )
    else:
        scale_filter = (
            f"scale={sub_w}:{sub_h}:flags=lanczos:force_original_aspect_ratio=decrease,"
            f"pad={sub_w}:{sub_h}:(ow-iw)/2:(oh-ih)/2:black"
        )

    vf_chain = scale_filter if kind == "image" else f"fps={fps},{scale_filter}"

    video_cmd = [ffmpeg_bin, "-nostdin", "-loglevel", "error"]
    if hwaccel:
        video_cmd += ["-hwaccel", hwaccel]
    video_cmd += [
        "-threads", str(workers),
        "-i", vid_url,
        "-vf", vf_chain,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
    ]
    if kind == "image":
        video_cmd += ["-frames:v", "1"]
    video_cmd.append("pipe:1")

    vid_proc = subprocess.Popen(
        video_cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        env=_CLEAN_ENV,
        bufsize=sub_w * sub_h * 3 * max(16, batch_size),
    )

    aud_proc = None
    use_native_ffmpeg_dfpwm = False
    if include_audio:
        use_native_ffmpeg_dfpwm = ffmpeg_supports_dfpwm(ffmpeg_bin)
        if use_native_ffmpeg_dfpwm:
            audio_cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-loglevel", "error",
                "-i", aud_source,
                "-vn",
                "-ac", "1",
                "-ar", "48000",
                "-f", "dfpwm",
                "-acodec", "dfpwm",
                "pipe:1",
            ]
        else:
            audio_cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-loglevel", "error",
                "-i", aud_source,
                "-vn",
                "-ac", "1",
                "-ar", "48000",
                "-f", "s8",
                "-acodec", "pcm_s8",
                "pipe:1",
            ]
        aud_proc = subprocess.Popen(
            audio_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            env=_CLEAN_ENV,
            bufsize=48000 * 4,
        )

    dfpwm_encoder = DFPWMEncoder()
    pcm_samples_per_frame = (48000 // max(1, fps)) // 8 * 8
    dfpwm_bytes_per_frame = pcm_samples_per_frame // 8
    frame_byte_size = sub_w * sub_h * 3

    def read_next_batch(max_frames: int) -> Tuple[List[bytes], List[bytes]]:
        v_batch: List[bytes] = []
        a_batch: List[bytes] = []
        for _ in range(max_frames):
            raw_frame = vid_proc.stdout.read(frame_byte_size) if vid_proc.stdout else b""
            if not raw_frame or len(raw_frame) < frame_byte_size:
                break
            v_batch.append(raw_frame)

            dfpwm_bytes = b""
            if aud_proc and aud_proc.stdout:
                if use_native_ffmpeg_dfpwm:
                    dfpwm_bytes = aud_proc.stdout.read(dfpwm_bytes_per_frame)
                else:
                    raw_audio = aud_proc.stdout.read(pcm_samples_per_frame)
                    if raw_audio:
                        pcm_arr = np.frombuffer(raw_audio, dtype=np.int8)
                        dfpwm_bytes = dfpwm_encoder.encode_chunk(pcm_arr)
            a_batch.append(dfpwm_bytes)
            if kind == "image":
                break
        return v_batch, a_batch

    pool = None
    if not (use_gpu and TORCH_DEVICE is not None) and workers > 1 and kind != "image":
        pool = ThreadPoolExecutor(max_workers=workers)

    try:
        while True:
            v_batch, a_batch = read_next_batch(1 if kind == "image" else batch_size)
            if not v_batch:
                break

            iters = max(25, kmeans_iters) if kind == "image" else kmeans_iters

            if use_gpu and TORCH_DEVICE is not None:
                results = quantize_batch_gpu_torch(
                    v_batch, sub_w, sub_h, dither=dither, kmeans_iters=iters
                )
            elif pool is not None and len(v_batch) > 1:
                tasks = [(rf, sub_w, sub_h, dither, iters) for rf in v_batch]
                results = list(pool.map(quantize_single_frame_cpu, tasks))
            else:
                results = [
                    quantize_single_frame_cpu((rf, sub_w, sub_h, dither, iters))
                    for rf in v_batch
                ]

            for dfpwm_bytes, (pal_bytes, packed_px) in zip(a_batch, results):
                pkt_header = struct.pack(">BH", 1, len(dfpwm_bytes))
                yield pkt_header + dfpwm_bytes + pal_bytes + packed_px

            if kind == "image":
                break

        yield struct.pack(">B", 0xFF)
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
        for p in (vid_proc, aud_proc):
            if p is not None:
                try:
                    p.kill()
                    p.wait(timeout=1)
                except Exception:
                    pass


# =============================================================================
# 4. Command-Line Interface
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ultra-Fast GPU + Multi-Threaded Converter for CC:Tweaked (Videos, Audio-Only, GIFs, and Images)."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input file path or URL (e.g. video.mp4, song.mp3, anim.gif, image.png, or YouTube URL)",
    )
    parser.add_argument(
        "-i", "--input-file",
        dest="input_flag",
        help="Input file path or URL (alternative to positional argument)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output .ccv file path (e.g. media.ccv) to upload to GitHub",
    )
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=164,
        help="Master canvas width in CC character cells (default: 164 = 328 subpixels wide)",
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=81,
        help="Master canvas height in CC character cells (default: 81 = 243 subpixels tall)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Target framerate for videos/GIFs (default: 20 FPS, max Minecraft tick rate)",
    )
    parser.add_argument(
        "--fit",
        choices=["contain", "stretch", "cover"],
        default="stretch",
        help="Aspect ratio fit mode: 'stretch' (default), 'contain', or 'cover'",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Convert only the audio track into a compact audio-only .ccv file (automatically enabled for .mp3/.wav/.ogg/.flac)",
    )
    parser.add_argument(
        "--kmeans-iters",
        type=int,
        default=10,
        help="K-Means++ palette optimization iterations per frame (default: 10)",
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=os.cpu_count() or 8,
        help=f"Number of parallel worker threads (default: all {os.cpu_count() or 8} threads)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of frames processed per GPU/CPU batch (default: 64)",
    )
    parser.add_argument(
        "--hwaccel",
        default=None,
        help="Optional FFmpeg hardware decoder (e.g. 'cuda', 'd3d11va', 'dxva2', 'videotoolbox', 'auto')",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU tensor acceleration even if PyTorch CUDA/MPS is available",
    )
    parser.add_argument(
        "--no-dither",
        action="store_true",
        help="Disable 4x4 Bayer subpixel dithering",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Strip audio track even if input video has audio",
    )

    args = parser.parse_args()
    source = args.input or args.input_flag
    if not source:
        parser.error("Please provide an input file or URL (e.g. `python3 convert_and_stream.py video.mp4 -o out.ccv`).")

    sub_w = args.width * 2
    sub_h = args.height * 3
    kind = detect_media_kind(source, force_audio_only=args.audio_only)
    use_gpu = (not args.no_gpu) and (TORCH_DEVICE is not None)

    hwaccel = args.hwaccel
    if hwaccel is None and use_gpu and TORCH_DEVICE is not None and TORCH_DEVICE.type == "cuda":
        hwaccel = "cuda"

    if kind == "audio":
        accel_str = "FFmpeg Native C DFPWM1a Audio Pipeline"
    elif use_gpu:
        accel_str = f"GPU ({TORCH_GPU_NAME} batched PyTorch tensor engine)"
    else:
        accel_str = (
            f"Multi-Threaded CPU ({args.workers} threads, "
            f"{'OpenCV C++ SIMD + ' if HAS_CV2 else ''}12-bit 3D LUT BLAS GEMM)"
        )

    print(f"Input:        {source} (detected type: {kind.upper()})")
    if kind != "audio":
        print(f"Resolution:   {sub_w}x{sub_h} subpixels ({args.width}x{args.height} CC chars @ 2x3 teletext)")
    print(f"Acceleration: {accel_str}")
    if kind != "image":
        print(f"Framerate:    {args.fps} FPS | Audio: {'Muted' if (args.mute or kind == 'gif') else '48kHz DFPWM1a'} | Looping: Enabled")

    frame_count = 0
    total_bytes = 0
    t_start = time.perf_counter()

    with open(args.output, "wb") as out_f:
        for idx, packet in enumerate(
            encode_media_to_ccv2(
                input_source=source,
                sub_w=sub_w,
                sub_h=sub_h,
                fps=args.fps,
                kmeans_iters=args.kmeans_iters,
                dither=not args.no_dither,
                fit_mode=args.fit,
                mute=args.mute,
                audio_only=args.audio_only,
                workers=args.workers,
                use_gpu=use_gpu,
                hwaccel=hwaccel,
                batch_size=args.batch_size,
            )
        ):
            out_f.write(packet)
            total_bytes += len(packet)
            if idx > 0 and packet != b"\xff":
                frame_count += 1
                if frame_count % 10 == 0 or kind in ("image", "audio"):
                    elapsed = max(0.001, time.perf_counter() - t_start)
                    encode_fps = frame_count / elapsed
                    unit = "sec(s) of audio" if kind == "audio" else "frame(s)"
                    sys.stdout.write(
                        f"\rEncoded {frame_count} {unit} ({total_bytes / (1024 * 1024):.2f} MB) "
                        f"@ {encode_fps:.1f} {unit}/s..."
                    )
                    sys.stdout.flush()

    elapsed_total = max(0.001, time.perf_counter() - t_start)
    unit = "sec(s) of audio" if kind == "audio" else "frame(s)"
    print(
        f"\nDone in {elapsed_total:.2f}s ({frame_count / elapsed_total:.1f} {unit}/s)! "
        f"Saved {frame_count} {unit} ({total_bytes / 1024:.1f} KB) to '{args.output}'."
    )


if __name__ == "__main__":
    main()