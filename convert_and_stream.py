#!/usr/bin/env python3
"""
convert_and_stream.py
Local High-Quality Media Converter for CC:Tweaked (Videos with Audio, GIFs, and Static Images)

Designed for workflows where you convert media locally on your PC, upload the resulting `.ccv`
file to GitHub (or any static host), and stream/play it directly in Minecraft (e.g., on PebbleHost)
using `player.lua`—with `player.lua` dynamically scaling the subpixel frame to fit ANY monitor size!

Supported Media Types:
1. Videos with Audio (.mp4, .mkv, .webm, .mov, .avi, or YouTube/Web URLs via yt-dlp):
   - Encoded at 20 FPS (Minecraft's maximum 20 ticks/sec hardware refresh rate).
   - Synchronized 48,000 Hz DFPWM1a 1-bit audio (300 bytes/frame) across all attached speakers,
     encoded using FFmpeg's native `-acodec dfpwm` (with an exact reference DFPWM1a fallback).
   - Loops automatically in `player.lua` unless `--once` is passed!
2. Animated GIFs / Silent Animations (.gif, .webp, .apng):
   - Encoded at 20 FPS (or custom `--fps`), automatically muted (`audio_mode = 0`), with looping.
3. Static Images (.png, .jpg, .jpeg, .bmp, .tiff):
   - Automatically detected as a 1-frame static image (`media_type = 2`) with extra K-Means++
     iterations for maximum color accuracy.

Why `player.lua` Can Scale Static GitHub Files to ANY Monitor Size:
- Each frame in the `.ccv` file stores a compact 4-bit-per-subpixel index grid (`subpixel_w x subpixel_h`
  defaulting to 328x243, the maximum 8x6 CC:Tweaked monitor resolution at 0.5x text scale) plus its
  16-color RGB palette.
- When `player.lua` downloads the file from GitHub, it detects its own monitor's exact dimensions,
  resamples the subpixel grid to `(monitor_char_w * 2) x (monitor_char_h * 3)` on the fly, and packs
  the 2x3 teletext characters (128..159) in real time!

Usage Examples:
  # Convert a video (local file or URL) with audio:
  python3 convert_and_stream.py video.mp4 -o video.ccv

  # Convert an animated GIF:
  python3 convert_and_stream.py animation.gif -o anim.ccv

  # Convert a static image (PNG/JPG):
  python3 convert_and_stream.py photo.png -o photo.ccv
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image, ImageSequence


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
        """
        Encode an array of signed 8-bit PCM samples (-128..127) into DFPWM1a bytes.
        Length of `pcm_s8` must be a multiple of 8.
        """
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
                # Set target (+127 for 1-bit, -128 for 0-bit)
                t_val = 127 if (v > q or (v == q and v == 127)) else -128
                if t_val == 127:
                    d |= (1 << bit_i)

                # Adjust charge (q)
                nq = q + ((s * (t_val - q) + 512) >> 10)
                if nq == q and nq != t_val:
                    nq += 1 if t_val == 127 else -1
                q = nq

                # Adjust strength/response (s): DFPWM1a increments/decrements by 1 with floor 8
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
            [ffmpeg_bin, "-hide_banner", "-codecs"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in proc.stdout.splitlines():
            if "dfpwm" in line and "E" in line[:8]:
                return True
    except Exception:
        pass
    return False


# =============================================================================
# 2. Perceptual K-Means++ 16-Color Quantizer & Subpixel Index Packer
# =============================================================================

def quantize_palette_kmeans(
    frame_rgb: np.ndarray,
    n_colors: int = 16,
    max_iter: int = 16,
    sample_limit: int = 8000,
) -> np.ndarray:
    """
    Find the optimal 16-color palette for `frame_rgb` (H, W, 3) using perceptual
    weighted K-Means++ clustering. Returns array of shape (16, 3) uint8.
    """
    pixels = frame_rgb.reshape(-1, 3).astype(np.float32)
    n_pixels = pixels.shape[0]

    if n_pixels > sample_limit:
        step = max(1, n_pixels // sample_limit)
        sample = pixels[::step]
    else:
        sample = pixels

    # Perceptual RGB weights (approximating CIELAB luminance/chrominance sensitivity)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sqrt_w = np.sqrt(weights)
    w_sample = sample * sqrt_w

    # Deterministic K-Means++ initialization for smooth frame-to-frame stability
    centers = np.empty((n_colors, 3), dtype=np.float32)
    centers[0] = w_sample[len(w_sample) // 2]
    dist_sq = np.sum((w_sample - centers[0]) ** 2, axis=1)

    for c in range(1, n_colors):
        idx = int(np.argmax(dist_sq))
        centers[c] = w_sample[idx]
        new_dist_sq = np.sum((w_sample - centers[c]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)

    for _ in range(max_iter):
        d2 = np.sum((w_sample[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new_centers = np.copy(centers)
        for c in range(n_colors):
            mask = labels == c
            if np.any(mask):
                new_centers[c] = w_sample[mask].mean(axis=0)
        if np.allclose(centers, new_centers, atol=0.4):
            centers = new_centers
            break
        centers = new_centers

    rgb_palette = np.clip(np.round(centers / sqrt_w), 0, 255).astype(np.uint8)

    # Sort palette by perceptual luminance so indices stay coherent across frames
    luma = (
        0.299 * rgb_palette[:, 0]
        + 0.587 * rgb_palette[:, 1]
        + 0.114 * rgb_palette[:, 2]
    )
    order = np.argsort(luma)
    return rgb_palette[order]


def quantize_frame_to_4bit_subpixels(
    frame_rgb: np.ndarray,
    dither: bool = True,
    kmeans_iters: int = 16,
) -> Tuple[bytes, bytes]:
    """
    Quantize an RGB frame of shape (sub_h, sub_w, 3) into:
      1. `palette_bytes` (48 bytes = 16 RGB triplets)
      2. `packed_indices` (ceil(sub_w * sub_h / 2) bytes, packing two 4-bit color
         indices 0..15 per byte: high nibble = even pixel, low nibble = odd pixel).

    Storing the full 4-bit subpixel grid allows `player.lua` to dynamically scale
    any pre-converted `.ccv` file hosted on GitHub to ANY monitor size in-game!
    """
    palette = quantize_palette_kmeans(frame_rgb, n_colors=16, max_iter=kmeans_iters)
    pal_f = palette.astype(np.float32)

    if dither:
        # 4x4 Bayer ordered dithering in perceptual space for crisp, artifact-free gradients
        bayer4 = (
            np.array(
                [
                    [0, 8, 2, 10],
                    [12, 4, 14, 6],
                    [3, 11, 1, 9],
                    [15, 7, 13, 5],
                ],
                dtype=np.float32,
            )
            / 16.0
            - 0.5
        )
        h_px, w_px, _ = frame_rgb.shape
        tiled_bayer = np.tile(bayer4, (h_px // 4 + 1, w_px // 4 + 1))[:h_px, :w_px, None]
        work_rgb = np.clip(frame_rgb.astype(np.float32) + tiled_bayer * 10.0, 0, 255)
    else:
        work_rgb = frame_rgb.astype(np.float32)

    flat_pixels = work_rgb.reshape(-1, 3)
    w_rgb = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # Compute perceptual distance to all 16 palette colors in chunks to keep RAM low
    n_px = flat_pixels.shape[0]
    indices = np.empty(n_px, dtype=np.uint8)
    chunk_size = 32768
    for start in range(0, n_px, chunk_size):
        end = min(n_px, start + chunk_size)
        sub = flat_pixels[start:end]
        dists = np.sum(((sub[:, None, :] - pal_f[None, :, :]) ** 2) * w_rgb, axis=2)
        indices[start:end] = np.argmin(dists, axis=1).astype(np.uint8)

    # Pack 2 pixels per byte (high nibble = pixel 2*i, low nibble = pixel 2*i + 1)
    if n_px % 2 == 1:
        indices = np.append(indices, np.uint8(0))
    packed = (indices[0::2] << 4) | (indices[1::2] & 0x0F)
    return palette.tobytes(), packed.tobytes()


# =============================================================================
# 3. Media Type Detection & Decoders (Static Image, Animated GIF, Video + Audio)
# =============================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
GIF_EXTENSIONS = {".gif", ".apng"}


def resize_with_aspect_fit(
    img: Image.Image, target_w: int, target_h: int, fit_mode: str = "contain"
) -> np.ndarray:
    """
    Resize a PIL Image to (target_w, target_h) using high-quality Lanczos resampling.
    - `contain`: preserves aspect ratio with black letterboxing/pillarboxing.
    - `stretch`: fills the exact target subpixel dimensions.
    - `cover`: crops to fill the target subpixel dimensions while preserving aspect ratio.
    """
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


def detect_media_kind(input_source: str) -> str:
    """
    Return 'image', 'gif', or 'video' based on file extension and PIL inspection.
    """
    if input_source.startswith(("http://", "https://")):
        parsed_path = input_source.split("?")[0].lower()
        ext = os.path.splitext(parsed_path)[1]
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in GIF_EXTENSIONS:
            return "gif"
        return "video"

    ext = os.path.splitext(input_source)[1].lower()
    if ext in IMAGE_EXTENSIONS or ext in GIF_EXTENSIONS or ext == ".webp":
        try:
            with Image.open(input_source) as im:
                is_anim = getattr(im, "is_animated", False) and getattr(im, "n_frames", 1) > 1
                return "gif" if is_anim else "image"
        except Exception:
            pass
    return "video"


def resolve_media_url(url: str) -> Tuple[str, Optional[str]]:
    """
    If `yt-dlp` is installed and `url` is a web video URL, resolve direct video/audio streams.
    """
    ytdlp_bin = shutil.which("yt-dlp")
    if ytdlp_bin and url.startswith(("http://", "https://")):
        try:
            proc = subprocess.run(
                [ytdlp_bin, "-g", "-f", "bestvideo+bestaudio/best", "--no-playlist", url],
                capture_output=True,
                text=True,
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
    """Check if the input video file/URL contains an audio track."""
    if not ffprobe_bin:
        return True
    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                input_path,
            ],
            capture_output=True,
            text=True,
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
    kmeans_iters: int = 16,
    dither: bool = True,
    fit_mode: str = "contain",
    mute: bool = False,
) -> Iterator[bytes]:
    """
    Encode a static image, animated GIF, or video (with or without audio) into CCV2 packets.

    CCV2 Binary Format Specification:
      Header (12 bytes):
        - Magic: "CCV2" (4 bytes)
        - Source Subpixel Width (`sub_w`): uint16 big-endian (always even)
        - Source Subpixel Height (`sub_h`): uint16 big-endian
        - FPS: uint8 (0 for static image, 20 for video/GIF)
        - Flags: uint8
            bit 0 (val 1): Has 48kHz DFPWM Audio
            bit 1 (val 2): Is Static Image (single frame, hold on screen)
            bit 2 (val 4): Is Looping Media (set for both Videos and GIFs so they loop by default)
        - Reserved: uint16 (0)

      Per-Frame Packet:
        - Chunk Type: uint8 (1 = Frame, 0xFF = End of Stream)
        - Audio Length: uint16 big-endian (bytes of DFPWM audio for this frame; 0 if silent)
        - Audio Data: [Audio Length] bytes
        - Palette Data: 48 bytes (16 colors x 3 RGB bytes)
        - Subpixel 4-bit Grid: `(sub_w * sub_h) // 2` bytes (2 subpixels per byte, row-major)
    """
    # Ensure sub_w is even so each row has an exact integer number of packed bytes
    if sub_w % 2 != 0:
        sub_w += 1

    kind = detect_media_kind(input_source)

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
    ffmpeg_bin = shutil.which("ffmpeg")
    if kind == "gif" and not ffmpeg_bin and not input_source.startswith(("http://", "https://")):
        flags = 0x04  # bit 2 = Looping Animation
        header = struct.pack(">4sHHBBH", b"CCV2", sub_w, sub_h, fps, flags, 0)
        yield header

        with Image.open(input_source) as img:
            for frame in ImageSequence.Iterator(img):
                frame_np = resize_with_aspect_fit(frame, sub_w, sub_h, fit_mode=fit_mode)
                pal_bytes, packed_px = quantize_frame_to_4bit_subpixels(
                    frame_np, dither=dither, kmeans_iters=kmeans_iters
                )
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
        # Set bit 2 (0x04 = loop) on videos as well so videos loop automatically!
        flags = (0x01 if include_audio else 0x00) | 0x04
        effective_fps = fps

    header = struct.pack(">4sHHBBH", b"CCV2", sub_w, sub_h, effective_fps, flags, 0)
    yield header

    # Build FFmpeg Lanczos scaling filter
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

    video_cmd = [
        ffmpeg_bin,
        "-loglevel", "error",
        "-i", vid_url,
        "-vf", vf_chain,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
    ]
    if kind == "image":
        video_cmd += ["-frames:v", "1"]
    video_cmd.append("pipe:1")

    vid_proc = subprocess.Popen(
        video_cmd, stdout=subprocess.PIPE, bufsize=sub_w * sub_h * 3 * 4
    )

    aud_proc = None
    use_native_ffmpeg_dfpwm = False
    if include_audio:
        use_native_ffmpeg_dfpwm = ffmpeg_supports_dfpwm(ffmpeg_bin)
        if use_native_ffmpeg_dfpwm:
            # FFmpeg 5.1+ has a native C DFPWM1a encoder (`-f dfpwm -acodec dfpwm`)
            audio_cmd = [
                ffmpeg_bin,
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
                "-loglevel", "error",
                "-i", aud_source,
                "-vn",
                "-ac", "1",
                "-ar", "48000",
                "-f", "s8",
                "-acodec", "pcm_s8",
                "pipe:1",
            ]
        aud_proc = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, bufsize=48000 * 2)

    dfpwm_encoder = DFPWMEncoder()
    pcm_samples_per_frame = (48000 // max(1, fps)) // 8 * 8
    dfpwm_bytes_per_frame = pcm_samples_per_frame // 8
    frame_byte_size = sub_w * sub_h * 3

    try:
        while True:
            raw_frame = vid_proc.stdout.read(frame_byte_size) if vid_proc.stdout else b""
            if not raw_frame or len(raw_frame) < frame_byte_size:
                break

            dfpwm_bytes = b""
            if aud_proc and aud_proc.stdout:
                if use_native_ffmpeg_dfpwm:
                    dfpwm_bytes = aud_proc.stdout.read(dfpwm_bytes_per_frame)
                else:
                    raw_audio = aud_proc.stdout.read(pcm_samples_per_frame)
                    if raw_audio:
                        pcm_arr = np.frombuffer(raw_audio, dtype=np.int8)
                        dfpwm_bytes = dfpwm_encoder.encode_chunk(pcm_arr)

            frame_np = np.frombuffer(raw_frame, dtype=np.uint8).reshape((sub_h, sub_w, 3))
            pal_bytes, packed_px = quantize_frame_to_4bit_subpixels(
                frame_np,
                dither=dither,
                kmeans_iters=max(25, kmeans_iters) if kind == "image" else kmeans_iters,
            )

            pkt_header = struct.pack(">BH", 1, len(dfpwm_bytes))
            yield pkt_header + dfpwm_bytes + pal_bytes + packed_px

            if kind == "image":
                break

        yield struct.pack(">B", 0xFF)
    finally:
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
            "Convert Videos (with Audio), Animated GIFs, or Static Images into a single "
            "high-density .ccv file to upload to GitHub and play in CC:Tweaked."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input file path or URL (e.g. video.mp4, anim.gif, image.png, or YouTube URL)",
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
        help=(
            "Master canvas width in CC character cells (default: 164 = 328 subpixels wide, "
            "matching the maximum 8x6 monitor at 0.5x scale; player.lua auto-scales this "
            "to any monitor size in-game!)"
        ),
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=81,
        help=(
            "Master canvas height in CC character cells (default: 81 = 243 subpixels tall, "
            "matching the maximum 8x6 monitor at 0.5x scale; player.lua auto-scales this "
            "to any monitor size in-game!)"
        ),
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
        help=(
            "How to fit source aspect ratio into master canvas: "
            "'stretch' (default, fills entire subpixel grid so player.lua can scale to full monitor), "
            "'contain' (letterbox), or 'cover' (crop)"
        ),
    )
    parser.add_argument(
        "--kmeans-iters",
        type=int,
        default=16,
        help="K-Means++ palette optimization iterations per frame (default: 16)",
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
    kind = detect_media_kind(source)

    print(f"Input:      {source} (detected type: {kind.upper()})")
    print(f"Resolution: {sub_w}x{sub_h} subpixels ({args.width}x{args.height} CC chars @ 2x3 teletext)")
    if kind != "image":
        print(f"Framerate:  {args.fps} FPS | Audio: {'Muted' if (args.mute or kind == 'gif') else '48kHz DFPWM1a'} | Looping: Enabled")

    frame_count = 0
    total_bytes = 0
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
            )
        ):
            out_f.write(packet)
            total_bytes += len(packet)
            if idx > 0 and packet != b"\xff":
                frame_count += 1
                if frame_count % 10 == 0 or kind == "image":
                    sys.stdout.write(
                        f"\rEncoded {frame_count} frame(s) ({total_bytes / (1024 * 1024):.2f} MB)..."
                    )
                    sys.stdout.flush()

    print(f"\nDone! Saved {frame_count} frame(s) ({total_bytes / 1024:.1f} KB) to '{args.output}'.")
    print("\nNext steps:")
    print(f"  1. Upload '{args.output}' to a GitHub repository.")
    print("  2. Copy the 'Raw' URL (e.g. https://raw.githubusercontent.com/<user>/<repo>/main/" + os.path.basename(args.output) + ")")
    print("  3. On your PebbleHost CC:Tweaked computer, run:")
    print('     player "https://raw.githubusercontent.com/<user>/<repo>/main/' + os.path.basename(args.output) + '"')


if __name__ == "__main__":
    main()