"""Standalone quality evaluation and PDF reporting for AdFidelity outputs.

This script does not import or change the generation pipeline.  Run it after
an image or video has been generated, supplying an optional reference asset
when a full-reference metric (PSNR, SSIM, LPIPS, FVD) is required.

Examples
--------
Image evaluation:
  python eval/generate_metrics_report.py --image outputs/ad.png \
      --reference-image data/reference.png --prompt "..." --generation-time 12.4

Video evaluation:
  python eval/generate_metrics_report.py --video outputs/animation.mp4 \
      --reference-video data/reference.mp4 --source-image outputs/ad.png \
      --prompt "..." --generation-time 48.2 --inference-cost 0.12

Optional packages unlock research metrics (the script still runs without them):
  pip install torch torchvision transformers lpips torchmetrics pyiqa

Metric conventions
------------------
* PSNR, SSIM, LPIPS and FVD compare generated output to a matched reference.
* CLIP prompt adherence uses CLIP when transformers is installed.
* FVD is calculated only through torchmetrics' I3D implementation.  A proxy
  is deliberately not labelled FVD, so the report is suitable for a thesis.
* User-study/advertising values are intentionally recorded as "Not measured";
  they need participant or analytics data and cannot be inferred from a file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _missing(reason: str) -> dict[str, Any]:
    return {"value": None, "status": "Not measured", "reason": reason}


def _value(value: Any, unit: str = "", note: str = "") -> dict[str, Any]:
    return {"value": value, "unit": unit, "status": "Measured", "note": note}


def load_image(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM, averaged across RGB channels; dependency-free implementation."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    scores = []
    for channel in range(3):
        x, y = a[:, :, channel], b[:, :, channel]
        mux, muy = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - mux) * (y - muy)).mean()
        scores.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))
    return float(np.mean(scores))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(20 * math.log10(255.0 / math.sqrt(mse)))


def no_reference_quality(image: np.ndarray) -> dict[str, Any]:
    """NIQE and BRISQUE using pyiqa; both are no-reference quality metrics."""
    try:
        import pyiqa  # type: ignore
        import torch  # type: ignore
    except ImportError:
        return {
            "niqe": _missing("Install optional package: pip install pyiqa"),
            "brisque": _missing("Install optional package: pip install pyiqa"),
        }
    try:
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        return {
            "niqe": _value(round(float(pyiqa.create_metric("niqe")(tensor).item()), 4), "lower is better"),
            "brisque": _value(round(float(pyiqa.create_metric("brisque")(tensor).item()), 4), "lower is better"),
        }
    except Exception as exc:
        return {"niqe": _missing(f"NIQE failed: {exc}"), "brisque": _missing(f"BRISQUE failed: {exc}")}


def aesthetic_proxy(image: np.ndarray) -> dict[str, Any]:
    """A transparent, lightweight composition/exposure/sharpness proxy (not a learned aesthetic model)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sharp = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
    exposure = max(0.0, 1.0 - abs(float(gray.mean()) - 127.5) / 127.5)
    saturation = min(float(hsv[:, :, 1].mean()) / 100.0, 1.0)
    score = 100.0 * (0.45 * sharp + 0.35 * exposure + 0.20 * saturation)
    return _value(round(score, 2), "0-100", "Heuristic aesthetic proxy from sharpness, exposure and colour; do not present as a learned aesthetic score.")


def image_quality(image_path: str, reference_path: str | None) -> dict[str, Any]:
    image = load_image(image_path)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result: dict[str, Any] = {
        "resolution": _value(f"{width} x {height}", "pixels"),
        "sharpness_variance_laplacian": _value(round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2), "higher is sharper"),
        "aesthetic_quality_proxy": aesthetic_proxy(image),
    }
    result.update(no_reference_quality(image))
    if not reference_path:
        result.update({
            "psnr": _missing("A matched --reference-image is required."),
            "ssim": _missing("A matched --reference-image is required."),
            "lpips": _missing("A matched --reference-image is required."),
        })
        return result
    reference = load_image(reference_path, (width, height))
    score = psnr(image, reference)
    result["psnr"] = _value("∞" if math.isinf(score) else round(score, 3), "dB")
    result["ssim"] = _value(round(ssim(image, reference), 5), "0–1")
    result["lpips"] = lpips_score(image, reference)
    return result


def lpips_score(image: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    try:
        import lpips  # type: ignore
        import torch  # type: ignore
    except ImportError:
        return _missing("Install optional packages: pip install torch lpips")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = lpips.LPIPS(net="alex").to(device).eval()
        def tensor(x: np.ndarray):
            return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 127.5 - 1
        with torch.no_grad():
            score = model(tensor(image), tensor(reference)).item()
        return _value(round(float(score), 5), "lower is better", "AlexNet LPIPS")
    except Exception as exc:
        return _missing(f"LPIPS failed: {exc}")


def clip_prompt_score(image_path: str, prompt: str | None) -> dict[str, Any]:
    if not prompt:
        return _missing("Provide --prompt to measure prompt adherence.")
    try:
        import torch  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except ImportError:
        return _missing("Install optional packages: pip install torch transformers")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_id)
        inputs = processor(text=[prompt], images=Image.open(image_path).convert("RGB"), return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            score = model(**inputs).logits_per_image[0, 0].item()
        return _value(round(float(score), 3), "CLIP logit", "Higher means stronger image–prompt alignment; compare within one study.")
    except Exception as exc:
        return _missing(f"CLIP evaluation failed: {exc}")


def video_frames(path: str, count: int = 16) -> tuple[list[np.ndarray], float, int, int, int]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"Cannot read video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    positions = np.linspace(0, max(0, total - 1), min(count, max(1, total))).astype(int)
    frames = []
    for pos in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = capture.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, fps, total, width, height


def fvd_score(video_path: str, reference_path: str | None) -> dict[str, Any]:
    if not reference_path:
        return _missing("A matched --reference-video is required.")
    try:
        import torch  # type: ignore
        from torchmetrics.video.fvd import FrechetVideoDistance  # type: ignore
    except ImportError:
        return _missing("Install optional packages: pip install torch torchmetrics[video]")
    try:
        generated, *_ = video_frames(video_path, 16)
        reference, *_ = video_frames(reference_path, 16)
        if len(generated) < 2 or len(reference) < 2:
            return _missing("Each video needs at least two readable frames.")
        n = min(len(generated), len(reference))
        shape = (224, 224)
        def tensor(frames: list[np.ndarray]):
            resized = np.stack([cv2.resize(frame, shape) for frame in frames[:n]])
            return torch.from_numpy(resized).permute(0, 3, 1, 2).unsqueeze(0)
        metric = FrechetVideoDistance(feature=400)
        metric.update(tensor(reference), real=True)
        metric.update(tensor(generated), real=False)
        return _value(round(float(metric.compute().item()), 4), "lower is better", "I3D FVD; meaningful for sets of videos, not one pair.")
    except Exception as exc:
        return _missing(f"FVD failed: {exc}")


def _center_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    y0, y1 = int(height * 0.20), int(height * 0.80)
    x0, x1 = int(width * 0.25), int(width * 0.75)
    return frame[y0:y1, x0:x1]


def _border_region(frame: np.ndarray) -> np.ndarray:
    """Keeps the outside border; center is replaced with its mean colour."""
    output = frame.copy()
    height, width = frame.shape[:2]
    y0, y1 = int(height * 0.20), int(height * 0.80)
    x0, x1 = int(width * 0.25), int(width * 0.75)
    output[y0:y1, x0:x1] = output.mean(axis=(0, 1), keepdims=True)
    return output


def video_stability_metrics(frames: list[np.ndarray]) -> dict[str, Any]:
    """Reference-free temporal measurements on sampled adjacent frames.

    Subject/background values are central-crop/border proxies. They are useful
    for a controlled, centred advertisement setup, but are not segmentation or
    face-recognition claims.
    """
    if len(frames) < 2:
        return {key: _missing("At least two readable video frames are required.") for key in (
            "temporal_flicker", "motion_smoothness_optical_flow", "subject_consistency_proxy", "background_consistency_proxy")}
    flicker, flows, subject, background = [], [], [], []
    previous_flow = None
    for before, after in zip(frames[:-1], frames[1:]):
        before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(before_gray, after_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # Median motion suppresses intended local movement and emphasizes broad flicker.
        flicker.append(float(np.mean(np.abs(after_gray.astype(float) - before_gray.astype(float))) / 255.0))
        if previous_flow is not None:
            flows.append(float(np.mean(np.linalg.norm(flow - previous_flow, axis=2))))
        previous_flow = flow
        subject.append(ssim(_center_crop(before), _center_crop(after)))
        background.append(ssim(_border_region(before), _border_region(after)))
    return {
        "temporal_flicker": _value(round(float(np.mean(flicker)), 5), "0-1, lower is better", "Mean adjacent-frame luminance change; intended motion can increase this value."),
        "motion_smoothness_optical_flow": _value(round(float(np.mean(flows)), 5) if flows else 0.0, "pixels/frame, lower is smoother", "Mean change in dense optical flow between adjacent samples."),
        "subject_consistency_proxy": _value(round(float(np.mean(subject)), 5), "0-1, higher is more stable", "Central 50% x 60% crop SSIM proxy; assumes the person is centred."),
        "background_consistency_proxy": _value(round(float(np.mean(background)), 5), "0-1, higher is more stable", "Outer-border SSIM proxy; assumes a mostly fixed camera."),
    }


def video_quality(video_path: str, reference_path: str | None) -> dict[str, Any]:
    frames, fps, total, width, height = video_frames(video_path)
    duration = total / fps if fps else 0
    consecutive = [ssim(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    result: dict[str, Any] = {
        "resolution": _value(f"{width} x {height}", "pixels"),
        "duration": _value(round(duration, 3), "seconds"),
        "frame_rate": _value(round(fps, 3), "FPS"),
        "sampled_frames": _value(len(frames), "frames"),
        "scene_consistency_temporal_ssim": _value(round(float(np.mean(consecutive)), 5) if consecutive else None, "0–1", "Adjacent sampled frames; high values indicate stable appearance, not necessarily realism."),
        "fvd": fvd_score(video_path, reference_path),
    }
    result.update(video_stability_metrics(frames))
    if reference_path:
        ref_frames, *_ = video_frames(reference_path, len(frames))
        n = min(len(frames), len(ref_frames))
        if n:
            aligned = [cv2.resize(ref_frames[i], (frames[i].shape[1], frames[i].shape[0])) for i in range(n)]
            average_psnr = float(np.mean([psnr(frames[i], aligned[i]) for i in range(n)]))
            result["aligned_frame_psnr"] = _value("∞" if math.isinf(average_psnr) else round(average_psnr, 3), "dB")
            result["aligned_frame_ssim"] = _value(round(float(np.mean([ssim(frames[i], aligned[i]) for i in range(n)])), 5), "0–1")
    else:
        result["aligned_frame_psnr"] = _missing("A matched --reference-video is required.")
        result["aligned_frame_ssim"] = _missing("A matched --reference-video is required.")
    return result


def gpu_usage() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return _missing("nvidia-smi is unavailable (common for cloud/API generation).")
    try:
        output = subprocess.check_output([executable, "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=10)
        return _value(output.strip(), "GPU / utilization% / memory MiB", "Snapshot taken when this report ran, not during generation.")
    except Exception as exc:
        return _missing(f"GPU query failed: {exc}")


def metric_line(label: str, data: dict[str, Any]) -> str:
    if data.get("status") != "Measured":
        return f"{label}: Not measured — {data.get('reason', '')}"
    value = data.get("value")
    value = "∞" if value == float("inf") else value
    suffix = f" {data['unit']}" if data.get("unit") else ""
    note = f" ({data['note']})" if data.get("note") else ""
    return f"{label}: {value}{suffix}{note}"


def render_pdf(report: dict[str, Any], path: Path) -> None:
    navy, purple, pink, ink, muted, pale = "#101828", "#5925DC", "#D444F1", "#182230", "#667085", "#F9FAFB"
    font_path = "C:/Windows/Fonts/arial.ttf"
    bold_path = "C:/Windows/Fonts/arialbd.ttf"
    def font(size: int, bold: bool = False):
        try: return ImageFont.truetype(bold_path if bold else font_path, size)
        except OSError: return ImageFont.load_default()
    title_font, h_font, body_font, small_font = font(48, True), font(25, True), font(18), font(14)
    page = Image.new("RGB", (1240, 1754), pale)
    draw = ImageDraw.Draw(page)
    pages = []
    y = 0
    def header(title: str, subtitle: str = "") -> None:
        nonlocal y
        draw.rectangle((0, 0, 1240, 178), fill=navy)
        draw.rectangle((0, 170, 1240, 178), fill=pink)
        draw.text((68, 45), title, fill="white", font=title_font)
        draw.text((70, 116), subtitle, fill="#D0D5DD", font=body_font)
        y = 225
    def footer() -> None:
        draw.line((68, 1680, 1172, 1680), fill="#D0D5DD", width=2)
        draw.text((68, 1700), "AdFidelity | Automated evaluation report", fill=muted, font=small_font)
        draw.text((1040, 1700), f"Page {len(pages) + 1}", fill=muted, font=small_font)
    def new_page(title: str, subtitle: str = "") -> None:
        nonlocal page, draw
        footer(); pages.append(page)
        page = Image.new("RGB", (1240, 1754), pale); draw = ImageDraw.Draw(page); header(title, subtitle)
    def wrapped(text: str, max_width: int, text_font) -> list[str]:
        words, out, current = text.split(), [], ""
        for word in words or [""]:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=text_font) > max_width and current:
                out.append(current); current = word
            else: current = candidate
        return out + [current]
    def section(name: str, metrics: dict[str, Any]) -> None:
        nonlocal y
        if y > 1450: new_page("Metric details", "continued")
        draw.text((68, y), name, fill=purple, font=h_font); y += 46
        for key, data in metrics.items():
            text = metric_line(key.replace("_", " ").title(), data)
            lines = wrapped(text, 990, body_font)
            height = max(46, 22 * len(lines) + 22)
            if y + height > 1650: new_page("Metric details", "continued")
            colour = "#FFFFFF" if data.get("status") == "Measured" else "#F2F4F7"
            draw.rounded_rectangle((68, y, 1172, y + height), radius=12, fill=colour, outline="#EAECF0", width=2)
            draw.ellipse((88, y + 17, 101, y + 30), fill="#12B76A" if data.get("status") == "Measured" else "#98A2B3")
            text_y = y + 13
            for line in lines:
                draw.text((120, text_y), line, fill=ink, font=body_font); text_y += 22
            y += height + 11
        y += 14
    header("AdFidelity", "AI-generated advertisement quality evaluation")
    draw.text((68, y), "Evaluation Report", fill=ink, font=font(34, True)); y += 60
    draw.text((68, y), f"Created  {report['created_at']}", fill=muted, font=body_font); y += 50
    image_path = report.get("assets", {}).get("image")
    if image_path and Path(image_path).is_file():
        try:
            thumb = Image.open(image_path).convert("RGB"); thumb.thumbnail((420, 420))
            draw.rounded_rectangle((68, y, 520, y + 452), radius=18, fill="white", outline="#EAECF0", width=2)
            page.paste(thumb, (84, y + 16))
        except Exception: pass
    draw.rounded_rectangle((560, y, 1172, y + 230), radius=18, fill="#F4F3FF")
    draw.text((592, y + 32), "Research-ready metrics", fill=purple, font=h_font)
    draw.text((592, y + 82), "Prompt adherence, image quality,", fill=ink, font=body_font)
    draw.text((592, y + 110), "temporal stability and efficiency.", fill=ink, font=body_font)
    draw.text((592, y + 166), "Green = measured | Grey = needs data", fill=muted, font=small_font)
    y += 490
    section("Generation metrics", report["generation_metrics"])
    if report.get("image_metrics"): section("Image quality", report["image_metrics"])
    if report.get("video_metrics"): section("Video quality and stability", report["video_metrics"])
    if report.get("ai_specific_metrics"): section("AI-specific metrics", report["ai_specific_metrics"])
    section("Methodology note", {"interpretation": _value("No-reference scores evaluate the generated asset itself. PSNR, SSIM, LPIPS and FVD need valid matched references or a dataset. Report mean and standard deviation over at least 30 prompts.")})
    footer(); pages.append(page)
    pages[0].save(path, "PDF", save_all=True, append_images=pages[1:], resolution=150.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate AdFidelity output metrics and create a PDF report.")
    parser.add_argument("--image", help="Generated image path")
    parser.add_argument("--video", help="Generated video path")
    parser.add_argument("--reference-image", help="Matched reference image path")
    parser.add_argument("--reference-video", help="Matched reference video path")
    parser.add_argument("--source-image", help="Image used to create the video (used for CLIP prompt adherence)")
    parser.add_argument("--prompt", help="Exact generation prompt")
    parser.add_argument("--generation-time", type=float, help="Generation/inference duration in seconds")
    parser.add_argument("--inference-cost", type=float, help="Provider cost in USD")
    parser.add_argument("--output-dir", default="eval/reports", help="Directory for PDF and JSON (default: eval/reports)")
    args = parser.parse_args()
    if not args.image and not args.video:
        parser.error("Provide --image and/or --video.")
    for file_path in (args.image, args.video, args.reference_image, args.reference_video, args.source_image):
        if file_path and not Path(file_path).is_file(): parser.error(f"File not found: {file_path}")
    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "assets": {"image": args.image, "video": args.video},
        "generation_metrics": {
            "generation_time": _value(args.generation_time, "seconds") if args.generation_time is not None else _missing("Pass --generation-time from your run log."),
            "inference_cost": _value(args.inference_cost, "USD") if args.inference_cost is not None else _missing("Pass the provider-reported value with --inference-cost."),
            "gpu_usage": gpu_usage(),
        },
        "image_metrics": image_quality(args.image, args.reference_image) if args.image else {},
        "video_metrics": video_quality(args.video, args.reference_video) if args.video else {},
        "ai_specific_metrics": {
            "prompt_adherence_clip": clip_prompt_score(args.image or args.source_image, args.prompt) if (args.image or args.source_image) else _missing("An image is required."),
            "object_consistency": _missing("Requires labelled object detections across a test set; do not infer it from a single asset."),
            "character_consistency": _missing("Requires identity annotations or face embeddings across video frames/test cases."),
            "hallucination_rate": _missing("Requires a labelled set of requested objects versus generated objects."),
        },
        "study_metrics": {key: _missing("Collect through a participant study or platform analytics.") for key in ("engagement", "watch_time", "brand_recall", "persuasiveness", "conversion_intent", "realism", "creativity", "trust", "purchase_intention", "audio_quality", "lip_sync_accuracy")},
    }
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, pdf_path = output_dir / f"metrics_{stamp}.json", output_dir / f"metrics_{stamp}.pdf"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    render_pdf(report, pdf_path)
    print(f"PDF report:  {pdf_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
