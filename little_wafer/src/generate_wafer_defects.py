"""
Wafer Defect Pattern Generator (Modified Version)
- รวม 'none' และ 'random' เข้าด้วยกันในโฟลเดอร์ 'random'
- สร้างข้อมูล clean และ noisy แยกกัน
- ใช้แนวทางตาม paper: Data Selection -> Zero Padding -> Augmentation
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from datetime import datetime
from typing import Tuple
import shutil
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


# ---------------------- Configuration Constants ----------------------
DEFAULT_RADIUS = 60
IMG_SIZE = 224
SQUARE_SIZE = 6
BASE_DIR = "./Data/Synthetic"
DIE_PITCH = 1

# ---------------------- Core Data Generation ----------------------
def apply_dummy_die_mask(df: pd.DataFrame, mask_type: str = "random", strip_width: int = 1) -> pd.DataFrame:
    """Randomly removes rows/columns of dies to simulate dummy or no-data dies (black gaps)."""
    out = df.copy()
    unique_x = sorted(out["x_coord"].unique())
    unique_y = sorted(out["y_coord"].unique())

    if not unique_x or not unique_y:
        return out
    if mask_type == "none":
        return out

    if mask_type in ["random", "vertical"]:
        remove_x_start = np.random.choice(range(len(unique_x) - strip_width + 1))
        remove_x_indices = range(remove_x_start, remove_x_start + strip_width)
        remove_x_coords = [unique_x[i] for i in remove_x_indices]
        out = out[~out["x_coord"].isin(remove_x_coords)]

    if mask_type == "vertical":
        return out

    if mask_type in ["random", "horizontal"]:
        remove_y_start = np.random.choice(range(len(unique_y) - strip_width + 1))
        remove_y_indices = range(remove_y_start, remove_y_start + strip_width)
        remove_y_coords = [unique_y[i] for i in remove_y_indices]
        out = out[~out["y_coord"].isin(remove_y_coords)]

    if mask_type == "horizontal":
        return out

    return out


def generate_wafer_points(radius: float = DEFAULT_RADIUS, pitch: float = DIE_PITCH) -> pd.DataFrame:
    """Generate a grid of points within a circular wafer."""
    max_coord = int(radius / pitch) * pitch
    xs = np.arange(-max_coord, max_coord + pitch, pitch)
    ys = np.arange(-max_coord, max_coord + pitch, pitch)

    xv, yv = np.meshgrid(xs, ys)
    coords = np.column_stack((xv.flatten(), yv.flatten()))
    df = pd.DataFrame(coords, columns=["x_coord", "y_coord"])
    dist = np.sqrt(df["x_coord"]**2 + df["y_coord"]**2)
    df = df[dist <= radius].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No wafer points generated: check pitch and radius")

    df["Hard_bin"] = 1
    return df


def _data_center(df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate the center coordinates of the wafer."""
    return 0.5 * (df["x_coord"].max() + df["x_coord"].min()), 0.5 * (df["y_coord"].max() + df["y_coord"].min())

# ---------------------- Defect Pattern Generators ----------------------
def generate_center_defect(df: pd.DataFrame, radius: float = 5, density: float = 1.0, noise_level: float = 0.05) -> pd.DataFrame:
    """Generate a center defect pattern using a Gaussian distribution."""
    out = df.copy()
    cx, cy = _data_center(out)
    angle = np.random.uniform(0, np.pi)
    aspect_ratio = np.random.uniform(0.6, 1.4)
    x_shift = out["x_coord"] - cx
    y_shift = out["y_coord"] - cy
    x_rot = x_shift * np.cos(angle) - y_shift * np.sin(angle)
    y_rot = (x_shift * np.sin(angle) + y_shift * np.cos(angle)) * aspect_ratio
    dist = np.sqrt(x_rot**2 + y_rot**2)
    prob = np.exp(-0.5 * (dist / radius)**2) * density
    prob += np.random.normal(0, noise_level, len(prob))
    prob = np.clip(prob, 0, 1)
    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 5
    return out


def generate_donut_defect(df: pd.DataFrame, inner_r: float = None, ring_width: float = None, 
                          density: float = 1.0, noise_level: float = 0.15) -> pd.DataFrame:
    """Generate a diverse donut-shaped defect with deformations, holes, and texture."""
    out = df.copy()
    cx, cy = _data_center(out)
    x = out["x_coord"] - cx
    y = out["y_coord"] - cy

    if inner_r is None:
        inner_r = np.random.uniform(20, 40)
    if ring_width is None:
        ring_width = np.random.uniform(6, 14)
    outer_r = inner_r + ring_width

    rx_factor = np.random.uniform(0.9, 1.2)
    ry_factor = np.random.uniform(0.9, 1.2)
    angle = np.random.uniform(0, np.pi)
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    theta = np.arctan2(y_rot, x_rot)
    base_radius = np.sqrt((x_rot / rx_factor)**2 + (y_rot / ry_factor)**2)

    low_freqs = np.random.uniform(1, 3, size=np.random.randint(2, 4))
    high_freqs = np.random.uniform(5, 10, size=np.random.randint(1, 3))
    all_freqs = np.concatenate([low_freqs, high_freqs])

    radial_noise = np.zeros_like(theta)
    for f in all_freqs:
        amp = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        radial_noise += amp * np.sin(f * theta + phase)

    radial_noise = np.convolve(radial_noise, np.ones(5) / 5, mode="same")
    radial_noise *= ring_width * np.random.uniform(0.1, 0.25)
    r_eff = base_radius + radial_noise
    r_eff = r_eff * np.random.uniform(0.9, 1.1) + np.random.uniform(-3, 3)

    if np.random.rand() < 0.5:
        r_eff *= 1 + np.random.uniform(0.05, 0.2) * np.cos(theta - np.random.uniform(0, 2 * np.pi))

    mask_ring = (r_eff >= inner_r) & (r_eff <= outer_r)
    prob = np.zeros(len(out))
    prob[mask_ring] = density * np.random.uniform(0.4, 1.0, mask_ring.sum())

    if np.random.rand() < 0.5:
        gap_angle = np.random.uniform(0, 2 * np.pi)
        gap_width = np.random.uniform(0.5, 1.5)
        gap_mask = np.abs((theta - gap_angle + np.pi) % (2 * np.pi) - np.pi) > gap_width / 2
        prob *= gap_mask

    prob[mask_ring] *= np.clip(1 + np.random.normal(0, 0.2, mask_ring.sum()), 0.2, 1.8)
    prob += np.random.normal(0, noise_level, len(prob))
    prob = np.clip(prob, 0, 1)
    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 5
    return out


def generate_loc_defect(df: pd.DataFrame, radius: float = 3.0, density: float = 1.0, 
                       sigma: float = 5.0, noise_level: float = 0.03) -> pd.DataFrame:
    """Generate a localized defect at a random position."""
    out = df.copy()
    cx, cy = _data_center(out)
    wafer_radius = np.sqrt((out["x_coord"] - cx)**2 + (out["y_coord"] - cy)**2).max()

    angle = np.random.uniform(0, 2 * np.pi)
    dist_from_edge = np.random.uniform(wafer_radius * 0.4, wafer_radius * 0.7)
    x0, y0 = cx + dist_from_edge * np.cos(angle), cy + dist_from_edge * np.sin(angle)

    dist = np.sqrt((out["x_coord"] - x0)**2 + (out["y_coord"] - y0)**2)
    prob = np.exp(-0.5 * (dist / radius)**2) * density
    prob += np.random.normal(0, noise_level, len(prob))
    prob = np.clip(prob, 0, 1)

    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 5
    return out


def generate_edge_loc_defect(df: pd.DataFrame, radius: float = 5.0, density: float = 1.0, 
                             edge_offset: float = 8, noise_level: float = 0.05) -> pd.DataFrame:
    """Generate a defect localized at the wafer's edge."""
    out = df.copy()
    cx, cy = _data_center(out)
    wafer_radius = np.sqrt((out["x_coord"] - cx)**2 + (out["y_coord"] - cy)**2).max()

    angle = np.random.uniform(0, 2 * np.pi)
    x0, y0 = cx + (wafer_radius - edge_offset) * np.cos(angle), cy + (wafer_radius - edge_offset) * np.sin(angle)
    aspect = np.random.uniform(0.5, 1.6)
    rot = np.random.uniform(0, np.pi)

    dx = out["x_coord"] - x0
    dy = out["y_coord"] - y0
    x_rot = dx * np.cos(rot) - dy * np.sin(rot)
    y_rot = (dx * np.sin(rot) + dy * np.cos(rot)) * aspect

    dist = np.sqrt(x_rot**2 + y_rot**2)
    edge_mask = np.sqrt((out["x_coord"] - cx)**2 + (out["y_coord"] - cy)**2) > (wafer_radius - edge_offset - 6)
    prob = np.exp(-0.5 * (dist / radius)**2) * density * edge_mask

    prob += np.random.normal(0, noise_level, len(prob))
    prob = np.clip(prob, 0, 1)

    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 5
    return out


def generate_near_full_defect(df: pd.DataFrame, defect_fraction: float = 0.7) -> pd.DataFrame:
    """Generate a near-full defect pattern covering most of the wafer."""
    out = df.copy()
    prob = np.full(len(out), defect_fraction)
    mask = np.random.rand(len(out)) < prob
    out.loc[mask, "Hard_bin"] = 5
    return out


def generate_random_defect(df: pd.DataFrame, defect_value: int = 5, pass_value: int = 1, 
                          density: float = 0.005) -> pd.DataFrame:
    """Generate a random defect pattern with adjustable density."""
    out = df.copy()
    out["Hard_bin"] = pass_value
    mask = np.random.rand(len(out)) < density
    out.loc[mask, "Hard_bin"] = defect_value
    return out


def generate_normal_defect(df: pd.DataFrame, noise_level: float = 0.002) -> pd.DataFrame:
    """Normal wafer: nearly flawless, with minimal random noise."""
    out = df.copy()
    mask = np.random.rand(len(out)) < noise_level
    out.loc[mask, "Hard_bin"] = 5
    return out


def generate_scratch_defect(df: pd.DataFrame, width: float = 2.0, density: float = 1.0, 
                           curvature: float = 0.02, jitter: float = 3.0, 
                           noise_level: float = 0.05, num_lines: int = 1) -> pd.DataFrame:
    """Generate a scratch-like defect pattern with curvature and jitter."""
    out = df.copy()
    cx, cy = _data_center(out)

    offset_x, offset_y = np.random.uniform(-15, 15), np.random.uniform(-15, 15)
    xs = out["x_coord"] - cx - offset_x
    ys = out["y_coord"] - cy - offset_y

    angle = np.random.uniform(0, np.pi)
    xs_rot = xs * np.cos(angle) - ys * np.sin(angle)
    ys_rot = xs * np.sin(angle) + ys * np.cos(angle)

    local_curvature = curvature * np.random.uniform(0.5, 1.5)
    dist = np.abs(ys_rot - np.sin(xs_rot * local_curvature) * 10 - np.random.normal(0, jitter, len(xs_rot)))

    width_local = width * np.random.uniform(0.8, 1.2, len(xs_rot))
    prob = np.exp(-0.5 * (dist / width_local) ** 2) * density

    r = np.sqrt(xs**2 + ys**2)
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-6)
    fade_strength = np.exp(-r_norm * np.random.uniform(1.0, 3.0))

    if np.random.rand() < 0.5:
        fade_strength *= (0.5 + 0.5 * np.cos(xs_rot / (np.max(np.abs(xs_rot)) + 1e-6) * np.pi))

    prob *= fade_strength
    prob += np.random.normal(0, noise_level, len(prob))
    prob = np.clip(prob, 0, 1)

    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 5
    return out


def generate_edge_ring_defect(df: pd.DataFrame, ring_width: float = 5.0, density: float = 0.02, 
                              arc_ratio: float = 0.23, noise_level: float = 0.01) -> pd.DataFrame:
    """Generate an edge ring defect pattern with an arc-shaped defect."""
    out = df.copy()
    cx, cy = _data_center(out)
    xs = out["x_coord"] - cx
    ys = out["y_coord"] - cy
    r = np.sqrt(xs**2 + ys**2)
    theta = np.arctan2(ys, xs)
    max_r = np.max(r)

    arc_center = np.random.uniform(0, 2 * np.pi)
    arc_half = arc_ratio * np.pi
    arc_angle = theta - arc_center
    arc_angle = (arc_angle + np.pi) % (2 * np.pi) - np.pi

    prob_arc = np.exp(-0.5 * (arc_angle / arc_half)**2)
    dist_from_edge = np.abs(r - max_r)
    prob_edge = np.exp(-0.5 * (dist_from_edge / (ring_width / 2))**2)
    prob = prob_arc * prob_edge * density

    prob += np.random.normal(0, noise_level, len(prob))
    prob = np.clip(prob, 0, 1)

    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 5
    return out


# ---------------------- Utility Functions ----------------------
def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Shift coordinates to positive values and round to integers."""
    df = df.copy()
    df["x_coord"] = (df["x_coord"] - df["x_coord"].min()).round().astype(int)
    df["y_coord"] = (df["y_coord"] - df["y_coord"].min()).round().astype(int)
    return df[["x_coord", "y_coord", "Hard_bin"]]


def add_zero_padding(img: np.ndarray, padding_percent: float = 0.06) -> np.ndarray:
    """
    🔹 ตาม Paper: เพิ่ม zero padding รอบขอบภาพ 6% ของขนาดภาพ
    เพื่อป้องกันความเสียหายจากการหมุนและรักษาข้อมูลที่ขอบ
    """
    h, w = img.shape[:2]
    pad_size = int(h * padding_percent)
    
    if len(img.shape) == 2:  # Grayscale
        padded = np.pad(img, pad_size, mode='constant', constant_values=0)
    else:  # RGB
        padded = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 
                       mode='constant', constant_values=0)
    return padded


def visualize_wafer_as_image(df: pd.DataFrame, savepath: str, img_size: int = IMG_SIZE, 
                            square_size: int = SQUARE_SIZE, apply_padding: bool = True):
    """Render wafer map as a grayscale image with optional zero padding."""
    cx = 0.5 * (df["x_coord"].max() + df["x_coord"].min())
    cy = 0.5 * (df["y_coord"].max() + df["y_coord"].min())
    data_radius = np.sqrt(((df["x_coord"] - cx)**2 + (df["y_coord"] - cy)**2)).max()
    scale = (img_size / 2 - 2) / (data_radius + 1e-6)

    img = np.zeros((img_size, img_size), dtype=np.uint8)
    half_size = square_size // 2

    for _, row in df.iterrows():
        x_centered_scaled = (row["x_coord"] - cx) * scale
        y_centered_scaled = (row["y_coord"] - cy) * scale
        x = int(round(x_centered_scaled + img_size / 2))
        y = int(round(-y_centered_scaled + img_size / 2))

        color = 255 if row["Hard_bin"] == 5 else 127
        x_start = max(0, x - half_size)
        x_end = min(img_size, x + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(img_size, y + half_size + 1)
        img[y_start:y_end, x_start:x_end] = color

    # 🔹 เพิ่ม zero padding ตาม paper (6%)
    if apply_padding:
        img = add_zero_padding(img, padding_percent=0.06)

    Image.fromarray(img, mode="L").save(savepath)


# ---------------------- Main Generation Function ----------------------
def generate_and_save(pattern: str, num_samples: int = 1, randomize: bool = True, 
                     base_dir: str = BASE_DIR):
    """
    🔹 แก้ไขตาม requirement:
    - รวม 'none' และ 'random' ไว้ใน folder เดียวกันชื่อ 'random'
    - สร้างทั้ง clean และ noisy แยกกัน
    - เมื่อเรียก pattern='random' จะสร้างทั้ง random และ none อย่างละครึ่ง
    """
    patterns = [
        "center", "donut", "edge_ring", "scratch",
        "near_full", "loc", "edge_loc", "random"  # ลบ "none" ออก
    ]

    if pattern == "all":
        for p in patterns:
            generate_and_save(p, num_samples, randomize, base_dir)
        return

    if pattern not in patterns:
        print(f"❌ Unknown pattern: {pattern}")
        return

    # Create directories
    csv_clean_dir = os.path.join(base_dir, "csv", "clean", pattern)
    csv_noisy_dir = os.path.join(base_dir, "csv", "noisy", pattern)
    png_clean_dir = os.path.join(base_dir, "png", "clean", pattern)
    png_noisy_dir = os.path.join(base_dir, "png", "noisy", pattern)
    
    for d in [csv_clean_dir, csv_noisy_dir, png_clean_dir, png_noisy_dir]:
        os.makedirs(d, exist_ok=True)

    # 🔹 ถ้า pattern='random' จะสร้างทั้ง random และ none อย่างละครึ่ง
    if pattern == "random":
        samples_per_type = num_samples // 2
        print(f"🎲 Generating {samples_per_type} random + {samples_per_type} normal (combined as 'random')")
        
        for sub_pattern, sub_count in [("random", samples_per_type), ("none", num_samples - samples_per_type)]:
            _generate_pattern_samples(
                sub_pattern, sub_count, randomize, base_dir, pattern,
                csv_clean_dir, csv_noisy_dir, png_clean_dir, png_noisy_dir
            )
    else:
        _generate_pattern_samples(
            pattern, num_samples, randomize, base_dir, pattern,
            csv_clean_dir, csv_noisy_dir, png_clean_dir, png_noisy_dir
        )


def _generate_pattern_samples(pattern_type, num_samples, randomize, base_dir, output_pattern,
                              csv_clean_dir, csv_noisy_dir, png_clean_dir, png_noisy_dir):
    """Helper function to generate samples for a specific pattern type."""
    
    for i in range(num_samples):
        for is_noisy, noise_tag in [(False, "clean"), (True, "noisy")]:
            df = generate_wafer_points()

            # Apply dummy die mask
            if randomize:
                mask_type = np.random.choice(
                    ["horizontal", "vertical", "none", "random"], p=[0.2, 0.2, 0.4, 0.2]
                )
                strip_width = np.random.randint(1, 3)
            else:
                mask_type = "vertical"
                strip_width = 1

            df = apply_dummy_die_mask(df, mask_type=mask_type, strip_width=strip_width)

            # Randomized parameters
            params = {
                "density": np.random.uniform(0.6, 1.0) if randomize else 0.8,
                "ring_thickness_ratio": np.random.uniform(0.2, 0.4) if randomize else 0.3,
                "arc_ratio": np.random.uniform(0.2, 0.5) if randomize else 0.3,
                "curvature": np.random.uniform(0.01, 0.05) if randomize else 0.02,
                "jitter": np.random.uniform(1.0, 4.0) if randomize else 2.0,
                "scratch_width": np.random.uniform(1.0, 3.0) if randomize else 2.0,
            }
            noise_scale = 1.0 if is_noisy else 0.0

            # 🔹 Generate defect pattern ตาม pattern_type
            if pattern_type == "center":
                df = generate_center_defect(df, radius=np.random.uniform(3, 6) if randomize else 5,
                                          density=params["density"], noise_level=0.08 * noise_scale)
            elif pattern_type == "donut":
                df = generate_donut_defect(df, inner_r=np.random.uniform(25, 35) if randomize else 30,
                                         ring_width=np.random.uniform(6, 10) if randomize else 8,
                                         density=params["density"], 
                                         noise_level=np.random.uniform(0.02, 0.03) * noise_scale if randomize else 0.025 * noise_scale)
            elif pattern_type == "edge_ring":
                df = generate_edge_ring_defect(df, ring_width=np.random.uniform(2, 4) if randomize else 3,
                                             density=params["density"], arc_ratio=params["arc_ratio"],
                                             noise_level=0.01 * noise_scale)
            elif pattern_type == "scratch":
                df = generate_scratch_defect(df, width=params["scratch_width"], density=params["density"],
                                           curvature=params["curvature"], jitter=params["jitter"],
                                           noise_level=0.05 * noise_scale)
            elif pattern_type == "near_full":
                df = generate_near_full_defect(df, defect_fraction=np.random.uniform(0.6, 0.75) * noise_scale if randomize else 0.7 * noise_scale)
            elif pattern_type == "loc":
                df = generate_loc_defect(df, radius=np.random.uniform(4, 6) if randomize else 5,
                                       density=params["density"], noise_level=0.05 * noise_scale)
            elif pattern_type == "edge_loc":
                df = generate_edge_loc_defect(df, radius=np.random.uniform(3, 6) if randomize else 5,
                                            density=params["density"], noise_level=0.05 * noise_scale)
            elif pattern_type == "random":
                df = generate_random_defect(df, density=np.random.uniform(0.02, 0.07) * noise_scale if randomize else 0.05 * noise_scale)
            elif pattern_type == "none":
                df = generate_normal_defect(df, noise_level=0.003 * noise_scale)

            # Save files
            df = normalize_coordinates(df)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{output_pattern}_{timestamp}_{i:03d}"
            
            csv_dir = csv_noisy_dir if is_noisy else csv_clean_dir
            png_dir = png_noisy_dir if is_noisy else png_clean_dir
            
            #csv_path = os.path.join(csv_dir, f"{base_name}_{noise_tag}.csv")
            png_path = os.path.join(png_dir, f"{base_name}_{noise_tag}.png")

            #df.to_csv(csv_path, index=False)
            visualize_wafer_as_image(df, png_path, apply_padding=True)
            
            print(f"✅ [{output_pattern}] {noise_tag} saved: {png_path}")


# ---------------------- Command Line Interface ----------------------
def main():
    """Parse command-line arguments and run the wafer defect generator."""
    parser = argparse.ArgumentParser(description="Wafer defect synthesis tool (Modified)")
    parser.add_argument("--step", type=str, default="generate", help="Process step")
    parser.add_argument("--pattern", type=str, required=True, 
                       help="Pattern type: center, donut, edge_ring, scratch, near_full, loc, edge_loc, random, all")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per pattern")
    parser.add_argument("--randomize", action="store_true", help="Randomize defect parameters")

    args = parser.parse_args()

    if args.step == "generate":
        generate_and_save(
            pattern=args.pattern,
            num_samples=args.num_samples,
            randomize=args.randomize
        )
    else:
        print("❌ Unknown step. Use --step generate")


if __name__ == "__main__":
    main()