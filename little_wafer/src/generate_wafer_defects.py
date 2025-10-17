import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from typing import List, Tuple
from datetime import datetime
import time
from pathlib import Path
import argparse


# ---------------------- Helper functions ----------------------

def generate_wafer_points(size: int = 100, radius: float = 20) -> pd.DataFrame:
    xs = np.linspace(-radius, radius, size)
    ys = np.linspace(-radius, radius, size)
    xv, yv = np.meshgrid(xs, ys)
    coords = np.column_stack((xv.flatten(), yv.flatten()))
    df = pd.DataFrame(coords, columns=["x_coord", "y_coord"])
    dist = np.sqrt(df["x_coord"]**2 + df["y_coord"]**2)
    df = df[dist <= radius].reset_index(drop=True)
    if df.shape[0] == 0:
        raise RuntimeError("No wafer points generated: check grid_size and radius")
    df["Hard_bin"] = 1
    return df

def _data_center(df: pd.DataFrame):
    cx = 0.5 * (df["x_coord"].max() + df["x_coord"].min())
    cy = 0.5 * (df["y_coord"].max() + df["y_coord"].min())
    return cx, cy

def generate_center_defect(df, radius=5, density=1.0):
    out = df.copy()
    cx, cy = _data_center(out)
    dist = np.sqrt((out["x_coord"] - cx)**2 + (out["y_coord"] - cy)**2)
    prob = np.exp(-0.5 * (dist / radius)**2) * density
    out["Hard_bin"] = 1
    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 0
    return out

def generate_donut_defect(df, inner_r=30, ring_width=8, density=1.0, noise_level=0.15):
    out = df.copy()
    cx, cy = _data_center(out)
    r = np.sqrt((out["x_coord"] - cx)**2 + (out["y_coord"] - cy)**2)
    
    # กำหนดขอบในและขอบนอก
    outer_r = inner_r + ring_width
    
    # ขอบคม + เพิ่ม noise นิดหน่อย
    prob = np.zeros(len(out))
    mask_ring = (r >= inner_r) & (r <= outer_r)
    
    # ใช้ step function + random noise เพื่อให้ขอบคมขึ้น
    prob[mask_ring] = density * np.random.uniform(0.9, 1.0, mask_ring.sum())
    center_radius = inner_r * 0.4  # ขนาดของ center defect
    mask_center = r <= center_radius
    prob[mask_center] += density * 0.6 * np.random.uniform(0.8, 1.0, mask_center.sum())  # เพิ่มความหนาแน่นในศูนย์กลาง
    
    # 3. เพิ่ม noise นิดหน่อยเพื่อให้ขอบไม่เรียบเกินไป
    prob += np.random.normal(0, noise_level, len(prob))
    prob = np.clip(prob, 0, 1)
    
    out["Hard_bin"] = 1
    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 0
    
    return out

def generate_loc_defect(df, radius=3.0, density=1.0,centers=[(1,0)], sigma=5.0):
    out = df.copy()
    cx, cy = _data_center(out)
    shift_x = np.random.uniform(-10, 10)
    shift_y = np.random.uniform(-10, 10)
    dist = np.sqrt((out["x_coord"] - (cx + shift_x))**2 + (out["y_coord"] - (cy + shift_y))**2)
    prob = np.exp(-0.5 * (dist / radius)**2) * density
    out["Hard_bin"] = 1
    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 0
    return out

def generate_edge_loc_defect(df, radius=5.0, density=1.0, edge_offset=8):
    out = df.copy()
    cx, cy = _data_center(out)
    wafer_radius = np.sqrt((out["x_coord"] - cx)**2 + (out["y_coord"] - cy)**2).max()

    # เลือกตำแหน่งรอย defect ให้อยู่ "เฉพาะขอบ"
    angle = np.random.uniform(0, 2*np.pi)
    x0 = cx + (wafer_radius - edge_offset) * np.cos(angle)
    y0 = cy + (wafer_radius - edge_offset) * np.sin(angle)

    # ระยะจากจุดศูนย์กลาง defect
    dist = np.sqrt((out["x_coord"] - x0)**2 + (out["y_coord"] - y0)**2)

    # สร้าง Gaussian เฉพาะขอบ
    edge_mask = np.sqrt((out["x_coord"] - cx)**2 + (out["y_coord"] - cy)**2) > (wafer_radius - edge_offset - 5)
    prob = np.exp(-0.5 * (dist / radius)**2) * density * edge_mask

    out["Hard_bin"] = 1
    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 0
    return out


def generate_near_full_defect(df, defect_fraction=0.9):
    out = df.copy()
    mask = np.random.rand(len(out)) < defect_fraction
    out["Hard_bin"] = 1
    out.loc[mask, "Hard_bin"] = 0
    return out

def generate_random_defect(df,
                          defect_value=0, pass_value=1,
                          low_density=0.05, high_density=0.25,
                          normal_threshold=0.10,
                          randomize=True):
    out = df.copy()
    out["Hard_bin"] = pass_value
    density = np.random.uniform(low_density, high_density) if randomize else high_density
    mask = np.random.rand(len(out)) < density
    out.loc[mask, "Hard_bin"] = defect_value
    fail_ratio = np.mean(out["Hard_bin"] == defect_value)
    defect_type = "normal" if fail_ratio < normal_threshold else "random"
    return out, defect_type

def generate_scratch_defect(df, width=2.0, density=1.0, curvature=0.02, jitter=3.0):
    out = df.copy()
    cx, cy = _data_center(out)
    
    # เลือกจุดศูนย์กลางการขีดแบบสุ่ม
    offset_x = np.random.uniform(-5, 5)
    offset_y = np.random.uniform(-5, 5)

    xs = out["x_coord"] - cx - offset_x
    ys = out["y_coord"] - cy - offset_y
    
    # เพิ่มเส้นโค้งแบบ sine wave
    angle = np.random.uniform(0, np.pi)
    xs_rot = xs * np.cos(angle) - ys * np.sin(angle)
    ys_rot = xs * np.sin(angle) + ys * np.cos(angle)

    # ความโค้งของเส้น (ใช้ sine wave)
    curve = curvature * xs_rot**2  # quadratic or sine curve
    dist = np.abs(ys_rot - np.sin(xs_rot * curvature) * 10 - np.random.normal(0, jitter, len(xs_rot)))
    
    # ความหนาไม่สม่ำเสมอ
    width_local = width * np.random.uniform(0.6, 1.4, len(xs_rot))

    # ความน่าจะเป็นของ defect
    prob = np.exp(-0.5 * (dist / width_local)**2) * density

    out["Hard_bin"] = 1
    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 0

    return out

def generate_edge_ring_defect(df, ring_width=3.0, density=0.6, arc_ratio=0.3):
    out = df.copy()
    cx, cy = _data_center(out)
    xs = out["x_coord"] - cx
    ys = out["y_coord"] - cy
    r = np.sqrt(xs**2 + ys**2)
    theta = np.arctan2(ys, xs)
    max_r = np.max(r)
    arc_ratio = np.random.uniform(0.2, 0.5)
    arc_center = np.random.uniform(0, 2*np.pi)
    arc_half = arc_ratio * np.pi
    arc_angle = theta - arc_center
    arc_angle = (arc_angle + np.pi) % (2*np.pi) - np.pi
    prob_arc = np.exp(-0.5 * (arc_angle / arc_half)**2)
    dist_from_edge = np.abs(r - max_r)
    prob_edge = np.exp(-0.5 * (dist_from_edge / (ring_width/2))**2)
    prob = prob_arc * prob_edge * density
    out["Hard_bin"] = 1
    out.loc[np.random.rand(len(out)) < prob, "Hard_bin"] = 0
    return out
    
def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Shift coordinates to all positive values and rename columns."""
    df = df.copy()
    df["x_coord"] = df["x_coord"] - df["x_coord"].min()
    df["y_coord"] = df["y_coord"] - df["y_coord"].min()
    return df[["x_coord", "y_coord", "Hard_bin"]]

def visualize_wafer_as_image(df, savepath, img_size=224, square_size=3):
    cx = 0.5 * (df["x_coord"].max() + df["x_coord"].min())
    cy = 0.5 * (df["y_coord"].max() + df["y_coord"].min())
    data_radius = np.sqrt(((df["x_coord"] - cx)**2 + (df["y_coord"] - cy)**2)).max()
    scale = (img_size / 2 - 2) / data_radius
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    for _, row in df.iterrows():
        x = int(round((row["x_coord"] - cx) * scale + img_size / 2))
        y = int(round((row["y_coord"] - cy) * scale + img_size / 2))
        if 0 <= x < img_size and 0 <= y < img_size:
            color = 255 if row["Hard_bin"] == 0 else 127
            img[y:y+square_size, x:x+square_size] = color

    Image.fromarray(img, mode="L").save(savepath)

# ---------------------- Main Generation Function ----------------------

def generate_and_save(pattern, num_samples=1, randomize=True, base_dir="./Data/Synthetic"):
    
    patterns = ["center", "donut", "edge_ring", "scratch", "near_full", "loc", "edge_loc", "random", "normal"]

    if pattern == "all":
        for p in patterns:
            generate_and_save(p, num_samples, randomize)
        return

    base_csv_dir = os.path.join(base_dir, "csv")
    base_png_dir = os.path.join(base_dir, "png")

    csv_dir = os.path.join(base_csv_dir, pattern)
    png_dir = os.path.join(base_png_dir, pattern)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    for i in range(num_samples):
        df = generate_wafer_points(radius=20)

        if randomize:
            density = np.random.uniform(0.6, 1.0)
            ring_thickness_ratio = np.random.uniform(0.2, 0.4)
            arc_ratio = np.random.uniform(0.2, 0.5)
            curvature = np.random.uniform(0.01, 0.05)
            jitter = np.random.uniform(1.0, 4.0)
            scratch_width = np.random.uniform(1.0, 3.0)
        else:
            density = 0.8
            ring_thickness_ratio = 0.3
            arc_ratio = 0.3
            curvature = 0.02
            jitter = 2.0
            scratch_width = 2.0
        
        if pattern == "center":
            df = generate_center_defect(df, radius=np.random.uniform(3, 6), density=density)
        
        elif pattern == "donut":
            inner_r = np.random.uniform(25, 35)
            ring_width = np.random.uniform(6, 10)
            df = generate_donut_defect(df, inner_r=inner_r, ring_width=ring_width,
                               density=np.random.uniform(0.8, 1.0),
                               noise_level=np.random.uniform(0.2, 0.5))
        
        elif pattern == "edge_ring":
            df = generate_edge_ring_defect(df, ring_width=np.random.uniform(2, 4),
                                           density=density, arc_ratio=arc_ratio)
        
        elif pattern == "scratch":
            df = generate_scratch_defect(df, width=scratch_width,
                                         density=density, curvature=curvature, jitter=jitter)
        
        elif pattern == "near_full":
            df = generate_near_full_defect(df, defect_fraction=np.random.uniform(0.85, 0.95))
        
        elif pattern == "loc":
            centers = [(np.random.uniform(-5, 5), np.random.uniform(-5, 5))]
            df = generate_loc_defect(df,centers=centers, sigma=np.random.uniform(4, 7),
                                     density=density)
        
        elif pattern == "edge_loc":
            df = generate_edge_loc_defect(df, radius=np.random.uniform(3, 6),
                                  density=np.random.uniform(0.7, 1.0),
                                  edge_offset=np.random.uniform(10, 18))
        
        elif pattern == "random":
            df, defect_type = generate_random_defect(df, randomize=randomize,high_density=density,low_density=0.05)
            pattern = defect_type  # อาจเปลี่ยนชื่อ folder เป็น “normal” ถ้า fail <10%
            csv_dir = os.path.join(base_csv_dir, pattern)
            png_dir = os.path.join(base_png_dir, pattern)
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)
        
        elif pattern == "normal":
            low_d = 0.01
            high_d = 0.08
            df, defect_type = generate_random_defect(
                df,
                low_density=low_d,
                high_density=high_d,
                randomize=True
            )
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        df["x_coord"] -= df["x_coord"].min()
        df["y_coord"] -= df["y_coord"].min()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{pattern}_{timestamp}_{i:03d}"
        csv_path = os.path.join(csv_dir, f"{base_name}.csv")
        png_path = os.path.join(png_dir, f"{base_name}.png")

        df.to_csv(csv_path, index=False)
        visualize_wafer_as_image(df, png_path)

        print(f"✅ [{pattern}] saved {i+1}/{num_samples}: {png_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wafer defect synthesis tool")
    parser.add_argument("--step", type=str, default="generate", help="Process step")
    parser.add_argument("--pattern", type=str, required=True,
                        help="Pattern type (e.g. donut, all)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples per pattern")
    parser.add_argument("--randomize", action="store_true",
                        help="Randomize parameters")

    args = parser.parse_args()

    if args.step == "generate":
        generate_and_save(args.pattern, args.num_samples, args.randomize)
    else:
        print("❌ Unknown step. Use --step generate")