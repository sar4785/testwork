# synthesis.py (อยู่ที่ root: little_wafer/synthesis.py)

import os
import argparse
import yaml
from typing import List, Tuple
from src.generate_wafer_defects import generate_and_save

def main():
    parser = argparse.ArgumentParser(description="Synthetic wafer defect pipeline")
    parser.add_argument("--step", required=True, choices=["generate"], help="Pipeline step")
    parser.add_argument("--pattern", required=True, choices=["center", "donut", "near_full", "loc",
                                                             'random', 'scratch', 'edge_loc', 'edge_ring',
                                                             'all','normal'], help="Defect pattern to generate")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--inner_r", type=float, default=30.0)
    parser.add_argument("--outer_r", type=float, default=60.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--defect_fraction", type=float, default=0.9)
    parser.add_argument("--loc_centers", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--radius_data", type=float, default=20.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--randomize", action="store_true", help="Enable random variation")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_dir = config.get("synthetis", "./Data/Synthetic/").rstrip('/')
    csv_dir = os.path.join(base_dir, "csv")
    png_dir = os.path.join(base_dir, "png")

    loc_centers = None
    if args.loc_centers:
        loc_centers = [(float(x), float(y)) for x, y in [c.split(',') for c in args.loc_centers.split(';')]]

    generate_and_save(
        pattern=args.pattern,
        num_samples=args.num_samples,
        randomize=args.randomize,
        base_dir=base_dir
    )
if __name__ == "__main__":
    main()