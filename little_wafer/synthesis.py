import os
import argparse
import yaml
from src.generate_wafer_defects import generate_and_save

def main():
    """
    Wafer Defect Pattern Synthesizer
    Generate synthetic wafer defect datasets.
    """
    parser = argparse.ArgumentParser(description="Synthetic wafer defect pipeline")
    parser.add_argument("--step", required=True, choices=["generate"], help="Pipeline step")
    parser.add_argument("--pattern", required=True,
                        choices=["center", "donut", "near_full", "loc",
                                 "random", "scratch", "edge_loc", "edge_ring",
                                 "all", "none"], help="Defect pattern to generate")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--randomize", action="store_true", help="Enable random variation in parameters")
    parser.add_argument("--filter", action="store_true", help="Apply denoising filter before saving images")
    parser.add_argument("--filter_strength", type=float, default=10.0, help="Strength of denoising filter")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_dir = config.get("synthetic", "./Data/Synthetic/").rstrip('/')

    if args.step == "generate":
        generate_and_save(
            pattern=args.pattern,
            num_samples=args.num_samples,
            randomize=args.randomize,
            base_dir=base_dir,
           # apply_filter=args.filter,
            #filter_strength=args.filter_strength
        )
    else:
        print("Unknown step. Use --step generate")

if __name__ == "__main__":
    main()
