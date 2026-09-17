#!/usr/bin/env python3
"""
main.py
-------
Main CLI dispatcher for the RAID library.
"""

import sys
import argparse
from raid import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="raid",
        description=f"RAID (v{__version__}): Response to Affine Transforms of Image Distance Metrics and Humans"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: translation
    sub_trans = subparsers.add_parser("translation", help="Run translation sweep benchmark across references")
    sub_trans.add_argument("--max_shift", type=int, default=100, help="Maximum pixel translation shift (default: 100)")
    sub_trans.add_argument("--ref", type=str, default=None, help="Evaluate single reference (e.g. I01) or all if omitted")

    # Command: rotation
    sub_rot = subparsers.add_parser("rotation", help="Run rotation sweep benchmark across references")
    sub_rot.add_argument("--crop_size", type=int, default=256, help="Central crop size (default: 256)")
    sub_rot.add_argument("--ref", type=str, default=None, help="Evaluate single reference (e.g. I01) or all if omitted")

    # Command: calibrate
    subparsers.add_parser("calibrate", help="Run psychometric calibration on TID2013 across RMSE, SSIM, LPIPS, DISTS")

    # Command: demo
    subparsers.add_parser("demo", help="Run demo benchmarking on RAID dataset")

    args = parser.parse_args()

    if args.command == "translation":
        if args.ref:
            from evaluate_translation_sweep import evaluate_single_image
            evaluate_single_image(args.ref, max_shift=args.max_shift)
        else:
            from run_all_metrics_including_rmse import main as run_trans
            run_trans()
    elif args.command == "rotation":
        if args.ref:
            from evaluate_rotation_sweep import evaluate_single_image
            evaluate_single_image(args.ref, crop_size=args.crop_size)
        else:
            from run_all_metrics_rotation_benchmark import main as run_rot
            run_rot()
    elif args.command == "calibrate":
        from raid import compute_or_load_tid2013_metrics, calibrate_all_metrics
        df_tid = compute_or_load_tid2013_metrics()
        calibrate_all_metrics(df_tid)
    elif args.command == "demo":
        from benchmark_metrics import run_demo
        run_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
