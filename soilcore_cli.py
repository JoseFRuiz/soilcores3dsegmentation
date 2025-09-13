# Example usage:
# python soilcore_cli.py --nifti-dir D:\monailabel\datasets\soilcores\test --model unet_dataset_2_default --lower 0.0 --upper 100.0 --view horizontal --pixels-per-range 2 --num-ranges 5 --outputs-dir outputs --gt-csv D:\monailabel\soilcores3dsegmentation\gt\CoresGT.csv
# python soilcore_cli.py --nifti-dir D:\monailabel\datasets\soilcores\test --model unet_dataset_2_default --lower 0.0 --upper 60.0 --view vertical --pixels-per-range 2 --num-ranges 5 --outputs-dir outputs --gt-csv D:\monailabel\soilcores3dsegmentation\gt\CoresGT.csv
# python soilcore_cli.py --nifti-dir D:\monailabel\datasets\soilcores\test --model dataset_2adamw_100k_num_heads_2 --lower 0.0 --upper 60.0 --view vertical --pixels-per-range 2 --num-ranges 5 --outputs-dir outputs --gt-csv D:\monailabel\soilcores3dsegmentation\gt\CoresGT.csv

import argparse
import os
import sys
from typing import List

from utils import (
    MODEL_CONFIGS,
    collect_nifti_files_from_dir,
    segment_multiple_files,
    save_thresholded_slices_from_nifti,
    analyze_core_folders,
    run_pipeline_for_niftis,
    find_core_folders_in_parent,
    compute_and_save_correlations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI for Soil Core segmentation and root topology analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nifti", nargs="*", help="One or more NIfTI files to segment and analyze")
    group.add_argument("--nifti-dir", help="Directory with .nii/.nii.gz files (non-recursive)")
    group.add_argument("--cores", nargs="*", help="Existing core folders with 2D images to analyze")
    group.add_argument("--cores-parent", help="Parent directory containing multiple core folders")

    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="dataset_2adamw_100k_num_heads_2",
        help="Model name to use when segmenting NIfTI files",
    )

    # Thresholding for slice export
    parser.add_argument("--lower", type=float, default=0.0, help="Lower threshold percentile (0-100) for slice export")
    parser.add_argument("--upper", type=float, default=100.0, help="Upper threshold percentile (0-100) for slice export")
    parser.add_argument("--view", choices=["horizontal", "vertical"], default="horizontal", 
                       help="Slicing direction for topology analysis: horizontal (6×6 cm slices) or vertical (30×6 cm slices)")

    # Topology analysis params
    parser.add_argument("--pixels-per-range", type=int, default=2, help="Pixels per diameter range")
    parser.add_argument("--num-ranges", type=int, default=5, help="Number of diameter ranges")

    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Base outputs directory")
    parser.add_argument("--gt-csv", type=str, help="Ground truth CSV path for correlation analysis (e.g., gt/CoresGT.csv)")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.nifti is not None or args.nifti_dir is not None:
        # End-to-end pipeline: segment -> export slices -> analyze
        nifti_files: List[str] = []
        if args.nifti is not None and len(args.nifti) > 0:
            nifti_files = args.nifti
        elif args.nifti_dir is not None:
            if not os.path.isdir(args.nifti_dir):
                print(f"Directory does not exist: {args.nifti_dir}")
                return 2
            nifti_files = collect_nifti_files_from_dir(args.nifti_dir)

        if not nifti_files:
            print("No NIfTI files found to process.")
            return 1

        summary = run_pipeline_for_niftis(
            nifti_files=nifti_files,
            model_name=args.model,
            lower_percent=args.lower,
            upper_percent=args.upper,
            pixels_per_range=args.pixels_per_range,
            num_ranges=args.num_ranges,
            outputs_dir=args.outputs_dir,
            view=args.view,
        )

        print("\nSegmentation outputs:")
        for p in summary["nifti_outputs"]:
            print(f" - {p}")
        print(f"Slices saved under: {summary['png_root']}")
        print("Analysis outputs:")
        print(f" - per-core CSVs: {summary['analysis_outputs'].get('per_core_csvs', [])}")
        print(f" - aggregated CSV: {summary['analysis_outputs'].get('aggregated_csv')}")
        print(f" - aggregated plot: {summary['analysis_outputs'].get('aggregated_plot')}")

        # Optional correlations against GT
        if args.gt_csv and summary['analysis_outputs'].get('aggregated_csv'):
            corr_out = compute_and_save_correlations(
                gt_csv_path=args.gt_csv,
                summary_csv_path=summary['analysis_outputs']['aggregated_csv'],
                save_dir=os.path.dirname(summary['analysis_outputs']['aggregated_csv']),
                lower_percent=args.lower,
                upper_percent=args.upper,
                view=args.view,
            )
            print("Correlation matrices saved:")
            print(f" - Pearson: {corr_out['pearson']}")
            print(f" - Spearman: {corr_out['spearman']}")
        return 0

    # Analyze existing cores (no segmentation step)
    core_folders: List[str] = []
    if args.cores is not None and len(args.cores) > 0:
        core_folders = args.cores
    elif args.cores_parent is not None:
        if not os.path.isdir(args.cores_parent):
            print(f"Directory does not exist: {args.cores_parent}")
            return 2
        core_folders = find_core_folders_in_parent(args.cores_parent)

    if not core_folders:
        print("No core folders found to analyze.")
        return 1

    analysis_outputs = analyze_core_folders(
        selected_folders=core_folders,
        pixels_per_range=args.pixels_per_range,
        num_ranges=args.num_ranges,
        view=args.view,
    )

    print("\nAnalysis outputs:")
    print(f" - per-core CSVs: {analysis_outputs.get('per_core_csvs', [])}")
    print(f" - aggregated CSV: {analysis_outputs.get('aggregated_csv')}")
    print(f" - aggregated plot: {analysis_outputs.get('aggregated_plot')}")

    # Optional correlations if summary exists
    if args.gt_csv and analysis_outputs.get('aggregated_csv'):
        corr_out = compute_and_save_correlations(
            gt_csv_path=args.gt_csv,
            summary_csv_path=analysis_outputs['aggregated_csv'],
            save_dir=os.path.dirname(analysis_outputs['aggregated_csv']),
            lower_percent=args.lower,
            upper_percent=args.upper,
            view=args.view,
        )
        print("Correlation matrices saved:")
        print(f" - Pearson: {corr_out['pearson']}")
        print(f" - Spearman: {corr_out['spearman']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


