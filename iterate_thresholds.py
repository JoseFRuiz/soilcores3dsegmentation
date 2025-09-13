import subprocess
import sys
import os
import shutil
import argparse
from typing import List

def move_results_to_unique_folder(outputs_dir: str, model: str, view: str, lower: int, upper: int) -> None:
    """Move the results to a unique folder to prevent overwriting"""
    # Get the network type from model name
    net_type = "unetr" if "unetr" in model.lower() else "unet" if "unet" in model.lower() else "dynunet" if "dynunet" in model.lower() else "segresnet"
    
    # Source folders (original structure)
    nifti_source = os.path.join(outputs_dir, model)
    png_source = os.path.join(outputs_dir, f"{net_type}_{model}_{view}_png")
    
    # Destination folders (with threshold suffix)
    nifti_dest = os.path.join(outputs_dir, f"{model}_l{lower}_u{upper}")
    png_dest = os.path.join(outputs_dir, f"{net_type}_{model}_{view}_l{lower}_u{upper}_png")
    
    # Move NIfTI files
    if os.path.exists(nifti_source):
        if os.path.exists(nifti_dest):
            shutil.rmtree(nifti_dest)
        shutil.move(nifti_source, nifti_dest)
        print(f"✓ Moved NIfTI files to: {nifti_dest}")
    
    # Move PNG files
    if os.path.exists(png_source):
        if os.path.exists(png_dest):
            shutil.rmtree(png_dest)
        shutil.move(png_source, png_dest)
        print(f"✓ Moved PNG files to: {png_dest}")

def run_cli_with_threshold(lower: int, upper: int, nifti_files: List[str], model: str, 
                          pixels_per_range: int, num_ranges: int, outputs_dir: str, 
                          view: str = "horizontal", gt_csv: str = None) -> bool:
    """Run the CLI script with specific threshold values"""
    
    cmd = [
        sys.executable, "soilcore_cli.py",
        "--nifti"
    ] + nifti_files + [
        "--model", model,
        "--lower", str(lower),
        "--upper", str(upper),
        "--pixels-per-range", str(pixels_per_range),
        "--num-ranges", str(num_ranges),
        "--outputs-dir", outputs_dir,
        "--view", view
    ]
    
    if gt_csv:
        cmd.extend(["--gt-csv", gt_csv])
    
    print(f"\n{'='*60}")
    print(f"Running with lower={lower}, upper={upper}, view={view}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success!")
        print("Output:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        return False

def parse_threshold_values(values_str: str) -> List[int]:
    """Parse comma-separated threshold values from string"""
    try:
        return [int(x.strip()) for x in values_str.split(',')]
    except ValueError:
        raise ValueError(f"Invalid threshold values format: {values_str}. Use comma-separated integers (e.g., '0,20,40,60,80')")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Iterate through different threshold values for soil core segmentation')
    
    parser.add_argument('--model', required=True, 
                       help='Model name (e.g., segresnet_dataset_2_default, dynunet_dataset_2_100k, unet_dataset_2_100k, unet_dataset_2_default, dataset_2adamw_100k_num_heads_2)')
    
    parser.add_argument('--lower-values', required=True, 
                       help='Comma-separated lower threshold values (e.g., "0,20,40,60,80")')
    
    parser.add_argument('--upper-values', required=True, 
                       help='Comma-separated upper threshold values (e.g., "100" or "80,60,40,20")')
    
    parser.add_argument('--view', required=True, choices=['horizontal', 'vertical'],
                       help='View type: horizontal (6×6 cm slices) or vertical (30×6 cm slices)')
    
    parser.add_argument('--disk', default='H:', 
                       help='Disk drive letter (default: H:)')
    
    parser.add_argument('--pixels-per-range', type=int, default=2,
                       help='Pixels per range (default: 2)')
    
    parser.add_argument('--num-ranges', type=int, default=5,
                       help='Number of ranges (default: 5)')
    
    parser.add_argument('--outputs-dir', default='outputs',
                       help='Output directory (default: outputs)')
    
    parser.add_argument('--gt-csv', 
                       help='Ground truth CSV file path (optional)')
    
    parser.add_argument('--no-confirm', action='store_true',
                       help='Skip confirmation prompt')
    
    return parser.parse_args()

def main():
    """Main function to iterate through different threshold values"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Parse threshold values
    try:
        lower_values = parse_threshold_values(args.lower_values)
        upper_values = parse_threshold_values(args.upper_values)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Build NIfTI file paths
    nifti_files = [
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\C1216.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\C2003.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\C2609.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\C2803.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\C5608.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\C6405.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\S1510.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\S1919.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\S2307.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\S2403.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\S3117.nii.gz",
        f"{args.disk}\\monailabel\\datasets\\soilcores\\test\\S3603.nii.gz",
    ]
    
    # Set ground truth CSV path if not provided
    if not args.gt_csv:
        args.gt_csv = f"{args.disk}\\monailabel\\soilcores3dsegmentation\\gt\\CoresGT.csv"
    print("Soil Core Threshold Iteration Script")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"View: {args.view}")
    print(f"Pixels per range: {args.pixels_per_range}")
    print(f"Number of ranges: {args.num_ranges}")
    print(f"Outputs directory: {args.outputs_dir}")
    if args.gt_csv:
        print(f"Ground truth CSV: {args.gt_csv}")
    print(f"Lower threshold values to test: {lower_values}")
    print(f"Upper threshold values to test: {upper_values}")
    print(f"NIfTI files: {nifti_files}")
    print()
    
    # Ask for confirmation unless --no-confirm is used
    if not args.no_confirm:
        response = input("Proceed with these settings? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Aborted.")
            return
    
    # Track results
    results = []
    
    # Iterate through threshold values
    for lower in lower_values:
        for upper in upper_values:
            print(f"Running with lower={lower}, upper={upper}, view={args.view}")
            success = run_cli_with_threshold(
                lower=lower,
                upper=upper,
                nifti_files=nifti_files,
                model=args.model,
                pixels_per_range=args.pixels_per_range,
                num_ranges=args.num_ranges,
                outputs_dir=args.outputs_dir,
                gt_csv=args.gt_csv,
                view=args.view
            )
        
            # If successful, move results to unique folder to prevent overwriting
            if success:
                print(f"Moving results for l{lower}_u{upper} to unique folder...")
                move_results_to_unique_folder(args.outputs_dir, args.model, args.view, lower, upper)
        
            results.append({
                'lower': lower,
                'upper': upper,
                'success': success
            })
            
            # Small delay between runs
            import time
            time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("ITERATION SUMMARY")
    stalled = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successful runs: {len(stalled)}/{len(results)}")
    for r in stalled:
        print(f"  ✓ lower={r['lower']}, upper={r['upper']}, view={args.view}")
    
    if failed:
        print(f"Failed runs: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  ✗ lower={r['lower']}, upper={r['upper']}, view={args.view}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
