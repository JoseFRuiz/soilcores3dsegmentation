import subprocess
import sys
import os
from typing import List

def run_cli_with_threshold(lower: int, upper: int, nifti_files: List[str], model: str, 
                          pixels_per_range: int, num_ranges: int, outputs_dir: str, gt_csv: str = None) -> bool:
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
        "--outputs-dir", outputs_dir
    ]
    
    if gt_csv:
        cmd.extend(["--gt-csv", gt_csv])
    
    print(f"\n{'='*60}")
    print(f"Running with lower={lower}, upper={upper}")
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

def main():
    """Main function to iterate through different threshold values"""
    
    # Configuration - modify these values as needed
    nifti_files = [
        "D:\\monailabel\\datasets\\soilcores\\test\\C1216.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\C2003.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\C2609.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\C2803.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\C5608.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\C6405.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\S1510.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\S1919.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\S2307.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\S2403.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\S3117.nii.gz",
        "D:\\monailabel\\datasets\\soilcores\\test\\S3603.nii.gz",
    ]
    
    model = "dataset_2adamw_100k_num_heads_2"
    upper = 100  # Keep upper threshold constant
    pixels_per_range = 2
    num_ranges = 5
    outputs_dir = "outputs"
    gt_csv = "D:\\monailabel\\soilcores3dsegmentation\\gt\\CoresGT.csv"  # Optional: set to None if not needed
    
    # Threshold values to iterate through
    lower_values = [0, 20, 40, 60, 80]
    
    print("Soil Core Threshold Iteration Script")
    print("=" * 50)
    print(f"Model: {model}")
    print(f"Upper threshold: {upper}")
    print(f"Pixels per range: {pixels_per_range}")
    print(f"Number of ranges: {num_ranges}")
    print(f"Outputs directory: {outputs_dir}")
    if gt_csv:
        print(f"Ground truth CSV: {gt_csv}")
    print(f"Lower threshold values to test: {lower_values}")
    print(f"NIfTI files: {nifti_files}")
    print()
    
    # Ask for confirmation
    response = input("Proceed with these settings? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Aborted.")
        return
    
    # Track results
    results = []
    
    # Iterate through threshold values
    for lower in lower_values:
        success = run_cli_with_threshold(
            lower=lower,
            upper=upper,
            nifti_files=nifti_files,
            model=model,
            pixels_per_range=pixels_per_range,
            num_ranges=num_ranges,
            outputs_dir=outputs_dir,
            gt_csv=gt_csv
        )
        
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
        print(f"  ✓ lower={r['lower']}, upper={r['upper']}")
    
    if failed:
        print(f"Failed runs: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  ✗ lower={r['lower']}, upper={r['upper']}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
