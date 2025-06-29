import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def safe_correlation(x, y):
    """
    Calculate correlation between two arrays, returning 0 if correlation cannot be computed
    due to zero variance in either array.
    """
    # Check if either array has zero variance
    if np.var(x) == 0 or np.var(y) == 0:
        return 0.0
    
    try:
        correlation, _ = pearsonr(x, y)
        return correlation
    except:
        return 0.0

def map_gt_to_prediction_ranges():
    """
    Map ground truth ranges to prediction ranges based on the data structure.
    Ground truth ranges: 0<L<=1, 1<L<=2, 2<L<=3, 3<L<=4, L>4
    Prediction ranges: Range 1, Range 2, Range 3, Range 4, Range 5
    """
    # Based on the data, it appears the ranges are mapped as follows:
    # GT: 0<L<=1 -> Pred: Range 1 (smallest roots)
    # GT: 1<L<=2 -> Pred: Range 2 
    # GT: 2<L<=3 -> Pred: Range 3
    # GT: 3<L<=4 -> Pred: Range 4
    # GT: L>4 -> Pred: Range 5 (largest roots)
    
    gt_columns = [
        'Sum of 0<.L.<=1.000000',
        'Sum of 1.0000000<.L.<=2.0000000', 
        'Sum of 2.0000000<.L.<=3.0000000',
        'Sum of 3.0000000<.L.<=4.0000000',
        'Sum of .L.>4.0000000'
    ]
    
    pred_columns = [
        'Root Length Diameter Range 1 (px)',
        'Root Length Diameter Range 2 (px)',
        'Root Length Diameter Range 3 (px)',
        'Root Length Diameter Range 4 (px)',
        'Root Length Diameter Range 5 (px)'
    ]
    
    return dict(zip(gt_columns, pred_columns))

def load_and_prepare_data():
    """Load ground truth and prediction data, align them by core names"""
    
    # Load data
    gt_path = os.path.join('gt', 'CoresGT.csv')
    pred_path = os.path.join('outputs', 'unet_dataset_2_default_png', 'soil_cores_summary.csv')
    
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)
    
    print(f"Ground truth data shape: {gt_df.shape}")
    print(f"Prediction data shape: {pred_df.shape}")
    
    # Rename columns for consistency
    gt_df = gt_df.rename(columns={'corename': 'soil_core'})
    
    # Get common cores
    gt_cores = set(gt_df['soil_core'])
    pred_cores = set(pred_df['soil_core'])
    common_cores = gt_cores.intersection(pred_cores)
    
    print(f"Common cores: {sorted(common_cores)}")
    print(f"GT only cores: {sorted(gt_cores - pred_cores)}")
    print(f"Pred only cores: {sorted(pred_cores - gt_cores)}")
    
    # Filter to common cores
    gt_filtered = gt_df[gt_df['soil_core'].isin(common_cores)].copy()
    pred_filtered = pred_df[pred_df['soil_core'].isin(common_cores)].copy()
    
    # Sort by core name for alignment
    gt_filtered = gt_filtered.sort_values('soil_core').reset_index(drop=True)
    pred_filtered = pred_filtered.sort_values('soil_core').reset_index(drop=True)
    
    return gt_filtered, pred_filtered

def calculate_correlations(gt_df, pred_df):
    """Calculate correlations between ground truth and prediction features"""
    
    # Get the mapping between GT and prediction columns
    range_mapping = map_gt_to_prediction_ranges()
    
    # Initialize correlation matrix
    gt_columns = list(range_mapping.keys())
    pred_columns = list(range_mapping.values())
    
    # Add total columns
    gt_columns.append('Sum of .L.>0.0000000')  # Total from GT
    pred_columns.append('total_images')  # Total from predictions
    
    # Create correlation matrix
    correlation_matrix = np.zeros((len(gt_columns), len(pred_columns)))
    
    # Calculate correlations
    for i, gt_col in enumerate(gt_columns):
        for j, pred_col in enumerate(pred_columns):
            if gt_col in gt_df.columns and pred_col in pred_df.columns:
                correlation = safe_correlation(gt_df[gt_col].values, pred_df[pred_col].values)
                correlation_matrix[i, j] = correlation
            else:
                correlation_matrix[i, j] = 0.0
    
    return correlation_matrix, gt_columns, pred_columns

def create_correlation_heatmap(correlation_matrix, gt_columns, pred_columns, save_path='correlation_heatmap.png'):
    """Create and save correlation heatmap"""
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r', 
                center=0,
                xticklabels=pred_columns,
                yticklabels=gt_columns,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Correlation Heatmap: Ground Truth vs Predictions', fontsize=14, pad=20)
    plt.xlabel('Prediction Features', fontsize=12)
    plt.ylabel('Ground Truth Features', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Correlation heatmap saved to: {save_path}")

def print_correlation_summary(correlation_matrix, gt_columns, pred_columns):
    """Print summary statistics of correlations"""
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*60)
    
    # Get the mapping for better interpretation
    range_mapping = map_gt_to_prediction_ranges()
    
    print("\nFeature-wise correlations:")
    print("-" * 40)
    
    for i, gt_col in enumerate(gt_columns[:-1]):  # Exclude total
        if gt_col in range_mapping:
            pred_col = range_mapping[gt_col]
            corr_value = correlation_matrix[i, pred_columns.index(pred_col)]
            print(f"{gt_col} <-> {pred_col}: {corr_value:.3f}")
    
    # Total correlation
    total_corr = correlation_matrix[gt_columns.index('Sum of .L.>0.0000000'), 
                                   pred_columns.index('total_images')]
    print(f"Total (Sum of .L.>0.0000000 <-> total_images): {total_corr:.3f}")
    
    # Summary statistics
    print(f"\nCorrelation Statistics:")
    print(f"Mean correlation: {np.mean(correlation_matrix):.3f}")
    print(f"Max correlation: {np.max(correlation_matrix):.3f}")
    print(f"Min correlation: {np.min(correlation_matrix):.3f}")
    print(f"Standard deviation: {np.std(correlation_matrix):.3f}")

def main():
    """Main function to run the correlation analysis"""
    
    print("Loading and preparing data...")
    gt_df, pred_df = load_and_prepare_data()
    
    print("\nCalculating correlations...")
    correlation_matrix, gt_columns, pred_columns = calculate_correlations(gt_df, pred_df)
    
    print("\nCreating correlation heatmap...")
    create_correlation_heatmap(correlation_matrix, gt_columns, pred_columns)
    
    print_correlation_summary(correlation_matrix, gt_columns, pred_columns)
    
    # Save correlation matrix to CSV for further analysis
    corr_df = pd.DataFrame(correlation_matrix, 
                          index=gt_columns, 
                          columns=pred_columns)
    corr_df.to_csv('correlation_matrix.csv')
    print(f"\nCorrelation matrix saved to: correlation_matrix.csv")

if __name__ == "__main__":
    main() 