#!/usr/bin/env python3
"""
Calculate confusion matrix metrics (TP, TN, FP, FN) for desk rejection detection.

True values: submissions.csv (status == 'Desk Rejected')
Predicted values: evaluation_results_ddr_1_iteration.csv (status_match == 1.0)
"""

import pandas as pd
import numpy as np
from pathlib import Path

__level_dict = {
    1 : "status_match",
    2: "category_match",
    3: "similarity_score"
}

def calculate_metrics(system_to_evaluate: str, level: int = 1):
    """Calculate TP, TN, FP, FN metrics."""
    # Define paths
    data_dir = Path(__file__).parent.parent / "data" / "iclr" / "data"
    submissions_path = data_dir / "submissions.csv"
    evaluation_path = data_dir / f"evaluation_results_{system_to_evaluate}.csv"
    
    # Load data
    submissions_df = pd.read_csv(submissions_path)
    evaluation_df = pd.read_csv(evaluation_path)
    
    # Extract submission IDs from directory names
    # directory_name format: "data/iclr/data/submission_<id>"
    
    # Merge datasets
    merged_df = pd.merge(
        submissions_df,
        evaluation_df,
        on="directory_name",
        how="inner"
    )
    
    
    # Check for missing submissions
    only_submissions = pd.merge(
        submissions_df,
        evaluation_df,
        on="directory_name",
        how="left",
        indicator=True
    )
    
    # Define true labels (1 = Desk Rejected, 0 = Not Desk Rejected)
    y_true = (merged_df["status"] == "Desk Rejected").astype(int)
    
    # Show distribution of tested submissions
    desk_rejected_count = (y_true == 1).sum()
    not_desk_rejected_count = (y_true == 0).sum()
    
    # Define predicted labels (1 = Match detected, 0 = No match)
    print(f'Evaluating for {__level_dict[level]}')
    y_pred = merged_df[__level_dict[level]]

    error_raise = (merged_df["total_input_tokens"] > 0.0).astype(int)
    # Calculate metrics
    tp = ((y_true == 1) & (y_pred > 0) & error_raise).sum()  # Correctly predicted desk rejected
    tn = ((y_true == 0) & (y_pred > 0) & error_raise).sum()  # Correctly predicted not desk rejected
    fp = ((y_true == 0) & (y_pred == 0) & error_raise).sum()  # Falsely predicted desk rejected
    fn = ((y_true == 1) & (y_pred == 0) & error_raise).sum()  # Missed desk rejections
    error_raise = len(merged_df) - error_raise.sum()
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(denom) if denom > 0 else 0.0
    
    # Filter for successful executions (no errors)
    successful_df = merged_df[merged_df["total_input_tokens"] > 0.0]
    
    # Calculate token averages (only for successful executions)
    avg_input_tokens = successful_df["total_input_tokens"].sum() / 100 if len(successful_df) > 0 else 0
    avg_output_tokens = successful_df["total_output_tokens"].sum() / 100 if len(successful_df) > 0 else 0
    
    # Calculate costs: $0.30 per 1M input tokens, $2.50 per 1M output tokens
    cost_per_1m_input = 0.30
    cost_per_1m_output = 2.50
    avg_input_cost = (avg_input_tokens * cost_per_1m_input) / 1_000_000 if len(successful_df) > 0 else 0
    avg_output_cost = (avg_output_tokens * cost_per_1m_output) / 1_000_000 if len(successful_df) > 0 else 0
    avg_total_cost = avg_input_cost + avg_output_cost
    
    # Calculate total costs
    total_input_tokens = successful_df["total_input_tokens"].sum() if len(successful_df) > 0 else 0
    total_output_tokens = successful_df["total_output_tokens"].sum() if len(successful_df) > 0 else 0
    total_input_cost = (total_input_tokens * cost_per_1m_input) / 1_000_000
    total_output_cost = (total_output_tokens * cost_per_1m_output) / 1_000_000
    total_cost = total_input_cost + total_output_cost
    
    # Calculate average execution time (only for successful executions)
    avg_execution_time = successful_df["total_elapsed_time"].mean() if len(successful_df) > 0 else 0
    
    # Display results
    print("=" * 50)
    print("CONFUSION MATRIX METRICS")
    print("=" * 50)
    print(f"True Positives (TP):  {tp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f'Error Raise: {error_raise}')
    print("=" * 50)
    print()
    
    print("=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print
    print("=" * 50)
    print()
    
    # Display token and cost metrics
    print("=" * 50)
    print("TOKEN AND COST METRICS (Successful Executions Only)")
    print("=" * 50)
    print(f"Total Successful Executions: {len(successful_df)}")
    print()
    print(f"Average Input Tokens:  {avg_input_tokens:,.0f}")
    print(f"Average Output Tokens: {avg_output_tokens:,.0f}")
    print()
    print(f"Total Input Tokens:  {total_input_tokens:,.0f}")
    print(f"Total Output Tokens: {total_output_tokens:,.0f}")
    print()
    print(f"Average Input Cost (@ $0.30/1M):  ${avg_input_cost:.6f}")
    print(f"Average Output Cost (@ $2.50/1M): ${avg_output_cost:.6f}")
    print(f"Average Total Cost per Submission: ${avg_total_cost:.6f}")
    print()
    print(f"Total Input Cost (@ $0.30/1M):  ${total_input_cost:.2f}")
    print(f"Total Output Cost (@ $2.50/1M): ${total_output_cost:.2f}")
    print(f"Total Cost: ${total_cost:.2f}")
    print()
    print(f"Average Execution Time: {avg_execution_time:.2f} seconds")
    print("=" * 50)
    print()
    
    # Show breakdown by category
    print("=" * 50)
    print("BREAKDOWN BY CATEGORY")
    print("=" * 50)


if __name__ == "__main__":
    for system_used in ["sasp", "sacp", "ddr_1_iteration", "ddr"]:
        for l in [3]:
            print(f"Evaluating for level={l} and system_used={system_used}")
            calculate_metrics(system_to_evaluate=system_used, level=l)
