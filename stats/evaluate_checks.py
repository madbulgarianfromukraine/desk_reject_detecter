#!/usr/bin/env python3
"""
Evaluate desk reject detection systems with per-check metrics calculation.

Calculates TP, TN, FP, FN for each check type (formatting, policy, scope, anonymity)
and tracks iteration counts for multi-iteration systems.

Usage:
    python evaluate_checks.py                  # Evaluate both systems
    python evaluate_checks.py ddr_1_iteration  # Evaluate single-iteration system
    python evaluate_checks.py ddr              # Evaluate multi-iteration system

The script will:
1. Calculate best-result metrics for each check type
2. For multi-iteration systems (ddr), also calculate:
   - Iteration counts (how many iterations each check needed)
   - Per-iteration metrics for each check
3. Save LaTeX tables to files for easy inclusion in papers

Output includes:
- Console output with detailed metrics
- LaTeX table files (latex_tables_{system}.tex) for copy-pasting into papers
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from core.config import VertexEngine


def load_data(system_used: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load submissions and evaluation results."""
    data_dir = Path(__file__).parent.parent / "data" / "iclr" / "data"
    submissions_path = data_dir / "submissions.csv"
    evaluation_path = data_dir / f"evaluation_results_{system_used}.csv"
    
    if not submissions_path.exists():
        raise FileNotFoundError(f"Submissions file not found: {submissions_path}")
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {evaluation_path}")
    
    submissions_df = pd.read_csv(submissions_path)
    evaluation_df = pd.read_csv(evaluation_path)
    
    return submissions_df, evaluation_df


def merge_datasets(submissions_df: pd.DataFrame, evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Merge submissions and evaluation data on directory_name."""
    merged_df = pd.merge(
        submissions_df,
        evaluation_df,
        on="directory_name",
        how="inner"
    )
    return merged_df


def calculate_check_metrics(merged_df: pd.DataFrame, check_name: str, use_similarity: bool = True) -> Dict[str, int]:
    """Calculate TP, TN, FP, FN for a specific check."""
    tp = tn = fp = fn = 0
    
    if use_similarity:
        engine = VertexEngine()
    
    for _, row in merged_df.iterrows():
        category = row['category']
        result_best = row.get(f'{check_name}_result_best', False)
        evidence_snippet = row.get(f'{check_name}_evidence_snippet_best', '')
        desk_reject_comment = row.get('desk_reject_comments', '')
        
        # Convert result_best to boolean
        if pd.isna(result_best):
            result_best = False
        else:
            result_best = bool(result_best)
        
        if pd.isna(category) or category is None:
            # No violation expected - all checks should be False
            if result_best:
                fp += 1
            else:
                tn += 1
        else:
            # Check if this is the expected violation category
            expected_check = category.lower()
            current_check = check_name.replace('_check', '').lower()
            
            if expected_check == current_check:
                # This check should detect the violation
                if result_best:
                    # Check semantic similarity if evidence is available
                    if use_similarity and evidence_snippet and desk_reject_comment:
                        try:
                            similarity = engine.get_semantic_similarity(evidence_snippet, desk_reject_comment)
                            if similarity > 0.5:
                                tp += 1
                            else:
                                fn += 1
                        except Exception as e:
                            print("I was here! error:", e)
                            # Fall back to simple match if similarity fails
                            if result_best:
                                tp += 1
                            else:
                                fn += 1
                    else:
                        tp += 1
                else:
                    fn += 1
            else:
                # Other checks should be False for this submission
                if result_best:
                    fp += 1
                else:
                    tn += 1
    
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def calculate_iteration_counts(merged_df: pd.DataFrame, check_name: str, max_iterations: int = 3) -> Dict[int, int]:
    """Calculate how many iterations each check needed per submission."""
    iteration_counts = {}
    
    for i in range(1, max_iterations + 1):
        iteration_counts[i] = 0
    
    for _, row in merged_df.iterrows():
        iterations_used = 0
        
        for i in range(1, max_iterations + 1):
            result_col = f'{check_name}_result_{i}'
            if not pd.isna(row[result_col]):
                iterations_used = i
        
        if iterations_used > 0:
            iteration_counts[iterations_used] += 1
    
    return iteration_counts


def calculate_per_iteration_metrics(merged_df: pd.DataFrame, check_name: str, iteration: int) -> Dict[str, int]:
    """Calculate metrics for a specific iteration of a check."""
    tp = tn = fp = fn = 0
    
    result_col = f'{check_name}_result_{iteration}'
    evidence_col = f'{check_name}_evidence_snippet_{iteration}'
    
    if result_col not in merged_df.columns:
        return {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    
    for _, row in merged_df.iterrows():
        if pd.isna(row[result_col]):
            continue
            
        category = row['category']
        result = bool(row[result_col])
        
        if pd.isna(category) or category is None:
            if result:
                fp += 1
            else:
                tn += 1
        else:
            expected_check = category.lower()
            current_check = check_name.replace('_check', '').lower()
            
            if expected_check == current_check:
                if result:
                    tp += 1
                else:
                    fn += 1
            else:
                if result:
                    fp += 1
                else:
                    tn += 1
    
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def format_metrics_table(results: Dict, title: str = "Results") -> str:
    """Format results into a nice table."""
    # Plain pipe-separated table (easy to paste into LaTeX if needed)
    rows = []
    rows.append(f"{title}")
    header = ["Check", "TP", "TN", "FP", "FN", "Precision", "Recall", "F1", "MCC", "Accuracy"]
    rows.append(" | ".join(header))
    rows.append("-" * 80)

    for check_name, metrics in results.items():
        if isinstance(metrics, dict) and 'tp' in metrics:
            tp, tn, fp, fn = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            # Matthews correlation coefficient
            denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            mcc = (tp * tn - fp * fn) / np.sqrt(denom) if denom > 0 else 0.0

            check_display = check_name.replace('_check', '').title()
            # accuracy
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0

            rows.append(" | ".join([
                check_display,
                str(tp),
                str(tn),
                str(fp),
                str(fn),
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{f1:.3f}",
                f"{mcc:.3f}",
                f"{accuracy:.3f}"
            ]))

    return "\n".join(rows)


def format_iteration_table(iteration_counts: Dict) -> str:
    """Format iteration counts into a table."""
    rows = []
    rows.append("Iteration Counts")
    rows.append("Check | 1 Iteration | 2 Iterations | 3 Iterations")
    rows.append("-" * 40)

    for check_name, counts in iteration_counts.items():
        check_display = check_name.replace('_check', '').title()
        iter1 = counts.get(1, 0)
        iter2 = counts.get(2, 0)
        iter3 = counts.get(3, 0)
        rows.append(f"{check_display} | {iter1} | {iter2} | {iter3}")

    return "\n".join(rows)


def save_latex_tables(system_used: str, best_results: Dict, iteration_counts: Dict = None, per_iteration_results: List = None):
    """Save LaTeX tables to a file for easy copying."""
    output_file = Path(__file__).parent / f"latex_tables_{system_used}.tex"
    
    with open(output_file, 'w') as f:
        f.write(f"% LaTeX tables for {system_used.upper()}\n\n")
        
        f.write(f"% Best Results\n")
        f.write(format_metrics_table(best_results, f"{system_used.upper()} Best Results"))
        f.write("\n\n")
        
        if iteration_counts:
            f.write(f"% Iteration Counts\n")
            f.write(format_iteration_table(iteration_counts))
            f.write("\n\n")
        
        if per_iteration_results:
            for iteration, results in enumerate(per_iteration_results, 1):
                if any(sum(m.values()) > 0 for m in results.values()):
                    f.write(f"% Iteration {iteration} Results\n")
                    f.write(format_metrics_table(results, f"{system_used.upper()} Iteration {iteration}"))
                    f.write("\n\n")
    
    print(f"LaTeX tables saved to: {output_file}")


def evaluate_system(system_used: str) -> None:
    """Main evaluation function for a system."""
    if system_used not in ["ddr_1_iteration", "ddr"]:
        raise ValueError(f"Invalid system_used: {system_used}. Must be 'ddr_1_iteration' or 'ddr'")
    
    print(f"=" * 60)
    print(f"EVALUATING SYSTEM: {system_used}")
    print(f"=" * 60)
    
    # Load and merge data
    submissions_df, evaluation_df = load_data(system_used)
    merged_df = merge_datasets(submissions_df, evaluation_df)
    
    print(f"Total submissions: {len(submissions_df)}")
    print(f"Evaluated submissions: {len(merged_df)}")
    print()
    
    # Define check names
    check_names = ['formatting_check', 'policy_check', 'scope_check', 'anonymity_check']
    
    # Calculate metrics for best results
    print("BEST RESULTS EVALUATION")
    print("-" * 30)
    
    best_results = {}
    for check_name in check_names:
        metrics = calculate_check_metrics(merged_df, check_name)
        best_results[check_name] = metrics
        
        check_display = check_name.replace('_check', '').title()
        print(f"{check_display}: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    
    print("\nBest Results Table:")
    print(format_metrics_table(best_results, f"{system_used.upper()} Best Results"))
    print()
    
    # For ddr system, also calculate per-iteration metrics and iteration counts
    per_iteration_results = []
    iteration_counts = {}
    
    if system_used == "ddr":
        print("ITERATION ANALYSIS")
        print("-" * 30)
        
        # Calculate iteration counts
        for check_name in check_names:
            iteration_counts[check_name] = calculate_iteration_counts(merged_df, check_name)
            
            check_display = check_name.replace('_check', '').title()
            counts = iteration_counts[check_name]
            print(f"{check_display}: 1 iter={counts.get(1, 0)}, 2 iter={counts.get(2, 0)}, 3 iter={counts.get(3, 0)}")
        
        print("\nIteration Counts Table:")
        print(format_iteration_table(iteration_counts))
        print()
        
        # Calculate per-iteration metrics
        for iteration in [1, 2, 3]:
            print(f"ITERATION {iteration} RESULTS")
            print("-" * 30)
            
            iteration_results = {}
            for check_name in check_names:
                metrics = calculate_per_iteration_metrics(merged_df, check_name, iteration)
                iteration_results[check_name] = metrics
                
                check_display = check_name.replace('_check', '').title()
                if sum(metrics.values()) > 0:  # Only show if there are results
                    print(f"{check_display}: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
            
            per_iteration_results.append(iteration_results)
            
            if any(sum(m.values()) > 0 for m in iteration_results.values()):
                print(f"\nIteration {iteration} Results Table:")
                print(format_metrics_table(iteration_results, f"{system_used.upper()} Iteration {iteration}"))
                print()
    else:
        # Save LaTeX tables to file
        save_latex_tables(system_used, best_results, None, None)


def evaluate_all_systems():
    """Evaluate both systems and provide a comprehensive comparison."""
    systems = ["ddr_1_iteration", "ddr"]
    
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("=" * 80)
    print()
    
    for system in systems:
        try:
            evaluate_system(system)
            print()
        except Exception as e:
            print(f"Error evaluating {system}: {e}")
            print()
    
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments - run all systems
        evaluate_all_systems()
    elif len(sys.argv) == 2:
        system_used = sys.argv[1]
        try:
            evaluate_system(system_used)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python evaluate_checks.py [<system_used>]")
        print("system_used must be either 'ddr_1_iteration' or 'ddr'")
        print("If no system_used is provided, both systems will be evaluated")
        sys.exit(1)