"""
Script to generate a pie chart of submissions based on desk rejection categories.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path


def plot_desk_rejection_pie(
    csv_file: Union[str, Path],
    output_path: Union[str, Path] = None,
    title: str = "Submissions by Desk Rejection Category"
) -> None:
    """
    Generate a pie chart of submissions based on desk rejection categories.
    
    Parameters:
    -----------
    csv_file : str or Path
        Path to the CSV file containing submission data.
        Expected columns: 'status' and 'category'
        - 'status': Should contain 'Desk Rejected' or 'Not Desk Rejected'
        - 'category': Contains the rejection category (Formatting, Anonymity, Policy, etc.)
        
    output_path : str or Path, optional
        Path where the pie chart will be saved. If None, displays the chart instead.
        
    title : str, optional
        Title for the pie chart. Default is "Submissions by Desk Rejection Category"
    
    Returns:
    --------
    None
    
    Example:
    --------
    >>> plot_desk_rejection_pie('submissions.csv')
    >>> plot_desk_rejection_pie('submissions.csv', output_path='desk_rejection_pie.png')
    """
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a new column for rejection category
    # Use 'category' if the submission was desk rejected, otherwise use 'None'
    df['rejection_category'] = df.apply(
        lambda row: row['category'] if row['status'] == 'Desk Rejected' and pd.notna(row['category']) and row['category'] != ''
        else 'Not Desk Rejected',
        axis=1
    )
    
    # Count submissions by rejection category
    category_counts = df['rejection_category'].value_counts()
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(range(len(category_counts)))
    total = int(category_counts.values.sum())
    wedges, texts, autotexts = ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct=lambda pct: str(int(round(pct * total / 100.0))),
        colors=colors,
        startangle=90
    )
    
    # Enhance the appearance
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Make labels more readable
    for text in texts:
        text.set_fontsize(11)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to: {output_path}")
    else:
        plt.show()
    
    # Print statistics
    print("\nSubmission Statistics:")
    print("-" * 50)
    print(f"Total submissions: {len(df)}")
    print(f"\nBreakdown:")
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    plot_desk_rejection_pie('data/iclr/data/submissions.csv', output_path='desk_rejection_pie.png')


def plot_evaluated_desk_rejection_pie(
    evaluation_csv: Union[str, Path],
    submissions_csv: Union[str, Path] = None,
    output_path: Union[str, Path] = None,
    title: str = "Evaluated Submissions by Desk Rejection Category",
    system_used: str | None = None,
) -> None:
    """Plot pie chart for submissions that were evaluated (present in evaluation CSV).

    This function loads `evaluation_csv` (e.g. evaluation_results_ddr.csv),
    merges it with `submissions_csv` (defaults to data/iclr/data/submissions.csv),
    and plots the distribution of `category` for those evaluated submissions.
    """

    eval_path = Path(evaluation_csv)
    if not eval_path.exists():
        # if user passed a system name instead of path, try to construct filename
        if system_used is not None:
            eval_path = Path(__file__).parent.parent / "data" / "iclr" / "data" / f"evaluation_results_{system_used}.csv"
        else:
            raise FileNotFoundError(f"Evaluation CSV not found: {evaluation_csv}")

    eval_df = pd.read_csv(eval_path)

    if submissions_csv is None:
        submissions_path = Path(__file__).parent.parent / "data" / "iclr" / "data" / "submissions.csv"
    else:
        submissions_path = Path(submissions_csv)

    submissions_df = pd.read_csv(submissions_path)

    # Keep only evaluated submissions
    merged = pd.merge(submissions_df, eval_df[['directory_name']], on='directory_name', how='inner')

    # Map categories: if desk rejected and category available, use it; else 'Not Desk Rejected'
    merged['rejection_category'] = merged.apply(
        lambda r: r['category'] if (r.get('status') == 'Desk Rejected' and pd.notna(r.get('category')) and r.get('category') != '') else 'Not Desk Rejected',
        axis=1
    )

    category_counts = merged['rejection_category'].value_counts()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(range(len(category_counts)))
    total = int(category_counts.values.sum())
    wedges, texts, autotexts = ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct=lambda pct: str(int(round(pct * total / 100.0))),
        colors=colors,
        startangle=90
    )
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(11)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to: {output_path}")
    else:
        plt.show()

    # Print statistics for evaluated subset
    print("\nEvaluated Submission Statistics:")
    print("-" * 50)
    print(f"Total evaluated submissions: {len(merged)}")
    print(f"\nBreakdown:")
    for category, count in category_counts.items():
        percentage = (count / len(merged)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    # Provide a simple CLI for plotting evaluated submissions for a given system
    import argparse

    parser = argparse.ArgumentParser(description='Plot evaluated submissions pie chart')
    parser.add_argument('--system', type=str, help='system name (e.g. ddr, ddr_1_iteration) to load evaluation_results_{system}.csv')
    parser.add_argument('--evaluation-csv', type=str, help='Path to evaluation CSV (overrides --system)')
    parser.add_argument('--output', type=str, help='Output image path (optional)')
    args = parser.parse_args()

    if args.evaluation_csv:
        plot_evaluated_desk_rejection_pie(args.evaluation_csv, output_path=args.output)
    elif args.system:
        plot_evaluated_desk_rejection_pie(args.system, output_path=args.output, system_used=args.system)
    else:
        # default: use ddr evaluation file
        plot_evaluated_desk_rejection_pie('ddr', output_path='evaluated_desk_rejection_pie.png', system_used='ddr')
