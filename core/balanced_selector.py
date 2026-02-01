"""
Balanced sampling module for desk rejection submissions.

This module provides functionality to select submissions in a balanced way,
ensuring equal representation of desk-rejected and non-desk-rejected papers.
"""

import random
import pandas as pd
import os
from typing import List, Tuple, Optional
from core.log import LOG

# Path to the submissions metadata CSV
SUBMISSIONS_CSV = "data/iclr/data/submissions.csv"


def load_submissions_metadata(csv_path: str = SUBMISSIONS_CSV) -> pd.DataFrame:
    """
    Load the submissions metadata from CSV file.
    
    :param csv_path: Path to the submissions.csv file
    :return: DataFrame containing submission metadata
    :raises FileNotFoundError: If the CSV file is not found
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Submissions CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    LOG.debug(f"Loaded {len(df)} submissions from {csv_path}")
    return df


def select_balanced_submissions(
    num_per_class: int = 35,
    csv_path: str = SUBMISSIONS_CSV,
    random_seed: Optional[int] = 42,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Select balanced submissions: equal number of desk rejections and non-desk rejections.
    Also ensures ALL desk-rejection categories are represented.
    
    Desk-rejected submissions are allocated proportionally across categories (Formatting, 
    Anonymity, Policy, Scope, etc.). If a category has fewer submissions than its allocation,
    it gets all available submissions and the remaining slots are filled from other categories.
    
    :param num_per_class: Number of submissions to select from each class (default: 35)
    :param csv_path: Path to the submissions.csv file
    :param random_seed: Random seed for reproducibility (default: 42)
    :return: Tuple of (desk_rejected_paths, not_rejected_paths, full_dataframe)
    :raises ValueError: If there aren't enough submissions in either class
    """
    # Load metadata
    df = load_submissions_metadata(csv_path)
    
    # Set random seed for reproducibility
    if random_seed is not None:
        import numpy as np
        np.random.seed(random_seed)
    
    # Separate submissions by status
    desk_rejected_df = df[df['status'] == 'Desk Rejected'].copy()
    not_rejected_df = df[df['status'] != 'Desk Rejected'].copy()
    
    desk_rejected_count = len(desk_rejected_df)
    not_rejected_count = len(not_rejected_df)
    
    LOG.info(f"Total desk rejected submissions: {desk_rejected_count}")
    LOG.info(f"Total non-desk rejected submissions: {not_rejected_count}")
    
    # Validate that we have enough non-desk rejected submissions
    if not_rejected_count < num_per_class:
        raise ValueError(
            f"Not enough non-desk rejected submissions. "
            f"Requested: {num_per_class}, Available: {not_rejected_count}"
        )
    
    # --- DESK REJECTED: Select with category stratification ---
    desk_rejected_sample = _select_stratified_desk_rejected(
        desk_rejected_df, num_per_class, random_seed
    )
    
    # Sample non-desk rejected (no stratification needed)
    not_rejected_sample = not_rejected_df.sample(n=num_per_class, random_state=random_seed)
    
    # Extract directory paths
    desk_rejected_paths = desk_rejected_sample['directory_name'].tolist()
    not_rejected_paths = not_rejected_sample['directory_name'].tolist()
    
    # Combine for returning full dataset
    combined_df = pd.concat([desk_rejected_sample, not_rejected_sample], ignore_index=True)
    
    LOG.info(f"Selected {len(desk_rejected_paths)} desk rejected submissions")
    LOG.info(f"  Categories: {desk_rejected_sample['category'].value_counts().to_dict()}")
    LOG.info(f"Selected {len(not_rejected_paths)} non-desk rejected submissions")
    LOG.info(f"Total balanced selection: {len(combined_df)} submissions")
    
    return desk_rejected_paths, not_rejected_paths, combined_df


def _select_stratified_desk_rejected(
    desk_rejected_df: pd.DataFrame, 
    num_per_class: int = 35,
    random_seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Select desk-rejected submissions ensuring all categories are represented.
    
    Strategy:
    1. Count submissions per category.
    2. Allocate slots proportionally to each category (at least 1 per category).
    3. If a category has fewer submissions than allocated, give it all and redistribute.
    4. Fill remaining slots from any available categories.
    
    :param desk_rejected_df: DataFrame of desk-rejected submissions.
    :param num_per_class: Total number of desk-rejected submissions to select.
    :param random_seed: Random seed for reproducibility.
    :return: DataFrame with selected desk-rejected submissions (stratified by category).
    """
    categories = desk_rejected_df['category'].unique()
    category_counts = desk_rejected_df['category'].value_counts().to_dict()
    
    LOG.info(f"Desk-rejected categories: {dict(category_counts)}")
    
    # Ensure at least 1 slot per category; distribute remaining proportionally
    num_categories = len(categories)
    base_allocation = num_per_class // num_categories
    remainder = num_per_class % num_categories
    
    # Initial allocation: base_allocation per category, +1 for first `remainder` categories
    allocations = {}
    for i, cat in enumerate(sorted(categories)):
        allocations[cat] = base_allocation + (1 if i < remainder else 0)
    
    LOG.debug(f"Initial allocations: {allocations}")
    
    # Adjust: if category has fewer submissions than allocation, cap at available count
    final_allocations = {}
    unfilled_slots = 0
    
    for cat in allocations:
        available = category_counts.get(cat, 0)
        allocated = allocations[cat]
        if available < allocated:
            final_allocations[cat] = available
            unfilled_slots += allocated - available
            LOG.debug(f"Category '{cat}': capping {allocated} -> {available} (surplus {allocated - available})")
        else:
            final_allocations[cat] = allocated
    
    LOG.debug(f"After adjustment: {final_allocations}, unfilled: {unfilled_slots}")
    
    # Fill unfilled slots from categories with surplus
    for cat in sorted(categories):
        if unfilled_slots <= 0:
            break
        available = category_counts.get(cat, 0)
        allocated = final_allocations[cat]
        surplus = available - allocated
        if surplus > 0:
            add = min(surplus, unfilled_slots)
            final_allocations[cat] += add
            unfilled_slots -= add
            LOG.debug(f"Category '{cat}': adding {add} from surplus")
    
    LOG.info(f"Final allocations: {final_allocations}")
    
    # Sample from each category according to allocation
    samples = []
    for cat in final_allocations:
        cat_df = desk_rejected_df[desk_rejected_df['category'] == cat]
        n_to_sample = final_allocations[cat]
        if len(cat_df) > 0 and n_to_sample > 0:
            sample = cat_df.sample(n=n_to_sample, random_state=random_seed)
            samples.append(sample)
    
    result_df = pd.concat(samples, ignore_index=True)
    return result_df


def get_balanced_submission_dirs(
    num_per_class: int = 35,
    csv_path: str = SUBMISSIONS_CSV,
    random_seed: Optional[int] = 42,
) -> List[str]:
    """
    Get a list of directory paths for balanced submissions (desk rejected and non-desk rejected).
    
    This is a convenience function that returns all selected submission directories
    in a single list.
    
    :param num_per_class: Number of submissions to select from each class
    :param csv_path: Path to the submissions.csv file
    :param random_seed: Random seed for reproducibility
    :return: List of full directory paths for selected submissions
    """
    desk_rejected_paths, not_rejected_paths, _ = select_balanced_submissions(
        num_per_class=num_per_class,
        csv_path=csv_path,
        random_seed=random_seed
    )
    
    return get_shuffled_paths(desk_rejected_paths, not_rejected_paths)


def get_shuffled_paths(*args):
    combined = [item for lst in args for item in lst]
    random.shuffle(combined)
    return combined

def get_balanced_submission_info(
    num_per_class: int = 35,
    csv_path: str = SUBMISSIONS_CSV,
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Get detailed information about the balanced selection as a DataFrame.
    
    :param num_per_class: Number of submissions to select from each class
    :param csv_path: Path to the submissions.csv file
    :param random_seed: Random seed for reproducibility
    :return: DataFrame with selected submissions and their metadata
    """
    _, _, combined_df = select_balanced_submissions(
        num_per_class=num_per_class,
        csv_path=csv_path,
        random_seed=random_seed
    )
    
    return combined_df

def find_unfinished_submissions(system_used: str = 'ddr', subdirs: List[str] = None, ) -> List[str]:
    eval_results = pd.read_csv(f'data/iclr/data/evaluation_results_{system_used}.csv')

    finished = eval_results['directory_name'].unique()

    return list(set(subdirs) - set(finished))