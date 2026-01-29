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
    
    This function ensures unbiased class representation by selecting submissions
    such that exactly num_per_class submissions are desk rejected and num_per_class
    are not desk rejected.
    
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
    
    # Validate that we have enough submissions in each class
    if desk_rejected_count < num_per_class:
        raise ValueError(
            f"Not enough desk rejected submissions. "
            f"Requested: {num_per_class}, Available: {desk_rejected_count}"
        )
    
    if not_rejected_count < num_per_class:
        raise ValueError(
            f"Not enough non-desk rejected submissions. "
            f"Requested: {num_per_class}, Available: {not_rejected_count}"
        )
    
    # Sample from each class
    desk_rejected_sample = desk_rejected_df.sample(n=num_per_class, random_state=random_seed)
    not_rejected_sample = not_rejected_df.sample(n=num_per_class, random_state=random_seed)
    
    # Extract directory paths
    desk_rejected_paths = desk_rejected_sample['directory_name'].tolist()
    not_rejected_paths = not_rejected_sample['directory_name'].tolist()
    
    # Combine for returning full dataset
    combined_df = pd.concat([desk_rejected_sample, not_rejected_sample], ignore_index=True)
    
    LOG.info(f"Selected {len(desk_rejected_paths)} desk rejected submissions")
    LOG.info(f"Selected {len(not_rejected_paths)} non-desk rejected submissions")
    LOG.info(f"Total balanced selection: {len(combined_df)} submissions")
    
    return desk_rejected_paths, not_rejected_paths, combined_df


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
