# Balanced Selector Module Documentation

## Overview

The `core/balanced_selector.py` module provides functionality for **balanced sampling** of academic paper submissions for evaluation in the desk rejection detection system. It ensures that evaluation datasets contain equal representation of desk-rejected and non-desk-rejected papers, with stratification across different rejection categories.

## Purpose

When evaluating a machine learning system for desk rejection detection, it's crucial to test it on a balanced dataset to get accurate performance metrics (Precision, Recall, F1). This module solves the problem of:

1. **Class Imbalance**: Preventing over-representation of one class (e.g., more non-rejected papers than rejected papers)
2. **Category Stratification**: Ensuring all desk rejection categories (Formatting, Anonymity, Policy, Scope, etc.) are represented
3. **Reproducibility**: Using random seeds to ensure the same evaluation set can be recreated
4. **Resumability**: Finding and re-evaluating incomplete submissions after interruptions

## Key Functions

### 1. `load_submissions_metadata(csv_path)`

**Purpose**: Loads submission metadata from a CSV file.

**Parameters**:
- `csv_path`: Path to the submissions.csv file (default: `"data/iclr/data/submissions.csv"`)

**Returns**: A pandas DataFrame containing submission metadata

**Usage**: This is an internal helper function called by other functions in the module.

```python
df = load_submissions_metadata("data/iclr/data/submissions.csv")
```

### 2. `select_balanced_submissions(num_per_class, csv_path, random_seed)`

**Purpose**: The core function that selects balanced submissions with stratification.

**Parameters**:
- `num_per_class`: Number of submissions to select from each class (default: 35)
- `csv_path`: Path to the submissions.csv file
- `random_seed`: Random seed for reproducibility (default: 42)

**Returns**: A tuple of:
- `desk_rejected_paths`: List of directory names for desk-rejected submissions
- `not_rejected_paths`: List of directory names for non-desk-rejected submissions  
- `combined_df`: DataFrame with all selected submissions and their metadata

**Algorithm**:
1. Separates submissions into desk-rejected and non-desk-rejected groups
2. For desk-rejected papers: Uses stratified sampling to ensure all categories are represented
3. For non-desk-rejected papers: Uses simple random sampling
4. Returns equal numbers from each class

**Example**:
```python
desk_rejected, not_rejected, full_df = select_balanced_submissions(
    num_per_class=50,
    random_seed=42
)
# Returns 50 desk-rejected + 50 non-desk-rejected = 100 total submissions
```

### 3. `_select_stratified_desk_rejected(desk_rejected_df, num_per_class, random_seed)`

**Purpose**: Internal function that performs stratified sampling of desk-rejected submissions.

**Strategy**:
1. Counts submissions per rejection category (Formatting, Anonymity, Policy, etc.)
2. Allocates slots proportionally to each category (minimum 1 per category)
3. If a category has fewer submissions than allocated, it gets all available and remaining slots are redistributed
4. Fills any remaining slots from categories with surplus submissions

**Example**: If you want 35 desk-rejected submissions and there are 5 categories:
- Initial allocation: 7 per category (35/5)
- If "Formatting" only has 3 submissions: it gets 3, and the remaining 4 are distributed to other categories
- Final result: All categories represented, 35 total submissions

### 4. `get_balanced_submission_dirs(num_per_class, csv_path, random_seed)`

**Purpose**: Convenience function that returns a single shuffled list of all selected submission directories.

**Parameters**:
- `num_per_class`: Number of submissions per class (default: 35)
- `csv_path`: Path to submissions CSV
- `random_seed`: Random seed (default: 42)

**Returns**: A shuffled list of directory paths containing both desk-rejected and non-desk-rejected submissions

**Usage**: This is the primary function used by `main.py` for balanced evaluation.

```python
submission_dirs = get_balanced_submission_dirs(num_per_class=50, random_seed=42)
# Returns a shuffled list of 100 submission directory names
```

### 5. `get_balanced_submission_info(num_per_class, csv_path, random_seed)`

**Purpose**: Returns detailed metadata about the balanced selection as a DataFrame.

**Returns**: DataFrame with columns like:
- `directory_name`: Submission directory
- `status`: "Desk Rejected" or other status
- `category`: Rejection category (for desk-rejected papers)
- Other metadata fields

**Usage**: Useful for analyzing the composition of your evaluation set.

### 6. `find_unfinished_submissions(system_used, subdirs)`

**Purpose**: Identifies submissions that haven't been fully evaluated yet.

**Parameters**:
- `system_used`: Name of the evaluation system (e.g., 'ddr', 'sasp', 'sacp')
- `subdirs`: List of submission directories to check

**Returns**: List of submission directories that are NOT in the evaluation results CSV

**How it works**:
1. Reads the evaluation results CSV for the specified system: `data/iclr/data/evaluation_results_{system_used}.csv`
2. Extracts the list of directories that have been evaluated
3. Returns the set difference: `subdirs - finished_submissions`

**Usage**: Allows resuming evaluations after interruptions or errors.

```python
subdirs = ["/path/to/sub1", "/path/to/sub2", "/path/to/sub3"]
unfinished = find_unfinished_submissions(system_used='ddr', subdirs=subdirs)
# Returns directories that haven't been evaluated yet
```

## Where This Logic Is Used

### In `main.py` - The `evaluate_desk_rejection()` Method

The balanced selector is integrated into the main CLI evaluation pipeline:

#### 1. **Balanced Selection Mode** (lines 144-151 in main.py)

When `--balanced` flag is set:

```bash
python main.py evaluate_desk_rejection ./submissions --balanced --per_class 50
```

Code flow:
```python
if balanced:
    balanced_dirs = get_balanced_submission_dirs(num_per_class=per_class, random_seed=42)
    subdirs = [os.path.join(directory, os.path.basename(d)) for d in balanced_dirs]
```

This ensures:
- Equal representation of desk-rejected and non-desk-rejected papers
- All rejection categories are included
- Reproducible evaluation sets

#### 2. **Resume Incomplete Evaluations** (lines 160-161 in main.py)

When `--find_unfinished` flag is set:

```bash
python main.py evaluate_desk_rejection ./submissions --find_unfinished
```

Code flow:
```python
if find_unfinished:
    subdirs = find_unfinished_submissions(system_used=system_used, subdirs=subdirs)
```

This allows:
- Resuming evaluations after crashes or interruptions
- Re-evaluating only failed submissions
- Saving time by skipping already-processed papers

## Complete Usage Examples

### Example 1: Balanced Evaluation of 70 Papers

```bash
python main.py evaluate_desk_rejection ./data/iclr/submissions \
    --balanced \
    --per_class 35 \
    --system_used ddr
```

This will:
- Select 35 desk-rejected papers (stratified across all categories)
- Select 35 non-desk-rejected papers (random sampling)
- Total: 70 papers for evaluation

### Example 2: Resume Interrupted Evaluation

```bash
# First run (interrupted)
python main.py evaluate_desk_rejection ./data/iclr/submissions \
    --limit 100 \
    --system_used ddr

# Resume later
python main.py evaluate_desk_rejection ./data/iclr/submissions \
    --find_unfinished \
    --system_used ddr
```

This will:
- Check `data/iclr/data/evaluation_results_ddr.csv` for completed submissions
- Only evaluate submissions that are missing from the results
- Save evaluation time by not re-processing

### Example 3: Combining Balanced and Unfinished

```bash
python main.py evaluate_desk_rejection ./data/iclr/submissions \
    --balanced \
    --per_class 50 \
    --find_unfinished \
    --system_used ddr
```

Note: When `find_unfinished` is set, it takes precedence over `balanced` selection (see line 160-162 in main.py).

## Data Structure Requirements

The module expects a CSV file at `data/iclr/data/submissions.csv` with at least these columns:

- `directory_name`: Name of the submission directory
- `status`: Submission status (must contain "Desk Rejected" for rejected papers)
- `category`: Category of desk rejection (for rejected papers)

Example CSV structure:
```csv
directory_name,status,category
sub_001,Desk Rejected,Formatting
sub_002,Accepted,
sub_003,Desk Rejected,Anonymity
sub_004,Rejected,
sub_005,Desk Rejected,Policy
```

## Benefits of This Approach

1. **Accurate Metrics**: Balanced datasets provide realistic Precision/Recall/F1 scores
2. **Category Coverage**: Stratification ensures all rejection types are tested
3. **Reproducibility**: Random seeds allow recreating the exact same evaluation set
4. **Efficiency**: Resume capability saves time and compute resources
5. **Flexibility**: Multiple parameters allow customizing evaluation scope

## Related Files

- **`main.py`**: Uses this module in the `evaluate_desk_rejection()` CLI method
- **`core/metrics.py`**: Calculates evaluation metrics using the balanced datasets
- **`systems/ddr.py`**: The desk rejection detection system being evaluated
- **`data/iclr/data/submissions.csv`**: Source metadata file (not in repo, must be provided)
- **`data/iclr/data/evaluation_results_{system}.csv`**: Generated results files

## Summary

The `balanced_selector.py` module is a critical component for **fair and comprehensive evaluation** of the desk rejection detection system. It ensures that:

- ✅ Evaluation sets are balanced between positive and negative classes
- ✅ All desk rejection categories are represented
- ✅ Results are reproducible across runs
- ✅ Interrupted evaluations can be resumed efficiently

Without this module, evaluations could be biased by class imbalance or missing certain rejection categories, leading to misleading performance metrics.
