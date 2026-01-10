from typing import Dict, Any
from pandas import read_csv
from sklearn.metrics import precision_score, recall_score, f1_score

from core.schemas import FinalDecision


def evaluate_submission_answers_only(evaluation_results: Dict[str, FinalDecision]) -> None:
    """
    Evaluates the model's binary desk-rejection decisions against ground truth labels.

    This function compares the "YES/NO" decisions from the system against a reference 
    CSV file. It calculates standard classification metrics: Precision, Recall, and F1 Score.

    Input requirements:
    - Ground truth is expected in 'data/iclr/data/submissions.csv' with columns:
      'directory_name' and 'label' ('Desk Rejected' or 'Not Desk Rejected').

    :param evaluation_results: A dictionary mapping directory names to FinalDecision objects.
    """

    submissions_df = read_csv("data/iclr/data/submissions.csv",
                              true_values=["Desk Rejected"],
                              false_values=["Not Desk Rejected"])
    predictions_dict = evaluation_results

    submissions_df['y_pred'] = submissions_df['directory_name'].map(predictions_dict)
    submissions_df = submissions_df.dropna(subset=['y_pred'])

    # 5. Calculate Metrics
    # Note: Ensure your 'YES'/'NO' labels match the pos_label parameter
    y_true = submissions_df['label']
    y_pred = submissions_df['y_pred']

    precision = precision_score(y_true, y_pred, pos_label="YES")
    recall = recall_score(y_true, y_pred, pos_label="YES")
    f1 = f1_score(y_true, y_pred, pos_label="YES")

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def evaluate_submission_full(evaluation_results: Dict[str, FinalDecision]) -> None:
    """
    (Placeholder) Performs a deep evaluation of reasoning and evidence snippets.

    Intended Logic:
    - Compare the 'evidence_snippet' and 'reasoning' provided by the agents against 
      human-annotated justifications.
    - Measure the semantic similarity or overlap (e.g., using BERTScore or ROUGE)
      to verify if the agent found the correct violation reasons, not just the right label.

    :param evaluation_results: A dictionary mapping directory names to FinalDecision objects.
    """
    pass