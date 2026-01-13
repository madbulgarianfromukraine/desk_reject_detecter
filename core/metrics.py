from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.schemas import FinalDecision
from core.log import LOG

__EVALUATION_RESULT_CSV= "data/iclr/data/evaluation_results.csv"

def evaluate_submission_answers_only(evaluation_results: Dict[str, FinalDecision]) -> None:
    """
    Evaluates the model's binary desk-rejection decisions against ground truth labels.

    This function compares the "YES/NO" decisions from the system against a reference 
    CSV file. It calculates standard classification metrics: Precision, Recall, and F1 Score.

    Input requirements:
    - Ground truth is expected in 'data/iclr/data/submissions.csv' with columns:
      'directory_name' and 'status' ('Desk Rejected' or 'Not Desk Rejected').

    :param evaluation_results: A dictionary mapping directory names to FinalDecision objects.
    """

    submissions_df = pd.read_csv("data/iclr/data/submissions.csv")
    predictions_dict = evaluation_results

    submissions_df['y_pred_obj'] = submissions_df['directory_name'].map(predictions_dict)
    submissions_df = submissions_df.dropna(subset=['y_pred_obj'])

    # 5. Calculate Metrics
    y_true = submissions_df['status'].map({"Desk Rejected": "YES", "Not Desk Rejected": "NO"})
    y_pred = submissions_df['y_pred_obj'].apply(lambda x: x.desk_reject_decision)

    precision = precision_score(y_true, y_pred, pos_label="YES", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="YES", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="YES", zero_division=0)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculates the cosine similarity between two strings using TF-IDF vectorization.
    """
    if not text1 or not text2:
        return 1.0 if not text1 and not text2 else 0.0
    
    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0


def evaluate_submission_full(evaluation_results: Dict[str, FinalDecision], system_used: str = 'ddr') -> None:
    """
    Performs a deep evaluation of reasoning and evidence snippets.
    
    Formula: 
    score = (y_true_status == y_pred_status) * (y_true_category in y_pred_categories) * similarity(y_true_comment, y_pred_snippet)
    
    :param evaluation_results: A dictionary mapping directory names to FinalDecision objects.
    """
    # Load Ground Truth
    submissions_df = pd.read_csv("data/iclr/data/submissions.csv")
    evaluation_results_df = submissions_df

    evaluation_results_df = evaluation_results_df[['directory_name']]

    # Mapping CSV status to model decision
    STATUS_MAP = {
        "Desk Rejected": "YES",
        "Not Desk Rejected": "NO"
    }
    
    # Mapping Category to AnalysisReport attribute
    CATEGORY_TO_ATTR = {
        "Formatting": "formatting_check",
        "Anonymity": "anonymity_check",
        "Policy": "policy_check",
        "Scope": "scope_check",
        "Code_of_Ethics": "safety_check",
        "Visual_Integrity": "visual_integrity_check"
    }

    total_scores = []

    for directory_name, decision in evaluation_results.items():
        # Get ground truth
        row = submissions_df[submissions_df['directory_name'] == directory_name]
        if row.empty:
            continue
        row = row.iloc[0]

        y_true_status = STATUS_MAP.get(row["status"], "NO")
        y_true_category = row["category"] if pd.notna(row["category"]) and row["category"] != "" else "None"
        y_true_comment = row["desk_reject_comments"] if pd.notna(row["desk_reject_comments"]) else ""

        # 1. Status Match
        status_match = 1 if y_true_status == decision.desk_reject_decision else 0
        
        # 2. Category Match
        category_match = 1 if y_true_category in decision.categories else 0
        
        # 3. Evidence Similarity
        similarity_score = 0.0
        if y_true_status == "YES":
            if status_match and category_match:
                attr_name = CATEGORY_TO_ATTR.get(y_true_category)
                if attr_name and hasattr(decision.analysis, attr_name):
                    check_result = getattr(decision.analysis, attr_name)
                    y_pred_snippet = check_result.evidence_snippet
                    similarity_score = calculate_similarity(y_true_comment, y_pred_snippet)
                else:
                    similarity_score = 0.0
            else:
                similarity_score = 0.0
        else: # y_true_status == "NO"
            if status_match:
                # If correctly not desk rejected, we expect "None" category and empty comment
                if "None" in decision.categories:
                    category_match = 1
                    similarity_score = 1.0
                else:
                    category_match = 0
                    similarity_score = 0.0
            else:
                similarity_score = 0.0

        score = status_match * category_match * similarity_score
        evaluation_results_df[evaluation_results_df['directory_name'] == directory_name, 'category_match'] = category_match
        evaluation_results_df[evaluation_results_df['directory_name'] == directory_name, 'status_match'] = status_match
        evaluation_results_df[evaluation_results_df['directory_name'] == directory_name, 'similarity_score'] = similarity_score

        total_scores.append(score)

    if total_scores:
        final_score = np.mean(total_scores)
        LOG.debug(f"Full Evaluation Score: {final_score:.4f}")

        evaluation_results_df.to_csv(path_or_buf=__EVALUATION_RESULT_CSV)
        LOG.debug(f"Saved evaluation results to {__EVALUATION_RESULT_CSV}")

    else:
        LOG.warn("No matching submissions found for full evaluation.")