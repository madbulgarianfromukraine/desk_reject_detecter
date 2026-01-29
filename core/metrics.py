from typing import Dict, Any, List
import re
from dataclasses import dataclass
import threading

# data manipulation libraries
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from core.config import VertexEngine
from core.schemas import FinalDecision
from core.log import LOG

__EVALUATION_RESULT_CSV= "data/iclr/data/evaluation_results"

# global variables used for token counting
__SUBMISSION_INPUT_TOKENS : int = 0
__SUBMISSION_OUTPUT_TOKENS : int = 0
# Locks to prevent race conditions for variable changes.
__INPUT_TOKENS_LOCK : threading.Lock = threading.Lock()
__OUTPUT_TOKENS_LOCK : threading.Lock = threading.Lock()

@dataclass
class SubmissionMetrics:

    def __init__(self, final_decision: FinalDecision = None, total_input_token_count: int = 0,
                 total_output_token_count: int = 0,
                 total_elapsed_time: float = 0.0,
                 submission_id: str = None,
                 system_name: str = None,
                 category: str = None,
                 sub_category: str = None,
                 reasoning: str = None,
                 confidence_score: float = None,
                 error_type: str = None,
                 error_message: str = None):
        self.final_decision = final_decision
        self.total_input_token_count = total_input_token_count
        self.total_output_token_count = total_output_token_count
        self.total_elapsed_time = total_elapsed_time
        # Additional fields for SASP/SACP compatibility
        self.submission_id = submission_id
        self.system_name = system_name
        self.category = category
        self.sub_category = sub_category
        self.reasoning = reasoning
        self.confidence_score = confidence_score
        # Error tracking fields
        self.error_type = error_type
        self.error_message = error_message
    
    def to_final_decision(self, status: str) -> FinalDecision:
        """
        Converts SASP/SACP metrics to FinalDecision format for evaluation compatibility.
        
        :param status: The status from SASP/SACP (ACCEPT/REJECT/UNCERTAIN)
        :return: FinalDecision object compatible with evaluate_submission_full
        """
        from core.schemas import AnalysisReport, SafetyCheck, AnonymityCheck, FormattingCheck, PolicyCheck, VisualIntegrityCheck, ScopeCheck
        
        # Map SASP/SACP category to FinalDecision category
        category_map = {
            "Code_of_Ethics": "Code_of_Ethics",
            "Anonymity": "Anonymity",
            "Formatting": "Formatting",
            "Visual_Integrity": "Visual_Integrity",
            "Policy": "Policy",
            "Scope": "Scope",
            "None": "None"
        }
        
        mapped_category = category_map.get(self.category, "None")
        
        # Create empty check templates
        empty_check_template = {
            "safety": SafetyCheck(violation_found=False, issue_type="None", evidence_snippet="", reasoning="", confidence_score=0.0),
            "anonymity": AnonymityCheck(violation_found=False, issue_type="None", evidence_snippet="", reasoning="", confidence_score=0.0),
            "visual": VisualIntegrityCheck(violation_found=False, issue_type="None", evidence_snippet="", reasoning="", confidence_score=0.0),
            "formatting": FormattingCheck(violation_found=False, issue_type="None", evidence_snippet="", reasoning="", confidence_score=0.0),
            "policy": PolicyCheck(violation_found=False, issue_type="None", evidence_snippet="", reasoning="", confidence_score=0.0),
            "scope": ScopeCheck(violation_found=False, issue_type="None", evidence_snippet="", reasoning="", confidence_score=0.0),
        }
        
        # Create the actual check for the identified category
        category_to_check_map = {
            "Code_of_Ethics": "safety",
            "Anonymity": "anonymity",
            "Formatting": "formatting",
            "Visual_Integrity": "visual",
            "Policy": "policy",
            "Scope": "scope"
        }
        
        # Build the analysis report
        if mapped_category != "None" and mapped_category in category_to_check_map:
            check_type = category_to_check_map[mapped_category]
            if check_type == "safety":
                check = SafetyCheck(violation_found=True, issue_type=self.sub_category if self.sub_category != "None" else "None", 
                                   evidence_snippet=self.reasoning or "", reasoning=self.reasoning or "", confidence_score=self.confidence_score or 0.0)
                analysis = AnalysisReport(
                    safety_check=check,
                    anonymity_check=empty_check_template["anonymity"],
                    visual_integrity_check=empty_check_template["visual"],
                    formatting_check=empty_check_template["formatting"],
                    policy_check=empty_check_template["policy"],
                    scope_check=empty_check_template["scope"]
                )
            elif check_type == "anonymity":
                check = AnonymityCheck(violation_found=True, issue_type=self.sub_category if self.sub_category != "None" else "None",
                                      evidence_snippet=self.reasoning or "", reasoning=self.reasoning or "", confidence_score=self.confidence_score or 0.0)
                analysis = AnalysisReport(
                    safety_check=empty_check_template["safety"],
                    anonymity_check=check,
                    visual_integrity_check=empty_check_template["visual"],
                    formatting_check=empty_check_template["formatting"],
                    policy_check=empty_check_template["policy"],
                    scope_check=empty_check_template["scope"]
                )
            elif check_type == "formatting":
                check = FormattingCheck(violation_found=True, issue_type=self.sub_category if self.sub_category != "None" else "None",
                                       evidence_snippet=self.reasoning or "", reasoning=self.reasoning or "", confidence_score=self.confidence_score or 0.0)
                analysis = AnalysisReport(
                    safety_check=empty_check_template["safety"],
                    anonymity_check=empty_check_template["anonymity"],
                    visual_integrity_check=empty_check_template["visual"],
                    formatting_check=check,
                    policy_check=empty_check_template["policy"],
                    scope_check=empty_check_template["scope"]
                )
            elif check_type == "visual":
                check = VisualIntegrityCheck(violation_found=True, issue_type=self.sub_category if self.sub_category != "None" else "None",
                                            evidence_snippet=self.reasoning or "", reasoning=self.reasoning or "", confidence_score=self.confidence_score or 0.0)
                analysis = AnalysisReport(
                    safety_check=empty_check_template["safety"],
                    anonymity_check=empty_check_template["anonymity"],
                    visual_integrity_check=check,
                    formatting_check=empty_check_template["formatting"],
                    policy_check=empty_check_template["policy"],
                    scope_check=empty_check_template["scope"]
                )
            elif check_type == "policy":
                check = PolicyCheck(violation_found=True, issue_type=self.sub_category if self.sub_category != "None" else "None",
                                   evidence_snippet=self.reasoning or "", reasoning=self.reasoning or "", confidence_score=self.confidence_score or 0.0)
                analysis = AnalysisReport(
                    safety_check=empty_check_template["safety"],
                    anonymity_check=empty_check_template["anonymity"],
                    visual_integrity_check=empty_check_template["visual"],
                    formatting_check=empty_check_template["formatting"],
                    policy_check=check,
                    scope_check=empty_check_template["scope"]
                )
            else:  # scope
                check = ScopeCheck(violation_found=True, issue_type=self.sub_category if self.sub_category != "None" else "None",
                                  evidence_snippet=self.reasoning or "", reasoning=self.reasoning or "", confidence_score=self.confidence_score or 0.0)
                analysis = AnalysisReport(
                    safety_check=empty_check_template["safety"],
                    anonymity_check=empty_check_template["anonymity"],
                    visual_integrity_check=empty_check_template["visual"],
                    formatting_check=empty_check_template["formatting"],
                    policy_check=empty_check_template["policy"],
                    scope_check=check
                )
        else:
            # No violation found
            analysis = AnalysisReport(
                safety_check=empty_check_template["safety"],
                anonymity_check=empty_check_template["anonymity"],
                visual_integrity_check=empty_check_template["visual"],
                formatting_check=empty_check_template["formatting"],
                policy_check=empty_check_template["policy"],
                scope_check=empty_check_template["scope"]
            )
        
        # Convert status to YES/NO
        
        return FinalDecision(
            desk_reject_decision=status,
            categories=mapped_category,
            confidence_score=self.confidence_score or None,
            analysis=analysis
        )


def increase_total_input_tokens(additional_tokens: int):
    global __INPUT_TOKENS_LOCK, __SUBMISSION_INPUT_TOKENS
    if additional_tokens > 0:
        with __INPUT_TOKENS_LOCK:
            __SUBMISSION_INPUT_TOKENS += additional_tokens

def get_total_input_tokens():
    global __SUBMISSION_INPUT_TOKENS
    old_total_input_tokens = __SUBMISSION_INPUT_TOKENS
    __SUBMISSION_INPUT_TOKENS = 0
    return  old_total_input_tokens

def increase_total_output_tokens(additional_tokens: int):
    global __OUTPUT_TOKENS_LOCK, __SUBMISSION_OUTPUT_TOKENS
    if additional_tokens > 0:
        with __OUTPUT_TOKENS_LOCK:
            __SUBMISSION_OUTPUT_TOKENS += additional_tokens

def get_total_output_tokens():
    global __SUBMISSION_OUTPUT_TOKENS
    old_total_input_tokens = __SUBMISSION_OUTPUT_TOKENS
    __SUBMISSION_OUTPUT_TOKENS = 0
    return  old_total_input_tokens


def evaluate_submission_answers_only(evaluation_results: Dict[str, SubmissionMetrics]) -> None:
    """
    Evaluates the model's binary desk-rejection decisions against ground truth labels.

    This function compares the "YES/NO" decisions from the system against a reference 
    CSV file. It calculates standard classification metrics: Precision, Recall, and F1 Score.

    Input requirements:
    - Ground truth is expected in 'data/iclr/data/submissions.csv' with columns:
      'directory_name' and 'status' ('Desk Rejected' or 'Not Desk Rejected').

    :param evaluation_results: A dictionary mapping directory names to SubmissionMetrics objects.
    """

    submissions_df = pd.read_csv("data/iclr/data/submissions.csv")

    # Convert SASP/SACP metrics to FinalDecision format if needed
    predictions_dict = {}
    for dir_name, metrics in evaluation_results.items():
        if metrics:
            if metrics.final_decision is None and metrics.system_name in ['SASP', 'SACP']:
                # Convert SASP/SACP metrics
                predictions_dict[dir_name] = metrics.to_final_decision(status="REJECT" if metrics.category != "None" else "ACCEPT")
            else:
                predictions_dict[dir_name] = metrics.final_decision
        else:
            predictions_dict[dir_name] = None

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


def evaluate_submission_full(evaluation_results: Dict[str, SubmissionMetrics], system_used: str = 'ddr',
                             skip: int = 0) -> None:
    """
    Performs a deep evaluation of reasoning and evidence snippets.
    
    Formula: 
    score = (y_true_status == y_pred_status) * (y_true_category in y_pred_categories) * similarity(y_true_comment, y_pred_snippet)
    
    :param evaluation_results: A dictionary mapping directory names to SubmissionMetrics objects.
    :param system_used: to be able to create an identifiable csv file
    :param skip: identifies whether to append or to write new csv file
    """
    # Load Ground Truth
    submissions_df = pd.read_csv("data/iclr/data/submissions.csv")

    evaluation_results_df = pd.DataFrame({'directory_name' : list(evaluation_results.keys())})
    evaluation_results_df.set_index('directory_name', inplace=True)

    evaluation_results_df.loc[:, 'category_match'] = 0.0
    evaluation_results_df.loc[:, 'status_match'] = 0.0
    evaluation_results_df.loc[:, 'similarity_score'] = 0.0
    evaluation_results_df.loc[:, 'total_input_tokens'] = 0
    evaluation_results_df.loc[:, 'total_output_tokens'] = 0
    evaluation_results_df.loc[:, 'total_elapsed_time']  = 0.0
    evaluation_results_df.loc[:, "error_status"] = None
    evaluation_results_df.loc[:, "error_message"] = None

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

    for directory_name, metrics in evaluation_results.items():
        # Get ground truth
        if metrics:
            # Store error information if present
            if metrics.error_type:
                evaluation_results_df.loc[directory_name, "error_status"] = metrics.error_type
                evaluation_results_df.loc[directory_name, "error_message"] = metrics.error_message or ""
                LOG.warning(f"{directory_name}: {metrics.error_type} - {metrics.error_message}")
                continue  # Skip evaluation for failed submissions
            
            # Convert SASP/SACP metrics to FinalDecision if needed
            if metrics.final_decision is None and metrics.system_name in ['SASP', 'SACP']:
                # This is a SASP/SACP metrics object, convert it
                # Determine status: REJECT if a violation category found, else ACCEPT
                status = "YES" if metrics.category != "None" else "NO"
                decision = metrics.to_final_decision(status=status)
                LOG.debug(f"Converted {metrics.system_name} metrics to FinalDecision for {directory_name} with status {status}")
            else:
                decision = metrics.final_decision
        else:
            decision = None

        row = submissions_df[submissions_df['directory_name'] == directory_name]
        if row.empty:
            continue
        row = row.iloc[0]
        if decision:
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

                        engine = VertexEngine()
                        similarity_score = engine.get_semantic_similarity(text_1=y_true_comment, text_2=y_pred_snippet)
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
            evaluation_results_df.loc[directory_name, 'category_match'] = category_match
            evaluation_results_df.loc[directory_name, 'status_match'] = status_match
            evaluation_results_df.loc[directory_name, 'similarity_score'] = similarity_score
            evaluation_results_df.loc[directory_name, 'total_input_tokens'] = metrics.total_input_token_count
            evaluation_results_df.loc[directory_name, 'total_output_tokens']  = metrics.total_output_token_count
            evaluation_results_df.loc[directory_name, 'total_elapsed_time'] = metrics.total_elapsed_time


            total_scores.append(score)

    if total_scores:
        final_score = np.mean(total_scores)
        LOG.debug(f"Full Evaluation Score: {final_score:.4f}")

        filename_suffix= re.sub(r'[^a-zA-Z0-9]', '_', system_used)
        evaluation_results_df.to_csv(path_or_buf=f"{__EVALUATION_RESULT_CSV}_{filename_suffix}.csv", index=True,
                                     mode = "w" if not skip else "a",
                                     header= not skip )
        LOG.debug(f"Saved evaluation results to {__EVALUATION_RESULT_CSV}_{filename_suffix}.csv")

    else:
        LOG.warn("No matching submissions found for full evaluation.")