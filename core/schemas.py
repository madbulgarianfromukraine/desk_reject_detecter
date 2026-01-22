from typing import Literal, Tuple, Any, Type, List, get_args, get_origin, Optional
from pydantic import BaseModel, Field

# Base schemas for individual checks
class SafetyCheck(BaseModel):
    """
    Schema for Ethical and Safety audit results.
    Checks for:
    - Privacy violations (PII).
    - Harmful content.
    - Research misconduct.
    """
    violation_found: bool
    issue_type: Literal["Privacy", "Harm", "Misconduct", "None"] = "None"
    evidence_snippet: str = Field(description="Quote text or describe unethical image")
    reasoning: str
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class AnonymityCheck(BaseModel):
    """
    Schema for Double-Blind Anonymity audit results.
    Checks for:
    - Author names or affiliations.
    - Non-anonymous links (e.g., lab websites).
    - Identifying self-citations.
    - Visual cues in images (e.g., logos).
    """
    violation_found: bool
    issue_type: Literal["Author_Names", "Visual_Anonymity", "Self-Citation", "Links", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class VisualIntegrityCheck(BaseModel):
    """
    Schema for Technical and Visual Quality audit results.
    Checks for:
    - LaTeX rendering artifacts (e.g., ??).
    - Unreadable figures or math.
    - Placeholder graphics.
    """
    violation_found: bool
    issue_type: Literal["Placeholder_Figures", "Unreadable_Content", "Broken_Rendering", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class FormattingCheck(BaseModel):
    """
    Schema for Document Layout and Formatting audit results.
    Checks for:
    - Page limits (main content).
    - Statement limits (Ethics/Reproducibility).
    - Line numbers (required for review).
    - Font/margin cheating.
    """
    violation_found: bool
    issue_type: Literal["Page_Limit", "Statement_Limit", "Margins/Spacing", "Line_Numbers", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class PolicyCheck(BaseModel):
    """
    Schema for Conference Policy and Integrity audit results.
    Checks for:
    - Placeholder text (e.g., "Lorem Ipsum").
    - Plagiarism or dual submission indicators.
    """
    violation_found: bool
    issue_type: Literal["Placeholder_Text", "Dual_Submission", "Plagiarism", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class ScopeCheck(BaseModel):
    """
    Schema for Subject-Matter Scope and Language audit results.
    Checks for:
    - Scientific relevance to the conference.
    - Language quality and professional tone.
    """
    violation_found: bool
    issue_type: Literal["Scope", "Language", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

# Single Agent Single Prompt (SASP) Schema
class SASPReport(BaseModel):
    """
    Single Agent Single Prompt report containing desk rejection analysis.
    Returns a simplified single-pass evaluation of the paper.
    """
    violation_found: Literal["YES", "NO"]
    issue_type: Literal["Code_of_Ethics", "Anonymity", "Formatting", "Visual_Integrity", "Policy", "Scope", "None"]
    sub_category: Literal[
        # Code_of_Ethics sub-categories
        "Privacy", "Harm", "Misconduct",
        # Anonymity sub-categories
        "Author_Names", "Visual_Anonymity", "Self-Citation", "Links",
        # Formatting sub-categories
        "Page_Limit", "Statement_Limit", "Margins/Spacing", "Line_Numbers",
        # Visual_Integrity sub-categories
        "Placeholder_Figures", "Unreadable_Content", "Broken_Rendering",
        # Policy sub-categories
        "Placeholder_Text", "Dual_Submission", "Plagiarism",
        # Scope sub-categories
        "Scope", "Language",
        # Default
        "None"
    ] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

# Single Agent Complex Prompt (SACP) Schema
class SACPReport(BaseModel):
    """
    Single Agent Single Prompt report containing desk rejection analysis.
    Returns a simplified single-pass evaluation of the paper.
    """
    violation_found: Literal["YES", "NO"]
    issue_type: Literal["Code_of_Ethics", "Anonymity", "Formatting", "Visual_Integrity", "Policy", "Scope", "None"]
    sub_category: Literal[
        # Code_of_Ethics sub-categories
        "Privacy", "Harm", "Misconduct",
        # Anonymity sub-categories
        "Author_Names", "Visual_Anonymity", "Self-Citation", "Links",
        # Formatting sub-categories
        "Page_Limit", "Statement_Limit", "Margins/Spacing", "Line_Numbers",
        # Visual_Integrity sub-categories
        "Placeholder_Figures", "Unreadable_Content", "Broken_Rendering",
        # Policy sub-categories
        "Placeholder_Text", "Dual_Submission", "Plagiarism",
        # Scope sub-categories
        "Scope", "Language",
        # Default
        "None"
    ] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

# Final Report Schema
class AnalysisReport(BaseModel):
    """
    Aggregated report containing results from all six specialized auditor agents.
    """
    safety_check: SafetyCheck
    anonymity_check: AnonymityCheck
    visual_integrity_check: VisualIntegrityCheck
    formatting_check: FormattingCheck
    policy_check: PolicyCheck
    scope_check: ScopeCheck

class FinalDecision(BaseModel):
    """
    Terminal decision schema produced by the Program Chair (Final Agent).
    """
    desk_reject_decision: Literal["YES", "NO"]
    categories: Literal["Code_of_Ethics", "Anonymity", "Formatting", "Visual_Integrity", "Policy", "Scope", "None"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    analysis: AnalysisReport


AGENT_SCHEMAS = {
    'formatting_check': FormattingCheck,
    'policy_check': PolicyCheck,
    'visual_integrity_check': VisualIntegrityCheck,
    'anonymity_check': AnonymityCheck,
    'scope_check': ScopeCheck,
    'safety_check': SafetyCheck,
}

def extract_possible_values(pydantic_scheme: Type[BaseModel], target_field: str) -> Tuple[Any]:
    """
    Extracts the allowed values from a Pydantic field annotated with 'Literal'.

    This utility uses Python's typing inspection (`get_origin`, `get_args`) to 
    programmatically determine valid options for a field. This is used during
    confidence scoring to validate LLM outputs against the expected schema.

    :param pydantic_scheme: The Pydantic model class to inspect.
    :param target_field: The name of the field to extract values for.
    :return: A tuple of allowed values, or an empty tuple if not a Literal.
    """
    field_info = pydantic_scheme.model_fields[target_field]

    if get_origin(field_info.annotation) is Literal:
        return get_args(field_info.annotation)
    else:
        return tuple()
