from typing import Literal, Tuple, Any, Type, get_args, get_origin
from pydantic import BaseModel, Field

# Base schemas for individual checks
class SafetyCheck(BaseModel):
    violation_found: bool
    issue_type: Literal["Privacy", "Harm", "Misconduct", "None"] = "None"
    evidence_snippet: str = Field(description="Quote text or describe unethical image")
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class AnonymityCheck(BaseModel):
    violation_found: bool
    issue_type: Literal["Author_Names", "Visual_Anonymity", "Self-Citation", "Links", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class VisualIntegrityCheck(BaseModel):
    violation_found: bool
    issue_type: Literal["Placeholder_Figures", "Unreadable_Content", "Broken_Rendering", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class FormattingCheck(BaseModel):
    violation_found: bool
    issue_type: Literal["Page_Limit", "Statement_Limit", "Margins/Spacing", "Line_Numbers", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class PolicyCheck(BaseModel):
    violation_found: bool
    issue_type: Literal["Placeholder_Text", "Dual_Submission", "Plagiarism", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class ScopeCheck(BaseModel):
    violation_found: bool
    issue_type: Literal["Scope", "Language", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

# Final Report Schema
class AnalysisReport(BaseModel):
    safety_check: SafetyCheck
    anonymity_check: AnonymityCheck
    visual_integrity_check: VisualIntegrityCheck
    formatting_check: FormattingCheck
    policy_check: PolicyCheck
    scope_check: ScopeCheck

class FinalDecision(BaseModel):
    desk_reject_decision: Literal["YES", "NO"]
    primary_reason_category: Literal["Code_of_Ethics", "Anonymity", "Formatting", "Visual_Integrity", "Policy", "Scope", "None"]
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
    field_info = pydantic_scheme.model_fields[target_field]

    if get_origin(field_info.annotation) is Literal:
        return get_args(field_info.annotation)
    else:
        return tuple()
