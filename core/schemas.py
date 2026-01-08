from typing import Literal
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
    issue_type: Literal["Author Names", "Visual Anonymity", "Self-Citation", "Links", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class VisualIntegrityCheck(BaseModel):
    violation_found: bool
    issue_type: Literal["Placeholder Figures", "Unreadable Content", "Broken Rendering", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class FormattingCheck(BaseModel):
    violation_found: bool
    issue_type: Literal["Page Limit", "Statement Limit", "Margins/Spacing", "Line Numbers", "None"] = "None"
    evidence_snippet: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class PolicyCheck(BaseModel):
    violation_found: bool
    issue_type: Literal["Placeholder Text", "Dual Submission", "Plagiarism", "None"] = "None"
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
    primary_reason_category: Literal["Code of Ethics", "Anonymity", "Formatting", "Visual Integrity", "Policy", "Scope", "None"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    analysis: AnalysisReport