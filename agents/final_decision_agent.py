from google.genai import types
from core.schemas import FinalDecision, AnalysisReport
from core.utils import create_chat, ask_final

SYSTEM_PROMPT = """You are a strict and meticulous Program Chair for a top-tier AI conference ICLR. 
Your task is to combine the results of the checks of 6 major desk rejection categories described here in pydantic(this is the format in which you will recieve the checks):
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

Return JSON matching FinalDecision schema."""

def create_chat_settings(model_id: str = 'gemini-2.5-flash', search_included: bool = False, thinking_included: bool = False):
    return create_chat(pydantic_model=FinalDecision, system_instructions=SYSTEM_PROMPT, model_id=model_id,
                       search_included=search_included, thinking_included=thinking_included)

def ask_final_decision_agent(analysis_report: AnalysisReport) -> types.GenerateContentResponse:
    return ask_final(analysis_report=analysis_report)