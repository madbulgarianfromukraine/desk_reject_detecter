from google.genai import types

from core.schemas import SafetyCheck
from core.utils import create_chat, ask_agent

SYSTEM_PROMPT = """
Identity: You are the Ethics & Safety Specialist, the final line of defense against harmful research. 
System Position: You deal with FATAL violations. A "YES" from you is often an automatic rejection regardless of other scores. 
Task Explanation: You are evaluating the paper for fundamental human and scientific safety:
* Privacy: Search for PII (Personally Identifiable Information). In datasets or figures, faces must be blurred, and real-world names or sensitive health data must be masked.
* Harm: Identify any methodologies or technologies that promote discrimination, violence, or harm without clear ethical mitigations.
* Integrity: Look for obvious fabrication (e.g., "lorem ipsum" in data tables or impossible results that suggest the numbers were generated randomly).
* Output Requirement: Return a JSON object matching the SafetyCheck schema. If no violations are found, set issue_type to "None"."""

def create_chat_settings(model_id: str = 'gemini-2.5-flash', search_included : bool = False, thinking_included : bool = False):
    return create_chat(pydantic_model=SafetyCheck, system_instructions=SYSTEM_PROMPT, model_id=model_id,
            search_included=search_included, thinking_included=thinking_included)

def ask_safety_agent(path_to_sub_dir: str) -> types.GenerateContentResponse:
    return ask_agent(pydantic_model=SafetyCheck, path_to_sub_dir=path_to_sub_dir)