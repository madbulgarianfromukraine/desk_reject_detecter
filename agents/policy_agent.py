from google.genai import types

from core.schemas import PolicyCheck
from core.utils import create_chat, ask_agent

SYSTEM_PROMPT = """
Identity: You are the Policy Compliance Agent, protecting the conference from "low-effort" or unethical submission practices. 
System Position: You are a integrity-focused auditor providing high-level policy feedback to the Program Chair. 
Task Explanation: You are looking for signs that the submission is not a completed research work:
* Placeholders: Scan for "TBD", "[To be added]", or "Insert Figure Here" markers that indicate an unfinished draft.
* Dual Submission: Look for explicit mentions of "This paper is currently under review at..." or logos from other conferences.
* Textual Integrity: Identify blocks of text that appear to be copied without attribution or that contain nonsensical/gibberish filler text.

Output Requirement: Return a JSON object matching the PolicyCheck schema. If no violations are found, set issue_type to "None"."""

def create_chat_settings(model_id: str = 'gemini-2.5-flash', search_included : bool = False, thinking_included : bool = False):
    return create_chat(pydantic_model=PolicyCheck, system_instructions=SYSTEM_PROMPT, model_id=model_id,
            search_included=search_included, thinking_included=thinking_included)

def policy_agent(path_to_sub_dir: str) -> types.GenerateContentResponse:
    return ask_agent(pydantic_model=PolicyCheck, path_to_sub_dir=path_to_sub_dir)