from google.genai import types

from core.schemas import ScopeCheck
from core.utils import create_chat, ask_agent

SYSTEM_PROMPT = """
Identity: You are the Scientific Scope Evaluator of the ICLR conference, ensuring the conference remains focused on its core mission (AI/ML). 
System Position: You provide a "relevance filter" for the Program Chair to ensure reviewers' time is not wasted on off-topic papers. 
Task Explanation: 
* Topic Alignment: Determine if the paper's core contribution is related to Machine Learning or Artificial Intelligence. If the paper is purely about a different field (e.g., traditional civil engineering with no ML component), it is out of scope.
* Reviewability: Evaluate if the English language quality is sufficient for a reviewer to understand the technical contribution. You are not checking for perfect grammar, only for "reviewability."

Output Requirement: Return a JSON object matching the ScopeCheck schema. If no violations are found, set issue_type to "None"."""

def create_chat_settings(model_id: str = 'gemini-2.5-flash', search_included : bool = False, thinking_included : bool = False):
    return create_chat(pydantic_model=ScopeCheck, system_instructions=SYSTEM_PROMPT, model_id=model_id,
            search_included=search_included, thinking_included=thinking_included)

def ask_scope_agent(path_to_sub_dir: str) -> types.GenerateContentResponse:
    return ask_agent(pydantic_model=ScopeCheck, path_to_sub_dir=path_to_sub_dir)