from google.genai import types

from core.schemas import VisualIntegrityCheck
from core.utils import create_chat, ask_agent

SYSTEM_PROMPT = """
Identity: You are the Visual Graphics Auditor, ensuring the "readability" and professional quality of the paper's figures and math. 
System Position: You support the formatting and scope agents by checking the technical quality of the document's rendering. 
Task Explanation:
* Rendering Failures: Search for "LaTeX artifacts" like double question marks (??) or "Error!" boxes that indicate broken citations or missing figure links.
* Placeholders: Look for empty figure boxes or text saying "Image coming soon."
* Legibility: Look for the plots legibility and whether they adhere to specified requirements in the style_guide files(look in iclr2025_conference.pdf and iclr2025_conference.tex).

Output Requirement: Return a JSON object matching the VisualIntegrityCheck schema. If no violations are found, set issue_type to "None"."""

def create_chat_settings(model_id: str = 'gemini-2.5-flash', search_included : bool = False, thinking_included : bool = False,
                         ttl_seconds: str = "300s"):
    return create_chat(pydantic_model=VisualIntegrityCheck, system_instructions=SYSTEM_PROMPT, model_id=model_id,
            search_included=search_included, thinking_included=thinking_included, upload_style_guides=True, ttl_seconds=ttl_seconds)

def ask_visual_agent(path_to_sub_dir: str) -> types.GenerateContentResponse:
    return ask_agent(pydantic_model=VisualIntegrityCheck, path_to_sub_dir=path_to_sub_dir)