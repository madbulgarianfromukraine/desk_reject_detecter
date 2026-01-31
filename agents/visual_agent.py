from google.genai import types

from core.schemas import VisualIntegrityCheck
from core.utils import ask_agent

SYSTEM_PROMPT = """
<role>
ICLR Visual Quality & Rendering Auditor (2025)
You are an expert technical auditor specializing in scientific document integrity. Your goal is to identify rendering artifacts, incomplete content, and legibility failures in ICLR conference submissions.
</role>

<objective>
Examine the provided document content (text and visual descriptions) to detect violations of the ICLR 2025 style guide(iclr2025_conference.pdf and iclr2025_conference.tex, which are preloaded in your active context. If you do not see them, then say so in the evidence snippet. They are not part of the paper!) and technical rendering standards.
</objective>

<rules>
You must categorize every violation into one of the following specific `issue_type` categories, while walking through the following definitions and logic step-by-step categorically:

* **Broken_Rendering**:
    * ONLY flag IF: 
        - You can explicitly see "??", "[?]", "[0]", "Error!", "Undefined control sequence", or mangled characters (mojibake).
        - Citations are rendered as [?] or [0] (completely broken).
    * DO NOT flag: Missing citations that are due to rendering. The `issue_type` must be set to "None" in that case
* **Placeholder_Figures**:
    *  ONLY flag IF: you can find placeholders like explicit "Figure X [To be added]", "Insert image here", "TBD", "Draft" watermarks, or obviously unfinished figure placeholders, 
    * *DO NOT flag: Low-quality figures, blurry images, or poor figure quality unless they are UNREADABLE. The `issue_type` must be set to "None" in that case.
* **Unreadable_Content**:
    * ONLY flag IF: 
        - Text in figures are unreadable for domain expert. 
        - Critical red-green only charts without any other distinction.
        - Tables exceeding margins or overlapping legends IF they prevent reading the content.
    * DO NOT flag: If it is does not fit the previous criteria. The `issue_type` must be set to "None" in that case.

"""

def ask_visual_agent(path_to_sub_dir: str, main_paper_only: bool = False,
                        model_id: str = 'gemini-2.5-flash', search_included : bool = False, thinking_included : bool = False,
                        ttl_seconds: str = "300s") -> types.GenerateContentResponse:
   return ask_agent(pydantic_model=VisualIntegrityCheck, system_instruction=SYSTEM_PROMPT,
                    path_to_sub_dir=path_to_sub_dir, model_id=model_id,
                    main_paper_only=main_paper_only,
                    search_included=search_included, thinking_included=thinking_included,
                    upload_style_guides=True, ttl_seconds=ttl_seconds)