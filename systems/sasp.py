from typing import Union, Optional
import os
import time
from google.genai import types

from core.schemas import SASPReport
from core.metrics import SubmissionMetrics, get_total_input_tokens, get_total_output_tokens
from core.utils import ask_agent
from core.log import LOG

# Model configuration
MODEL_ID = "gemini-2.5-flash"

SASP_SYSTEM_PROMPT = """
Identity: You are an expert desk rejection reviewer for the ICLR conference.

Task: Perform a single-pass evaluation of the submitted paper to determine if it should be desk-rejected based on ICLR conference guidelines.

Evaluation Criteria (pick ONE category with the most critical issue if found):
1. Code_of_Ethics: Violations of privacy (PII), harmful content, or research misconduct
2. Anonymity: Author identification, visual cues, self-citations, or non-anonymous links
3. Formatting: Page limits, line numbers, font/margin violations, or statement limits
4. Visual_Integrity: Placeholder figures, unreadable content, or broken LaTeX rendering
5. Policy: Placeholder text (Lorem Ipsum), plagiarism, or dual submission indicators
6. Scope: Papers not aligned with ICLR or language/professionalism issues

Output Requirements:
- Set 'status' to:
  * "YES" if you find a critical violation that warrants desk rejection
  * "NO" if the paper appears compliant with guidelines
- Identify the PRIMARY category and sub-category affected
- Provide evidence snippet and clear reasoning
- Confidence score should reflect certainty in your assessment (0.0-1.0)
- Return a JSON object matching the SASPReport schema
"""

def ask_sasp_agent(path_to_sub_dir: str, model_id: str = MODEL_ID, main_paper_only: bool = False,
                   search_included: bool = False, thinking_included: bool = False) -> types.GenerateContentResponse:
    """Send evaluation request to SASP agent."""
    return ask_agent(pydantic_model=SASPReport, system_instruction=SASP_SYSTEM_PROMPT,
                    path_to_sub_dir=path_to_sub_dir, model_id=model_id,
                    main_paper_only=main_paper_only,
                    search_included=search_included, thinking_included=thinking_included)

def sasp(path_sub_dir: Union[os.PathLike, str], think: bool = False, search: bool = False) -> Optional[SubmissionMetrics]:
    """
    Single Agent Single Prompt (SASP) system for desk rejection.
    
    A simplified single-pass evaluation using one agent with a straightforward prompt.
    This system returns a single verdict on whether to desk-reject the paper.
    
    :param path_sub_dir: Path to the directory containing 'main_paper.pdf' and supplemental files.
    :param think: Boolean flag to enable Gemini's 'thinking' (reasoning) capabilities.
    :param search: Boolean flag to enable Google Search grounding.
    :return: A SubmissionMetrics object containing the evaluation result compatible with evaluate_submission_full.
    """
    LOG.info(f"--- Starting SASP (Single Agent Single Prompt) for submission={path_sub_dir} ---")
    
    start_time = time.time()
    
    try:
        # Get evaluation from SASP agent
        response = ask_sasp_agent(path_to_sub_dir=str(path_sub_dir), model_id=MODEL_ID,
                                 main_paper_only=False, search_included=search,
                                 thinking_included=think)

        # Parse the response
        parsed_response: SASPReport = response.parsed

        LOG.info(f"SASP decision: {parsed_response.violation_found} (Category: {parsed_response.issue_type})")
        
        elapsed_time = time.time() - start_time
        
        # Create and return metrics
        metrics = SubmissionMetrics(
            submission_id=str(path_sub_dir),
            system_name="SASP",
            total_elapsed_time=elapsed_time,
            total_input_token_count=get_total_input_tokens(),
            total_output_token_count=get_total_output_tokens(),
            category=parsed_response.issue_type,
            sub_category=parsed_response.sub_category,
            reasoning=parsed_response.evidence_snippet,
            confidence_score=parsed_response.confidence_score
        )
        
        return metrics
        
    except Exception as e:
        LOG.error(f"Error during SASP evaluation: {e}")
        return SubmissionMetrics(
            submission_id=str(path_sub_dir),
            system_name="SACP",
            total_elapsed_time=time.time() - start_time,
            total_input_token_count=get_total_input_tokens(),
            total_output_token_count=get_total_output_tokens(),
            error_type=str(e),
            error_message=str(e),
            confidence_score=0.0
        )
