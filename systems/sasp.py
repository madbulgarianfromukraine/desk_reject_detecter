from typing import Union, Optional, Type
import os
import time
import pydantic
from google.genai import types

from core.schemas import SASPReport
from core.metrics import SubmissionMetrics, get_total_input_tokens, get_total_output_tokens
from core.utils import create_chat, ask_agent
from core.log import LOG
from core.logprobs import combine_confidences

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

def create_chat_settings(model_id: str = MODEL_ID, search_included: bool = False, thinking_included: bool = False):
    """Initialize chat session for SASP."""
    return create_chat(
        pydantic_model=SASPReport,
        system_instructions=SASP_SYSTEM_PROMPT,
        model_id=model_id,
        search_included=search_included,
        thinking_included=thinking_included
    )

def ask_sasp_agent(path_to_sub_dir: str) -> types.GenerateContentResponse:
    """Send evaluation request to SASP agent."""
    return ask_agent(pydantic_model=SASPReport, path_to_sub_dir=path_to_sub_dir)

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
    
    # Initialize chat session
    create_chat_settings(model_id=MODEL_ID, search_included=search, thinking_included=think)
    
    start_time = time.time()
    
    try:
        # Get evaluation from SASP agent
        response = ask_sasp_agent(path_sub_dir)
        
        # Compute logprob-based confidence score
        #new_confidence = combine_confidences(response, SASPReport)
        parsed_response: SASPReport = response.parsed
        #parsed_response.confidence_score = new_confidence
        
        #LOG.debug(f"SASP evaluation confidence: {new_confidence}")
        LOG.info(f"SASP decision: {parsed_response.violation_found} (Category: {parsed_response.issue_type})")
        
        elapsed_time = time.time() - start_time
        
        # Create and return metrics (without final_decision - will be converted during evaluation)
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
        return None
