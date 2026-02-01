from typing import Union, Optional
import os
import time
from google.genai import types

from core.schemas import SACPReport
from core.metrics import SubmissionMetrics, get_total_input_tokens, get_total_output_tokens
from core.utils import ask_agent
from core.log import LOG

# Model configuration
MODEL_ID = "gemini-2.5-flash"

SACP_SYSTEM_PROMPT = """
Identity: You are a Senior Program Chair for the ICLR conference with expertise in:
- Research ethics and academic integrity
- Conference formatting standards and technical quality
- Double-blind review protocols
- Machine learning scope and relevance assessment
- Academic writing standards

Complex Multi-Perspective Analysis Task:
Conduct a COMPREHENSIVE desk rejection analysis by examining the paper from six distinct angles:

1. **Code_of_Ethics (Research Integrity)**
   - Evaluate for Privacy violations: Are there exposed PII, unblurred faces, or identifiable health data?
   - Check for Harmful Content: Does the work promote discrimination, violence, or misuse without ethical mitigations?
   - Assess Research Misconduct: Are there indicators of fabrication (e.g., "lorem ipsum" in results) or impossible claims?
   Sub-categories: "Privacy", "Harm", "Misconduct"

2. **Anonymity (Double-Blind Review)**
   - Author Identification: Do author names or affiliations appear anywhere?
   - Visual_Anonymity: Are institutional logos, unique images, or identifying visual elements present?
   - Self-Citation: Are authors revealing themselves through citation patterns or self-referential content?
   - Links: Are there traceable URLs, GitHub profiles, or institutional links revealing identity?
   Sub-categories: "Author_Names", "Visual_Anonymity", "Self-Citation", "Links"

3. **Formatting (Document Compliance)**
   - Page_Limit: Does the paper exceed ICLR's page limit for main content?
   - Statement_Limit: Are ethics/reproducibility statements within required limits?
   - Margins/Spacing: Are margins or font sizes artificially reduced to fit more content?
   - Line_Numbers: Are required line numbers present for review?
   Sub-categories: "Page_Limit", "Statement_Limit", "Margins/Spacing", "Line_Numbers"

4. **Visual_Integrity (Technical Quality)**
   - Placeholder_Figures: Are there unfinished, placeholder, or generic figures?
   - Unreadable_Content: Are figures, tables, or mathematical equations illegible or too small?
   - Broken_Rendering: Are there LaTeX artifacts (??), formatting errors, or display issues?
   Sub-categories: "Placeholder_Figures", "Unreadable_Content", "Broken_Rendering"

5. **Policy (Conference Integrity)**
   - Placeholder_Text: Presence of Lorem Ipsum, template text, or incomplete sections?
   - Dual_Submission: Indicators of simultaneous submission to other venues?
   - Plagiarism: Text or figures suspiciously similar to existing works?
   Sub-categories: "Placeholder_Text", "Dual_Submission", "Plagiarism"

6. **Scope (Domain Relevance & Language)**
   - Scope: Is the research aligned with ICLR's ML/AI scope? Or is it completely outside the conference domain?
   - Language: Is the paper professionally written with proper English/grammar, or does it have severe language barriers?
   Sub-categories: "Scope", "Language"

Analysis Framework:
- First, scan across ALL six categories to identify potential issues
- Determine which category contains the MOST CRITICAL or DISQUALIFYING issue
- If multiple categories have issues, prioritize by severity: Code_of_Ethics > Anonymity > Policy > Scope > Formatting > Visual_Integrity
- Synthesize evidence from multiple areas to make a holistic judgment

Output Requirements:
- Set 'status' to:
  * "REJECT": If you find evidence requiring mandatory desk rejection (violations of ethics, anonymity, policy, or clear policy breaches)
  * "ACCEPT": If the paper appears fully compliant with ICLR guidelines across all six criteria
  * "UNCERTAIN": If evidence is ambiguous or requires human expert review to disambiguate
- Identify the PRIMARY category with the most critical issue
- Provide evidence snippets and clear, well-reasoned justification for your assessment
- Confidence score should reflect certainty (0.0-1.0): Higher for clear-cut cases, lower for borderline situations
- Return a JSON object matching the SACPReport schema
"""

def ask_sacp_agent(path_to_sub_dir: str, model_id: str = MODEL_ID, main_paper_only: bool = False,
                   search_included: bool = False, thinking_included: bool = False) -> types.GenerateContentResponse:
    """Send comprehensive evaluation request to SACP agent."""
    return ask_agent(pydantic_model=SACPReport, system_instruction=SACP_SYSTEM_PROMPT,
                    path_to_sub_dir=path_to_sub_dir, model_id=model_id,
                    main_paper_only=main_paper_only,
                    search_included=search_included, thinking_included=thinking_included)

def sacp(path_sub_dir: Union[os.PathLike, str], think: bool = False, search: bool = False) -> Optional[SubmissionMetrics]:
    """
    Single Agent Complex Prompt (SACP) system for desk rejection.
    
    A comprehensive multi-perspective evaluation using one agent with a detailed, 
    multi-dimensional prompt. This system examines the paper across six categories
    and returns a holistic verdict on desk rejection.
    
    :param path_sub_dir: Path to the directory containing 'main_paper.pdf' and supplemental files.
    :param think: Boolean flag to enable Gemini's 'thinking' (reasoning) capabilities.
    :param search: Boolean flag to enable Google Search grounding.
    :return: A SubmissionMetrics object containing the comprehensive evaluation result compatible with evaluate_submission_full.
    """
    LOG.info(f"--- Starting SACP (Single Agent Complex Prompt) for submission={path_sub_dir} ---")
    
    start_time = time.time()
    
    try:
        # Get comprehensive evaluation from SACP agent
        response = ask_sacp_agent(path_to_sub_dir=str(path_sub_dir), model_id=MODEL_ID,
                                 main_paper_only=False, search_included=search,
                                 thinking_included=think)

        # Parse the response
        parsed_response: SACPReport = response.parsed

        LOG.info(f"SACP decision: {parsed_response.violation_found} (Category: {parsed_response.issue_type}, Sub-category: {parsed_response.sub_category})")
        
        elapsed_time = time.time() - start_time
        
        # Create and return metrics
        metrics = SubmissionMetrics(
            submission_id=str(path_sub_dir),
            system_name="SACP",
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
        LOG.error(f"Error during SACP evaluation: {e}")
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
