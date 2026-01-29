from typing import Union, Dict, Optional, Type, List

import pydantic
import time

from core.schemas import (
    FinalDecision, AnalysisReport,
    AGENT_SCHEMAS
)
from core.logprobs import combine_confidences
from core.log import LOG
from core.metrics import SubmissionMetrics, get_total_input_tokens, get_total_output_tokens
from core.rate_limiter import RateLimitError
# Import Agents
from agents import final_decision_agent
from agents.utils import AGENT_MAPPING, create_chats
import os
import concurrent.futures


# Model configuration
MODEL_ID = "gemini-2.5-flash"


def ddr(path_sub_dir: Union[os.PathLike, str], think: bool = False, search: bool = False, iterations: int = 3, ttl_seconds: str = "300s") -> Optional[SubmissionMetrics]:
    """
    Main Orchestrator for the ddr system.

    This function implements a multi-agent, iterative workflow to determine if a scientific
    paper should be desk-rejected based on conference-specific guidelines (e.g., ICLR).

    Workflow Logic:
    1.  Initialization: Creates chat sessions for all specialized auditor agents (Safety,
        Anonymity, Formatting, etc.) with optional thinking/search capabilities.
    2.  Iterative Evaluation (Self-Correction):
        - It runs agents in parallel using a ThreadPoolExecutor.
        - For each agent's response, it calculates a logprob-based confidence score.
        - If an agent's confidence score is below the `CONFIDENCE_THRESHOLD`, it will be
          re-run in the next iteration (up to `MAX_ITERATIONS`).
        - It keeps the result with the highest confidence score for each category.
    3.  Aggregation: Once all agents satisfy the threshold or max iterations are reached,
        the results are compiled into an `AnalysisReport`.
    4.  Final Decision: The `AnalysisReport` is sent to the `final_decision_agent`, which
        acts as the "Program Chair" to make the terminal "YES/NO" decision.

    :param path_sub_dir: Path to the directory containing 'main_paper.pdf' and supplemental files.
    :param think: Boolean flag to enable Gemini's 'thinking' (reasoning) capabilities.
    :param search: Boolean flag to enable Google Search grounding for relevant agents.
    :param iterations: The maximum number of self-correction iterations for agents.
    :param ttl_seconds: TTL (time-to-live) for uploaded files in the agents that upload style guides.
    :return: A FinalDecision object containing the terminal decision and aggregated analysis.
    :raises RuntimeError: If an agent fails to provide a result after all iterations.
    """
    LOG.info(f"--- Starting Desk Rejection Protocol for submission={path_sub_dir}---")

    create_chats(model_id=MODEL_ID, include_thinking=think, include_search=search, ttl_seconds=ttl_seconds)

    CONFIDENCE_THRESHOLD = 0.95 # probably make it configurable, if have time.
    MAX_ITERATIONS = iterations
    
    agent_results : Dict[str, Optional[Type[pydantic.BaseModel]]] = {key: None for key in AGENT_MAPPING.keys()}
    agent_errors : Dict[str, Optional[Dict[str, str]]] = {key: None for key in AGENT_MAPPING.keys()}  # Store errors per agent
    start_time = time.time()
    for iteration in range(MAX_ITERATIONS):
        
        # Identify agents that still need to be run
        agents_to_run = {}
        for key, result in agent_results.items():
            if result is None or result.confidence_score < CONFIDENCE_THRESHOLD:
                agents_to_run[key] = AGENT_MAPPING[key]
        
        if not agents_to_run:
            LOG.info("All agents satisfied confidence threshold.")
            break
        LOG.info(f"--- Iteration {iteration + 1} ---")
        LOG.info(f"Running agents: {list(agents_to_run.keys())}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5, thread_name_prefix="AgentThread") as executor:
            future_to_agent = {
                executor.submit(func, path_sub_dir): key 
                for key, func in agents_to_run.items()
            }

            for future in concurrent.futures.as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    response = future.result()
                    # The response is a GenerateContentResponse, we need the parsed object
                    parsed_response = response.parsed

                    # Compute logprob-based confidence score and substitute it
                    agent_schema = AGENT_SCHEMAS.get(agent_name)
                    if agent_schema:
                        new_confidence = combine_confidences(response, agent_schema)
                        parsed_response.confidence_score = new_confidence
                        LOG.debug(f"Substituted {agent_name} confidence with logprob-based score: {new_confidence}")

                    # Update only if confidence is higher or if it's the first result
                    if agent_results[agent_name] is None or parsed_response.confidence_score > agent_results[agent_name].confidence_score:
                        agent_results[agent_name] = parsed_response
                        agent_errors[agent_name] = None  # Clear error if successful
                        LOG.debug(f"{agent_name} updated with confidence {parsed_response.confidence_score}.")
                    else:
                        LOG.debug(f"{agent_name} current confidence ({parsed_response.confidence_score}) is not higher than existing ({agent_results[agent_name].confidence_score}). Keeping existing.")

                except Exception as exc:
                    LOG.error(f"{agent_name} generated an exception: {exc}")
                    agent_results[agent_name] = None

    # Ensure all results are present, even if some failed (fallback or re-raise)
    submission_error_type = None
    submission_error_message = None
    failed_agents = [key for key, result in agent_results.items() if result is None]
    
    if failed_agents:
        LOG.error(f"Failed agents: {failed_agents}")
        # Return SubmissionMetrics with error information
        submission_error_type = "AgentFailure"
        submission_error_message = f"Agents failed: {', '.join(failed_agents)}"
        return SubmissionMetrics(
            final_decision=None,
            total_input_token_count=get_total_input_tokens(),
            total_output_token_count=get_total_output_tokens(),
            total_elapsed_time=time.time() - start_time,
            error_type=submission_error_type,
            error_message=submission_error_message
        )

    # Remove confidence scores from all checks before passing to final agent
    for key, result in agent_results.items():
        if result is not None:
            result.confidence_score = None
            LOG.debug(f"Set confidence_score to None for {key}")

    analysis_report = AnalysisReport(
        safety_check=agent_results["safety_check"],
        anonymity_check=agent_results["anonymity_check"],
        visual_integrity_check=agent_results["visual_integrity_check"],
        formatting_check=agent_results["formatting_check"],
        policy_check=agent_results["policy_check"],
        scope_check=agent_results["scope_check"]
    )

    final_decision_response = final_decision_agent.ask_final_decision_agent(analysis_report=analysis_report, submission_id=str(path_sub_dir))
    final_decision_response.parsed.confidence_score = combine_confidences(llm_response=final_decision_response, pydantic_scheme=FinalDecision, final_agent=True)

    end_time = time.time()
    return SubmissionMetrics(final_decision=final_decision_response.parsed, total_input_token_count=get_total_input_tokens(),
                             total_output_token_count=get_total_output_tokens(), total_elapsed_time=end_time - start_time,
                             error_type=submission_error_type,
                             error_message=submission_error_message)

