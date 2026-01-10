from typing import Union, Dict, Optional, Type, List

import pydantic

from core.schemas import (
    FinalDecision, AnalysisReport,
    AGENT_SCHEMAS
)
from core.logprobs import combine_confidences
from core.log import LOG

# Import Agents
from agents import final_decision_agent
from agents.utils import AGENT_MAPPING, create_chats
import os
import concurrent.futures


def desk_rejection_system(path_sub_dir: Union[os.PathLike, str], think: bool = False, search: bool = False) -> FinalDecision:
    """
    Main Orchestrator for the Desk Rejection System.

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
    :return: A FinalDecision object containing the terminal decision and aggregated analysis.
    :raises RuntimeError: If an agent fails to provide a result after all iterations.
    """
    LOG.info("--- Starting Desk Rejection Protocol ---")

    create_chats(include_thinking=think, include_search=search)

    CONFIDENCE_THRESHOLD = 0.7 # probably make it configurable, if have time.
    MAX_ITERATIONS = 3
    
    agent_results : Dict[str, Optional[Type[pydantic.BaseModel]]] = {key: None for key in AGENT_MAPPING.keys()}
    
    for iteration in range(MAX_ITERATIONS):
        LOG.info(f"--- Iteration {iteration + 1} ---")
        
        # Identify agents that still need to be run
        agents_to_run = {}
        for key, result in agent_results.items():
            if result is None or result.confidence_score < CONFIDENCE_THRESHOLD:
                agents_to_run[key] = AGENT_MAPPING[key]
        
        if not agents_to_run:
            LOG.info("All agents satisfied confidence threshold.")
            break
            
        LOG.info(f"Running agents: {list(agents_to_run.keys())}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents_to_run)) as executor:
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
                        LOG.debug(f"{agent_name} updated with confidence {parsed_response.confidence_score}.")
                    else:
                        LOG.debug(f"{agent_name} current confidence ({parsed_response.confidence_score}) is not higher than existing ({agent_results[agent_name].confidence_score}). Keeping existing.")
                except Exception as exc:
                    LOG.error(f"{agent_name} generated an exception: {exc}")

    # Ensure all results are present, even if some failed (fallback or re-raise)
    for key, result in agent_results.items():
        if result is None:
            LOG.error(f"Agent {key} failed to provide a result after {MAX_ITERATIONS} iterations.")
            # In a real system, you might want to raise an error or provide a default fail-safe result
            raise RuntimeError(f"Agent {key} failed to provide a result.")

    analysis_report = AnalysisReport(
        safety_check=agent_results["safety_check"],
        anonymity_check=agent_results["anonymity_check"],
        visual_integrity_check=agent_results["visual_integrity_check"],
        formatting_check=agent_results["formatting_check"],
        policy_check=agent_results["policy_check"],
        scope_check=agent_results["scope_check"]
    )

    final_decision_response = final_decision_agent.ask_final_decision_agent(analysis_report=analysis_report)
    final_decision_response.parsed.confidence_score = combine_confidences(llm_response=final_decision_response, pydantic_scheme=FinalDecision, final_agent=True)

    return final_decision_response.parsed

