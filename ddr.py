from typing import Union, Dict, Optional, Type

import pydantic

from core.schemas import FinalDecision, AnalysisReport
from core.log import LOG

# Import Agents
import agents
import os
import concurrent.futures


def create_chats(include_thinking: bool = False, include_search: bool = False) -> None:
    """
    Initializes the chat settings for all agents in the desk rejection system.

    This function sets up each agent with the specified capabilities (thinking and search)
    by calling their respective `create_chat_settings` methods in parallel.

    :param include_thinking: Whether to enable thinking/reasoning capabilities for applicable agents.
    :param include_search: Whether to enable search capabilities for applicable agents.
    """
    # 1. Initialize all agents in parallel
    initialization_tasks = [
        (agents.formatting_agent.create_chat_settings, {}),
        (agents.policy_agent.create_chat_settings, {'search_included': include_search}),
        (agents.visual_agent.create_chat_settings, {}),
        (agents.anonymity_agent.create_chat_settings, {'search_included': include_search}),
        (agents.scope_agent.create_chat_settings, {'thinking_included': include_thinking}),
        (agents.safety_agent.create_chat_settings, {'thinking_included': include_thinking}),
        (agents.final_decision_agent.create_chat_settings, {'thinking_included': include_thinking}),
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(initialization_tasks)) as executor:
        futures = [executor.submit(func, **kwargs) for func, kwargs in initialization_tasks]
        concurrent.futures.wait(futures)

def desk_rejection_system(path_sub_dir: Union[os.PathLike, str], think: bool = False, search: bool = False) -> FinalDecision:
    """
    Main Orchestrator: Calls all agents and aggregates the decision.
    """
    LOG.info("--- Starting Desk Rejection Protocol ---")

    create_chats(include_thinking=think, include_search=search)

    agent_mapping = {
        'formatting_check': agents.formatting_agent.ask_formatting_agent,
        'policy_check': agents.policy_agent.ask_policy_agent,
        'visual_integrity_check': agents.visual_agent.ask_visual_agent,
        'anonymity_check': agents.anonymity_agent.ask_anonymity_agent,
        'scope_check': agents.scope_agent.ask_scope_agent,
        'safety_check': agents.safety_agent.ask_safety_agent,
    }

    CONFIDENCE_THRESHOLD = 0.7 # probably make it configurable, if have time.
    MAX_ITERATIONS = 3
    
    agent_results : Dict[str, Optional[Type[pydantic.BaseModel]]] = {key: None for key in agent_mapping.keys()}
    
    for iteration in range(MAX_ITERATIONS):
        LOG.info(f"--- Iteration {iteration + 1} ---")
        
        # Identify agents that still need to be run
        agents_to_run = {}
        for key, result in agent_results.items():
            if result is None or result.confidence_score < CONFIDENCE_THRESHOLD:
                agents_to_run[key] = agent_mapping[key]
        
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
                    agent_results[agent_name] = response.parsed
                    LOG.debug(f"{agent_name} completed with confidence {agent_results[agent_name].confidence_score}.")
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

    final_decision_response = agents.final_decision_agent.ask_final_decision_agent(analysis_report=analysis_report)
    return final_decision_response.parsed

