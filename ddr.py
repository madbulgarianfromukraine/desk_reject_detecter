from typing import Union
from core.schemas import FinalDecision, AnalysisReport

# Import Agents
import agents
import os
import concurrent.futures


def desk_rejection_system(path_sub_dir: Union[os.PathLike, str]) -> FinalDecision:
    """
    Main Orchestrator: Calls all agents and aggregates the decision.
    """
    print("--- Starting Desk Rejection Protocol ---")

    agent_funcs = [
        agents.formatting_agent,
        agents.policy_agent,
        agents.visual_agent,
        agents.anonymity_agent,
        agents.scope_agent,
        agents.safety_agent
    ]

    # 2. Run them in parallel
    # max_workers=None defaults to a sensible number based on your CPU
    agent_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all functions with the same argument
        future_to_agent = {executor.submit(func, path_sub_dir): func.__name__ for func in agent_funcs}

        for future in concurrent.futures.as_completed(future_to_agent):
            agent_name = future_to_agent[future]
            try:
                agent_results[agent_name] = future.result()
                print(f"{agent_name} Check completed.")
            except Exception as exc:
                print(f"{agent_name} generated an exception: {exc}")


    analysis_report = AnalysisReport(
        safety_check=agent_results["safety_check"],
        anonymity_check=agent_results["anonymity_check"],
        visual_integrity_check=agent_results["visual_integrity_check"],
        formatting_check=agent_results["formatting_check"],
        policy_check=agent_results["policy_check"],
        scope_check=agent_results["scope_check"]
    )

    final_decision = agents.final_decision_agent(analysis_report=analysis_report)
    return final_decision

