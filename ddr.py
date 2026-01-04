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

    agent_funcs = {
        agents.formatting_agent : 'formatting_check',
        agents.policy_agent : 'policy_check',
        agents.visual_agent : 'visual_integrity_check',
        agents.anonymity_agent : 'anonymity_check',
        agents.scope_agent : 'scope_check',
        agents.safety_agent : 'safety_check',
    }

    # 2. Run them in parallel
    # max_workers=None defaults to a sensible number based on your CPU
    agent_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all functions with the same argument
        future_to_agent = {executor.submit(func, path_sub_dir): key for func, key in agent_funcs.items()}

        for future in concurrent.futures.as_completed(future_to_agent):
            agent_name = future_to_agent[future]
            try:
                agent_results[agent_name] = future.result()
                print(f"{agent_name} completed.")
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

