from typing import List
from core.utils import encode_image
from core.schemas import FinalDecision, AnalysisReport

# Import Agents
import agents
import os


def desk_rejection_system(path_sub_dir: os.PathLike, image_paths: List[str]) -> Any:
    """
    Main Orchestrator: Calls all agents and aggregates the decision.
    """
    print("--- Starting Desk Rejection Protocol ---")

    # Encode images once
    encoded_images = [encode_image(path) for path in image_paths]

    # 1. Run all agents
    print("Running Formatting Check...")
    formatting_res = agents.formatting_agent(path_sub_dir)

    print("Running Policy Check...")
    policy_res = agents.policy_agent(path_sub_dir)

    print("Running Visual Integrity Check...")
    visual_res = agents.visual_agent(path_sub_dir)

    print("Running Anonymity Check...")
    anonymity_res = agents.anonymity_agent(path_sub_dir)

    print("Running Scope Check...")
    scope_res = agents.scope_agent(path_sub_dir)

    print("Running Safety Check...")
    safety_res = agents.safety_agent(path_sub_dir)

    # 2. Decision Logic
    # Define Fatal Categories
    fatal_checks = [
        (safety_res, "Code of Ethics"),
        (anonymity_res, "Anonymity"),
        (formatting_res, "Formatting"),
        (visual_res, "Visual Integrity"),
        (policy_res, "Policy")
    ]






    return "".model_dump_json(indent=2)

