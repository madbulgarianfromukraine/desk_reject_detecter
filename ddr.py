from typing import List
from core.utils import encode_image
from core.schemas import FinalDecision, AnalysisReport

# Import Agents
from agents.formatting_agent import formatting_agent
from agents.policy_agent import policy_agent
from agents.visual_agent import visual_agent
from agents.anonymity_agent import anonymity_agent
from agents.scope_agent import scope_agent
from agents.safety_agent import safety_agent


def desk_rejection_system(paper_text_content: str, image_paths: List[str] = []):
    """
    Main Orchestrator: Calls all agents and aggregates the decision.
    """
    print("--- Starting Desk Rejection Protocol ---")

    # Encode images once
    encoded_images = [encode_image(path) for path in image_paths]

    # 1. Run all agents
    print("Running Formatting Check...")
    formatting_res = formatting_agent(paper_text_content, encoded_images)

    print("Running Policy Check...")
    policy_res = policy_agent(paper_text_content, encoded_images)

    print("Running Visual Integrity Check...")
    visual_res = visual_agent(paper_text_content, encoded_images)

    print("Running Anonymity Check...")
    anonymity_res = anonymity_agent(paper_text_content, encoded_images)

    print("Running Scope Check...")
    scope_res = scope_agent(paper_text_content, encoded_images)

    print("Running Safety Check...")
    safety_res = safety_agent(paper_text_content, encoded_images)

    # 2. Decision Logic
    # Define Fatal Categories
    fatal_checks = [
        (safety_res, "Code of Ethics"),
        (anonymity_res, "Anonymity"),
        (formatting_res, "Formatting"),
        (visual_res, "Visual Integrity"),
        (policy_res, "Policy")
    ]

    final_decision = "NO"
    primary_reason = "None"
    confidence = 0.95

    # Check Fatal First
    for res, category in fatal_checks:
        if res.violation_found:
            final_decision = "YES"
            primary_reason = category
            confidence = 1.0
            break

    # Check Subjective (Scope)
    if final_decision == "NO" and scope_res.violation_found:
        final_decision = "YES"
        primary_reason = "Scope"
        confidence = 0.85

    # 3. Build Final Report
    final_report = FinalDecision(
        desk_reject_decision=final_decision,
        primary_reason_category=primary_reason,
        confidence_score=confidence,
        analysis=AnalysisReport(
            safety_check=safety_res,
            anonymity_check=anonymity_res,
            visual_integrity_check=visual_res,
            formatting_check=formatting_res,
            policy_check=policy_res,
            scope_check=scope_res
        )
    )

    return final_report.model_dump_json(indent=2)


if __name__ == "__main__":
    # --- USAGE EXAMPLE ---
    sample_text = """
    Abstract. This paper presents a new algorithm.
    Introduction. We optimize X using Y...
    """
    # Replace [] with ["page1.png"] if you have files
    result = desk_rejection_system(sample_text, [])

    print("\n=== FINAL REPORT ===\n")
    print(result)