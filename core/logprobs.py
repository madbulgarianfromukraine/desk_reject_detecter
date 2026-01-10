from typing import List, Type, Dict
import math
from pydantic import BaseModel
from google.genai import types

from core.schemas import extract_possible_values
from core.log import LOG


__LOGPROB_CANDIDATES = {
    "violation_found" : 0.15,
    "issue_type" : 0.4,
    "evidence_snippet" : 0.1,
}
def get_field_confidence(logprob_candidates: List, target_field: str, pydantic_scheme : Type[BaseModel]) -> float:
    """
    Parses a list of LogprobsResultCandidate to find a specific JSON field
    and returns the average probability of its value tokens.
    """
    possible_values = extract_possible_values(pydantic_scheme=pydantic_scheme, target_field=target_field)
    target_field_clean = target_field.lower()
    tokens = [c.token for c in logprob_candidates]
    logprobs = [c.log_probability for c in logprob_candidates]

    field_values = []
    field_value_logprobs = []
    found_key = False
    idx = 0

    while idx < len(tokens):
        # --- PHASE 1: Find the Key ---
        if not found_key:
            # Check if current token (cleaned) matches part of our target
            current_token_clean = tokens[idx].strip().strip('"').lower()

            # If it's a partial match, we check subsequent tokens to rebuild the full key
            if current_token_clean and target_field_clean.startswith(current_token_clean):
                reconstructed_key = current_token_clean
                temp_idx = idx + 1
                while temp_idx < len(tokens) and reconstructed_key != target_field_clean:
                    next_clean = tokens[temp_idx].strip().strip('"').lower()
                    reconstructed_key += next_clean
                    if not next_clean or not target_field_clean.startswith(reconstructed_key): break
                    temp_idx += 1

                if reconstructed_key == target_field_clean:
                    found_key = True
                    idx = temp_idx  # Jump past the key
                    continue
            idx += 1
            continue

        # --- PHASE 2: Skip structural JSON punctuation (": " or " ") ---
        if tokens[idx].strip() in [":", '"', '{', ' ', '":', ' "']:
            idx += 1
            continue

        # --- PHASE 3: Capture the Value Tokens ---
        while idx < len(tokens):
            t = tokens[idx]
            if t in [',', '}', '],', '\n', '",']:
                break
            field_values.append(t)
            field_value_logprobs.append(logprobs[idx])
            idx += 1

        break

    # --- PHASE 4: Final Calculation ---


    if not field_value_logprobs:
        LOG.warn(f"The field={target_field} or the value of it was not found")
        return 0.0  # Return 0 if the field was never found or value was empty

    possible_values = extract_possible_values(pydantic_scheme=pydantic_scheme, target_field=target_field)

    if possible_values and not "".join(field_values) in possible_values:
        LOG.warn(f"The field_value={"".join(field_values)} is not one of the possible for the field.")
        return 0.0


    # Geometric mean of probabilities = e^(average of logprobs)
    avg_logprob = sum(map(math.exp, field_value_logprobs)) / len(field_value_logprobs)
    return avg_logprob

def combine_confidences(llm_response: types.GenerateContentResponse,
                        pydantic_scheme: Type[BaseModel]) -> float:
    final_confidence = 0.0
    logprob_candidates = llm_response.candidates[0].logprobs_result.chosen_candidates

    for target_field, weight in __LOGPROB_CANDIDATES.items():
        final_confidence += weight * get_field_confidence(logprob_candidates=logprob_candidates, target_field=target_field, pydantic_scheme=pydantic_scheme)

    return final_confidence
