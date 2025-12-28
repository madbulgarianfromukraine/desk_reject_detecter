from schemas import SafetyCheck
from utils import create_agent_chain

SYSTEM_PROMPT = """You are the Safety Agent. You deal with FATAL ethical violations.
Analyze for:
1. Data Privacy (PII): Unmasked faces, real names, addresses, health data.
2. Harmful Content: Discriminatory language or tech designed for harm without mitigation.
3. Scientific Misconduct: Obvious fabrication or 'lorem ipsum' in results.

Return JSON matching SafetyCheck schema."""

safety_agent = create_agent_chain(SafetyCheck, SYSTEM_PROMPT)