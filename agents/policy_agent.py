from schemas import PolicyCheck
from utils import create_agent_chain

SYSTEM_PROMPT = """You are the Policy & Integrity Agent.
Analyze for:
1. Placeholder Text: "TBD", "Draft", gibberish in abstract/content.
2. Dual Submission: Evidence of submission elsewhere.
3. Plagiarism: Evidence of large-scale text copying.

Return JSON matching PolicyCheck schema."""

policy_agent = create_agent_chain(PolicyCheck, SYSTEM_PROMPT)