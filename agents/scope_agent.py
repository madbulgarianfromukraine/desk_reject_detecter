from schemas import ScopeCheck
from utils import create_agent_chain

SYSTEM_PROMPT = """You are the Scope & Quality Agent.
Analyze for:
1. Scope: Is the topic clearly unrelated to ML or AI?
2. Language: Is English quality so poor it is unreviewable?

Return JSON matching ScopeCheck schema."""

scope_agent = create_agent_chain(ScopeCheck, SYSTEM_PROMPT)