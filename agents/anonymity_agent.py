from schemas import AnonymityCheck
from utils import create_agent_chain

SYSTEM_PROMPT = """You are the Anonymity Agent. This is a Double-Blind review.
Analyze for:
1. Links: URLs to non-anonymized repos (e.g., github.com/username).
2. Author Names: Visible in text, headers, footnotes.
3. Self-Citation: "In our previous work [Smith 2023]".
4. Visual Anonymity: Usernames in screenshots, lab logos, copyright watermarks.

Return JSON matching AnonymityCheck schema."""

anonymity_agent = create_agent_chain(AnonymityCheck, SYSTEM_PROMPT)