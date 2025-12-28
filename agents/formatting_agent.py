from schemas import FormattingCheck
from utils import create_agent_chain

SYSTEM_PROMPT = """You are the Formatting Check Agent for ICLR. 
Analyze the submission for:
1. Main Pages Limit: Does main content exceed 10 pages? (Refs/Appendix/Ethics/Reproducibility exclude).
2. Statement Length: Do Ethics/Reproducibility statements exceed 1 page each?
3. Line Numbers: Are they missing?
4. Margins/Spacing: Is there visual evidence of cheating margins or reduced font size?

Return a JSON object matching the FormattingCheck schema. If no violation, set issue_type to 'None'."""

formatting_agent = create_agent_chain(FormattingCheck, SYSTEM_PROMPT)