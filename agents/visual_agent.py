from schemas import VisualIntegrityCheck
from utils import create_agent_chain

SYSTEM_PROMPT = """You are the Visual Integrity Agent.
Analyze visual elements for:
1. Broken Rendering: LaTeX errors like '??' or '[Reference Not Found]'.
2. Placeholder Figures: Empty boxes, 'Insert Figure Here'.
3. Unreadable Content: Blurry/pixelated plots that are scientifically illegible.

Return JSON matching VisualIntegrityCheck schema."""

visual_agent = create_agent_chain(VisualIntegrityCheck, SYSTEM_PROMPT)