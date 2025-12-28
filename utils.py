import base64
from typing import List
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from config import llm  # Import the configured LLM

__REQUIREMENTS_DIR = 'data/iclr/requirements/'
STYLE_GUIDES_DEFAULT = [f for f in Path(__REQUIREMENTS_DIR).iterdir() if f.is_file()]

# Exhaustive list of standard supported types for Gemini 2.5
SUPPORTED_MIME_TYPES = {
    'application/pdf', 'text/plain',
    'image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif',
    'video/mp4', 'video/mpeg', 'video/mov', 'video/avi', 'video/x-flv',
    'video/mpg', 'video/webm', 'video/wmv', 'video/3gpp',
    'audio/wav', 'audio/mp3', 'audio/aiff', 'audio/aac', 'audio/ogg',
    'audio/flac', 'audio/m4a', 'audio/mpga', 'audio/pcm'
}

def encode_image(image_path: str):
    """Encodes a local image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_agent_chain(pydantic_model, system_instructions):
    """Factory function that creates a specialized agent."""
    structured_llm = llm.with_structured_output(pydantic_model)

    def run_agent(paper_text: str, paper_images: List[str]):
        content = [{"type": "text", "text": f"Analyze this paper content:\n\n{paper_text}"}]

        for img_b64 in paper_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

        messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=content)
        ]
        return structured_llm.invoke(messages)

    return run_agent