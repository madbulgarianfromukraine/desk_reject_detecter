import base64
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from config import llm  # Import the configured LLM


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