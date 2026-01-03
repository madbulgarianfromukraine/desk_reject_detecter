import base64
from typing import List, Union, Dict, Any, Callable
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage

from core.cache_manager import get_optimized_fallback_mime
from core.config import llm  # Import the configured LLM
import os

from core.schemas import AnalysisReport

__REQUIREMENTS_DIR = '../data/iclr/requirements/'
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
#Skip dirs for efficient loading of supplemental files
SKIP_DIRS = {'.venv', 'CVS', '.git', '__pycache__', '.pytest_cache'}

def encode_image(image_path: str):
    """Encodes a local image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def add_supplemental_files(path_to_supplemental_files: Union[os.PathLike, str]) -> List[Union[os.PathLike, str]]:
    supplemental_files_paths = []

    for root, dirs, files in os.walk(f".{path_to_supplemental_files}"):
        # Modifying dirs[:] in-place prunes the search tree
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

        for file in files:
            if not file.startswith("."):
                supplemental_files_paths.append(os.path.join(root, file))

    return supplemental_files_paths

def create_agent_chain(pydantic_model, system_instructions) -> Callable:
    """Factory function that creates a specialized agent."""
    structured_llm = llm.with_structured_output(pydantic_model)

    def run_agent(path_to_sub_dir):
        main_pdf_bytes = open("/path/to/your/test.pdf", "rb").read()
        main_pdf_base64 = base64.b64encode(main_pdf_bytes).decode("utf-8")

        content = [
            {"type": "text", "text": f"Here is the main.pdf for the paper"},
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": "application/pdf",
                "data": main_pdf_base64,
            }
        ]

        messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=content)
        ]

        if os.path.exists(f"{path_to_sub_dir}/supplemental_files"):
            supplemental_files = add_supplemental_files(f'{path_to_sub_dir}/supplemental_files')
            supplemental_files_dict_list : List[Dict[str, Any]] = []

            supplemental_file_bytes = open("/path/to/your/test.pdf", "rb").read()
            supplemental_file_base64 = base64.b64encode(supplemental_file_bytes).decode("utf-8")

            for supplemental_file in supplemental_files:
                supplemental_files_dict_list.append(
                    {
                        "type": "file",
                        "source_type": "base64",
                        "mime_type": get_optimized_fallback_mime(supplemental_file),
                        "data": supplemental_file_base64,
                    }
                )

            supplemental_files_dict_list.insert(0,
                                                {"type": "text", "text": "Here are the supplemental files for the paper"})

            messages.append(
                HumanMessage(
                    content=supplemental_files_dict_list
                )
            )

        return structured_llm.invoke(messages)

    return run_agent

def create_final_agent(pydantic_model, system_instructions) -> Callable:
    structured_llm = llm.with_structured_output(pydantic_model)

    def run_agent(analysis_report: AnalysisReport):
        human_message_content = [
            {
                "type": "text",
                "text": f"Here is the result of {key}\n{val}\n"
            }
            for key, val in vars(analysis_report).items()
            if key.endswith("_check")
        ]

        messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=human_message_content)
        ]

        return structured_llm.invoke(messages)

    return run_agent
