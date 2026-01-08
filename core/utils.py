import base64
from typing import List, Union, Dict, Any, Callable
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from core.cache_manager import get_optimized_fallback_mime, get_style_guide_content
from core.config import llm  # Import the configured LLM
from core.constants import SKIP_DIRS
import os

from core.schemas import AnalysisReport


def add_supplemental_files(path_to_supplemental_files: Union[os.PathLike, str]) -> List[Union[os.PathLike, str]]:
    supplemental_files_paths = []

    for root, dirs, files in os.walk(f"{path_to_supplemental_files}"):
        # Modifying dirs[:] in-place prunes the search tree
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.') and not d.startswith("_")]

        for file in files:
            if not file.startswith("."):
                supplemental_files_paths.append(os.path.join(root, file))

    return supplemental_files_paths

def create_agent_chain(pydantic_model, system_instructions) -> Callable:
    """Factory function that creates a specialized agent."""
    structured_llm = llm.with_structured_output(pydantic_model, include_raw=True)

    def run_agent(path_to_sub_dir):
        # 1. Start with the style guides (leveraging OpenAI Prompt Caching)
        messages : List[BaseMessage] = [
            SystemMessage(content=get_style_guide_content()),
            SystemMessage(content=system_instructions),
        ]

        # 2. Add the main paper
        main_pdf_path = f'{path_to_sub_dir}/main_paper.pdf'
        with open(main_pdf_path, "rb") as f:
            main_pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        main_content = [
            {"type": "text", "text": "Here is the main_paper.pdf"},
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": "application/pdf",
                "data": main_pdf_base64,
            }
        ]
        messages.append(HumanMessage(content=main_content))

        # 3. Add supplemental files
        if os.path.exists(f"{path_to_sub_dir}/supplemental_files"):
            supplemental_files = add_supplemental_files(f'{path_to_sub_dir}/supplemental_files')
            supplemental_content = [{"type": "text", "text": "Here are the supplemental files for the paper:"}]

            for supplemental_file in supplemental_files:
                mime = get_optimized_fallback_mime(supplemental_file)
                try:
                    with open(supplemental_file, "rb") as f:
                        file_data = base64.b64encode(f.read()).decode("utf-8")
                    
                    supplemental_content.append({
                        "type": "file",
                        "source_type": "base64",
                        "mime_type": mime,
                        "data": file_data,
                    })
                except Exception as e:
                    from core.log import LOG
                    LOG.error(f"Failed to read supplemental file {supplemental_file}: {e}")

            messages.append(HumanMessage(content=supplemental_content))

        return structured_llm.invoke(messages)

    return run_agent

def create_final_agent(pydantic_model, system_instructions) -> Callable:
    structured_llm = llm.with_structured_output(pydantic_model, include_raw=True)

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
