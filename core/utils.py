import mimetypes
from typing import List, Union, Dict, Any, Callable
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage

from core.cache_manager import get_optimized_fallback_mime
from core.config import llm  # Import the configured LLM
from core.constants import SKIP_DIRS
import os

from core.schemas import AnalysisReport


__CHATS : Dict[str, chats.Chat] = {}

def get_optimized_fallback_mime(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)

    if mime in SUPPORTED_MIME_TYPES:
        return mime

    # 2. Structural pattern matching for closest-category fallbacks
    match mime.split('/') if mime else []:
        case ['video', _]:
            return 'video/mp4'  # Best fallback for all unsupported video
        case ['audio', _]:
            return 'audio/mpeg'  # Best fallback for all unsupported audio
        case ['image', _]:
            return 'image/jpeg'  # Best fallback for all unsupported images
        case ['text', _]:
            return 'text/plain'  # Catch-all for varied text (logs, csv, etc.)
        case _:
            return 'text/plain'  # Ultimate default for unknown binaries

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
    structured_llm = llm.with_structured_output(pydantic_model)

    def run_agent(path_to_sub_dir):
        main_pdf_bytes = open(f'{path_to_sub_dir}/main_paper.pdf', "rb").read()
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


            for supplemental_file in supplemental_files:
                supplemental_file_bytes = open(supplemental_file, "rb").read()
                supplemental_file_base64 = base64.b64encode(supplemental_file_bytes).decode("utf-8")

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
