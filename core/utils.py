import mimetypes
from typing import List, Union, Dict, Any, Callable
import pydantic
from google.genai import types, chats
import os

from core.config import VertexEngine  # Import the configured LLM
from core.log import LOG
from core.constants import SKIP_DIRS, STYLE_GUIDES_DEFAULT, SUPPORTED_MIME_TYPES
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

def create_chat(pydantic_model, system_instructions, model_id: str = 'gemini-2.5-flash') -> None:
    """Create chat for a single agent."""

    if __CHATS.get(pydantic_model.__name__, None):
        LOG.info(f"Chat {__CHATS.get(pydantic_model.__name__)} already exists")

    engine = VertexEngine(model_id=model_id)
    structured_engine = engine.set_schema(schema=pydantic_model)
    structured_engine = structured_engine.set_system_instruction(instruction=types.Part.from_text(text=system_instructions))

    LOG.info(f"Creating chat for {pydantic_model.__name__}")
    __CHATS[pydantic_model.__name__] = structured_engine.get_chat_session()


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
