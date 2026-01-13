import mimetypes
from typing import List, Union, Dict, Type, Optional
import pydantic
from google.genai import types, chats
import os

from core.config import VertexEngine  # Import the configured LLM
from core.log import LOG
from core.constants import SKIP_DIRS, STYLE_GUIDES_DEFAULT, SUPPORTED_MIME_TYPES
from core.schemas import AnalysisReport, FinalDecision


__CHATS : Dict[str, chats.Chat] = {}
__ENGINES : Dict[str, VertexEngine] = {}
__STYLE_GUIDES_CACHE : List[types.Part] = []

def get_style_guides_parts() -> List[types.Part]:
    """Get style guides as a list of Parts, using cache if available."""
    global __STYLE_GUIDES_CACHE
    if not __STYLE_GUIDES_CACHE:
        LOG.info("Loading style guides into the prompt")
        for style_guide in STYLE_GUIDES_DEFAULT:
            with open(style_guide, "rb") as f:
                __STYLE_GUIDES_CACHE.append(types.Part.from_bytes(
                    data=f.read(),
                    mime_type=get_optimized_fallback_mime(str(style_guide))
                ))
    return __STYLE_GUIDES_CACHE

def get_optimized_fallback_mime(file_path: str) -> str:
    """
    Determines the best supported MIME type for a given file, falling back to safe defaults
    if the exact type is not supported by the Gemini API.

    Rationale:
    - Gemini has a specific list of supported MIME types.
    - For unsupported media, we map to a "best-fit" supported type (e.g., any video -> video/mp4)
      to allow the model to attempt processing.
    - text/plain is used as the ultimate fallback for unknown or varied text formats.

    :param file_path: Path to the file.
    :return: A supported MIME type string.
    """
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
    """
    Recursively gathers all files from the supplemental files directory.

    Implementation Details:
    - Uses os.walk to traverse the directory tree.
    - Prunes the search tree by modifying 'dirs' in-place to skip hidden directories
      and those listed in SKIP_DIRS (e.g., .venv, __pycache__).
    - Ignores hidden files (starting with '.').

    :param path_to_supplemental_files: Path to the directory containing supplemental materials.
    :return: A list of full file paths.
    """
    supplemental_files_paths = []

    for root, dirs, files in os.walk(f"{path_to_supplemental_files}"):
        # Modifying dirs[:] in-place prunes the search tree
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.') and not d.startswith("_")]

        for file in files:
            if not file.startswith("."):
                supplemental_files_paths.append(os.path.join(root, file))

    return supplemental_files_paths

def create_chat(pydantic_model: Type[pydantic.BaseModel], system_instructions, model_id: str = 'gemini-2.5-flash',
                search_included : bool = False, thinking_included : bool = False,
                upload_style_guides: bool = False, ttl_seconds: str = "180s") -> None:
    """
    Initializes and caches a chat session for a specific agent/schema.

    This setup includes:
    1. Schema binding for structured output.
    2. System instruction configuration.
    3. Logprobs enablement for confidence scoring.
    4. Optional Google Search grounding and Thinking capabilities.

    :param pydantic_model: The schema class defining the expected output.
    :param system_instructions: The identity and instructions for the agent.
    :param model_id: The model version to use.
    :param search_included: Whether to enable Google Search tool.
    :param thinking_included: Whether to enable thinking/reasoning config.
    :param upload_style_guides: Whether to upload the style_guides cache to use.
    :param ttl_seconds: How long to wait until the cache is deleted.
    """

    if __CHATS.get(pydantic_model.__name__, None):
        LOG.info(f"Chat for {pydantic_model.__name__} already exists. Skipping initialization.")
        return

    engine = VertexEngine(model_id=model_id)


    # NEW: Create cache for style guides before creating the chat
    if upload_style_guides:
        style_guides = get_style_guides_parts()
        if style_guides:
            LOG.info(f"Creating context cache with style guides for {pydantic_model.__name__}")
            cache = engine.create_cache(
                contents=style_guides,
                display_name=f"style_guides",
                ttl_seconds=ttl_seconds
            )
            engine.set_cache(cache.name)
            engine.set_system_instruction(instruction=system_instructions)
        else:
            # Fallback to non-cached settings
            engine.set_system_instruction(instruction=system_instructions)

    structured_engine = engine.set_schema(schema=pydantic_model)
    structured_engine = structured_engine.set_logprobs()

    if search_included:
        LOG.debug("Adding grounding search")
        google_search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        structured_engine.config.tools = [google_search_tool]

    if thinking_included:
        LOG.debug("Adding thinking availability")
        structured_engine.config.thinking_config = types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024
        )
    else:
        structured_engine.config.thinking_config = None

    LOG.info(f"Creating chat for {pydantic_model.__name__}")
    __ENGINES[pydantic_model.__name__] = structured_engine
    __CHATS[pydantic_model.__name__] = structured_engine.get_chat_session()


def send_message_with_splitting(chat: chats.Chat, engine: VertexEngine, prompt_parts: List[types.Part]) -> Optional[types.GenerateContentResponse]:
    """
    Sends a message to the model, splitting it into parts if it exceeds the token limit.
    Each part generates a structured response, which are then merged.

    :param chat: The chat session to use.
    :param engine: The engine containing model and configuration.
    :param prompt_parts: The list of Parts to send.
    :return: The merged response from the model.
    """
    limit = engine.get_model_limit()
    total_tokens = engine.count_tokens(prompt_parts)

    if total_tokens > limit:
        LOG.info(f"Prompt tokens ({total_tokens}) exceed limit ({limit}). Splitting into parts.")
        chunks = engine.split_contents(prompt_parts, limit)

        responses = []
        for i, chunk in enumerate(chunks):
            LOG.info(f"Sending part {i+1}/{len(chunks)}")
            # We use the full config (including schema) for each part
            response = chat.send_message(chunk)
            LOG.debug(f"Responded with: {response.parsed}")
            responses.append(response)

        if len(responses) <= 1:
            return responses[0]

        final_merge_prompt = f"""
            Below are several partial desk-reject analysis reports for the same main_paper.pdf and its supplemental files.
            Please merge them into a single, consistent JSON report as specified to you
            If categories conflict, prioritize 'Policy' or 'Scope' over 'Formatting'.

            PARTIAL REPORTS:
            {chr(10).join([response.parsed for response in responses])}
            """

        return chat.send_message(types.Part.from_text(text=final_merge_prompt))
    else:
        return chat.send_message(prompt_parts)


def ask_agent(pydantic_model: Type[pydantic.BaseModel], path_to_sub_dir: str) -> types.GenerateContentResponse:
    """
    Constructs a multi-modal prompt and sends it to the specified agent.

    Prompt Construction Sequence:
    1. Conference style guides and requirements.
    2. Main paper PDF (expected at `path_to_sub_dir/main_paper.pdf`).
    3. Supplemental files (recursively gathered from `path_to_sub_dir/supplemental_files`).

    :param pydantic_model: The schema class identifying which agent's chat to use.
    :param path_to_sub_dir: Path to the directory containing the paper and supplemental data.
    :return: The raw response from the LLM, containing parsed structured data and logprobs.
    """
    # We build the prompt as a flat list of native Parts
    agent_chat = __CHATS[pydantic_model.__name__]
    engine = __ENGINES[pydantic_model.__name__]
    prompt_parts: List[types.Part] = list()

    # --- 2. Main Paper (Sequence: Text -> PDF File) ---
    prompt_parts.append(types.Part.from_text(text="Here is the main_paper.pdf for the paper"))
    with open(f"{path_to_sub_dir}/main_paper.pdf", "rb") as f:
        prompt_parts.append(types.Part.from_bytes(
            data=f.read(),
            mime_type="application/pdf"
        ))

    # --- 3. Supplemental Files (Optional Sequence: Text -> Multiple Files) ---
    supp_path = os.path.join(path_to_sub_dir, "supplemental_files")
    if os.path.exists(supp_path):
        prompt_parts.append(types.Part.from_text(text="Here are the supplemental files for the paper"))
        supplemental_files = add_supplemental_files(supp_path)
        for s_file in supplemental_files:
            with open(s_file, "rb") as f:
                prompt_parts.append(types.Part.from_bytes(
                    data=f.read(),
                    mime_type=get_optimized_fallback_mime(s_file)
                ))

    return send_message_with_splitting(agent_chat, engine, prompt_parts)

def ask_final(analysis_report: AnalysisReport) -> types.GenerateContentResponse:
    """
    Constructs a prompt from the aggregated AnalysisReport and sends it to the Final Decision Agent.

    This function extracts all check results (Safety, Anonymity, etc.) from the report
    and presents them to the final agent to reach a terminal desk-rejection decision.

    :param analysis_report: The object containing results from all individual auditor agents.
    :return: The terminal decision response.
    """

    final_agent_chat = __CHATS[FinalDecision.__name__]
    engine = __ENGINES[FinalDecision.__name__]
    prompt_parts: List[types.Part] = list()

    for key, val in vars(analysis_report).items():
        if key.endswith("_check"):
            prompt_parts.append(types.Part.from_text(
                text=f"Here is the result of {key}:\n{val}\n"
            ))

    # 3. Native chat.send_message
    # This automatically adds the prompt and model response to the session history
    return send_message_with_splitting(final_agent_chat, engine, prompt_parts)
