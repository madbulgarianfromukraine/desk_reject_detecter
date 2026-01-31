import time
from typing import List, Union, Dict, Type, Optional, Any
import pydantic
import threading
from google.genai import types, chats
import os

from core.config import VertexEngine  # Import the configured LLM
from core.log import LOG
from core.schemas import AnalysisReport, FinalDecision
from core.metrics import increase_total_output_tokens, increase_total_input_tokens
from core.files import get_style_guides_parts, get_optimized_fallback_mime, try_decoding, add_supplemental_files
from core.rate_limiter import retry_with_backoff

__CHATS : Dict[str, chats.Chat] = {}
__ENGINES : Dict[str, VertexEngine] = {}
__CACHE : Dict[str, types.CachedContent] = {}

__API_LOCK: threading.Semaphore = threading.Semaphore(5)

def create_chat(pydantic_model: Type[pydantic.BaseModel], system_instructions: str, model_id: str = 'gemini-2.5-flash',
                search_included : bool = False, thinking_included : bool = False,
                upload_style_guides: bool = False, ttl_seconds: str = "300s") -> None:
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
    if isinstance(pydantic_model, FinalDecision):
        engine.set_temperature(temp=0.0)

    structured_engine = engine.set_schema(schema=pydantic_model)
    # NEW: Create cache for style guides before creating the chat
    if upload_style_guides:
        style_guides = get_style_guides_parts()
        if style_guides:

            style_guides.insert(0, types.Part.from_text(text=system_instructions))
            LOG.info(f"Creating context cache with style guides for {pydantic_model.__name__}")
            cache = structured_engine.create_cache(
                contents=style_guides,
                display_name=f"style_guides",
                ttl_seconds=ttl_seconds
            )
            structured_engine.set_cache(cache.name)
            __CACHE[pydantic_model.__name__] = cache
        else:
            # Fallback to non-cached settings
            structured_engine.set_system_instruction(instruction=system_instructions)
    else:
        structured_engine.set_system_instruction(instruction=system_instructions)
        if search_included:
            LOG.debug("Adding grounding search")
            google_search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            structured_engine.config.tools = [google_search_tool]

    structured_engine = structured_engine.set_logprobs()

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


def send_message_with_token_counting(chat: chats.Chat, message: Union[list[types.PartUnionDict], types.PartUnionDict],
                                     config: Optional[types.GenerateContentConfigOrDict] = None, wait: bool = False) -> types.GenerateContentResponse:

    @retry_with_backoff
    def _send_with_retry():
        """Inner function to apply retry logic to chat.send_message."""
        with __API_LOCK:
            response = chat.send_message(message=message, config=config)
            if wait:
                # Rate limit friendly delay: 5 req/sec = 200ms per request
                time.sleep(0.2)
            return response
    
    response = _send_with_retry()

    # add output tokens to the count
    additional_output_tokens = response.usage_metadata.candidates_token_count
    increase_total_output_tokens(additional_tokens=additional_output_tokens)

    # add input tokens to the count
    additional_input_tokens = response.usage_metadata.prompt_token_count
    increase_total_input_tokens(additional_tokens=additional_input_tokens)

    return response


def send_message_with_cutting(chat: chats.Chat, engine: VertexEngine, prompt_parts: List[types.Part]) -> Optional[types.GenerateContentResponse]:
    """
    Sends a message to the model, sending only main paper pdf if it exceeds the token limit.
    Keeps removing parts from the end until the remaining content fits within the context window.

    :param chat: The chat session to use.
    :param engine: The engine containing model and configuration.
    :param prompt_parts: The list of Parts to send.
    :return: The response from the model.
    """
    limit = engine.get_model_limit()

    valid_parts = [
    p for p in prompt_parts 
    if (getattr(p, 'text', None) and p.text.strip()) or getattr(p, 'inline_data', None) and len(p.inline_data.data) > 0
    ]
    total_tokens = engine.count_tokens(valid_parts)

    if total_tokens > limit:
        LOG.info(f"Prompt tokens ({total_tokens}) exceed limit ({limit}). Analyzing main_paper only")

        return send_message_with_token_counting(chat=chat, message=prompt_parts[:2])
    else:
        return send_message_with_token_counting(chat=chat, message=prompt_parts)


def ask_agent(pydantic_model: Type[pydantic.BaseModel], path_to_sub_dir: str, main_paper_only: bool = False) -> types.GenerateContentResponse:
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
    if not main_paper_only:
        supp_path = os.path.join(path_to_sub_dir, "supplemental_files")
        if os.path.exists(supp_path):
            prompt_parts.append(types.Part.from_text(text="Here are the supplemental files for the paper"))
            supplemental_files = add_supplemental_files(supp_path)
            for s_file in supplemental_files:
                s_file_mime = get_optimized_fallback_mime(s_file)
                prompt_parts.append(types.Part.from_text(text=f"The file {s_file}:"))

                with open(s_file, "rb") as f:
                    f_read = f.read()
                    if len(f_read) <= 0:
                        continue

                    if not s_file_mime:
                        file_part = try_decoding(binary_data=f_read)
                        if not file_part:
                            LOG.info(f"The file '{s_file}' couldn't be uploaded due to unsupported mime_type, but the notice was added.")
                            continue

                        prompt_parts.append(file_part)
                    else:
                        prompt_parts.append(types.Part.from_bytes(
                            data=f_read,
                            mime_type=s_file_mime
                        ))

    return send_message_with_cutting(agent_chat, engine, prompt_parts)

def ask_final(analysis_report: AnalysisReport, submission_id: str = None) -> types.GenerateContentResponse:
    """
    Constructs a prompt from the aggregated AnalysisReport and sends it to the Final Decision Agent.

    This function extracts all check results (Safety, Anonymity, etc.) from the report
    and presents them to the final agent to reach a terminal desk-rejection decision.

    :param analysis_report: The object containing results from all individual auditor agents.
    :param submission_id: Unique identifier for the submission. Used for caching chunks.
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

    return send_message_with_cutting(final_agent_chat, engine, prompt_parts)



def cleanup_caches() -> None:
    """
    Cleans up the shared cache to free resources.

    This function deletes the shared cached content from the Google Gemini API.
    Should be called on program exit or on KeyboardInterrupt to prevent resource wastage.

    :return: None
    """
    engine = VertexEngine()
    for cache in __CACHE.values():
        if cache is None:
            LOG.info("No cache to clean up.")
            return

        try:
            LOG.info(f"Deleting cache: {cache.name}")
            engine.client.caches.delete(name=cache.name)
            LOG.info("Cache cleanup completed successfully.")
        except Exception as e:
            LOG.error(f"Error during cache cleanup: {e}")

