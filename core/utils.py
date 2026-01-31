import time
from typing import List, Union, Dict, Type, Optional, Any
import pydantic
import threading
from google.genai import types, chats, errors
import os

from core.config import VertexEngine, create_engine  # Import the configured LLM
from core.log import LOG
from core.schemas import AnalysisReport, FinalDecision
from core.metrics import increase_total_output_tokens, increase_total_input_tokens
from core.files import process_supplemental_files
from core.rate_limiter import retry_with_backoff

__ENGINES : Dict[str, VertexEngine] = {}

__API_LOCK: threading.Semaphore = threading.Semaphore(5)


def send_message_with_token_counting(engine: VertexEngine, message: Union[list[types.PartUnionDict], types.PartUnionDict],
                                     config: Optional[types.GenerateContentConfigOrDict] = None, wait: bool = False) -> types.GenerateContentResponse:

    @retry_with_backoff
    def _send_with_retry():
        """Inner function to apply retry logic to engine.generate."""
        with __API_LOCK:
            response = engine.generate(contents=message)
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


def send_message_with_cutting(engine: VertexEngine, prompt_parts: List[types.Part]) -> Optional[types.GenerateContentResponse]:
    """
    Sends a message to the model, sending only main paper pdf if it exceeds the token limit.
    Keeps removing parts from the end until the remaining content fits within the context window.

    :param engine: The engine containing model and configuration.
    :param prompt_parts: The list of Parts to send.
    :return: The response from the model.
    """
    limit = engine.get_model_limit()

    valid_parts = [
    p for p in prompt_parts 
    if (getattr(p, 'text', None) and p.text.strip()) or getattr(p, 'inline_data', None) and len(p.inline_data.data) > 0
    ]
    try:
        total_tokens = engine.count_tokens(valid_parts)
    except errors.ClientError as e:
        LOG.error(f"Token counting failed: {e}. Sending main paper only.")
        total_tokens = limit + 1 # Force sending main paper only

    if total_tokens > limit:
        LOG.info(f"Prompt tokens ({total_tokens}) exceed limit ({limit}). Analyzing main_paper only")

        return send_message_with_token_counting(engine=engine, message=prompt_parts[:2])
    else:
        return send_message_with_token_counting(engine=engine, message=prompt_parts)

def ask_agent(pydantic_model: Type[pydantic.BaseModel], system_instruction: str,
              path_to_sub_dir: str,
              model_id: str = 'gemini-2.5-flash', main_paper_only: bool = False,
              search_included : bool = False, thinking_included : bool = False,
              upload_style_guides: bool = False, ttl_seconds: str = "300s") -> types.GenerateContentResponse:
    """
    Executes a multi-modal analysis by sending a paper and its associated files to a structured agent.

    This function initializes a specialized engine based on the provided Pydantic schema and constructs 
    a prompt sequence including the main PDF and optional supplemental materials.

    :param pydantic_model: Pydantic class used to define the structured output schema and engine identity.
    :param system_instruction: The core behavioral prompt for the agent.
    :param path_to_sub_dir: Local directory containing 'main_paper.pdf' and the 'supplemental_files' folder.
    :param model_id: Identifier for the generative model.
    :param main_paper_only: If True, skips processing the supplemental files directory.
    :param search_included: Whether to enable web search capabilities for the engine.
    :param thinking_included: Whether to enable extended reasoning/thinking features.
    :param upload_style_guides: Whether to include conference-specific style requirements in the initialization.
    :param ttl_seconds: Time-to-live for cached content or engine resources.
    :return: A GenerateContentResponse object containing structured data, text, and metadata.
    """
    if __ENGINES.get(pydantic_model.__name__, None):
        engine = __ENGINES[pydantic_model.__name__]
        LOG.info(f"Engine for {pydantic_model.__name__} already exists. Skipping initialization.")
    else:
        engine = create_engine(model_id=model_id, pydantic_model=pydantic_model,
                               system_instruction=system_instruction, thinking_included=thinking_included,
                               search_included=search_included, upload_style_guides=upload_style_guides,
                               ttl_seconds=ttl_seconds)
        __ENGINES[pydantic_model.__name__] = engine

    prompt_parts: List[types.Part] = list()

    # --- 2. Main Paper ---
    prompt_parts.append(types.Part.from_text(text="Here is the main_paper.pdf for the paper"))
    with open(f"{path_to_sub_dir}/main_paper.pdf", "rb") as f:
        prompt_parts.append(types.Part.from_bytes(
            data=f.read(),
            mime_type="application/pdf"
        ))

    # --- 3. Supplemental Files ---
    if not main_paper_only:
        supp_path = os.path.join(path_to_sub_dir, "supplemental_files")
        process_supplemental_files(supp_path, prompt_parts)

    return send_message_with_cutting(engine, prompt_parts)

def ask_final(analysis_report: AnalysisReport,
              system_instruction: str = "",
              model_id: str = 'gemini-2.5-flash',
              search_included : bool = False, thinking_included : bool = False,
              ) -> types.GenerateContentResponse:
    """
    Constructs a prompt from the aggregated AnalysisReport and sends it to the Final Decision Agent.

    This function extracts all check results (Safety, Anonymity, etc.) from the report
    and presents them to the final agent to reach a terminal desk-rejection decision.

    :param analysis_report: The object containing results from all individual auditor agents.
    :param submission_id: Unique identifier for the submission. Used for caching chunks.
    :return: The terminal decision response.
    """

    if __ENGINES.get(FinalDecision.__name__, None):
        engine = __ENGINES[FinalDecision.__name__]
        LOG.info(f"Engine for {FinalDecision.__name__} already exists. Skipping initialization.")
    else:
        engine = create_engine(model_id=model_id, pydantic_model=FinalDecision,
                               system_instruction=system_instruction, thinking_included=thinking_included,
                               search_included=search_included)
        __ENGINES[FinalDecision.__name__] = engine
        engine.set_temperature(temp=0.0)

    prompt_parts: List[types.Part] = list()

    for key, val in vars(analysis_report).items():
        if key.endswith("_check"):
            prompt_parts.append(types.Part.from_text(
                text=f"Here is the result of {key}:\n{val}\n"
            ))

    return send_message_with_cutting(engine, prompt_parts)

