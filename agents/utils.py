import concurrent.futures
from . import (
    formatting_agent,
    policy_agent,
    anonymity_agent,
    scope_agent,
    final_decision_agent
    # DISABLED: visual_agent, safety_agent
)

AGENT_MAPPING = {
    'formatting_check': formatting_agent.ask_formatting_agent,
    'policy_check': policy_agent.ask_policy_agent,
    'anonymity_check': anonymity_agent.ask_anonymity_agent,
    'scope_check': scope_agent.ask_scope_agent,
    # DISABLED: visual_integrity_check, safety_check
}

def create_chats(model_id: str = 'gemini-2.5-flash', include_thinking: bool = False, include_search: bool = False, ttl_seconds: str = "300s") -> None:
    """
    Initializes the chat settings for all agents in the desk rejection system.

    This function sets up each agent with the specified capabilities (thinking and search)
    by calling their respective `create_chat_settings` methods in parallel.

    :param model_id: The model ID to use for all agents (e.g., 'gemini-2.5-flash').
    :param include_thinking: Whether to enable thinking/reasoning capabilities for applicable agents.
    :param include_search: Whether to enable search capabilities for applicable agents.
    """
    # 1. Initialize all agents in parallel
    initialization_tasks = [
        (formatting_agent.create_chat_settings, {"model_id": model_id, "ttl_seconds": ttl_seconds}),
        (policy_agent.create_chat_settings, {'model_id': model_id, 'search_included': include_search}),
        (anonymity_agent.create_chat_settings, {'model_id': model_id, 'search_included': include_search}),
        (scope_agent.create_chat_settings, {'model_id': model_id, 'thinking_included': include_thinking}),
        (final_decision_agent.create_chat_settings, {'model_id': model_id, 'thinking_included': include_thinking}),
        # DISABLED: visual_agent, safety_agent initialization
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(initialization_tasks), thread_name_prefix="ChatCreationThread") as executor:
        futures = [executor.submit(func, **kwargs) for func, kwargs in initialization_tasks]
        concurrent.futures.wait(futures)
