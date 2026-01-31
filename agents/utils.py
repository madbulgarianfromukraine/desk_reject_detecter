from . import (
    formatting_agent,
    policy_agent,
    anonymity_agent,
    scope_agent,
    # DISABLED: visual_agent, safety_agent
)

AGENT_MAPPING = {
    'formatting_check': formatting_agent.ask_formatting_agent,
    'policy_check': policy_agent.ask_policy_agent,
    'anonymity_check': anonymity_agent.ask_anonymity_agent,
    'scope_check': scope_agent.ask_scope_agent,
    # DISABLED: visual_integrity_check, safety_check
}
