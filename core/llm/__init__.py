# Manage the importable files, functions, classes, and variables
"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT

GROQ EDITION - Converted from OpenAI to Groq + Open Source

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, 
                 Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

# import files
from . import llm
from . import custom_responses

# import functions from llm.py
from .llm import (
    groq_init,
    rag_load,
    create_chat_engine,
    get_msg_history,
    get_session_history,
    clear_session_history,
    get_system_prompt,
    check_groq_api_key,
    format_chat_history
)

# import functions from custom_responses.py
from .custom_responses import (
    response,
    get_parameter_string,
    get_modulation_info,
    format_performance_summary,
    compare_modulation_strategies,
    validate_performance_data,
    MODULATION_STRATEGIES
)


__all__ = [
    # importable modules
    "llm",
    "custom_responses",
    
    # importable functions from llm.py
    "groq_init",
    "rag_load",
    "create_chat_engine",
    "get_msg_history",
    "get_session_history",
    "clear_session_history",
    "get_system_prompt",
    "check_groq_api_key",
    "format_chat_history",
    
    # importable functions from custom_responses.py
    "response",
    "get_parameter_string",
    "get_modulation_info",
    "format_performance_summary",
    "compare_modulation_strategies",
    "validate_performance_data",
    
    # importable variables/constants
    "MODULATION_STRATEGIES",
]