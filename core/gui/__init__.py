"""
Manage the importable files, functions, classes, and variables

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

# Import functions from gui module
from .gui import (
    build_gui, 
    init_states, 
    display_history,
    upload_func,
    initialize_conversation,
    display_data_status,
    display_system_info,
    display_shortcuts
)

# Import functions and classes from design_stages module
from .design_stages import (
    design_flow,
    create_task_agent,
    TaskClassificationAgent,
    init_design,
    recommend_modulation,
    evaluate_dab,
    simulation_verification,
    pe_gpt_introduction,
    train_pann,
    other_tasks
)


__all__ = [
    # GUI functions
    'build_gui',
    'init_states',
    'display_history',
    'upload_func',
    'initialize_conversation',
    'display_data_status',
    'get_groq_api_key',
    'display_system_info',
    'display_shortcuts',
    
    # Design flow functions
    'design_flow',
    'create_task_agent',
    'init_design',
    'recommend_modulation',
    'evaluate_dab',
    'simulation_verification',
    'pe_gpt_introduction',
    'train_pann',
    'other_tasks',
    
    # Classes
    'TaskClassificationAgent',
]