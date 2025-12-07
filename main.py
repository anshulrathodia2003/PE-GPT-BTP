"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin
@github: https://github.com/XinzeLee/PE-GPT

GROQ EDITION - Converted from OpenAI to Groq + Open Source

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, 
                 Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import streamlit as st
import os
os.environ['GROQ_API_KEY'] = "your_groq_api"

# Import GUI functions
from core.gui import (
    build_gui, 
    init_states, 
    display_history,
)

# Import design flow functions
from core.gui.design_stages import (
    design_flow,
    create_task_agent
)

# Import LLM functions
from core.llm.llm import (
    create_chat_engine,
    get_system_prompt
)


# =====================================================================
# MAIN APPLICATION
# =====================================================================

if __name__ == "__main__":
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # Flexible-response mode: use an LLM agent to enrich
    # and enhance the PE expertise of predefined responses
    FlexRes = True
    
    # ChromaDB paths for different knowledge bases
    CHROMA_DB_PATH = "core/knowledge/kb/database"
    CHROMA_DB1_PATH = "core/knowledge/kb/database1"
    CHROMA_DB2_PATH = "core/knowledge/kb/introduction"
    
    # Temperature setting
    temperature = 0.0
    
    
    # ============================================================
    # BUILD GUI
    # ============================================================
    
    build_gui()
    
    
    # ============================================================
    # API KEY CHECK
    # ============================================================
    
    # Get Groq API key from user
    # api_key = get_groq_api_key()
    
    # if not api_key:
    #     st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")
    #     st.stop()
    
    
    # ============================================================
    # MODEL SELECTION
    # ============================================================
    
    # Get model from session state (set by sidebar in build_gui)
    llm_model = st.session_state.get("llm_model", "llama-3.1-8b-instant")
    
    
    # ============================================================
    # LOAD SYSTEM PROMPT
    # ============================================================
    
    # Load system prompt from file
    prompt_path = 'core/knowledge/prompts/prompt.txt'
    
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                system_prompt = file.read()
        except Exception as e:
            st.error(f"Error reading prompt file: {str(e)}")
            system_prompt = get_system_prompt("general")
    else:
        st.warning("⚠️ Prompt file not found, using default system prompt")
        system_prompt = get_system_prompt("general")
    
    
    # ============================================================
    # CREATE CHAT ENGINES WITH RAG
    # ============================================================
    
    # AGENT 0: Provides insights and PE-specific reasoning for selected modulations
    # Uses main knowledge base with RAG
    chat_engine0 = create_chat_engine(
        llm_model=llm_model,
        system_prompt=system_prompt,
        temperature=temperature,
        use_rag=True,
        database_folder=CHROMA_DB_PATH,
        session_id="agent0"
    )
    
    # AGENT 1: Specialized in modulation recommendation
    # Uses modulation-specific knowledge base
    chat_engine1 = create_chat_engine(
        llm_model=llm_model,
        system_prompt=system_prompt,
        temperature=temperature,
        use_rag=True,
        database_folder=CHROMA_DB1_PATH,
        session_id="agent1"
    )
    
    # AGENT 2: For PE-GPT introduction
    # Uses introduction knowledge base
    intro_prompt = get_system_prompt("general")
    chat_engine2 = create_chat_engine(
        llm_model=llm_model,
        system_prompt=intro_prompt,
        temperature=temperature,
        use_rag=True,
        database_folder=CHROMA_DB2_PATH,
        session_id="agent2"
    )
    
    
    # ============================================================
    # CREATE TASK CLASSIFICATION AGENT
    # ============================================================
    
    # Define an LLM agent to judge and keep track of the design stage/task
    # task_agent = create_task_agent(model=llm_model)
    task_agent = create_task_agent()

    
    # ============================================================
    # INITIALIZE SESSION STATE
    # ============================================================
    
    # Define electrical variables that might be used
    initial_values = {
        'M': 'TPS',      # Modulation strategy
        'Uin': None,    # Input voltage
        'Uo': None,     # Output voltage
        'P': None,      # Power level
        'fs': None,     # Switching frequency
        'vp': None,     # Primary voltage data
        'vs': None,     # Secondary voltage data
        'iL': None,     # Inductor current data
        'pos': None,     # Optimized modulation parameters
        'topology': 'unknown'
    }
    
    # Initialize st.session_state
    init_states(initial_values)
    
    
    # ============================================================
    # DISPLAY HISTORICAL MESSAGES
    # ============================================================
    
    # Display the historical chat messages
    display_history()
    
    
    # ============================================================
    # RUN PE-GPT DESIGN WORKFLOW
    # ============================================================
    
    # Prepare chat engines dictionary
    chat_engines = {
        'agent0': chat_engine0,
        'agent1': chat_engine1,
        'agent2': chat_engine2
    }
    
    # Run the PE-GPT engine to conduct the design workflow
    design_flow(
        chat_engines=chat_engines,
        task_agent=task_agent,
        FlexRes=FlexRes
    )