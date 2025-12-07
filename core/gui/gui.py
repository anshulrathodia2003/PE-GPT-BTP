"""
@functionality
    Create a graphical user interface (GUI) using Streamlit for PE-GPT
    Handles file uploads, conversation management, and display

@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT

GROQ EDITION - Converted from OpenAI to Groq + Open Source
Multi-Topology Support: DAB, Buck, Boost

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
import numpy as np



def build_gui():
    """
    Create a graphical user interface (GUI) using Streamlit
    Open-source version using Groq LLM backend with multi-topology support
    """
    
    # Graphical User Interface Configuration
    st.set_page_config(
        page_title="PE-GPT (Groq Edition)", 
        page_icon="💎", 
        layout="centered",
        initial_sidebar_state="auto", 
        menu_items=None
    )
    
    st.title("Chat with the Power Electronics Robot 🤖")
    st.info(
        "Hello, I am a robot specifically for power electronics design! "
        "Powered by Groq's open-source LLMs with support for DAB, Buck, and Boost converters.", 
        icon="🤟"
    )
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown(
            "<h1 style='color: #FF5733;'>PE-GPT (v2.0 - Groq Edition)</h1>", 
            unsafe_allow_html=True
        )
        st.markdown('---')
        
        # Feature Description - UPDATED
        st.markdown(
            "### 🚀 Features\n"
            "- **Multi-Topology Support**: DAB, Buck, Boost converters\n"
            "- **Modulation Strategies**: SPS, DPS, EPS, TPS, 5DOF (DAB)\n"
            "- **Open Source**: Powered by Groq & LangChain\n"
            "- **Model Zoo**: PANN models for DAB converters\n"
        )
        
        st.markdown('---')
        
        # LLM Model Selection
        st.markdown("### ⚙️ Model Settings")
        llm_model = st.selectbox(
            "Select LLM Model",
            [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "openai/gpt-oss-20b",
                "meta-llama/llama-4-maverick-17b-128e-instruct"
            ],
            index=0,
            help="Choose the Groq model for inference"
        )
        
        # Store model selection in session state
        if 'llm_model' not in st.session_state or st.session_state.llm_model != llm_model:
            st.session_state.llm_model = llm_model
            st.info(f"Model changed to: {llm_model}")
        
        st.markdown('---')
        
        # Project Information
        st.markdown(
            "### 📚 Reference\n"
            "**Paper**: PE-GPT: a New Paradigm for Power Electronics Design\n\n"
            "**Authors**: Fanfan Lin, Xinze Li, et al.\n\n"
            "**GitHub**: [PE-GPT Repository](https://github.com/XinzeLee/PE-GPT)\n\n"
            "**DOI**: 10.1109/TIE.2024.3454408"
        )
        
        st.markdown('---')
        
        # NEW: Display detected topology
        display_current_topology()
    
    # Clear conversation button
    clear_button = st.sidebar.button('Clear Conversation', key='clear')
    
    # # File Upload Section
    # st.sidebar.markdown("### 📁 Training Data Upload")
    # st.sidebar.markdown("*Upload experimental data for DAB PANN model training*")
    
    # # File type selection
    # file_type = st.sidebar.selectbox(
    #     "Select file type", 
    #     ("vp", "vs", "iL"),
    #     help="vp: Primary voltage, vs: Secondary voltage, iL: Inductor current"
    # )
    
    # # File uploader
    # uploaded_file = st.sidebar.file_uploader(
    #     "Upload CSV file", 
    #     key="file_uploader",
    #     type=['csv', 'txt'],
    #     help="Upload experimental waveform data for model training"
    # )
    
    # # Confirm upload button
    # if st.sidebar.button("Confirm Upload"):
    #     if uploaded_file is not None:
    #         try:
    #             # Load data for training
    #             upload_func(uploaded_file, file_type)
    #             st.sidebar.success(f"✅ {file_type} file uploaded successfully!")
    #             st.sidebar.info(f"Data shape: {st.session_state[file_type].shape}")
    #         except Exception as e:
    #             st.sidebar.error(f"❌ Upload failed: {str(e)}")
    #     else:
    #         st.sidebar.warning("⚠️ Please select a file first!")
    
    # # Display upload status
    # if uploaded_file is not None:
    #     st.sidebar.markdown(f"**Current file**: {uploaded_file.name}")

    st.sidebar.markdown("### 📁 Upload Experimental Data (DAB Training)")
    st.sidebar.markdown("*Please upload all three CSV files: vp, vs, iL*")

    vp_file = st.sidebar.file_uploader(
        "Primary Voltage (vp.csv)", type=['csv', 'txt'], key="vp_file")
    vs_file = st.sidebar.file_uploader(
        "Secondary Voltage (vs.csv)", type=['csv', 'txt'], key="vs_file")
    il_file = st.sidebar.file_uploader(
        "Inductor Current (iL.csv)", type=['csv', 'txt'], key="il_file")

    if vp_file is not None:
        st.session_state.vp = np.loadtxt(vp_file, skiprows=1, delimiter=',')
        st.sidebar.success("✅ Primary Voltage data uploaded!")

    if vs_file is not None:
        st.session_state.vs = np.loadtxt(vs_file, skiprows=1, delimiter=',')
        st.sidebar.success("✅ Secondary Voltage data uploaded!")

    if il_file is not None:
        st.session_state.iL = np.loadtxt(il_file, skiprows=1, delimiter=',')
        st.sidebar.success("✅ Inductor Current data uploaded!")

    # Check for training readiness
    if all([vp_file, vs_file, il_file]):
        st.sidebar.success("🎉 All files uploaded! Ready to train PANN.")
        if st.sidebar.button("Train PANN Model"):
            # Call your training function here, e.g.:
            # train_pann(st.session_state.vp, st.session_state.vs, st.session_state.iL)
            st.sidebar.info("Training has started. Check main window for results.")
    else:
        st.sidebar.info("Upload all three files to enable training.")

    # Initialize conversation with guiding prompts
    initialize_conversation(clear_button)


def upload_func(uploaded_file, file_type):
    """
    Handle file upload and store data in session state
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        file_type: Type of data (vp, vs, or iL)
    """
    try:
        # Load data from uploaded file
        data = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')
        
        # Store in session state based on file type
        if file_type == "vp":
            st.session_state.vp = data
        elif file_type == "vs":
            st.session_state.vs = data
        elif file_type == "iL":
            st.session_state.iL = data
        
        # Mark that new data is available for training
        st.session_state.new_data_uploaded = True
        
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")


def init_states(initial_values):
    """
    Initialize Streamlit session state variables
    
    Args:
        initial_values: Dictionary of key-value pairs to initialize
    """
    for key, value in initial_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_history():
    """
    Display the historical chat messages
    Shows messages after the initial system prompts (index 2+)
    """
    # Skip the first 2 messages (system prompts) and display the rest
    for msg in st.session_state.messages[2:]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Display images if present in the message
            if "images" in msg:
                for image in msg["images"]:
                    st.image(image)


def initialize_conversation(clear_button):
    """
    Initialize or reset the conversation with guiding prompts
    
    Args:
        clear_button: Boolean indicating if clear button was pressed
    """
    # Paths to prompt files
    prompt_path = 'core/knowledge/prompts/prompt.txt'
    reply_path = 'core/knowledge/prompts/init_reply.txt'
    
    # Check if files exist
    if not os.path.exists(prompt_path) or not os.path.exists(reply_path):
        # Fallback to default prompts if files don't exist - UPDATED
        content1 = """You are PE-GPT, an AI assistant specialized in power electronics design. 
        You help engineers design and optimize power electronic converters including 
        Dual-Active-Bridge (DAB), Buck, and Boost converters. You can recommend modulation 
        strategies, evaluate performance, and provide simulation verification."""
        
        reply = """Hello! I'm PE-GPT, your power electronics design assistant powered by Groq's 
        open-source LLMs. I can help you with:
        
        1. **Converter Design** - DAB, Buck, and Boost converters
        2. **Modulation Strategy** - Recommend optimal strategies (SPS, DPS, EPS, TPS, 5DOF for DAB)
        3. **Performance Evaluation** - Analyze efficiency, current stress, and soft-switching
        4. **Simulation Verification** - Validate DAB designs through PLECS simulation
        5. **Model Training** - Fine-tune PANN models with your experimental data
        
        **Supported topologies:**
        - **DAB** (Dual Active Bridge) - Isolated, bidirectional
        - **Buck** - Non-isolated, step-down (Vout < Vin)
        - **Boost** - Non-isolated, step-up (Vout > Vin)
        
        How can I assist with your power electronics design today?"""
    else:
        # Read from files
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                content1 = file.read()
            with open(reply_path, 'r', encoding='utf-8') as file:
                reply = file.read()
        except Exception as e:
            st.error(f"Error reading prompt files: {str(e)}")
            return
    
    # Initialize or clear message history
    if clear_button or ("messages" not in st.session_state):
        st.session_state.messages = [
            {"role": "user", "content": content1},
            {"role": "assistant", "content": reply},
        ]
        
        # Clear uploaded data if conversation is cleared
        if clear_button:
            for key in ['vp', 'vs', 'iL', 'topology']:
                if key in st.session_state:
                    del st.session_state[key]
            if 'new_data_uploaded' in st.session_state:
                del st.session_state['new_data_uploaded']
            # Reset topology to unknown
            st.session_state['topology'] = 'unknown'


def display_data_status():
    """
    Display status of uploaded training data in the sidebar
    """
    st.sidebar.markdown("### 📊 Data Status")
    
    data_types = ['vp', 'vs', 'iL']
    data_names = {
        'vp': 'Primary Voltage',
        'vs': 'Secondary Voltage', 
        'iL': 'Inductor Current'
    }
    
    uploaded_count = 0
    for data_type in data_types:
        if data_type in st.session_state:
            st.sidebar.success(f"✅ {data_names[data_type]}: Loaded")
            uploaded_count += 1
        else:
            st.sidebar.info(f"⏳ {data_names[data_type]}: Not uploaded")
    
    if uploaded_count == 3:
        st.sidebar.success("🎉 All training data uploaded! Ready to train DAB PANN.")
    elif uploaded_count > 0:
        st.sidebar.warning(f"⚠️ {uploaded_count}/3 datasets uploaded")


# def get_groq_api_key():
#     """
#     Get Groq API key from session state or prompt user to enter it
    
#     Returns:
#         str: Groq API key or None if not set
#     """
#     if 'GROQ_API_KEY' not in st.session_state:
#         with st.sidebar:
#             st.markdown("---")
#             st.markdown("### 🔑 API Configuration")
#             api_key = st.text_input(
#                 "Enter Groq API Key", 
#                 type="password",
#                 help="Get your free API key at https://console.groq.com"
#             )
#             if api_key:
#                 st.session_state.GROQ_API_KEY = api_key
#                 os.environ['GROQ_API_KEY'] = api_key
#                 st.success("✅ API Key configured!")
#                 return api_key
#             else:
#                 st.warning("⚠️ Please enter your Groq API key to continue")
#                 st.markdown("[Get a free API key](https://console.groq.com)")
#                 return None
#     return st.session_state.GROQ_API_KEY


# Additional utility functions for enhanced GUI

def display_system_info():
    """Display system information and settings"""
    with st.sidebar.expander("ℹ️ System Information"):
        st.markdown(f"""
        **Version**: PE-GPT v2.0 (Groq Edition)
        
        **LLM Backend**: Groq
        
        **Current Model**: {st.session_state.get('llm_model', 'Not set')}
        
        **Supported Topologies**:
        - DAB (Dual Active Bridge)
        - Buck (Step-down)
        - Boost (Step-up)
        
        **Features**:
        - Task Classification Agent
        - Topology Detection Agent
        - Multi-stage Design Flow
        - PANN Model Zoo (DAB)
        - PLECS Integration (DAB)
        """)


def display_shortcuts():
    """Display keyboard shortcuts and tips - UPDATED"""
    with st.sidebar.expander("⌨️ Tips & Shortcuts"):
        st.markdown("""
        **Usage Tips**:
        - Specify converter type (DAB, Buck, or Boost)
        - Provide voltage and power specifications
        - Upload experimental data for DAB training
        - Use simulation verification for DAB
        
        **Example Queries**:
        - "Help me design a DAB converter"
        - "Design a buck converter: 400V input, 48V output, 1000W"
        - "I need a boost converter from 24V to 48V, 500W"
        - "Recommend modulation for high efficiency DAB"
        - "Verify the DAB design in simulation"
        """)


def display_current_topology():
    """Display currently detected topology in sidebar - UPDATED"""
    topo = st.session_state.get('topology', 'unknown')
    
    if topo != 'unknown':
        # Expanded to include all topologies
        topo_info = {
            'dab': ('🔄', 'Dual Active Bridge'),
            'buck': ('⬇️', 'Buck Converter'),
            'boost': ('⬆️', 'Boost Converter'),
            'buckboost': ('↕️', 'Buck-Boost'),
            'flyback': ('🔌', 'Flyback')
        }
        
        if topo in topo_info:
            icon, name = topo_info[topo]
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔍 Detected Topology")
            st.sidebar.info(f"{icon} **{name}**")
# """
# @functionality
#     Create a graphical user interface (GUI) using Streamlit for PE-GPT
#     Handles file uploads, conversation management, and display

# @reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
# @code-author: Xinze Li, Fanfan Lin, Weihao Lei
# @github: https://github.com/XinzeLee/PE-GPT

# GROQ EDITION - Converted from OpenAI to Groq + Open Source

# @reference:
#     Following references are related to power electronics GPT (PE-GPT)
#     1: PE-GPT: a New Paradigm for Power Electronics Design
#         Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, 
#                  Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
#         Paper DOI: 10.1109/TIE.2024.3454408
# """

# import streamlit as st
# import numpy as np
# import os


# def build_gui():
#     """
#     Create a graphical user interface (GUI) using Streamlit
#     Open-source version using Groq LLM backend
#     """
    
#     # Graphical User Interface Configuration
#     st.set_page_config(
#         page_title="PE-GPT (Groq Edition)", 
#         page_icon="💎", 
#         layout="centered",
#         initial_sidebar_state="auto", 
#         menu_items=None
#     )
    
#     st.title("Chat with the Power Electronics Robot 🤖")
#     st.info(
#         "Hello, I am a robot specifically for power electronics design! "
#         "Powered by Groq's open-source LLMs.", 
#         icon="🤟"
#     )
    
#     # Sidebar Configuration
#     with st.sidebar:
#         st.markdown(
#             "<h1 style='color: #FF5733;'>PE-GPT (v2.0 - Groq Edition)</h1>", 
#             unsafe_allow_html=True
#         )
#         st.markdown('---')
        
#         # Feature Description
#         st.markdown(
#             "### 🚀 Features\n"
#             "- **Modulation Strategies**: SPS, EPS, DPS, TPS, 5DOF for DAB converters\n"
#             "- **Circuit Design**: Buck converter design support\n"
#             "- **Open Source**: Powered by Groq & LangChain\n"
#             "- **Model Zoo**: PANN models for power electronics\n"
#         )
        
#         st.markdown('---')
        
#         # LLM Model Selection
#         st.markdown("### ⚙️ Model Settings")
#         llm_model = st.selectbox(
#             "Select LLM Model",
#             [
#                 "llama-3.1-8b-instant",
#                 "llama-3.1-70b-versatile",
#                 "mixtral-8x7b-32768",
#                 "gemma2-9b-it"
#             ],
#             index=0,
#             help="Choose the Groq model for inference"
#         )
        
#         # Store model selection in session state
#         if 'llm_model' not in st.session_state or st.session_state.llm_model != llm_model:
#             st.session_state.llm_model = llm_model
#             st.info(f"Model changed to: {llm_model}")
        
#         st.markdown('---')
        
#         # Project Information
#         st.markdown(
#             "### 📚 Reference\n"
#             "**Paper**: PE-GPT: a New Paradigm for Power Electronics Design\n\n"
#             "**Authors**: Fanfan Lin, Xinze Li, et al.\n\n"
#             "**GitHub**: [PE-GPT Repository](https://github.com/XinzeLee/PE-GPT)\n\n"
#             "**DOI**: 10.1109/TIE.2024.3454408"
#         )
        
#         st.markdown('---')
    
#     # Clear conversation button
#     clear_button = st.sidebar.button('Clear Conversation', key='clear')
    
#     # File Upload Section
#     st.sidebar.markdown("### 📁 Training Data Upload")
#     st.sidebar.markdown("*Upload experimental data for PANN model training*")
    
#     # File type selection
#     file_type = st.sidebar.selectbox(
#         "Select file type", 
#         ("vp", "vs", "iL"),
#         help="vp: Primary voltage, vs: Secondary voltage, iL: Inductor current"
#     )
    
#     # File uploader
#     uploaded_file = st.sidebar.file_uploader(
#         "Upload CSV file", 
#         key="file_uploader",
#         type=['csv', 'txt'],
#         help="Upload experimental waveform data for model training"
#     )
    
#     # Confirm upload button
#     if st.sidebar.button("Confirm Upload"):
#         if uploaded_file is not None:
#             try:
#                 # Load data for training
#                 upload_func(uploaded_file, file_type)
#                 st.sidebar.success(f"✅ {file_type} file uploaded successfully!")
#                 st.sidebar.info(f"Data shape: {st.session_state[file_type].shape}")
#             except Exception as e:
#                 st.sidebar.error(f"❌ Upload failed: {str(e)}")
#         else:
#             st.sidebar.warning("⚠️ Please select a file first!")
    
#     # Display upload status
#     if uploaded_file is not None:
#         st.sidebar.markdown(f"**Current file**: {uploaded_file.name}")
    
#     # Initialize conversation with guiding prompts
#     initialize_conversation(clear_button)


# def upload_func(uploaded_file, file_type):
#     """
#     Handle file upload and store data in session state
    
#     Args:
#         uploaded_file: Streamlit UploadedFile object
#         file_type: Type of data (vp, vs, or iL)
#     """
#     try:
#         # Load data from uploaded file
#         data = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')
        
#         # Store in session state based on file type
#         if file_type == "vp":
#             st.session_state.vp = data
#         elif file_type == "vs":
#             st.session_state.vs = data
#         elif file_type == "iL":
#             st.session_state.iL = data
        
#         # Mark that new data is available for training
#         st.session_state.new_data_uploaded = True
        
#     except Exception as e:
#         raise Exception(f"Error loading file: {str(e)}")


# def init_states(initial_values):
#     """
#     Initialize Streamlit session state variables
    
#     Args:
#         initial_values: Dictionary of key-value pairs to initialize
#     """
#     for key, value in initial_values.items():
#         if key not in st.session_state:
#             st.session_state[key] = value


# def display_history():
#     """
#     Display the historical chat messages
#     Shows messages after the initial system prompts (index 2+)
#     """
#     # Skip the first 2 messages (system prompts) and display the rest
#     for msg in st.session_state.messages[2:]:
#         with st.chat_message(msg["role"]):
#             st.write(msg["content"])
            
#             # Display images if present in the message
#             if "images" in msg:
#                 for image in msg["images"]:
#                     st.image(image)


# def initialize_conversation(clear_button):
#     """
#     Initialize or reset the conversation with guiding prompts
    
#     Args:
#         clear_button: Boolean indicating if clear button was pressed
#     """
#     # Paths to prompt files
#     prompt_path = 'core/knowledge/prompts/prompt.txt'
#     reply_path = 'core/knowledge/prompts/init_reply.txt'
    
#     # Check if files exist
#     if not os.path.exists(prompt_path) or not os.path.exists(reply_path):
#         # Fallback to default prompts if files don't exist
#         content1 = """You are PE-GPT, an AI assistant specialized in power electronics design. 
#         You help engineers design and optimize power electronic converters, particularly 
#         Dual-Active-Bridge (DAB) converters and Buck converters. You can recommend modulation 
#         strategies, evaluate performance, and provide simulation verification."""
        
#         reply = """Hello! I'm PE-GPT, your power electronics design assistant powered by Groq's 
#         open-source LLMs. I can help you with:
        
#         1. **Modulation Strategy Selection** - Recommend optimal strategies (SPS, DPS, EPS, TPS, 5DOF)
#         2. **Performance Evaluation** - Analyze efficiency, current stress, and soft-switching
#         3. **Simulation Verification** - Validate designs through PLECS simulation
#         4. **Model Training** - Fine-tune PANN models with your experimental data
        
#         How can I assist with your power electronics design today?"""
#     else:
#         # Read from files
#         try:
#             with open(prompt_path, 'r', encoding='utf-8') as file:
#                 content1 = file.read()
#             with open(reply_path, 'r', encoding='utf-8') as file:
#                 reply = file.read()
#         except Exception as e:
#             st.error(f"Error reading prompt files: {str(e)}")
#             return
    
#     # Initialize or clear message history
#     if clear_button or ("messages" not in st.session_state):
#         st.session_state.messages = [
#             {"role": "user", "content": content1},
#             {"role": "assistant", "content": reply},
#         ]
        
#         # Clear uploaded data if conversation is cleared
#         if clear_button:
#             for key in ['vp', 'vs', 'iL']:
#                 if key in st.session_state:
#                     del st.session_state[key]
#             if 'new_data_uploaded' in st.session_state:
#                 del st.session_state['new_data_uploaded']


# def display_data_status():
#     """
#     Display status of uploaded training data in the sidebar
#     """
#     st.sidebar.markdown("### 📊 Data Status")
    
#     data_types = ['vp', 'vs', 'iL']
#     data_names = {
#         'vp': 'Primary Voltage',
#         'vs': 'Secondary Voltage', 
#         'iL': 'Inductor Current'
#     }
    
#     uploaded_count = 0
#     for data_type in data_types:
#         if data_type in st.session_state:
#             st.sidebar.success(f"✅ {data_names[data_type]}: Loaded")
#             uploaded_count += 1
#         else:
#             st.sidebar.info(f"⏳ {data_names[data_type]}: Not uploaded")
    
#     if uploaded_count == 3:
#         st.sidebar.success("🎉 All training data uploaded! Ready to train.")
#     elif uploaded_count > 0:
#         st.sidebar.warning(f"⚠️ {uploaded_count}/3 datasets uploaded")


# def get_groq_api_key():
#     """
#     Get Groq API key from session state or prompt user to enter it
    
#     Returns:
#         str: Groq API key or None if not set
#     """
#     if 'GROQ_API_KEY' not in st.session_state:
#         with st.sidebar:
#             st.markdown("---")
#             st.markdown("### 🔑 API Configuration")
#             api_key = st.text_input(
#                 "Enter Groq API Key", 
#                 type="password",
#                 help="Get your free API key at https://console.groq.com"
#             )
#             if api_key:
#                 st.session_state.GROQ_API_KEY = api_key
#                 os.environ['GROQ_API_KEY'] = api_key
#                 st.success("✅ API Key configured!")
#                 return api_key
#             else:
#                 st.warning("⚠️ Please enter your Groq API key to continue")
#                 st.markdown("[Get a free API key](https://console.groq.com)")
#                 return None
#     return st.session_state.GROQ_API_KEY


# # Additional utility functions for enhanced GUI

# def display_system_info():
#     """Display system information and settings"""
#     with st.sidebar.expander("ℹ️ System Information"):
#         st.markdown(f"""
#         **Version**: PE-GPT v2.0 (Groq Edition)
        
#         **LLM Backend**: Groq
        
#         **Current Model**: {st.session_state.get('llm_model', 'Not set')}
        
#         **Features**:
#         - Task Classification Agent
#         - Multi-stage Design Flow
#         - PANN Model Zoo
#         - PLECS Integration
#         """)


# def display_shortcuts():
#     """Display keyboard shortcuts and tips"""
#     with st.sidebar.expander("⌨️ Tips & Shortcuts"):
#         st.markdown("""
#         **Usage Tips**:
#         - Start with your design requirements
#         - Specify voltage, power, and objectives
#         - Upload experimental data for training
#         - Use simulation verification
        
#         **Example Queries**:
#         - "Help me design a DAB converter"
#         - "I need 200V input, 100V output, 500W"
#         - "Recommend modulation for high efficiency"
#         - "Verify the design in simulation"
#         """)

# def display_current_topology():
#     """Display currently detected topology"""
#     topo = st.session_state.get('topology', 'unknown')
    
#     if topo != 'unknown':
#         topo_info = {
#             'dab': ('🔄', 'Dual Active Bridge'),
#             'buck': ('⬇️', 'Buck Converter'),
#             'boost': ('⬆️', 'Boost Converter'),
#         }
        
#         if topo in topo_info:
#             icon, name = topo_info[topo]
#             st.sidebar.info(f"{icon} **Current Topology**: {name}")