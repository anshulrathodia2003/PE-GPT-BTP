"""
@functionality
    Mainly used for your custom design workflow.
    The design stages are defined in each individual function, 
    and another LLM agent is used to classify stage.
    Main hub to build GUI, LLM, interact with model zoo, simulation validation.

@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT

GROQ EDITION - Converted from OpenAI to Groq + Open Source
Multi-Topology Support: DAB, Buck, Boost
"""
try:
    from ..simulation import load_plecs
    HAS_PLECS = True
except ImportError:
    HAS_PLECS = False
    import warnings
    warnings.warn("PLECS simulation module not available")

import re
import json
import streamlit as st

# At the top after imports:
model_list = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-maverick-17b-128e-instruct"
]

if "model_index" not in st.session_state:
    st.session_state.model_index = 0

def get_next_model():
    idx = st.session_state.model_index
    chosen_model = model_list[idx]
    st.session_state.model_index = (idx + 1) % len(model_list)
    return chosen_model


from ..llm import custom_responses
from .llm_manager import (
    LLMManager, 
    intelligent_parameter_extraction,
    resolve_parameter_conflict,
    generate_natural_buck_response,
    generate_natural_boost_response
)

from ..llm.custom_responses import response_buck, response_boost

from ..optim import optimizers
from ..llm.llm import get_msg_history
from ..model_zoo.pann_dab import train_dab

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..llm.topology_agent import TopologyAgent

from .conversation_intelligence import ConversationIntelligenceAgent

def calculate_buck_components(Vin, Vout, P, performances, fs=100):
    """
    Calculate recommended component values for Buck converter.
    
    Args:
        Vin: Input voltage (V)
        Vout: Output voltage (V)
        P: Output power (W)
        performances: List of performance tuples from optimizer
        fs: Switching frequency (kHz), default=100
    
    Returns:
        dict: Component recommendations
    """
    import math
    
    # Helper function to safely convert to float
    def safe_float(value, default=0.0):
        try:
            if isinstance(value, str):
                # Remove any units or extra characters
                value = value.split()[0]  # Take first word if multiple
            return float(value)
        except (ValueError, TypeError, AttributeError):
            return default
    
    # Extract performance values and convert to float
    perf_dict = {name: value for name, value in performances}
    
    duty_cycle = safe_float(perf_dict.get('Duty Cycle (D)', Vout / Vin), Vout / Vin)
    Iout = safe_float(perf_dict.get('Output Current (A)', P / Vout), P / Vout)
    delta_iL = safe_float(perf_dict.get('Inductor Ripple ΔiL (A)', 0), 0)
    
    # Use the provided switching frequency
    fs_hz = fs * 1000
    
    # --- INDUCTOR CALCULATION ---
    # L = (Vin - Vout) * duty_cycle / (delta_iL * fs)
    if delta_iL > 0:
        L_calculated = ((Vin - Vout) * duty_cycle) / (delta_iL * fs_hz) * 1e6  # in µH
    else:
        # If not provided, assume 30% ripple
        ripple_ratio = 0.3
        delta_iL_assumed = Iout * ripple_ratio
        L_calculated = ((Vin - Vout) * duty_cycle) / (delta_iL_assumed * fs_hz) * 1e6
    
    # Standard inductor values: round to nearest standard value
    standard_L = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470]
    L_value = min(standard_L, key=lambda x: abs(x - L_calculated))
    
    Ipeak = Iout + delta_iL / 2
    
    inductor = {
        'value': L_value,
        'current_rating': Iout * 1.3,  # 30% margin
        'saturation_current': Ipeak * 1.2,  # 20% margin
        'dcr': 5  # mΩ - typical recommendation
    }
    
    # --- CAPACITOR CALCULATION ---
    # C = delta_iL / (8 * fs * delta_V)
    delta_V_target = Vout * 0.01  # 1% output ripple target
    C_calculated = delta_iL / (8 * fs_hz * delta_V_target) * 1e6  # in µF
    
    # Add margin and round up
    C_value = math.ceil(C_calculated * 1.5 / 10) * 10  # Round to nearest 10µF
    
    capacitor = {
        'value': C_value,
        'voltage_rating': Vout,
        'recommended_voltage': Vout * 1.5,  # 50% voltage derating
        'esr': max(5, 100 / C_value),  # ESR scales with capacitance
        'type': f"{C_value//2}µF ceramic (X7R) + {C_value//2}µF polymer"
    }
    
    # --- MOSFET CALCULATION ---
    # Voltage rating should exceed input voltage with margin
    V_rating_standard = [30, 60, 100, 150, 200, 250, 300, 600, 650]
    V_mosfet = min([v for v in V_rating_standard if v >= Vin * 1.3], default=650)
    
    # RDS(on) estimation for target efficiency
    # Conduction loss = I^2 * RDS(on)
    # For 98% efficiency, allow ~1% conduction loss
    Ploss_allowed = P * 0.01
    RDS_high = (Ploss_allowed / (Iout**2 * duty_cycle)) * 1000  # in mΩ
    RDS_low = (Ploss_allowed / (Iout**2 * (1 - duty_cycle))) * 1000
    
    mosfet = {
        'voltage_rating': V_mosfet,
        'current_rating': Iout * 1.5,  # 50% margin
        'rdson_high': max(10, RDS_high),
        'rdson_low': max(5, RDS_low)
    }
    
    components = {
        'inductor': inductor,
        'capacitor': capacitor,
        'mosfet': mosfet,
        'switching_freq': fs
    }
    
    return components



# =====================================================================
# TASK CLASSIFICATION AGENT (Replaces OpenAI Agent)
# =====================================================================

class TaskClassificationAgent:
    """
    Groq-based agent to classify user intent and route to appropriate task
    Replaces LlamaIndex OpenAIAgent
    Supports multiple converter topologies (DAB, Buck, Boost)
    """
    
    def __init__(self):
        # self.llm = ChatGroq(model=model, temperature=0.5)
        
        # Define all available tasks with descriptions
        self.tasks = {
            "Task 0": {
                "name": "init_design",
                "description": "Initialize the design process and provide guidance to users. Use when user requests design help or introduction to converter design."
            },
            "Task 1": {
                "name": "recommend_modulation",
                "description": "Understand user's requirements and recommend suitable modulation strategies (SPS, DPS, EPS, TPS, 5DOF) for DAB converters. Use when user mentions modulation performances or objectives like efficiency, power loss, current stress, soft switching, easy implementation. Only applicable to DAB converters."
            },
            "Task 2": {
                "name": "evaluate_converter",
                "description": "Evaluate waveforms and converter performances (DAB, Buck, or Boost) given operating conditions. Use when user specifies input voltage (Uin), output voltage (Uo), and power level (PL) for any converter topology."
            },
            "Task 3": {
                "name": "simulation_verification",
                "description": "Open simulation models and conduct simulation to validate the designed converter. Use when user wants to verify design through simulation (currently supports DAB only)."
            },
            "Task 4": {
                "name": "pe_gpt_introduction",
                "description": "Provide introduction and information about PE-GPT. Use when user asks about PE-GPT itself, its features, or capabilities."
            },
            "Task 5": {
                "name": "train_pann",
                "description": "Train or fine-tune the PANN models for DAB converters. Use ONLY after datasets are provided and user explicitly requests training. Do NOT use for guidance requests. Only applicable to DAB converters."
            },
            "Other": {
                "name": "other_tasks",
                "description": "Handle all other general queries not covered by specific tasks, including theoretical questions, explanations, and general power electronics knowledge."
            }
        }
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._build_system_prompt()),
            ("human", "{query}")
        ])
        
        # self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _build_system_prompt(self):
        task_descriptions = "\n".join([
            f"- **{task_id}** ({info['name']}): {info['description']}"
            for task_id, info in self.tasks.items()
        ])
        
        return f"""You are a task classification agent for PE-GPT (Power Electronics GPT).
    Your job is to analyze the user's request and classify it into ONE of the following tasks:

{task_descriptions}

CRITICAL INSTRUCTIONS:
1. Return ONLY the task ID (e.g., "Task 0", "Task 1", "Task 2", etc.)
2. Choose the MOST RELEVANT task based on the user's request
3. If no specific task matches, return "Other"
4. Do NOT provide explanations, just the task ID

IMPORTANT RULES:

Task 1 (Modulation Recommendation):
- ONLY for DAB converters with phase-shift modulation
- Use when user asks about SPS, DPS, EPS, TPS, 5DOF strategies
- NOT applicable to Buck/Boost (they use simple PWM)

Task 2 (Converter Evaluation):
- Use when user provides SPECIFIC numerical values for ANY converter type:
  * Input voltage (Uin/Vin) in Volts
  * Output voltage (Uo/Vout/Vo) in Volts  
  * Power level (PL/P) in Watts
- ALSO use for "what if" scenarios and design modifications:
  * "What if I use 200kHz?" → Task 2
  * "Change frequency to 300kHz" → Task 2
  * "What happens if I increase output to 24V?" → Task 2
  * "Try with 500W instead" → Task 2
  * "Recalculate with different frequency" → Task 2
  * "Compare 100kHz vs 200kHz" → Task 2
- Applicable to: DAB, Buck, Boost converters
- Topology will be auto-detected separately

Task 5 (Model Training):
- ONLY for DAB PANN models
- Buck/Boost do not require PANN training

"Other" Task - General Questions & Explanations:
- Component explanation questions: "Why 47µH?" → Other
- Theoretical explanations: "How does X work?" → Other
- Mathematical derivations: "Explain the formula" → Other
- General power electronics knowledge → Other
- Questions about existing design without modification → Other

CRITICAL DISTINCTION - What If vs Why:

"What if I use 200kHz?" → Task 2 (requires re-evaluation)
"Why did you choose 100kHz?" → Other (just explanation)
"Can I use 68µH instead?" → Other (asking for advice, not evaluation)
"Change to 68µH and recalculate" → Task 2 (explicit recalculation)

CLASSIFICATION EXAMPLES:

Task 0 - Design Initialization:
- "Help me design a DAB converter"
- "I want to design a buck converter"
- "Guide me through boost converter design"
- "How do I start designing a power converter?"

Task 1 - Modulation Recommendation (DAB only):
- "I need high efficiency and low current stress for DAB"
- "Which modulation strategy is best for my DAB converter?"
- "Recommend a phase shift strategy"
- "Compare TPS vs DPS modulation"

Task 2 - Converter Evaluation (All topologies):
Initial evaluations:
- "My input is 200V, output 100V, power 500W" (DAB)
- "Evaluate buck converter: 400V input, 48V output, 1000W"
- "Design boost with Vin=24V, Vout=48V, P=500W"
- "[400, 48, 1000]"

What-if scenarios (CRITICAL - these are Task 2):
- "What if I use 200kHz?"
- "What happens if I change output to 24V?"
- "Change frequency to 300kHz"
- "Increase power to 1500W"
- "Try with 500kHz switching frequency"
- "Recalculate with different parameters"
- "Compare 100kHz vs 200kHz"
- "Use 24V output instead"

Task 3 - Simulation:
- "Can you verify this in simulation?"
- "Run PLECS simulation"
- "Validate my design through simulation"

Task 4 - PE-GPT Info:
- "What is PE-GPT?"
- "What can PE-GPT do?"
- "Tell me about this tool"

Task 5 - Model Training:
- "Train the model with my data"
- "Fine-tune PANN with uploaded CSV"
- "Retrain the neural network"

Other - General Questions & Explanations:
Component questions (no recalculation):
- "Why did you choose 47µH inductor?"
- "Can I use 68µH instead of 47µH?" (asking advice)
- "What's the difference between 47µH and 68µH?"
- "Explain the inductor selection"

Theoretical questions:
- "How does a buck converter work?"
- "What are the equations for power transfer?"
- "Explain duty cycle calculation"
- "Derive the voltage transfer ratio"
- "What's CCM vs DCM?"

General knowledge:
- "What's the difference between buck and boost?"
- "Compare synchronous vs non-synchronous rectification"
- "Advantages of higher switching frequency"

KEYWORD DETECTION RULES:

If prompt contains these phrases, classify as Task 2:
- "what if" + number
- "change" + parameter + "to" + number
- "use" + number + unit (kHz, V, W)
- "increase/decrease" + parameter + "to" + number
- "recalculate"
- "try with" + number
- "compare" + number + "vs" + number
- Bracket format: [numbers]

If prompt contains these WITHOUT numbers for recalculation, classify as Other:
- "why" + component/parameter
- "can I use" + component (without "recalculate")
- "explain"
- "how does"
- "what is"
- "difference between"

EXAMPLES TO MEMORIZE:

Task 2: "What if I use 200kHz?" (has number, implies recalculation)
Other: "Why 100kHz?" (asking explanation only)

Task 2: "Change output to 24V" (has number, explicit change)
Other: "Can I change the output voltage?" (no number, asking if possible)

Task 2: "Compare 100kHz vs 200kHz" (has numbers, evaluate both)
Other: "What's the difference between 100kHz and 200kHz?" (theoretical)

Task 2: "Try with 68µH" (has number, implies testing)
Other: "Can I use 68µH?" (asking advice)

Task 2: "Recalculate with 300kHz" (explicit recalculation request)
Other: "Should I use a different frequency?" (asking advice)
"""

    def classify(self, user_query):
        """Classify user query into appropriate task (round robin per call)"""
        try:
            # Round robin: select the next model for every classification
            llm = ChatGroq(model=get_next_model(), temperature=0.5)
            chain = self.prompt | llm | StrOutputParser()
            response = chain.invoke({"query": user_query})
            
            # Extract task ID from response
            for task_id in self.tasks.keys():
                if task_id in response:
                    return task_id
            return "Other"
        except Exception as e:
            print(f"Classification error: {e}")
            return "Other"

    
    # def classify(self, user_query):
    #     """Classify user query into appropriate task"""
    #     try:
    #         response = self.chain.invoke({"query": user_query})
    #         # Extract task ID from response
    #         for task_id in self.tasks.keys():
    #             if task_id in response:
    #                 return task_id
    #         return "Other"
    #     except Exception as e:
    #         print(f"Classification error: {e}")
    #         return "Other"


# =====================================================================
# TASK FUNCTIONS
# =====================================================================

# Task-0: Initialize the design process
def init_design(chat_engine, prompt, messages_history, session_id="agent1"):
    """Initialize the design process and provide guidance to users"""
    with st.spinner("Thinking..."):
        response = chat_engine.invoke(
            {"input": prompt, "chat_history": messages_history},
            config={"configurable": {"session_id": session_id}}
        )
        st.write(response)
        return response


# Task-1: Recommend modulation strategy (DAB only)
def recommend_modulation(chat_engine, prompt, messages_history, session_id="agent1"):
    """Understand user's requirements and recommend suitable modulation strategies for DAB"""
    
    # Check if topology is DAB
    topo = st.session_state.get("topology", "unknown")
    
    if topo != "dab":
        warning = f"""⚠️ Modulation strategy recommendation is only applicable to DAB converters.
        
Your detected topology is: **{topo.upper()}**

{topo.upper()} converters use simple PWM control, not phase-shift modulation strategies.
Please specify operating conditions (Vin, Vout, Power) for {topo.upper()} evaluation instead."""
        st.warning(warning)
        return [{"role": "assistant", "content": warning}]
    
    with st.spinner("Thinking..."):
        # Get initial recommendation
        response = chat_engine.invoke(
            {"input": prompt, "chat_history": messages_history},
            config={"configurable": {"session_id": session_id}}
        )
        st.write(response)
        
        # Extract recommended modulation method
        modulation_methods = ["SPS", "DPS", "EPS", "TPS", "5DOF"]
        follow_up_prompt = f"""Based on your response quoted in '' below, which strategy in 
        the list {modulation_methods} do you recommend? 
        Attention: Only output the recommended strategy from the list, nothing else!!!
        """.replace("\n", "") + f"'{response}'"
        
        response2 = chat_engine.invoke(
            {"input": follow_up_prompt, "chat_history": []},
            config={"configurable": {"session_id": f"{session_id}_followup"}}
        )
        
        recommended_mod = "TPS"
        # Capture the recommended modulation
        for method in modulation_methods:
            if method.lower() in response2.lower():
                recommended_mod = method
                break
        
        # Store recommended modulation
        st.session_state.M = recommended_mod
        
        messages = [{"role": "assistant", "content": response}]
    return messages


# Task-2a: Evaluate DAB converter performances
def evaluate_dab(chat_engine, prompt, messages_history, session_id="agent0"):
    """Evaluate the waveforms and various converter performances for DAB"""
    
    re_specs = re.compile(r".*\[\D*(\d+)\D*\,\D*(\d+)\D*\,\D*(\d+)\D*\]")
    with st.spinner("Evaluating DAB converter..."):
        prompt = prompt + "\n Please be really careful about the response format for this request!!!! In the form of [Uin, Uo, PL]!!!"
        response = chat_engine.invoke(
            {"input": prompt, "chat_history": messages_history},
            config={"configurable": {"session_id": session_id}}
        )
        
        matched = re_specs.findall(response)
        if len(matched):
            st.session_state.Uin, st.session_state.Uo, \
                st.session_state.P = map(float, matched[0])
        
        # VALIDATION: Check if required values are set
        if (st.session_state.get('Uin') is None or 
            st.session_state.get('Uo') is None or 
            st.session_state.get('P') is None or
            st.session_state.get('M') is None):
            
            error_msg = """⚠️ I couldn't extract the required operating conditions from your request.
            
Please provide specific values for DAB converter:
- **Input Voltage (Uin)**: e.g., 200V
- **Output Voltage (Uo)**: e.g., 100V  
- **Power Level (PL)**: e.g., 500W

Example: "My input is 200V, output 100V, power 500W"
            """
            st.warning(error_msg)
            messages = [{"role": "assistant", "content": error_msg}]
            return messages
        
        Uins, Uos = [st.session_state.Uin]*2, [st.session_state.Uo]*2
        Ps = [st.session_state.P, st.session_state.P]
        Ms = [st.session_state.M, "SPS"]
        messages = []
        
        for Uin, Uo, P, M in zip(Uins, Uos, Ps, Ms):
            *performances, plot, updated_M = optimizers.optimize_mod_dab(Uin, Uo, P, M)
            if M == st.session_state.M:
                st.session_state["pos"] = performances[-1][1:]  # get the optimized modulation parameters
            response = custom_responses.response(performances, updated_M)
            
            st.write(response)
            st.image(plot)
            messages.append({"role": "assistant", "content": response, "images": [plot]})
    return messages


# Task-2b: Evaluate BUCK converter
def evaluate_buck(chat_engine, prompt, messages_history, session_id="agent0"):
    """Evaluate Buck converter with intelligent LLM-driven flow"""
    
    # Get or create LLM manager
    if "llm_manager" not in st.session_state:
        st.session_state.llm_manager = LLMManager()
    
    llm_manager = st.session_state.llm_manager
    
    with st.spinner("Analyzing your Buck converter request..."):
        # STEP 1: Intelligent parameter extraction using LLM
        param_data = intelligent_parameter_extraction(prompt, "buck", llm_manager)
        
        # STEP 2: Check for conflicts
        if param_data.get('conflict'):
            # Ask LLM to generate friendly clarification request
            clarification = resolve_parameter_conflict(param_data, prompt, "buck", llm_manager)
            st.warning(clarification)
            return [{"role": "assistant", "content": clarification}]
        # STEP 2.5: Validate that we have all required parameters (backup check)
        if param_data.get('vin') is None or param_data.get('vout') is None or param_data.get('power') is None:
            missing_msg = """Hmm, I need a bit more info! For a buck converter design, please provide:
        - Input voltage (Vin)
        - Output voltage (Vout)  
        - Power (P in watts)

        Example: "buck converter 48V to 30V, 500W"

        What are your specs?"""
            st.warning(missing_msg)
            return [{"role": "assistant", "content": missing_msg}]        
        # STEP 3: Store validated parameters
        st.session_state.Uin = param_data['vin']
        st.session_state.Uo = param_data['vout']
        st.session_state.P = param_data['power']
        st.session_state.fs = param_data.get('frequency') or 100
        
        Uin, Uo, P, fs = param_data['vin'], param_data['vout'], param_data['power'], param_data.get('frequency') or 100
        
        # STEP 4: Calculate performances
        performances, plot = optimizers.optimize_buck(Uin, Uo, P)
        components = calculate_buck_components(Uin, Uo, P, performances, fs)

        # STEP 5: Store components in session state for later retrieval
        st.session_state.components = components
        st.session_state.performances = performances

        # STEP 6: Generate natural response using smart LLM
        design_context = {
            'vin': Uin,
            'vout': Uo,
            'power': P,
            'frequency': fs
        }

        response_text = generate_natural_buck_response(
            performances, components, design_context, llm_manager
        )

        # STEP 7: Create brief component summary
        comp_summary = f"\n\n**Quick Component Summary:**\n"
        comp_summary += f"• Inductor: {components['inductor']['value']}µH\n"
        comp_summary += f"• Capacitor: {components['capacitor']['value']}µF\n"
        comp_summary += f"• MOSFETs: {components['mosfet']['voltage_rating']}V rated\n"
        comp_summary += f"• Switching Frequency: {components['switching_freq']}kHz\n\n"
        comp_summary += "_💡 Ask 'show component details' for full specifications_"

        # Display response with summary
        st.write(response_text)
        st.write(comp_summary)
        st.image(plot)

        return [{"role": "assistant", "content": response_text + "\n\n" + comp_summary, "images": [plot]}]



# Task-2c: Evaluate BOOST converter
def evaluate_boost(chat_engine, prompt, messages_history, session_id="agent0"):
    """Evaluate Boost converter with intelligent LLM-driven flow"""
    
    # Get or create LLM manager
    if "llm_manager" not in st.session_state:
        st.session_state.llm_manager = LLMManager()
    
    llm_manager = st.session_state.llm_manager
    
    with st.spinner("Analyzing your Boost converter request..."):
        # STEP 1: Intelligent parameter extraction using LLM
        param_data = intelligent_parameter_extraction(prompt, "boost", llm_manager)
        
        # STEP 2: Check for conflicts
        if param_data.get('conflict'):
            # Ask LLM to generate friendly clarification request
            clarification = resolve_parameter_conflict(param_data, prompt, "boost", llm_manager)
            st.warning(clarification)
            return [{"role": "assistant", "content": clarification}]
        # STEP 2.5: Validate required parameters
        if param_data.get('vin') is None or param_data.get('vout') is None or param_data.get('power') is None:
            missing_msg = """Hmm, I need a bit more info! For a boost converter, please provide:
        - Input voltage (Vin)
        - Output voltage (Vout)  
        - Power (P in watts)

        Example: "boost converter 24V to 48V, 200W"

        What are your specs?"""
            st.warning(missing_msg)
            return [{"role": "assistant", "content": missing_msg}]        
        # STEP 3: Store validated parameters
        st.session_state.Uin = param_data['vin']
        st.session_state.Uo = param_data['vout']
        st.session_state.P = param_data['power']
        st.session_state.fs = param_data.get('frequency') or 100
        
        Uin, Uo, P, fs = param_data['vin'], param_data['vout'], param_data['power'], param_data.get('frequency') or 100
        
        # STEP 4: Calculate performances
        performances, plot = optimizers.optimize_boost(Uin, Uo, P)

        # Store performances in session state
        st.session_state.performances = performances

        # STEP 5: Generate natural response using smart LLM
        design_context = {
            'vin': Uin,
            'vout': Uo,
            'power': P,
            'frequency': fs
        }

        response_text = generate_natural_boost_response(
            performances, design_context, llm_manager
        )

        # Brief performance summary
        perf_dict = {name: value for name, value in performances}
        perf_summary = f"\n\n**Key Metrics:**\n"
        perf_summary += f"• Duty Cycle: {perf_dict.get('Duty Cycle (D)', 'N/A')}\n"
        perf_summary += f"• Input Current: {perf_dict.get('Input Current (A)', 'N/A')}A\n"
        perf_summary += f"• Efficiency: {perf_dict.get('Efficiency (%)', 'N/A')}%\n"

        st.write(response_text)
        st.write(perf_summary)
        st.image(plot)

        return [{"role": "assistant", "content": response_text + "\n\n" + perf_summary, "images": [plot]}]




# Task-4: Introduction to PE-GPT
def pe_gpt_introduction(chat_engine, prompt, session_id="agent2"):
    """Provide introduction and information about PE-GPT"""
    with st.spinner("Thinking..."):
        response = chat_engine.invoke(
            {"input": prompt, "chat_history": []},
            config={"configurable": {"session_id": session_id}}
        )
        st.write(response)
        messages = [{"role": "assistant", "content": response}]
    return messages

# Task-3: Verify the designed modulation in simulation
def simulation_verification():
    """Open simulation models and conduct simulation to validate the designed converter"""
    if not HAS_PLECS:
        reply = "⚠️ PLECS simulation is not available. Please install required dependencies."
        st.warning(reply)
        messages = [{"role": "assistant", "content": reply}]
        return messages
    
    # Check topology
    topo = st.session_state.get("topology", "unknown")
    
    if topo != "dab":
        reply = f"""⚠️ PLECS simulation is currently only supported for DAB converters.

Your detected topology: **{topo.upper()}**

Simulation support for Buck/Boost converters is under development."""
        st.warning(reply)
        return [{"role": "assistant", "content": reply}]
    
    with st.spinner("Waiting... PLECS is starting up..."):
        load_plecs.dab_plecs(
            st.session_state.M,
            st.session_state.Uin,
            st.session_state.Uo,
            st.session_state.P,
            *st.session_state.pos
        )
    
    reply = "The PLECS simulation is running... Complete! You can now verify if the design is reasonable by observing the simulation waveforms."
    st.write(reply)
    messages = [{"role": "assistant", "content": reply}]
    return messages


# Task-5: Fine-tune/train the PANN model
def train_pann():
    """Train or fine-tune the PANN models in model zoo (DAB only)"""
    try:
        topo = st.session_state.get("topology", "dab")

        with st.spinner(f"Training PANN model... Please wait..."):

            # Only DAB converters use PANN models
            if topo == "dab":
                plot, test_loss, val_loss = train_dab()
                
                reply = f"""✅ DAB PANN retraining complete!

**Performance Metrics:**
- Test MAE: {test_loss:.3f}
- Validation MAE: {val_loss:.3f}

Predicted & experimental waveforms shown below."""
                st.write(reply)
                st.image(plot)

                return [{"role": "assistant", "content": reply, "images": [plot]}]

            else:
                reply = f"""⚠️ PANN training is only supported for DAB converters.

Your detected topology: **{topo.upper()}**

Buck and Boost converters use analytical models and do not require neural network training.
Only DAB converters with phase-shift modulation use PANN for waveform prediction."""
                st.warning(reply)
                return [{"role": "assistant", "content": reply}]

    except Exception as e:
        reply = f"❌ Training error: {str(e)}"
        st.error(reply)
        return [{"role": "assistant", "content": reply}]


# Other tasks - general LLM
def other_tasks(chat_engine, prompt, messages_history, session_id="agent1"):
    """Perform all other tasks through a general LLM - now truly conversational!"""
    
    # Get current topology
    topo = st.session_state.get("topology", "unknown")
    
    # Check stored data
    has_components = st.session_state.get('components') is not None
    has_stored_params = all([
        st.session_state.get('Uin') is not None,
        st.session_state.get('Uo') is not None,
        st.session_state.get('P') is not None
    ])
    
    # Detect INITIAL component detail request (not a follow-up question)
    initial_component_request = (
        ('show' in prompt.lower() or 'display' in prompt.lower()) and 
        ('component' in prompt.lower() or 'detail' in prompt.lower() or 'spec' in prompt.lower())
    )
    
    # If this is an INITIAL request to show components
    if initial_component_request and has_components:
        components = st.session_state.components
        Uin = st.session_state.Uin
        Uo = st.session_state.Uo
        P = st.session_state.P
        fs = st.session_state.get('fs', 100)
        
        # Use LLM to generate intelligent component recommendations
        if "llm_manager" not in st.session_state:
            st.session_state.llm_manager = LLMManager()
        
        llm = st.session_state.llm_manager.get_smart()
        
        recommendation_prompt = f"""You are an expert power electronics component selection engineer. Generate detailed component recommendations with reasoning.

DESIGN SPECIFICATIONS:
- Topology: {topo.upper()} Converter
- Input Voltage: {Uin}V
- Output Voltage: {Uo}V
- Output Power: {P}W
- Switching Frequency: {fs}kHz

CALCULATED COMPONENT REQUIREMENTS:
Inductor:
- Inductance: {components['inductor']['value']}µH
- RMS Current: {components['inductor']['current_rating']:.1f}A
- Peak Current: {components['inductor']['saturation_current']:.1f}A
- Max DCR: {components['inductor']['dcr']:.1f}mΩ

Capacitor:
- Capacitance: {components['capacitor']['value']}µF
- Min Voltage Rating: {components['capacitor']['voltage_rating']}V
- Max ESR: {components['capacitor']['esr']:.1f}mΩ

MOSFETs:
- Voltage Rating: {components['mosfet']['voltage_rating']}V
- Current Rating: {components['mosfet']['current_rating']:.1f}A
- High-Side RDS(on): < {components['mosfet']['rdson_high']:.1f}mΩ
- Low-Side RDS(on): < {components['mosfet']['rdson_low']:.1f}mΩ

Generate detailed component recommendations (300-400 words) with:
1. Inductor recommendation (core type, construction) and WHY
2. Capacitor recommendation (dielectric type) and WHY  
3. MOSFET recommendation and WHY
4. Additional components needed

Use markdown formatting. Be specific to THIS design.
"""
        
        try:
            with st.spinner("Generating intelligent component recommendations..."):
                response = llm.invoke(recommendation_prompt)
                detail_text = response.content
            
            st.markdown(detail_text)
            return [{"role": "assistant", "content": detail_text}]
        except Exception as e:
            error_msg = f"Error generating recommendations: {str(e)}"
            st.error(error_msg)
            return [{"role": "assistant", "content": error_msg}]
    
    # If no initial request - this is a CONVERSATION (follow-up question)
    # Build rich context for the LLM
    if has_stored_params:
        Uin = st.session_state.get('Uin')
        Uo = st.session_state.get('Uo')
        P = st.session_state.get('P')
        fs = st.session_state.get('fs', 100)
        
        # Include component data if available
        component_context = ""
        if has_components:
            components = st.session_state.components
            component_context = f"""

Component recommendations for this design:
- Inductor: {components['inductor']['value']}µH, {components['inductor']['current_rating']:.1f}A rated, DCR < {components['inductor']['dcr']:.1f}mΩ
- Capacitor: {components['capacitor']['value']}µF, {components['capacitor']['voltage_rating']}V rated, ESR < {components['capacitor']['esr']:.1f}mΩ, Type: {components['capacitor']['type']}
- MOSFETs: {components['mosfet']['voltage_rating']}V rated, High-side RDS(on) < {components['mosfet']['rdson_high']:.1f}mΩ, Low-side RDS(on) < {components['mosfet']['rdson_low']:.1f}mΩ
- Switching frequency: {components['switching_freq']}kHz
"""
        
        context_info = f"""You are PE-GPT, a friendly power electronics expert having a conversation with a user.

CURRENT DESIGN CONTEXT:
- Topology: {topo.upper()} Converter
- Input: {Uin}V, Output: {Uo}V, Power: {P}W, Frequency: {fs}kHz{component_context}

IMPORTANT RULES:
1. This is a CONVERSATION - answer the user's specific question naturally
2. Reference the design context above when relevant
3. Be conversational and friendly, not robotic
4. Keep answers SHORT (3-5 sentences) unless user asks for more detail
5. Answer ONLY about {topo.upper()} converters (NOT DAB or other topologies)
6. Use simple, clear language

User's question: {prompt}

Provide a natural, conversational answer:
"""
        
        final_prompt = context_info
    else:
        # No stored design - general question
        final_prompt = f"""You are PE-GPT, a friendly power electronics expert.

The user hasn't designed a converter yet.

User's question: {prompt}

Provide a helpful, conversational answer:
"""
    
    try:
        with st.spinner("Thinking..."):
            # Use smart LLM for conversational responses
            if "llm_manager" not in st.session_state:
                st.session_state.llm_manager = LLMManager()
            
            llm = st.session_state.llm_manager.get_smart()
            
            response = llm.invoke(final_prompt)
            response_text = response.content
            
            st.write(response_text)
            return [{"role": "assistant", "content": response_text}]
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.error(error_msg)
        return [{"role": "assistant", "content": error_msg}]




    

# newly added
def is_recalculation_request(prompt, session_state):
    """
    Detect if user is asking for recalculation with modified parameters
    Returns: (is_recalc, modified_params_dict)
    """
    # Check if we have existing design
    has_design = all([
        session_state.get('Uin') is not None,
        session_state.get('Uo') is not None,
        session_state.get('P') is not None
    ])
    
    if not has_design:
        return False, {}
    
    # Check for recalculation keywords
    recalc_keywords = ['what if', 'change', 'increase', 'decrease', 'use', 'with', 'recalculate', 'try', 'compare']
    if not any(keyword in prompt.lower() for keyword in recalc_keywords):
        return False, {}
    
    modifications = {}
    
    # Detect frequency changes
    freq_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:khz|frequency)', prompt, re.IGNORECASE)
    if freq_match:
        modifications['fs'] = float(freq_match.group(1))
    
    # Detect input voltage changes
    input_match = re.search(r'(?:input|vin).*?(\d+(?:\.\d+)?)\s*v', prompt, re.IGNORECASE)
    if input_match:
        modifications['Uin'] = float(input_match.group(1))
    
    # Detect output voltage changes
    output_match = re.search(r'(?:output|vout|vo).*?(\d+(?:\.\d+)?)\s*v', prompt, re.IGNORECASE)
    if output_match:
        modifications['Uo'] = float(output_match.group(1))
    
    # Detect power changes
    power_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:w|watt|power)', prompt, re.IGNORECASE)
    if power_match:
        modifications['P'] = float(power_match.group(1))
    
    return len(modifications) > 0, modifications

def user_wants_details(prompt):
    """Check if user is asking for detailed information"""
    detail_keywords = [
        'explain', 'detail', 'why', 'how does', 'how do', 
        'tell me more', 'elaborate', 'show component', 
        'specification', 'breakdown', 'in depth'
    ]
    return any(keyword in prompt.lower() for keyword in detail_keywords)


def get_truncated_history(max_chars=3000):
    """Get message history truncated to avoid token limits"""
    all_messages = st.session_state.get("messages", [])
    
    # Start from most recent and work backwards
    truncated = []
    total_chars = 0
    
    for msg in reversed(all_messages):
        content = str(msg.get('content', ''))
        msg_chars = len(content)
        
        if total_chars + msg_chars > max_chars:
            break
        
        truncated.insert(0, msg)
        total_chars += msg_chars
    
    return truncated

# =====================================================================
# MAIN DESIGN FLOW
# =====================================================================
def design_flow(chat_engines, task_agent, FlexRes=True):
    if 'intelligence_agent' not in st.session_state:
        st.session_state.intelligence_agent = ConversationIntelligenceAgent()

    intelligence_agent = st.session_state.intelligence_agent

    """
    Main customized design workflow supporting multiple converter topologies
    """
    
    chat_engine0 = chat_engines.get('agent0')
    chat_engine1 = chat_engines.get('agent1')
    chat_engine2 = chat_engines.get('agent2')
    
    # User textual input block for queries
    if prompt := st.chat_input("Your request:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # ---- 1. CHECK IF NOW WAITING FOR FILES FOR PANN TRAINING
        if st.session_state.get('expecting_pann_training', False):
            # Use your LLM-based analysis of user prompt to get the latest intent
            analysis = intelligence_agent.analyze(
                query=prompt,
                conversation_context=st.session_state.messages,
                stored_topology=st.session_state.get("topology", "unknown")
            )
            user_intent = analysis.get('intent', '')

            # Check that all three required files are present, and not None
            have_all_files = all(
                (k in st.session_state) and (st.session_state[k] is not None)
                for k in ['vp', 'vs', 'iL']
            )

            # Only trigger training if user confirmed upload (via LLM) and all files are present
            if user_intent == "confirm_upload" and have_all_files:
                pann_msgs = train_pann()
                for msg in pann_msgs:
                    st.session_state.messages.append(msg)
                st.session_state.expecting_pann_training = False
                return  # Don't process more, cycle ends here

            # If not ready, help user
            missing = [k for k in ['vp', 'vs', 'iL'] if (k not in st.session_state) or (st.session_state[k] is None)]
            if missing:
                st.info(f"Waiting for uploads: {', '.join(missing)} (use sidebar). After uploading, confirm in chat (e.g. 'I have uploaded').")
            else:
                st.info("All files uploaded! Type 'I have uploaded' or use the chat to confirm so training can begin.")
            return
     


        # Check if this is a recalculation
        is_recalc, modifications = is_recalculation_request(prompt, st.session_state)

        # -------------------------------
        # FULLY INTELLIGENT Conversation Analysis - NO HARDCODING
        # -------------------------------

        if "intelligence_agent" not in st.session_state:
            st.session_state["intelligence_agent"] = ConversationIntelligenceAgent()

        intelligence_agent = st.session_state["intelligence_agent"]

        # Get current state
        current_stored_topo = st.session_state.get('topology', 'unknown')

        # 🔒 CRITICAL: If this is a recalculation, LOCK topology (don't re-detect)
        if is_recalc and current_stored_topo != 'unknown':
            # Keep current topology for recalculations
            st.info(f"♻️ Recalculating {current_stored_topo.upper()} converter...")
            detected_topology = current_stored_topo  # Lock topology
            is_continuation = True
            intent = "modify"
            reasoning = "Recalculation request - maintaining current topology"
        else:
            # Ask LLM to analyze EVERYTHING
            analysis = intelligence_agent.analyze(
                query=prompt,
                conversation_context=st.session_state.messages,
                stored_topology=current_stored_topo
            )

        # Use LLM's reasoning
        detected_topology = analysis['topology']
        is_continuation = analysis['is_continuation']
        intent = analysis['intent']
        reasoning = analysis['reasoning']

        if intent == "upload_data":
            # Prompt user to upload vp, vs, iL using sidebar widgets
            st.info(
                "To improve your DAB model with experimental data, please upload three files in the sidebar:\n"
                "- Primary Voltage (vp.csv)\n"
                "- Secondary Voltage (vs.csv)\n"
                "- Inductor Current (iL.csv)\n"
                "After uploading all three, type 'I have uploaded' in the chat to continue training."
            )
            # Set session flag so next input triggers training
            st.session_state.expecting_pann_training = True
            messages = [{"role": "assistant", "content": "Waiting for data uploads..."}]
            for msg in messages:
                st.session_state.messages.append(msg)
            return

        # Update topology if needed
        if detected_topology != "unknown":
            # Only update if confident or if no topology stored
            if analysis['confidence'] in ['high', 'medium'] or current_stored_topo == 'unknown':
                st.session_state["topology"] = detected_topology
                
                # Show reasoning
                st.info(f"🧠 {detected_topology.upper()} converter | {reasoning}")
        elif current_stored_topo == 'unknown':
            st.warning("🤔 Please specify converter type: DAB, Buck, Boost, Buck-Boost, or Flyback")

        # Show current topology
        st.caption(f"📌 Topology: {st.session_state.get('topology', 'none')} | Intent: {intent}")

        # Get historical messages
        messages_history = get_truncated_history(max_chars=3000)

        # Classify the task
        with st.chat_message("assistant"):
            if is_recalc:
                task = "Task 2"
                for key, value in modifications.items():
                    st.session_state[key] = value
                
                topo = st.session_state.get("topology", "converter")
                change_desc = ', '.join([f'{k}={v}' for k, v in modifications.items()])
                st.info(f"♻️ Recalculating {topo} design with: {change_desc}")
            else:
                task = task_agent.classify(prompt)
            
            # Execute the appropriate task
            if task == "Task 0":
                response_pe = init_design(chat_engine1, prompt, messages_history, session_id="agent1")
                messages = [{"role": "assistant", "content": response_pe}]
                
            elif task == "Task 1":
                messages = recommend_modulation(chat_engine1, prompt, messages_history, session_id="agent1")
                
            elif task == "Task 2":
                topo = st.session_state.get("topology", "unknown")

                if topo == "dab":
                    messages = evaluate_dab(chat_engine0, prompt, messages_history, session_id="agent0")

                elif topo == "buck":
                    messages = evaluate_buck(chat_engine0, prompt, messages_history, session_id="agent0")

                elif topo == "boost":
                    messages = evaluate_boost(chat_engine0, prompt, messages_history, session_id="agent0")

                else:
                    warn = f"""⚠️ Converter topology '{topo}' evaluation is not implemented yet.
                    
**Currently supported topologies:**
- DAB (Dual Active Bridge)
- Buck (Step-down)
- Boost (Step-up)

Please specify one of these converter types with operating conditions."""
                    st.warning(warn)
                    messages = [{"role": "assistant", "content": warn}]
                
                # FlexRes: Enhanced responses with LLM insights
                if topo == "dab" and FlexRes and messages:
                    prompt_flex = """Attention!!! Now I have evaluated the current stress and soft switching
                    performances of the recommended modulation and the conventional SPS strategy, as the contents 
                    shown below. Please refer to your expertise for DAB modulations, completely rewrite the 
                    contents and provide more power electronics insights.""".replace("\n", "") + "\n" + \
                        "'" + "\n".join(item["content"] for item in messages) + "'"
                    
                    response = chat_engine0.invoke(
                        {"input": prompt_flex, "chat_history": messages_history},
                        config={"configurable": {"session_id": "agent0_flex"}}
                    )
                    st.write(response)
                    messages.append({"role": "assistant", "content": response})

                # ✅ Use intelligence agent's analysis instead of hardcoded keywords
                user_wants_detail = (intent == "question")

                if topo == "dab" and FlexRes and user_wants_detail and messages:
                    prompt_flex = (
                        "You are a DAB converter expert. "
                        "Provide a CONCISE explanation (3-4 sentences max) addressing:\n\n"
                        f"User question: {prompt}\n"
                        f"Design summary: {messages[0]['content'][:500]}"
                    )
                    
                    response = chat_engine0.invoke(
                        {"input": prompt_flex, "chat_history": []},
                        config={"configurable": {"session_id": "agent0_flex"}}
                    )
                    st.write(response)
                    messages.append({"role": "assistant", "content": response})

                elif topo == "boost" and FlexRes and user_wants_detail and messages:
                    explanation_prompt = (
                        "You are a senior power electronics engineer. "
                        "Provide a CONCISE explanation (3-4 sentences max) addressing the user's question:\n\n"
                        f"User question: {prompt}\n"
                        f"Design summary: {messages[0]['content'][:500]}"
                    )

                    response = chat_engine0.invoke(
                        {"input": explanation_prompt, "chat_history": []},
                        config={"configurable": {"session_id": "agent0_boost_flex"}}
                    )

                    st.write(response)
                    messages.append({"role": "assistant", "content": response})

            elif task == "Task 3":
                messages = simulation_verification()
                
            elif task == "Task 4":
                messages = pe_gpt_introduction(chat_engine2, prompt, session_id="agent2")
                
            elif task == "Task 5":
                messages = train_pann()
                
            else:  # Other tasks
                messages = other_tasks(chat_engine1, prompt, messages_history, session_id="agent1_other")
        
        # Append all messages to session state
        for msg in messages:
            st.session_state.messages.append(msg)



# =====================================================================
# INITIALIZATION HELPER
# =====================================================================

# def create_task_agent(model="llama-3.1-8b-instant"):
#     """Create and return a task classification agent"""
#     return TaskClassificationAgent(model=model)


def create_task_agent():                           
    """Create and return a task classification agent"""
    return TaskClassificationAgent()
