"""
LLM Manager for PE-GPT
Manages multiple LLM models for different tasks
"""

from langchain_groq import ChatGroq
import json
import streamlit as st

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
# ------------------------------------------

class LLMManager:
    """
    Manages multiple LLM models for different tasks:
    - Fast LLM for parameter extraction and classification
    - Smart LLM for natural response generation
    - Conflict resolver for ambiguous cases
    """
    
    def __init__(self):
        # Fast model for parameter extraction and classification (cheap & quick)
        self.fast_llm = ChatGroq(
            model= get_next_model(),
            temperature=0.3  # Low temp for precise extraction
        )
        
        # Smart model for natural response generation (better quality)
        self.smart_llm = ChatGroq(
            model= get_next_model(),
            temperature=0.5  # Higher temp for natural conversation
        )
        
        # Conflict resolver (alternative model for reasoning)
        self.resolver_llm = ChatGroq(
            model=get_next_model(),
            temperature=0.5  # Balanced


        )
        #         self.fast_llm = ChatGroq(
        #     model="llama-3.1-8b-instant",
        #     temperature=0.3  # Low temp for precise extraction
        # )
        
        # # Smart model for natural response generation (better quality)
        # self.smart_llm = ChatGroq(
        #     model="llama-3.3-70b-versatile",
        #     temperature=0.5  # Higher temp for natural conversation
        # )
        
        # # Conflict resolver (alternative model for reasoning)
        # self.resolver_llm = ChatGroq(
        #     model="openai/gpt-oss-20b",
        #     temperature=0.5  # Balanced
        # )
    
    def get_fast(self):
        """Get fast LLM for parameter extraction"""
        return ChatGroq(model=get_next_model(), temperature=0.3)
    
    def get_smart(self):
        """Get smart LLM for response generation"""
        return ChatGroq(model=get_next_model(), temperature=0.5)
    
    def get_resolver(self):
        """Get resolver LLM for conflicts"""
        return ChatGroq(model=get_next_model(), temperature=0.5)


def intelligent_parameter_extraction(prompt, topology, llm_manager):
    """
    Use LLM to intelligently extract and validate parameters
    Handles ambiguity, conflicts, and user mistakes
    
    Args:
        prompt: User's input text
        topology: Detected topology (buck, boost, dab, etc.)
        llm_manager: LLMManager instance
    
    Returns:
        dict: Extracted parameters and conflict status
    """
    
    extraction_prompt = f"""You are a parameter extraction expert for {topology} converters.

User input: "{prompt}"

Your task:
1. Extract input voltage (Vin), output voltage (Vout), power (P), and frequency (fs if mentioned)
2. Validate if these make sense for a {topology} converter
3. If there's a conflict or MISSING required parameter, identify it

{topology.upper()} converter rules:
- Buck: Vout MUST be < Vin (step-down). If you see Vout > Vin, that's a CONFLICT.
- Boost: Vout MUST be > Vin (step-up). If you see Vout < Vin, that's a CONFLICT.
- DAB: Can be bidirectional (any Vin/Vout ratio is ok)

REQUIRED PARAMETERS:
- Vin (input voltage) - REQUIRED
- Vout (output voltage) - REQUIRED  
- P (power in watts) - REQUIRED
- fs (frequency) - OPTIONAL (default 100kHz if not mentioned)

If ANY required parameter is missing, set conflict=true.

Respond in this EXACT JSON format (no extra text):

{{
  "vin": <number>,
  "vout": <number>,
  "power": <number>,
  "frequency": <number or null>,
  "conflict": <true or false>,
  "conflict_description": "<explanation if conflict=true, otherwise empty string>",
  "suggested_fix": "<your intelligent suggestion if conflict, otherwise empty string>"
}}

Examples:

User: "buck 24V, 48V, 500W"
Response: {{"vin": 24, "vout": 48, "power": 500, "frequency": null, "conflict": true, "conflict_description": "Buck converter requires Vout < Vin, but 48V > 24V", "suggested_fix": "User likely meant 48V→24V (swap values) OR intended boost converter"}}

User: "buck 48V to 24V, 500W"
Response: {{"vin": 48, "vout": 24, "power": 500, "frequency": null, "conflict": false, "conflict_description": "", "suggested_fix": ""}}

User: "boost 24V input 48V output 200W at 150kHz"
Response: {{"vin": 24, "vout": 48, "power": 200, "frequency": 150, "conflict": false, "conflict_description": "", "suggested_fix": ""}}

User: "buck converter 48v, 30v"
Response: {{"vin": 48, "vout": 30, "power": null, "frequency": null, "conflict": true, "conflict_description": "Missing required parameter: Power (P)", "suggested_fix": "Please specify power in watts, e.g., '48V to 30V, 500W' or '48V, 30V, 500W'"}}

Now process the user input above and respond with ONLY the JSON (no other text):

"""
    
    llm = llm_manager.get_fast()
    
    try:
        response = llm.invoke(extraction_prompt)
        # Parse JSON response
        result = json.loads(response.content.strip())
        return result
    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails
        st.error(f"Parameter extraction failed: {e}")
        return {
            "conflict": True, 
            "conflict_description": "Could not parse parameters from your input",
            "suggested_fix": "Please specify in format: [Vin, Vout, Power] or 'Vin input, Vout output, Power'"
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return {"conflict": True, "conflict_description": str(e), "suggested_fix": ""}


def resolve_parameter_conflict(conflict_data, prompt, topology, llm_manager):
    """
    Use LLM to resolve conflicts through friendly conversation
    
    Args:
        conflict_data: Dict with conflict info from parameter extraction
        prompt: Original user prompt
        topology: Detected topology
        llm_manager: LLMManager instance
    
    Returns:
        str: Friendly clarification message
    """
    
    resolution_prompt = f"""You're helping someone figure out their {topology} converter design. They seem a bit confused.

What they said: "{prompt}"

The issue: {conflict_data['conflict_description']}

Possible fix: {conflict_data['suggested_fix']}

Write a super friendly, SHORT message (2-3 sentences) asking them to clarify. Be warm and helpful, like talking to a friend who's learning.

Rules:
- Start casually (like "Hey," or "Hmm," or "Quick question -")
- Offer 2-3 clear options
- Make it SHORT
- End with a simple question
- Sound human, not robotic

Example for buck 24V→48V:
"Hmm, you mentioned a buck converter from 24V to 48V, but buck converters step voltage down (not up). Did you mean 48V→24V for buck, or 24V→48V for boost? Which direction are you going?"

Write your response (be casual!):
"""
    
    llm = llm_manager.get_resolver()
    response = llm.invoke(resolution_prompt)
    
    return response.content



def generate_natural_buck_response(performances, components, design_context, llm_manager):
    """
    Generate conversational response for Buck converter using smart LLM
    
    Args:
        performances: List of (name, value) tuples
        components: Dict with component recommendations
        design_context: Dict with vin, vout, power, frequency
        llm_manager: LLMManager instance
    
    Returns:
        str: Natural, conversational response
    """
    
    perf_dict = {name: value for name, value in performances}
    
    response_prompt = f"""You're PE-GPT, a friendly power electronics expert chatting with someone about their converter design. Be conversational like you're talking to a colleague over coffee.

THEIR BUCK CONVERTER DESIGN:
- {design_context['vin']}V → {design_context['vout']}V
- {design_context['power']}W at {design_context.get('frequency', 100)}kHz

KEY RESULTS:
- Duty Cycle: {perf_dict.get('Duty Cycle (D)', 'N/A')}
- Output Current: {perf_dict.get('Output Current (A)', 'N/A')}A
- Efficiency: {perf_dict.get('Efficiency (%)', 'N/A')}%
- Inductor Ripple: {perf_dict.get('Inductor Ripple ΔiL (A)', 'N/A')}A
- Output Ripple: {perf_dict.get('Output Ripple ΔV (V)', 'N/A')}mV

COMPONENTS:
- Inductor: {components['inductor']['value']}µH, {components['inductor']['current_rating']:.1f}A rated
- Capacitor: {components['capacitor']['value']}µF
- MOSFETs: {components['mosfet']['voltage_rating']}V rated

YOUR TASK: Write a super casual, friendly response (4-5 sentences) that:
- Starts warmly (like "Nice!", "Looking good!", "Sweet design!")
- Explains 2-3 key results in context (WHY they matter, not just listing numbers)
- Sounds like chatting with a friend, not giving a lecture
- Uses contractions (you're, it's, let's, we've)
- NO bullet points, NO formal lists - just natural conversation

Example style:
"Nice! Your 48V to 24V buck converter's running at a sweet spot with that 50% duty cycle - minimizes both switching and conduction losses. The 100µH inductor keeps ripple nice and tight at about 6% of output current, which is perfect for most apps. Efficiency's sitting at a solid 96%, which is pretty great for this voltage ratio. Want to see what happens if we crank up the frequency?"

Write YOUR response (be casual and friendly):
"""
    
    llm = llm_manager.get_smart()
    response = llm.invoke(response_prompt)
    
    return response.content



def generate_natural_boost_response(performances, design_context, llm_manager):
    """
    Generate conversational response for Boost converter using smart LLM
    
    Args:
        performances: List of (name, value) tuples
        design_context: Dict with vin, vout, power, frequency
        llm_manager: LLMManager instance
    
    Returns:
        str: Natural, conversational response
    """
    
    perf_dict = {name: value for name, value in performances}
    
    response_prompt = f"""You're PE-GPT, a friendly power electronics expert chatting casually about their boost converter.

THEIR BOOST CONVERTER:
- {design_context['vin']}V → {design_context['vout']}V (stepping up)
- {design_context['power']}W at {design_context.get('frequency', 100)}kHz

RESULTS:
- Duty Cycle: {perf_dict.get('Duty Cycle (D)', 'N/A')}
- Input Current: {perf_dict.get('Input Current (A)', 'N/A')}A
- Efficiency: {perf_dict.get('Efficiency (%)', 'N/A')}%
- Inductor Ripple: {perf_dict.get('Inductor Ripple ΔiL (A)', 'N/A')}A

Write a casual, friendly response (4-5 sentences) that:
- Starts warmly
- Explains key results naturally (not a list!)
- Mentions boost-specific stuff if relevant (high voltage stress, pulsating current)
- Sounds like talking to a friend
- Uses contractions

Be conversational and natural:
"""
    
    llm = llm_manager.get_smart()
    response = llm.invoke(response_prompt)
    
    return response.content
