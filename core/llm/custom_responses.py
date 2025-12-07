"""
@functionality
    Predefined response templates for modulation strategies and performance metrics

@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin
@github: https://github.com/XinzeLee/PE-GPT

GROQ EDITION - Enhanced with better formatting and extensibility

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, 
                 Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""


# =====================================================================
# DAB CONVERTER RESPONSES
# =====================================================================

def response(performances, modulation):
    """
    Generate formatted response for DAB converter modulation performance
    
    Args:
        performances: Tuple containing (ipp, nZVS, nZCS, P_required, pos)
            - ipp: Peak-to-peak current (A)
            - nZVS: Number of switches achieving Zero Voltage Switching
            - nZCS: Number of switches achieving Zero Current Switching
            - P_required: Required power level (W)
            - pos: Modulation parameters tuple
        modulation: Modulation strategy name (SPS, EPS1, EPS2, DPS, TPS, 5DOF)
    
    Returns:
        str: Formatted response string with performance metrics
    """
    
    ipp, nZVS, nZCS, P_required, pos = performances
    
    # Default response if no valid strategy found
    if not modulation or modulation not in MODULATION_STRATEGIES:
        return "No valid modulation strategy found."
    
    # Base answer format
    answer_format = """Under the {} modulation strategy, {}, 
    the number of switches that achieve zero-voltage turn-on is {:.0f}, 
    the number of switches that achieve zero-current turn-off is {:.0f}. 
    And the current stress performance is shown with the following figure. 
    At the power level (PL = {} W), the peak-to-peak current is {:.2f} A.""".replace("\n", "")
    
    # Get parameter string based on modulation type
    ps_str = get_parameter_string(modulation, pos)
    
    # Format and return response
    response_text = answer_format.format(
        modulation, 
        ps_str, 
        nZVS, 
        nZCS, 
        P_required, 
        ipp
    )
    
    return response_text


def get_parameter_string(modulation, pos):
    """
    Generate parameter description string for each modulation strategy
    
    Args:
        modulation: Modulation strategy name
        pos: Tuple of modulation parameters
    
    Returns:
        str: Formatted parameter description
    """
    
    if modulation == "SPS":
        D0 = pos[0]
        return f"the D0 is {D0:.3f}"
    
    elif modulation == "EPS1":
        D0, Din = pos[0], pos[1]
        return f"the D0 is {D0:.3f}, D2 is 1, the optimal D1 is designed to be {Din:.3f}"
    
    elif modulation == "EPS2":
        D0, Din = pos[0], pos[1]
        return f"the D0 is {D0:.3f}, D1 is 1, the optimal D2 is designed to be {Din:.3f}"
    
    elif modulation == "DPS":
        D0, Din = pos[0], pos[1]
        return f"the D0 is {D0:.3f}, the optimal D1 and D2 are designed to be {Din:.3f}"
    
    elif modulation == "TPS":
        D0, D1, D2 = pos[0], pos[1], pos[2]
        return f"the D0 is {D0:.3f}, the optimal D1 is {D1:.3f}, the optimal D2 is {D2:.3f}"
    
    elif modulation == "5DOF":
        D0, D1, D2, phi1, phi2 = pos[0], pos[1], pos[2], pos[3], pos[4]
        return f"the D0 is {D0:.3f}, the optimal D1 is {D1:.3f}, the optimal D2 is {D2:.3f}, the optimal phi1 is {phi1:.3f}, the optimal phi2 is {phi2:.3f}"
    
    return "parameters not specified"


# =====================================================================
# MODULATION STRATEGY METADATA
# =====================================================================

MODULATION_STRATEGIES = {
    "SPS": {
        "name": "Single Phase Shift",
        "parameters": ["D0"],
        "description": "Simplest modulation strategy with single phase shift control"
    },
    "EPS1": {
        "name": "Extended Phase Shift 1",
        "parameters": ["D0", "D1"],
        "description": "Extended phase shift with D2 fixed at 1"
    },
    "EPS2": {
        "name": "Extended Phase Shift 2",
        "parameters": ["D0", "D2"],
        "description": "Extended phase shift with D1 fixed at 1"
    },
    "DPS": {
        "name": "Dual Phase Shift",
        "parameters": ["D0", "D1=D2"],
        "description": "Two-degree-of-freedom control with symmetric duty cycles"
    },
    "TPS": {
        "name": "Triple Phase Shift",
        "parameters": ["D0", "D1", "D2"],
        "description": "Three-degree-of-freedom control for optimal performance"
    },
    "5DOF": {
        "name": "Five Degrees of Freedom",
        "parameters": ["D0", "D1", "D2", "phi1", "phi2"],
        "description": "Most flexible control with five independent parameters"
    }
}


def get_modulation_info(modulation):
    """
    Get detailed information about a modulation strategy
    
    Args:
        modulation: Modulation strategy name
    
    Returns:
        dict: Information about the modulation strategy
    """
    return MODULATION_STRATEGIES.get(modulation, {
        "name": "Unknown",
        "parameters": [],
        "description": "No information available"
    })


def format_performance_summary(performances, modulation):
    """
    Generate a concise summary of performance metrics
    
    Args:
        performances: Tuple containing performance metrics
        modulation: Modulation strategy name
    
    Returns:
        str: Concise performance summary
    """
    
    ipp, nZVS, nZCS, P_required, pos = performances
    
    summary = f"""
Performance Summary ({modulation}):
- Peak-to-peak current: {ipp:.2f} A
- Zero-voltage switching: {nZVS:.0f} switches
- Zero-current switching: {nZCS:.0f} switches
- Power level: {P_required} W
    """.strip()
    
    return summary


def compare_modulation_strategies(perf1, mod1, perf2, mod2):
    """
    Generate comparison between two modulation strategies
    
    Args:
        perf1: Performance tuple for first strategy
        mod1: First modulation strategy name
        perf2: Performance tuple for second strategy
        mod2: Second modulation strategy name
    
    Returns:
        str: Comparison text
    """
    
    ipp1, nZVS1, nZCS1, P1, _ = perf1
    ipp2, nZVS2, nZCS2, P2, _ = perf2
    
    comparison = f"""
Comparison: {mod1} vs {mod2}

Current Stress:
- {mod1}: {ipp1:.2f} A
- {mod2}: {ipp2:.2f} A
- Improvement: {((ipp1 - ipp2) / ipp1 * 100):.1f}%

Soft Switching (ZVS):
- {mod1}: {nZVS1:.0f} switches
- {mod2}: {nZVS2:.0f} switches

Soft Switching (ZCS):
- {mod1}: {nZCS1:.0f} switches
- {mod2}: {nZCS2:.0f} switches
    """.strip()
    
    return comparison


# =====================================================================
# BUCK CONVERTER RESPONSES (Future Extension)
# =====================================================================

def buck_response(performances, control_method):
    """
    Generate formatted response for Buck converter performance
    
    Args:
        performances: Performance metrics for buck converter
        control_method: Control method name
    
    Returns:
        str: Formatted response string
    
    Note:
        This is a placeholder for future buck converter support
    """
    
    # TODO: Implement buck converter response formatting
    return "Buck converter response formatting not yet implemented."


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def format_parameter_dict(pos_dict):
    """
    Format a dictionary of parameters into readable text
    
    Args:
        pos_dict: Dictionary with parameter names and values
    
    Returns:
        str: Formatted parameter string
    """
    
    param_strings = []
    for key, value in pos_dict.items():
        if isinstance(value, float):
            param_strings.append(f"{key} = {value:.3f}")
        else:
            param_strings.append(f"{key} = {value}")
    
    return ", ".join(param_strings)


def validate_performance_data(performances):
    """
    Validate that performance data has correct structure
    
    Args:
        performances: Performance tuple to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    
    try:
        if not isinstance(performances, tuple) or len(performances) != 5:
            return False
        
        ipp, nZVS, nZCS, P_required, pos = performances
        
        # Check types
        if not all(isinstance(x, (int, float)) for x in [ipp, nZVS, nZCS, P_required]):
            return False
        
        if not isinstance(pos, tuple):
            return False
        
        return True
    
    except Exception:
        return False



#######################################################
# Codes below are used for buck converters #
#######################################################



# =====================================================================
# BUCK & BOOST — CLEAN FORMATTED RESPONSES
# =====================================================================

def response_buck(performances, components=None, show_details=False):
    """
    Format output for BUCK converter evaluation with optional component recommendations.
    
    Args:
        performances: list of (label, value) tuples
        components: dict with component recommendations (optional)
        show_details: bool - if True, show full component details (default: False)
    """
    text = "⚡ BUCK Converter Evaluation\n\n"
    text += "Key specifications:\n\n"
    for name, value in performances:
        text += f"- {name}: {value}\n"
    
    # Only show full component recommendations if requested
    if components and show_details:
        text += "\n---\n\n🔧 Component Recommendations\n\n"
        
        if 'inductor' in components:
            L = components['inductor']
            text += f"Inductor: {L['value']:.1f} µH, {L['current_rating']:.1f}A rated, DCR < {L['dcr']:.1f}mΩ\n\n"
        
        if 'capacitor' in components:
            C = components['capacitor']
            text += f"Capacitor: {C['value']:.0f} µF, {C['voltage_rating']:.0f}V rated, ESR < {C['esr']:.0f}mΩ\n\n"
        
        if 'mosfet' in components:
            M = components['mosfet']
            text += f"MOSFETs: {M['voltage_rating']:.0f}V rated, High-side RDS(on) < {M['rdson_high']:.1f}mΩ, Low-side RDS(on) < {M['rdson_low']:.1f}mΩ\n\n"
        
        if 'switching_freq' in components:
            text += f"Switching Frequency: {components['switching_freq']:.0f} kHz\n\n"
    
    elif components:
        # Show brief summary only
        text += f"\n💡 Component summary: {components['inductor']['value']}µH inductor, {components['capacitor']['value']}µF capacitor, {components['mosfet']['voltage_rating']}V MOSFETs @ {components['switching_freq']}kHz\n"
        text += "(Ask 'show component details' for full specifications)\n"
    
    return text




def response_boost(performances):
    """
    Format output for BOOST converter evaluation.

    performances: list of (label, value)
    """

    text = "### ⚡ BOOST Converter Evaluation\n\n"

    text += "Here are the computed electrical characteristics:\n\n"

    for name, value in performances:
        text += f"- **{name}**: {value}\n"

    text += "\nThese results are based on classical boost converter steady-state equations."

    return text
