# """
# @reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
# @code-author: Xinze Li, Fanfan Lin, Weihao Lei
# @github: https://github.com/XinzeLee/PE-GPT

# @reference:
#     Following references are related to power electronics GPT (PE-GPT)
#     1: PE-GPT: a New Paradigm for Power Electronics Design
#         Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
#         Paper DOI: 10.1109/TIE.2024.3454408
# """

# import numpy as np
# import pyswarms as ps
# import streamlit as st

# from ..optim import obj_func
# from ..utils.plots import plot_modulation

# # Import model and constants at the TOP of the file
# from ..model_zoo.pann_dab import model_pann
# # from ..model_zoo.pann_dab import model_pann_onnx
# from ..model_zoo.pann_dab_vars import n


# def optimize_cs(nums, model_pann, PL, Vin, Vref, 
#                 modulation, upper_bound, lower_bound, 
#                 bh_strategy, vh_strategy, with_ZVS=False):
#     """
#     Optimize the current stress through particle swarm optimization algorithm
    
#     Args:
#         nums: Number of iterations
#         model_pann: PANN model for prediction
#         PL: Power level (W)
#         Vin: Input voltage (V)
#         Vref: Reference/output voltage (V)
#         modulation: Modulation strategy (SPS, DPS, EPS, TPS, 5DOF)
#         upper_bound: Upper bounds for optimization variables
#         lower_bound: Lower bounds for optimization variables
#         bh_strategy: Boundary handling strategy
#         vh_strategy: Velocity handling strategy
#         with_ZVS: Whether to include ZVS constraint
    
#     Returns:
#         cost: Optimized cost value
#         pos: Optimal position (modulation parameters)
#     """
#     upper_bounds = np.array(upper_bound)
#     lower_bounds = np.array(lower_bound)
    
#     PSO_optimizer = ps.single.GlobalBestPSO(
#         n_particles=50, 
#         dimensions=len(upper_bounds), 
#         bounds=(lower_bounds, upper_bounds),
#         options={'c1': 2.05, 'c2': 2.05, 'w': 0.9},
#         bh_strategy=bh_strategy,
#         velocity_clamp=None,
#         vh_strategy=vh_strategy,
#         oh_strategy={"w": "lin_variation"}
#     )
    
#     cost, pos = PSO_optimizer.optimize(
#         obj_func.obj_func, 
#         nums,
#         model_pann=model_pann,
#         PL=PL, 
#         Vin=Vin, 
#         Vref=Vref,
#         modulation=modulation,
#         with_ZVS=with_ZVS
#     )
    
#     # Alternative: ONNX objective function (commented out)
#     # cost, pos = PSO_optimizer.optimize(
#     #     obj_func.obj_func_onnx, 
#     #     nums,
#     #     model_pann_onnx=model_pann,
#     #     PL=PL, Vin=Vin, Vref=Vref,
#     #     modulation=modulation,
#     #     with_ZVS=with_ZVS
#     # )
    
#     return cost, pos


# def optimize_mod_dab(Vin, Vref, PL, modulation):
#     """
#     Optimize the modulation parameters for DAB converters with specified 
#     operating conditions and recommended modulation strategy
    
#     Args:
#         Vin: Input voltage (V)
#         Vref: Reference/output voltage (V)
#         PL: Power level (W)
#         modulation: Modulation strategy (SPS, DPS, EPS, TPS, 5DOF)
    
#     Returns:
#         ipp: Peak-to-peak current (A)
#         ZVS: Number of switches with zero-voltage switching
#         ZCS: Number of switches with zero-current switching
#         PL: Power level (W)
#         pos: Optimized modulation parameters (list)
#         plot: Matplotlib figure with waveforms
#         modulation: Updated modulation strategy name
#     """
    
#     # Define hyperparameters for optimizers
#     bh_strategy = "periodic"
#     vh_strategy = "unmodified"
#     num_iter = 50
    
#     st.write(f"Recommended Modulation is: {modulation}")
    
#     # Initialize bounds (prevent UnboundLocalError)
#     upper_bound = None
#     lower_bound = None
    
#     # Define the searching boundaries based on modulation strategy
#     if modulation == "SPS":
#         upper_bound = [0.35]
#         lower_bound = [-0.2]
        
#     elif modulation in ["EPS", "DPS"]:
#         upper_bound = [0.35, 1.0]
#         lower_bound = [-0.2, 0.6]
        
#         # Determine EPS variant based on voltage ratio
#         if modulation == "EPS":
#             if Vin > n * Vref:
#                 modulation = "EPS1"
#             else:
#                 modulation = "EPS2"
        
#     elif modulation == "TPS":
#         num_iter = 100
#         upper_bound = [0.35, 1.0, 1.0]
#         lower_bound = [-0.2, 0.6, 0.6]
        
#     elif modulation == "5DOF":
#         num_iter = 200
#         upper_bound = [0.35, 1.0, 1.0, 0.2, 0.2]
#         lower_bound = [-0.2, 0.6, 0.6, -0.2, -0.2]
    
#     else:
#         # Handle unknown modulation strategies
#         error_msg = f"⚠️ Unknown modulation strategy: {modulation}. Defaulting to TPS."
#         st.error(error_msg)
        
#         # Default to TPS parameters
#         modulation = "TPS"
#         num_iter = 100
#         upper_bound = [0.35, 1.0, 1.0]
#         lower_bound = [-0.2, 0.6, 0.6]
    
#     # Validate bounds are set
#     if upper_bound is None or lower_bound is None:
#         raise ValueError(f"Failed to set optimization bounds for modulation: {modulation}")
    
#     # Conduct the optimization algorithm
#     try:
#         obj, optimal_x = optimize_cs(
#             num_iter, 
#             model_pann, 
#             PL, 
#             Vin, 
#             Vref,
#             modulation, 
#             upper_bound, 
#             lower_bound,
#             bh_strategy, 
#             vh_strategy, 
#             with_ZVS=True
#         )
#     except Exception as e:
#         st.error(f"Optimization failed: {str(e)}")
#         raise
    
#     # Evaluate all performance metrics (PyTorch)
#     try:
#         ipp, P_pred, pred, inputs, ZVS, ZCS, penalty = obj_func.obj_func(
#             optimal_x[None], 
#             model_pann, 
#             PL, 
#             Vin, 
#             Vref, 
#             with_ZVS=True, 
#             modulation=modulation, 
#             return_all=True
#         )
#     except Exception as e:
#         st.error(f"Performance evaluation failed: {str(e)}")
#         raise
    
#     # Alternative: Evaluate using ONNX (commented out)
#     # ipp, P_pred, pred, inputs, ZVS, ZCS, penalty = obj_func.obj_func_onnx(
#     #     np.tile(optimal_x[None], (50, 1)), 
#     #     model_pann_onnx, 
#     #     PL, 
#     #     Vin, 
#     #     Vref, 
#     #     with_ZVS=True, 
#     #     modulation=modulation, 
#     #     return_all=True
#     # )
    
#     # Round optimal parameters to 3 decimal places
#     pos = [round(x, 3) for x in optimal_x]
    
#     # Generate waveform plot
#     try:
#         plot = plot_modulation(inputs, pred, Vin, Vref, PL, modulation)
#     except Exception as e:
#         st.warning(f"Plot generation failed: {str(e)}")
#         plot = None
    
#     return ipp[0], ZVS[0], ZCS[0], PL, pos, plot, modulation


# # Validation function (optional utility)
# def validate_modulation_strategy(modulation):
#     """
#     Validate if modulation strategy is supported
    
#     Args:
#         modulation: Modulation strategy name
    
#     Returns:
#         bool: True if valid, False otherwise
#     """
#     valid_strategies = ["SPS", "DPS", "EPS", "EPS1", "EPS2", "TPS", "5DOF"]
#     return modulation in valid_strategies


# def get_default_bounds(modulation):
#     """
#     Get default optimization bounds for a given modulation strategy
    
#     Args:
#         modulation: Modulation strategy name
    
#     Returns:
#         tuple: (upper_bound, lower_bound, num_iter)
#     """
#     bounds_map = {
#         "SPS": ([0.35], [-0.2], 50),
#         "DPS": ([0.35, 1.0], [-0.2, 0.6], 50),
#         "EPS": ([0.35, 1.0], [-0.2, 0.6], 50),
#         "EPS1": ([0.35, 1.0], [-0.2, 0.6], 50),
#         "EPS2": ([0.35, 1.0], [-0.2, 0.6], 50),
#         "TPS": ([0.35, 1.0, 1.0], [-0.2, 0.6, 0.6], 100),
#         "5DOF": ([0.35, 1.0, 1.0, 0.2, 0.2], [-0.2, 0.6, 0.6, -0.2, -0.2], 200),
#     }
    
#     return bounds_map.get(modulation, ([0.35, 1.0, 1.0], [-0.2, 0.6, 0.6], 100))
"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import numpy as np
import pyswarms as ps
import streamlit as st

from ..optim import obj_func
from ..utils.plots import plot_modulation

# Import model and constants at the TOP of the file
from ..model_zoo.pann_dab import model_pann
# from ..model_zoo.pann_dab import model_pann_onnx
from ..model_zoo.pann_dab_vars import n


def optimize_cs(nums, model_pann, PL, Vin, Vref, 
                modulation, upper_bound, lower_bound, 
                bh_strategy, vh_strategy, with_ZVS=False):
    """
    Optimize the current stress through particle swarm optimization algorithm
    
    Args:
        nums: Number of iterations
        model_pann: PANN model for prediction
        PL: Power level (W)
        Vin: Input voltage (V)
        Vref: Reference/output voltage (V)
        modulation: Modulation strategy (SPS, DPS, EPS, TPS, 5DOF)
        upper_bound: Upper bounds for optimization variables
        lower_bound: Lower bounds for optimization variables
        bh_strategy: Boundary handling strategy
        vh_strategy: Velocity handling strategy
        with_ZVS: Whether to include ZVS constraint
    
    Returns:
        cost: Optimized cost value
        pos: Optimal position (modulation parameters)
    """
    upper_bounds = np.array(upper_bound)
    lower_bounds = np.array(lower_bound)
    
    PSO_optimizer = ps.single.GlobalBestPSO(
        n_particles=50, 
        dimensions=len(upper_bounds), 
        bounds=(lower_bounds, upper_bounds),
        options={'c1': 2.05, 'c2': 2.05, 'w': 0.9},
        bh_strategy=bh_strategy,
        velocity_clamp=None,
        vh_strategy=vh_strategy,
        oh_strategy={"w": "lin_variation"}
    )
    
    cost, pos = PSO_optimizer.optimize(
        obj_func.obj_func, 
        nums,
        model_pann=model_pann,
        PL=PL, 
        Vin=Vin, 
        Vref=Vref,
        modulation=modulation,
        with_ZVS=with_ZVS
    )
    
    # Alternative: ONNX objective function (commented out)
    # cost, pos = PSO_optimizer.optimize(
    #     obj_func.obj_func_onnx, 
    #     nums,
    #     model_pann_onnx=model_pann,
    #     PL=PL, Vin=Vin, Vref=Vref,
    #     modulation=modulation,
    #     with_ZVS=with_ZVS
    # )
    
    return cost, pos


def optimize_mod_dab(Vin, Vref, PL, modulation):
    """
    Optimize the modulation parameters for DAB converters with specified 
    operating conditions and recommended modulation strategy
    
    Args:
        Vin: Input voltage (V)
        Vref: Reference/output voltage (V)
        PL: Power level (W)
        modulation: Modulation strategy (SPS, DPS, EPS, TPS, 5DOF)
    
    Returns:
        ipp: Peak-to-peak current (A)
        ZVS: Number of switches with zero-voltage switching
        ZCS: Number of switches with zero-current switching
        PL: Power level (W)
        pos: Optimized modulation parameters (list)
        plot: Matplotlib figure with waveforms
        modulation: Updated modulation strategy name
    """
    
    # Define hyperparameters for optimizers
    bh_strategy = "periodic"
    vh_strategy = "unmodified"
    num_iter = 50
    
    st.write(f"Recommended Modulation is: {modulation}")
    
    # Initialize bounds (prevent UnboundLocalError)
    upper_bound = None
    lower_bound = None
    
    # Define the searching boundaries based on modulation strategy
    if modulation == "SPS":
        upper_bound = [0.35]
        lower_bound = [-0.2]
        
    elif modulation in ["EPS", "DPS"]:
        upper_bound = [0.35, 1.0]
        lower_bound = [-0.2, 0.6]
        
        # Determine EPS variant based on voltage ratio
        if modulation == "EPS":
            if Vin > n * Vref:
                modulation = "EPS1"
            else:
                modulation = "EPS2"
        
    elif modulation == "TPS":
        num_iter = 100
        upper_bound = [0.35, 1.0, 1.0]
        lower_bound = [-0.2, 0.6, 0.6]
        
    elif modulation == "5DOF":
        num_iter = 200
        upper_bound = [0.35, 1.0, 1.0, 0.2, 0.2]
        lower_bound = [-0.2, 0.6, 0.6, -0.2, -0.2]
    
    else:
        # Handle unknown modulation strategies
        error_msg = f"⚠️ Unknown modulation strategy: {modulation}. Defaulting to TPS."
        st.error(error_msg)
        
        # Default to TPS parameters
        modulation = "TPS"
        num_iter = 100
        upper_bound = [0.35, 1.0, 1.0]
        lower_bound = [-0.2, 0.6, 0.6]
    
    # Validate bounds are set
    if upper_bound is None or lower_bound is None:
        raise ValueError(f"Failed to set optimization bounds for modulation: {modulation}")
    
    # Conduct the optimization algorithm
    try:
        obj, optimal_x = optimize_cs(
            num_iter, 
            model_pann, 
            PL, 
            Vin, 
            Vref,
            modulation, 
            upper_bound, 
            lower_bound,
            bh_strategy, 
            vh_strategy, 
            with_ZVS=True
        )
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        raise
    
    # Evaluate all performance metrics (PyTorch)
    try:
        ipp, P_pred, pred, inputs, ZVS, ZCS, penalty = obj_func.obj_func(
            optimal_x[None], 
            model_pann, 
            PL, 
            Vin, 
            Vref, 
            with_ZVS=True, 
            modulation=modulation, 
            return_all=True
        )
    except Exception as e:
        st.error(f"Performance evaluation failed: {str(e)}")
        raise
    
    # Alternative: Evaluate using ONNX (commented out)
    # ipp, P_pred, pred, inputs, ZVS, ZCS, penalty = obj_func.obj_func_onnx(
    #     np.tile(optimal_x[None], (50, 1)), 
    #     model_pann_onnx, 
    #     PL, 
    #     Vin, 
    #     Vref, 
    #     with_ZVS=True, 
    #     modulation=modulation, 
    #     return_all=True
    # )
    
    # Round optimal parameters to 3 decimal places
    pos = [round(x, 3) for x in optimal_x]
    
    # Generate waveform plot
    try:
        plot = plot_modulation(inputs, pred, Vin, Vref, PL, modulation)
    except Exception as e:
        st.warning(f"Plot generation failed: {str(e)}")
        plot = None
    
    return ipp[0], ZVS[0], ZCS[0], PL, pos, plot, modulation


# Validation function (optional utility)
def validate_modulation_strategy(modulation):
    """
    Validate if modulation strategy is supported
    
    Args:
        modulation: Modulation strategy name
    
    Returns:
        bool: True if valid, False otherwise
    """
    valid_strategies = ["SPS", "DPS", "EPS", "EPS1", "EPS2", "TPS", "5DOF"]
    return modulation in valid_strategies


def get_default_bounds(modulation):
    """
    Get default optimization bounds for a given modulation strategy
    
    Args:
        modulation: Modulation strategy name
    
    Returns:
        tuple: (upper_bound, lower_bound, num_iter)
    """
    bounds_map = {
        "SPS": ([0.35], [-0.2], 50),
        "DPS": ([0.35, 1.0], [-0.2, 0.6], 50),
        "EPS": ([0.35, 1.0], [-0.2, 0.6], 50),
        "EPS1": ([0.35, 1.0], [-0.2, 0.6], 50),
        "EPS2": ([0.35, 1.0], [-0.2, 0.6], 50),
        "TPS": ([0.35, 1.0, 1.0], [-0.2, 0.6, 0.6], 100),
        "5DOF": ([0.35, 1.0, 1.0, 0.2, 0.2], [-0.2, 0.6, 0.6, -0.2, -0.2], 200),
    }
    
    return bounds_map.get(modulation, ([0.35, 1.0, 1.0], [-0.2, 0.6, 0.6], 100))

# ================================================================
# SIMPLE ANALYTICAL OPTIMIZERS FOR BUCK & BOOST
# ================================================================

def optimize_buck(Vin, Vout, P):
    """
    Simple analytical BUCK converter evaluation.
    Returns:
        performances -> list of (name, value)
        plot -> PNG buffer
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import io

    # Load resistance
    Ro = (Vout ** 2) / P
    Iout = Vout / Ro

    # Duty cycle
    D = Vout / Vin

    # Assume typical components (user can change later)
    fs = 100e3
    L = 100e-6
    C = 100e-6

    # Ripple calculations
    delta_i = (Vin - Vout) * D / (L * fs)
    delta_v = delta_i / (8 * C * fs)

    # Rough efficiency model
    Rds = 40e-3
    P_cond = Iout**2 * Rds
    P_sw = 0.5 * Vin * Iout / fs
    eff = P / (P + P_cond + P_sw) * 100

    performances = [
        ("Duty Cycle (D)", f"{D:.3f}"),
        ("Inductor Ripple ΔiL (A)", f"{delta_i:.3f}"),
        ("Output Ripple ΔV (V)", f"{delta_v:.3f}"),
        ("Efficiency (%)", f"{eff:.2f}"),
        ("Output Current (A)", f"{Iout:.3f}")
    ]

    # Waveform plot
    t = np.linspace(0, 1/fs, 200)
    iL = Iout + (delta_i/2) * np.sin(2*np.pi*fs*t)

    fig, ax = plt.subplots()
    ax.plot(t * 1e6, iL)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Inductor Current (A)")
    ax.set_title("Buck Converter — Inductor Current Ripple")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return performances, buf



def optimize_boost(Vin, Vout, P):
    """
    Simple analytical BOOST converter evaluation.
    Returns:
        performances -> list
        plot -> PNG buffer
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import io

    # Load
    Ro = (Vout ** 2) / P
    Iout = Vout / Ro
    Iin = P / Vin

    # Duty cycle
    D = 1 - Vin / Vout

    # Component guesses
    fs = 100e3
    L = 100e-6
    C = 100e-6

    # Ripple calculations
    delta_i = Vin * D / (L * fs)
    delta_v = D * Iout / (C * fs)

    # Rough efficiency
    Rds = 40e-3
    P_cond = Iin**2 * Rds
    P_sw = 0.5 * Vout * Iin / fs
    eff = P / (P + P_cond + P_sw) * 100

    performances = [
        ("Duty Cycle (D)", f"{D:.3f}"),
        ("Inductor Ripple ΔiL (A)", f"{delta_i:.3f}"),
        ("Output Ripple ΔV (V)", f"{delta_v:.3f}"),
        ("Efficiency (%)", f"{eff:.2f}"),
        ("Input Current (A)", f"{Iin:.3f}")
    ]

    # Waveform
    t = np.linspace(0, 1/fs, 200)
    iL = Iin + (delta_i/2) * np.sin(2*np.pi*fs*t)

    fig, ax = plt.subplots()
    ax.plot(t * 1e6, iL)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Inductor Current (A)")
    ax.set_title("Boost Converter — Inductor Current Ripple")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return performances, buf
