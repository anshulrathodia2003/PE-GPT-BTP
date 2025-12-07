"""
@reference: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network, by Xinze Li, Fanfan Lin, et al.
@code-author: Xinze Li, Fanfan Lin
@github: https://github.com/XinzeLee/PE-GPT
         https://github.com/XinzeLee/PANN

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import torch
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from . import pann_train
from ..utils.pann_utils import evaluate
from .pann_net import PANN, EulerCell_DAB, WeightClamp



def train_data():
    """Load and prepare experimental data for PANN training"""
    
    # Check which files are missing
    missing = []
    if st.session_state.get('vp') is None:
        missing.append('vp')
    if st.session_state.get('vs') is None:
        missing.append('vs')
    if st.session_state.get('iL') is None:
        missing.append('iL')
    
    # If any files missing, show error and stop
    if missing:
        st.error(f"❌ Missing experimental data: **{', '.join(missing)}**")
        st.info("📁 Please upload CSV files using the sidebar **Training Data Upload** section")
        st.markdown("""
        Required files:
        1. **vp.csv** - Primary voltage waveform (batch_size × sequence_length)
        2. **vs.csv** - Secondary voltage waveform (batch_size × sequence_length)  
        3. **iL.csv** - Inductor current waveform (batch_size × sequence_length)
        
        **How to upload:**
        1. Go to sidebar → "Training Data Upload"
        2. Select file type from dropdown (vp, vs, or iL)
        3. Click "Browse files" and select your CSV
        4. Click "Confirm Upload"
        5. Repeat for all 3 files
        """)
        
        raise ValueError(f"Missing training data: {', '.join(missing)}")
    
    # Prepare data
    try:
        vp = st.session_state.vp
        vs = st.session_state.vs
        iL = st.session_state.iL
        
        # Validate shapes
        if vp.shape != vs.shape:
            raise ValueError(f"Shape mismatch: vp{vp.shape} vs vs{vs.shape}. They must have the same shape.")
        if vp.shape[0] != iL.shape[0]:
            raise ValueError(f"Batch size mismatch: vp/vs have {vp.shape[0]} samples but iL has {iL.shape[0]}")
        
        # Concatenate inputs and prepare states
        inputs = np.concatenate((vp.T[1:, :, None], vs.T[1:, :, None]), axis=-1)
        states = iL.T[1:, :, None]
        
        st.success(f"✅ Data loaded: {vp.shape[0]} samples × {vp.shape[1]} time steps")
        
        return inputs, states
        
    except Exception as e:
        st.error(f"❌ Error preparing data: {str(e)}")
        raise

# def train_data():
#     # check if all required files have been uploaded
#     if all(value is not None for value in (st.session_state.vp, 
#                                            st.session_state.vs, 
#                                            st.session_state.iL)):
#         inputs = np.concatenate((st.session_state.vp.T[1:, :, None], 
#                                  st.session_state.vs.T[1:, :, None]), axis=-1)
#         states = st.session_state.iL.T[1:, :, None]
#     return inputs, states

def train_dab():
    """Train PANN model for DAB with proper error handling"""
    
    # Step 1: Load training data with error handling
    try:
        inputs, states = train_data()
        
        # Validate data
        if inputs is None or states is None:
            raise ValueError("train_data() returned None - no experimental data loaded")
        
        if inputs.shape[0] == 0 or states.shape[0] == 0:
            raise ValueError("train_data() returned empty arrays - please upload CSV files")
        
        st.info(f"✓ Loaded {inputs.shape[0]} samples from experimental data")
        
    except Exception as e:
        st.error(f"❌ Failed to load experimental data: {str(e)}")
        st.info("Please upload the required CSV files: vp, vs, iL")
        raise ValueError(f"Data loading failed: {str(e)}")
    
    # Step 2: Train-test-val split
    train_pct, test_pct = 0.1, 0.45
    np.random.seed(888)
    idx = np.random.permutation(inputs.shape[0])
    
    train_inputs = inputs[idx[:round(train_pct * inputs.shape[0])]]
    test_inputs = inputs[idx[round(train_pct * inputs.shape[0]):
                             round((train_pct + test_pct) * inputs.shape[0])]]
    val_inputs = inputs[idx[round((train_pct + test_pct) * inputs.shape[0]):]]

    train_states = states[idx[:round(train_pct * inputs.shape[0])]]
    test_states = states[idx[round(train_pct * inputs.shape[0]):
                             round((train_pct + test_pct) * inputs.shape[0])]]
    val_states = states[idx[round((train_pct + test_pct) * inputs.shape[0]):]]
    
    # Step 3: Convert to torch Tensors
    train_inputs = torch.Tensor(train_inputs)
    test_inputs = torch.Tensor(test_inputs)
    val_inputs = torch.Tensor(val_inputs)
    train_states = torch.Tensor(train_states)
    test_states = torch.Tensor(test_states)
    val_states = torch.Tensor(val_states)
    
    # Step 4: Define data loader for training
    data_loader_train = DataLoader(
        dataset=pann_train.CustomDataset(train_states[:, :-1], train_inputs[:, 0:-1],
                                         train_states[:, 1:]),
        batch_size=10, shuffle=True, drop_last=False)
    
    # Step 5: Train the model
    test_data = (test_inputs, test_states)
    st.info("🔄 Training PANN model (50 epochs)...")
    
    best_pann_states = pann_train.train(
        st.session_state.model_pann, 
        st.session_state.clamper, 
        st.session_state.optimizer_pann, 
        data_loader_train, 
        test_data, 
        convert_to_mean=False, 
        epoch=50
    )
    
    # Step 6: Update model in session state
    scripted_EulerCell = torch.jit.script(EulerCell_DAB(dt, Lr, RL, n))
    best_pann = torch.jit.script(PANN(scripted_EulerCell))
    best_pann.load_state_dict(best_pann_states)
    st.session_state.model_pann = best_pann
    st.session_state.optimizer_pann = init_optim_pann_dab(best_pann)

    # Step 7: Evaluate on test and validation datasets
    test_pred, test_inputs, test_loss = evaluate(
        test_inputs[:, 0:], test_states, best_pann, 200, True
    )
    val_pred, val_inputs, val_loss = evaluate(
        val_inputs[:, 0:], val_states, best_pann, 200, True
    )
    
    # Step 8: Generate comparison plot
    selected_idx = 2
    plt.figure(figsize=(10, 6))
    plt.plot(val_pred[selected_idx, :, 0].detach().numpy(), 
             label='Predicted waveform', linewidth=2) 
    plt.plot(val_states[selected_idx, 1:, 0].detach().numpy(), 
             label='Experimental waveform', linewidth=2, alpha=0.7)
    plt.xlabel('Time steps')
    plt.ylabel('Current (A)')
    plt.title('PANN Training Results: Predicted vs Experimental')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    return buf, test_loss, val_loss

# def train_dab():
#     # training of the PANN model for DAB
#     inputs, states = train_data()
    
#     # train-test-val split
#     train_pct, test_pct = 0.1, 0.45 # train-test-val partition
#     np.random.seed(888)
#     idx = np.random.permutation(inputs.shape[0])
#     train_inputs = inputs[idx[:round(train_pct * inputs.shape[0])]]
#     test_inputs = inputs[idx[round(train_pct * inputs.shape[0]):
#                              round((train_pct + test_pct) * inputs.shape[0])]]
#     val_inputs = inputs[idx[round((train_pct + test_pct) * inputs.shape[0]):]]

#     train_states = states[idx[:round(train_pct * inputs.shape[0])]]
#     test_states = states[idx[round(train_pct * inputs.shape[0]):
#                              round((train_pct + test_pct) * inputs.shape[0])]]
#     val_states = states[idx[round((train_pct + test_pct) * inputs.shape[0]):]]
    
#     # convert type to torch Tensor types
#     train_inputs = torch.Tensor(train_inputs)
#     test_inputs = torch.Tensor(test_inputs)
#     val_inputs = torch.Tensor(val_inputs)
#     train_states = torch.Tensor(train_states)
#     test_states = torch.Tensor(test_states)
#     val_states = torch.Tensor(val_states)
    
#     # define data loader for training
#     data_loader_train = DataLoader(
#         dataset=pann_train.CustomDataset(train_states[:, :-1], train_inputs[:, 0:-1],
#                                          train_states[:, 1:]),
#         batch_size=10, shuffle=True, drop_last=False)
    
#     test_data = (test_inputs, test_states) # specify the test data
#     best_pann_states = pann_train.train(st.session_state.model_pann, st.session_state.clamper, 
#                                  st.session_state.optimizer_pann, data_loader_train, 
#                                  test_data, convert_to_mean=False, epoch=50) # start training
#     # print(best_pann.cell.Lr.item(), best_pann.cell.RL.item())
    
#     # update st.session_state
#     scripted_EulerCell = torch.jit.script(EulerCell_DAB(dt, Lr, RL, n)) # initialize a new model
#     best_pann = torch.jit.script(PANN(scripted_EulerCell))
#     best_pann.load_state_dict(best_pann_states) # load the best states
#     st.session_state.model_pann = best_pann
#     st.session_state.optimizer_pann = init_optim_pann_dab(best_pann)


#     # evaluate test and val datasets
#     test_pred, test_inputs, test_loss = evaluate(test_inputs[:, 0:], test_states, 
#                                                  best_pann, 200, True) # for collected dataset, Vin = 200 V
#     val_pred, val_inputs, val_loss = evaluate(val_inputs[:, 0:], val_states, 
#                                               best_pann, 200, True) # for collected dataset, Vin = 200 V
#     # print("Mean absolute errors are: ", test_loss, val_loss)
    
#     selected_idx = 2
#     plt.plot(val_pred[selected_idx, :, 0].detach().numpy(), 
#              label='Predicted waveform') 
#     plt.plot(val_states[selected_idx, 1:, 0].detach().numpy(), 
#              label='Experimental waveform')
#     plt.legend()
#     plt.show()
    
#     # save the file and transmit through io
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)

#     # return the buffer for plots
#     return buf, test_loss, val_loss


def init_optim_pann_dab(model_pann):
    param_list = ['cell.Lr']
    params = list(filter(lambda kv: kv[0] in param_list, model_pann.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in param_list, model_pann.named_parameters()))

    # define the optimizer of the PANN model
    optimizer_pann = torch.optim.Adam([{"params": [param[1] for param in params], "lr":5e-6},
                                       {"params": [base_param[1] for base_param in base_params]}], lr=1e-1)
    return optimizer_pann




################################################################################
from .pann_dab_vars import dt, Lr, RL, n


# Below uses pytorch and reveal all PANN codes
# define the PANN model for modeling DAB converters in time domain
scripted_EulerCell = torch.jit.script(EulerCell_DAB(dt, Lr, RL, n))
model_pann = torch.jit.script(PANN(scripted_EulerCell))


optimizer_pann = init_optim_pann_dab(model_pann)

clamper = WeightClamp(['cell.Lr', 'cell.RL', 'cell.n'],
                      [(10e-6, 200e-6), (1000e-6, 5e0), (0.85, 1.15)]) # clamp the weights of PA-RNN
    
# store these variables into st.session_state
st.session_state["model_pann"] = model_pann
st.session_state["optimizer_pann"] = optimizer_pann
st.session_state["clamper"] = clamper




# codes below utilize onnxruntime for PANN inference

# import onnxruntime
# import streamlit as st

# # initialize the ort inference session
# model_pann_onnx = onnxruntime.InferenceSession("core/model_zoo/pann_dab.onnx", 
#                                                 providers=["CPUExecutionProvider"])
# st.session_state["model_pann_onnx"] = model_pann_onnx