# PE-GPT: LLM-Powered Intelligent Design Assistant for Power Electronics Converters

PE-GPT (Power Electronics GPT) is an intelligent, conversational design assistant for designing and optimizing power electronic converters using natural language.  
This project re-implements the PE-GPT architecture using **Groq, LLaMA, LangChain, ChromaDB, Scipy, Streamlit, and PyTorch** — completely open-source.

---

## ✨ Features

### 1. Pure LLM-Based Conversation Intelligence
- Detects converter topology from user text (Buck, Boost, DAB).
- Performs intent recognition for design, optimization, simulation, or theory.
- Maintains multi-turn conversation memory.
- Supports What-If Analysis (dynamic parameter updates).

### 2. Multi-Topology Design & Optimization
- Supports: **Buck, Boost, Dual Active Bridge (DAB)**.
- Automatically sizes components and predicts performance.
- Optimization suggests best modulation:
  - SPS, DPS, EPS, TPS, 5DOF.
- Objective functions: minimum RMS current, improved ZVS, peak current reduction.

### 3. Physics-Informed Neural Network (PANN)
- Predicts converter waveforms ~1000× faster than PLECS.
- Physics-based model implemented in PyTorch.
- Can be trained on user experimental data (real parasitic modelling).
- (Currently in prototype/development stage.)

### 4. PLECS Simulation Integration
- Uses XML-RPC to control PLECS.
- Automatically sets parameters, runs simulations, retrieves waveforms.
- Provides high-fidelity validation of converter design.

---

## 🛠 Technology Stack

| Layer | Technology | Purpose |
|------|------------|---------|
| LLM Backend | Groq + LLaMA 3.1 (70B) | Reasoning & conversation |
| Orchestration | LangChain | Agent routing, tools, workflow |
| Vector Database | ChromaDB | RAG grounding with PE knowledge |
| Optimization | Scipy | Modulation + sizing optimization |
| Simulation | PLECS (XML-RPC) | High-fidelity verification |
| Neural Model | PyTorch | Physics-informed neural network |
| Frontend | Streamlit | User interface |


---

## 🧠 Workflow Overview

### Step 1 — Task Classification
Every query is classified into:

| Task ID | Description |
|--------|-------------|
| 1 | Design Evaluation |
| 2 | Modulation Recommendation |
| 3 | Simulation Verification |
| 4 | PANN Training |
| 5 | Theory / General Question |

### Step 2 — Routing to Modules

#### ✔ Design → Optimization Module  
Example: “Design DAB 400V to 48V at 2 kW”

#### ✔ Modulation → Modulation Strategy Module  
Example: “Which modulation gives lowest RMS current?”

#### ✔ Simulation → PLECS  
Example: “Verify this design in PLECS”

#### ✔ Theory → RAG  
Example: “Explain ZVS simply”

---

## 🔬 Core Components

### 1. Task Classification Agent
Uses LLaMA to detect user intent.

### 2. Optimization Engine
- Scipy-based nonlinear optimization.
- Supports SPS/DPS/EPS/TPS/5DOF comparison.
- Calculates RMS, ZVS regions, peak current, etc.

### 3. PANN Module
- Physics-informed deep neural network.
- Predicts waveforms far faster than circuit solvers.
- Adaptable using user’s lab data.

### 4. PLECS Integration
- Automatic simulation execution.
- Direct parameter control.
- Real waveforms for validation.

---

## 📅 Future Work

- Add support for resonant converters (LLC, CLLC, Flyback, etc.).
- Add automatic control design (PID, state-space, digital control).
- Add thermal & PCB layout assistance.
- Multilingual support.
- Enhance PANN generalization.

---

## 🧑‍💻 Contributors

- Anshul Rathodia 
- Angela Singhal  


