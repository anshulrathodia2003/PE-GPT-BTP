"""
Conversation Intelligence Agent
Handles ALL conversation understanding with LLM reasoning - NO hardcoded keywords
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
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

class ConversationIntelligenceAgent:
    """
    Single LLM agent that understands EVERYTHING about the conversation
    """
    
    def __init__(self):
        
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert power electronics conversation analyst.

Your job: Analyze the conversation and user intent, then output a JSON with your reasoning.

ANALYZE:
1. What converter topology is being discussed? (dab, buck, boost, buckboost, flyback, unknown)
2. Is this a continuation of previous discussion or a new topic?
3. What is the user trying to do? (design, modify, ask_question, train_model, simulate, upload_data, etc.)

TOPOLOGY REASONING:
- DAB: Phase-shift modulation (TPS/SPS/DPS/EPS/5DOF), soft switching, PANN model, experimental data
- Buck: Step-down, simple PWM, non-isolated
- Boost: Step-up, simple PWM, non-isolated
- Buck-Boost: Can step up or down, inverting or non-inverting
- Flyback: Isolated, transformer-based SMPS

IMPORTANT RULES:
1. Conversation Continuity: If previous messages discuss a topology, assume user is continuing UNLESS they explicitly switch
2. No Voltage Rules: Ignore Vin vs Vout - DAB can step down, Boost can be used in various ranges
3. Semantic Understanding: "experimental data", "train model", "PANN" → DAB (only topology using neural networks)
4. Modulation = DAB: Any mention of TPS/SPS/DPS/EPS/5DOF means DAB
5. Trust Context: Previous topology + no explicit switch = continue that topology
6. If the user expresses that they have measured or experimental data and wants to improve, upload, or provide data, set intent to "upload_data".
             
OUTPUT FORMAT - MUST be valid JSON with this structure:
{{"topology": "dab or buck or boost or buckboost or flyback or unknown", "confidence": "high or medium or low", "is_continuation": true or false, "intent": "design or modify or question or train or simulate or upload_data or other", "reasoning": "your explanation"}}

Examples:

Input: Design DAB with TPS modulation
Output: {{"topology": "dab", "confidence": "high", "is_continuation": false, "intent": "design", "reasoning": "User explicitly mentioned DAB and TPS"}}

Input: Previous conversation about DAB. Current query: "I have experimental data"
Output: {{"topology": "dab", "confidence": "high", "is_continuation": true, "intent": "train", "reasoning": "Continuing DAB discussion, experimental data implies PANN training"}}

Input: "Now I have some experimental data. Can you help me improve the accuracy?"
Output: {{"topology": "dab", "confidence": "high", "is_continuation": true, "intent": "upload_data", ...}}             
             
CRITICAL: Output ONLY the JSON object. No text before or after."""),
            ("human", """Previous conversation (last 5 messages):
{context}

Currently stored topology: {stored_topology}

Current user query: {query}

Analyze and output JSON:""")
        ])
        
        # self.chain = self.prompt | self.llm | StrOutputParser()
    
    def analyze(self, query: str, conversation_context: list = None, stored_topology: str = "unknown") -> dict:
        """
        Analyze conversation with pure LLM reasoning, rotating model every call
        Returns: dict with: topology, confidence, is_continuation, intent, reasoning
        """
        # Build context as before
        if conversation_context and len(conversation_context) > 0:
            recent = conversation_context[-5:]
            context_str = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}"
                for msg in recent
            ])
        else:
            context_str = "[First message - no previous context]"
        
        try:
            # ---- Round robin model selection for EVERY call ----
            llm = ChatGroq(model=get_next_model(), temperature=0.2)
            chain = self.prompt | llm | StrOutputParser()
            # ----------------------------------------------------
            
            result = chain.invoke({
                "query": query,
                "context": context_str,
                "stored_topology": stored_topology if stored_topology else "unknown"
            })
            
            # DEBUG: Show what LLM returned
            st.write(f"🔍 **LLM Raw Output:**")
            st.code(result, language="json")
            
            # Parse JSON
            analysis = self._parse_json(result)
            if analysis and 'topology' in analysis:
                st.success(f"✅ **Detected:** {analysis['topology'].upper()} | {analysis.get('reasoning', '')}")
                return analysis
            else:
                st.error(f"❌ **JSON Parse Failed** - Using stored topology: {stored_topology}")
                return {
                    "topology": stored_topology if stored_topology != "unknown" else "unknown",
                    "confidence": "low",
                    "is_continuation": True,
                    "intent": "other",
                    "reasoning": f"JSON parsing failed, using stored topology"
                }
        except Exception as e:
            st.error(f"❌ **LLM Error:** {str(e)}")
            return {
                "topology": stored_topology if stored_topology != "unknown" else "unknown",
                "confidence": "low",
                "is_continuation": True,
                "intent": "other",
                "reasoning": f"Error: {str(e)}, defaulting to stored topology"
            }
    
    def _parse_json(self, text: str) -> dict:
        """
        Parse JSON from LLM response - multiple methods
        """
        
        # Clean the text
        text = text.strip()
        
        # Method 1: Direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Find JSON object in text
        try:
            match = re.search(r'\{[^{}]*"topology"[^{}]*\}', text, re.DOTALL)
            if match:
                json_str = match.group()
                return json.loads(json_str)
        except:
            pass
        
        # Method 3: Extract between first { and last }
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except:
            pass
        
        # Failed all methods
        return None
