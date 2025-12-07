"""
@functionality
    Topology detection agent to identify power converter type from user query

@code-author: Modified for PE-GPT Groq Edition
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class TopologyAgent:
    """
    Intelligent topology detection using conversation understanding.
    NO hardcoded voltage rules, NO keyword matching - pure LLM reasoning.
    """
    
    def __init__(self, model="llama-3.1-70b-versatile"):  # Use smarter model
        self.llm = ChatGroq(model=model, temperature=0.2)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert power electronics engineer analyzing conversations to determine which converter topology the user is discussing.

Your job: Understand the CONVERSATION FLOW and DESIGN CONTEXT, then output the topology.

ALLOWED OUTPUTS (lowercase only):
- dab
- buck  
- boost
- buckboost
- flyback
- unknown

REASONING APPROACH (think like an engineer):

1. **Conversation Continuity** (MOST IMPORTANT):
   - If previous messages discuss a specific topology, assume user is continuing that discussion
   - Example: 
     * Previous: "Design a DAB with TPS modulation"
     * Current: "I have experimental data to improve accuracy"
     * Reasoning: User is clearly continuing DAB discussion → output: dab

2. **Design Goals & Features** (semantic understanding):
   - "Phase shift modulation", "TPS", "SPS", "DPS", "EPS", "5DOF" → ONLY used in DAB
   - "Soft switching range", "ZVS", "ZCS" with modulation → DAB
   - "PANN model", "experimental waveforms", "retrain" → DAB (only topology using neural networks)
   - "Isolated", "bidirectional" → DAB or flyback
   - "Simple step-down", "non-isolated" → buck or boost
   - "Transformer turns ratio" → DAB or flyback

3. **Ignore Voltage Values** (DO NOT use Vin vs Vout):
   - 400V→48V could be: DAB, buck, or flyback
   - 24V→48V could be: DAB, boost, or flyback
   - Voltage alone tells you NOTHING - focus on features and context

4. **Training/Experimental Data:**
   - "Train model", "experimental data", "improve accuracy", "PANN" → DAB
   - Buck/Boost use analytical models, not neural networks

5. **If Truly Ambiguous:**
   - No previous context + no distinctive features → output: unknown
   - Let the system ask the user for clarification

THINK STEP BY STEP:
1. What topology was discussed in previous messages?
2. What design features/goals are mentioned?
3. Is user continuing previous discussion or starting new?
4. What's the most logical topology?

OUTPUT FORMAT:
- ONLY output the topology: dab, buck, boost, buckboost, flyback, or unknown
- NO explanations in output
- Just the lowercase topology name

Examples:

Example 1:
Previous: "Let's design a DAB with optimal soft switching"
Current: "The operating conditions are 400V input, 48V output, 2kW"
Reasoning: User is providing specs for the DAB they mentioned → output: dab

Example 2:
Previous: "I want to design a 400V to 48V converter with TPS modulation"
Current: "What if I change the frequency to 150kHz?"
Reasoning: TPS is DAB-specific, user is modifying DAB design → output: dab

Example 3:
Previous: [no previous conversation]
Current: "I have experimental data to improve my design"
Reasoning: No context about which design → output: unknown

Example 4:
Previous: "Design buck converter 400V, 48V, 1000W" [shows results]
Current: "I want to try a different topology now. Can you help with phase shift modulation?"
Reasoning: User explicitly switching + "phase shift" is DAB → output: dab

Example 5:
Previous: [DAB conversation with TPS modulation]
Current: "Can I use the experimental waveforms to retrain?"
Reasoning: Retraining implies PANN, which is DAB-only → output: dab
"""),
            ("human", """Previous conversation (last 5 messages):
{context}

Current user query: {query}

Think: What topology is the user discussing?
Output:""")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def detect(self, query: str, conversation_context: list = None, stored_topology: str = None) -> str:
        """
        Detect topology using intelligent conversation reasoning
        
        Args:
            query: Current user query
            conversation_context: List of previous messages
            stored_topology: Currently stored topology (if any)
        
        Returns:
            Topology string
        """
        
        # Build rich context from conversation
        if conversation_context and len(conversation_context) > 0:
            # Get last 5 messages for better context
            recent_messages = conversation_context[-5:]
            context_lines = []
            
            for msg in recent_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:200]  # More context
                context_lines.append(f"{role}: {content}")
            
            context_str = "\n".join(context_lines)
        else:
            context_str = "[No previous conversation - this is the first message]"
        
        # Add stored topology info if exists
        if stored_topology and stored_topology != 'unknown':
            context_str += f"\n\nNOTE: System currently has stored topology: {stored_topology}"
        
        try:
            # Let LLM reason about topology
            result = self.chain.invoke({
                "query": query,
                "context": context_str
            })
            
            # Clean output
            topo = result.strip().lower()
            
            # Remove any explanation text (sometimes LLM adds it)
            valid_topologies = ['dab', 'buck', 'boost', 'buckboost', 'flyback', 'unknown']
            for valid_topo in valid_topologies:
                if valid_topo in topo:
                    return valid_topo
            
            # If LLM output is invalid, use stored topology if available
            if stored_topology and stored_topology != 'unknown':
                print(f"LLM returned invalid '{topo}', using stored: {stored_topology}")
                return stored_topology
            
            return 'unknown'
                
        except Exception as e:
            print(f"LLM topology detection failed: {e}")
            
            # Emergency fallback: use stored topology
            if stored_topology and stored_topology != 'unknown':
                return stored_topology
            
            return 'unknown'


# Helper function for testing
def test_topology_agent():
    """Test function to validate topology detection"""
    agent = TopologyAgent()
    
    test_cases = [
        ("Design a DAB converter with 200V input", "dab"),
        ("I need a buck converter for 12V to 5V", "buck"),
        ("Step up from 24V to 48V, 500W", "boost"),
        ("Phase shift modulation strategy", "dab"),
        ("What's the weather today?", "unknown"),
        ("Input is 400V, output 48V, power 1000W", "buck"),
        ("Boost converter analysis", "boost"),
    ]
    
    print("Testing Topology Agent:")
    print("-" * 50)
    
    for query, expected in test_cases:
        detected = agent.classify(query)
        status = "✅" if detected == expected else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected}, Got: {detected}\n")


if __name__ == "__main__":
    test_topology_agent()