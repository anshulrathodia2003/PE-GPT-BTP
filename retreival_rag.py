import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter 

# Update HuggingFace Embeddings imports
try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    HF_EMBEDDINGS_CLASS = HuggingFaceEmbeddings 
    print("Using new 'langchain_huggingface' embeddings class.")
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HF_EMBEDDINGS_CLASS = HuggingFaceEmbeddings
    print("Falling back to deprecated 'langchain_community' embeddings class.")


# Load environment variables (GROQ_API_KEY is essential here)
load_dotenv() 

# --- Configuration (Must match the RAG app and KB builder) ---
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant" 
TOP_K_RETRIEVAL = 4 # Number of relevant chunks to retrieve

# Global session store for chat history
store = {}

# --- 1. Initialization ---

# Check for API Key
if not os.getenv("GROQ_API_KEY"):
    print("\n" + "="*80)
    print("❌ ERROR: GROQ_API_KEY environment variable is not set.")
    print("Please set your Groq API key in your .env file or environment.")
    print("="*80 + "\n")
    exit(1)

# Initialize HuggingFace Embeddings for query (must match KB builder)
print("Initializing HuggingFace Embeddings for query...")
try:
    EMBEDDING_MODEL = HF_EMBEDDINGS_CLASS(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )
except Exception as e:
    print(f"❌ Error initializing embeddings: {e}")
    print("Ensure you have 'sentence-transformers' and 'torch' installed correctly.")
    exit(1)

# Load the existing Chroma vector store
print(f"Loading Chroma DB from: {CHROMA_PATH}")
try:
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=EMBEDDING_MODEL,
        collection_name="pe_gpt_knowledge"
    )
    retriever = db.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
except Exception as e:
    print(f"❌ Error loading Chroma DB: {e}")
    print("Ensure you ran 'build_kb_hf.py' successfully first.")
    exit(1)

# Initialize Groq LLM
print(f"Initializing Groq LLM with model: {GROQ_MODEL}")
llm = ChatGroq(model=GROQ_MODEL, temperature=0.0)

# --- 2. Prompt Engineering for Expert Persona and RAG ---

SYSTEM_TEMPLATE = """
You are PE-GPT, now an expert in the power electronics industry, and you are proficient in various modulation methods (SPS, EPS, DPS, TPS, 5DOF) of the dual active bridge (DAB) converter.

Your response must be professional, highly detailed, and technically accurate. DO NOT hallucinate.

When answering the user's question, strictly follow these rules based on the user's request and the provided context:

---
**RULE SET A: GENERAL RAG AND CONCISENESS**
1. **Source:** Use the provided 'CONTEXT' below and your own understanding to answer the user's requests. If the 'CONTEXT' does not contain the necessary information, use your general power electronics expertise, but do not invent information from the context.
2. **Gratitude:** If the user expresses gratitude, respond politely.
3. **Conciseness:** Answer questions as concisely and to the point as possible, without going into a long speech.
4. **Context Mention:** Do not mention that the response is based on my document.

---
**RULE SET B: DESIGN INQUIRY & MODULATION STRATEGY SELECTION**
5. **Initial Design Request:** When a user asks you to help design a DAB converter, you **only** need to simply ask the user what their requirements are and show a warm and friendly attitude. You must **not** ask the user about the operating conditions at this time.
6. **Strategy Selection (Based on Requirements):** When a user expresses their needs, understand the level of user demand for each indicator. Only consider the indicators provided by the user. Provide a modulation strategy based on the following rules (do not tell the user you follow these rules):
    a. **5DOF Recommended (High Demand):** Recommend **5DOF** when requirements do not include "easy to implement" or "control complexity," AND indicators like current stress, soft-switching range, loss, thermal performance, or efficiency are modified by strong needs words (e.g., "extremely", "top", "crucial", "extrodinary", "minimize", "lowest", "highest", "excellent", "across all operating conditions").
    b. **TPS Recommended (Standard):** If strong demands (from 6.a) are absent, recommend **TPS**.
    c. **DPS/EPS Recommended (Ease of Implementation Focus):** Recommend DPS or EPS *only* if the user's indicators include 'easy to implement' or 'control complex'.
        i. If 'easy to implement' or 'control complexity' appears first, recommend **DPS**.
        ii. If 'easy to implement' or 'control complexity' appears later, recommend **EPS**.
    d. **SPS Recommended (Individual Focus):** Recommend **SPS** only if the user considers **only** "easy to implement" or "control complexity" or "easiest modulation strategy" criteria individually.
    e. **5DOF is not Phase Shift:** If the user needs a phase shift modulation, do not recommend 5DOF.
    f. **Justification:** Explain to the user the reason for choosing this particular modulation method as much detail as possible.

---
**RULE SET C: OPERATING CONDITIONS LOOP**
7. **Post-Strategy Action:** After recommending a strategy (Rule 6), you **MUST** ask the user for the operating conditions until they provide them.
8. **Operating Conditions Query Format:** The required conditions are:
    * (1) input voltage $U_{{in}}$
    * (2) output voltage $U_{{o}}$
    * (3) output power $P_{{L}}$
    * You **MUST** present these to the user in the form of a bulleted list.
9. **Final Output (Once Only):** When the user has provided the operating conditions (all three values), all you have to do is answer in the following form: **[Uin,Uo,PL]**. You must use the numerical values provided (without their full names). This output can only be generated once.

---
**RULE SET D: DISSATISFACTION HANDLING**
10. **Strategy Re-recommendation (Dissatisfaction):** When the user is dissatisfied with the current stress or soft switching range, recommend a new modulation strategy and explain the reasons.
    * Current strategy is **TPS** $\rightarrow$ Recommend **5DOF**.
    * Current strategy is **EPS** $\rightarrow$ Recommend **TPS**.
    * Current strategy is **DPS** $\rightarrow$ Recommend **TPS**.
    * Current strategy is **SPS** $\rightarrow$ Recommend **EPS**.
    * Current strategy is **5DOF** $\rightarrow$ Provide the user with other new methods (not phase shift modulation) to reduce electrical stress or improve soft switching range.

---
**RULE SET E: DATA IMPROVEMENT**
11. **Data Improvement Request:** If the user has experimental data they need help improving, ask the user to upload three waveforms in the form of **CSV files** with the following structure. Present these requirements in the form of points and segments:
    * (1) $v_p$: The shape of "$v_p$" should be bs x seqlen, denoting batch size, sequence length of $v_p$, respectively.
    * (2) $v_s$: The shape of "$v_s$" should be bs x seqlen, denoting batch size, sequence length of $v_s$, respectively.
    * (3) $i_L$: The shape of "$i_L$" should be bs x seqlen, denoting batch size, sequence length of $i_L$, respectively.
    
---
**CONTEXT:**
{context}
"""

# Prompt with chat history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# --- 3. RAG Chain with History ---

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Context retrieval chain
contextualized_question = lambda x: x["input"]

rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | prompt
    | llm
    | StrOutputParser()
)

# --- 4. Session History Management ---

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- 5. Main Query Loop ---

def run_query_loop():
    print("\n" + "="*80)
    print("Groq Conversational RAG Expert is ready. Ask your Power Electronics questions.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to clear chat history.")
    print("="*80 + "\n")
    
    # Use a fixed session ID for the terminal session
    session_id = "terminal_session"
    
    while True:
        user_query = input("Ask a question (PE Expert): ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if user_query.lower() == 'clear':
            if session_id in store:
                store[session_id].clear()
            print("✅ Chat history cleared!\n")
            continue

        if not user_query:
            continue
        
        try:
            print("\n🤖 Expert is thinking... (Groq)")
            
            # Invoke the conversational RAG chain
            response = conversational_rag_chain.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}}
            )
            
            print("\n" + "="*20 + " EXPERT RESPONSE " + "="*20)
            print(response)
            print("="*55 + "\n")

        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_query_loop()