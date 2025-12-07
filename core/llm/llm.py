"""
@functionality
    LLM initialization and RAG setup for PE-GPT using Groq and LangChain with ChromaDB

@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, and Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT

GROQ EDITION - Converted from OpenAI to Groq + Open Source

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, 
                 Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import os
os.environ['GROQ_API_KEY'] = "your_groq_api"
import streamlit as st
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# LangChain and Groq imports
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# HuggingFace Embeddings with fallback
try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    HF_EMBEDDINGS_CLASS = HuggingFaceEmbeddings 
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HF_EMBEDDINGS_CLASS = HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

# Global session store for chat history
store = {}


# =====================================================================
# CONFIGURATION
# =====================================================================

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 4


# =====================================================================
# GROQ INITIALIZATION
# =====================================================================

def groq_init(api_key: Optional[str] = None, model: Optional[str] = None) -> ChatGroq:
    """
    Initialize Groq LLM client
    
    Args:
        api_key: Groq API key (if None, reads from environment or session state)
        model: Model name (if None, reads from session state or uses default)
    
    Returns:
        ChatGroq: Initialized Groq chat model
    """
    
    # Get API key from multiple sources
    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY") or st.session_state.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("❌ Groq API key not found! Please set it in the sidebar.")
        st.stop()
    
    # Set API key in environment
    os.environ["GROQ_API_KEY"] = api_key
    
    # Model selection
    if model is None:
        model = st.session_state.get("llm_model", "llama-3.1-8b-instant")
    
    # Store model in session state
    st.session_state["groq_model"] = model
    
    # Initialize Groq client
    llm = ChatGroq(
        model=model,
        temperature=0.0,
        api_key=api_key
    )
    
    return llm


# =====================================================================
# RAG SETUP WITH CHROMADB
# =====================================================================

@st.cache_resource(show_spinner=False)
def rag_load(
    database_folder: str, 
    llm_model: str,
    temperature: Optional[float] = None,
    chunk_size: Optional[int] = None,
    system_prompt: Optional[str] = None
):
    """
    Load documents and create Retrieval-Augmented Generation (RAG) chain using Groq + ChromaDB
    
    Args:
        database_folder: Path to ChromaDB folder (e.g., "chroma_db")
        llm_model: Groq model name
        temperature: LLM temperature (default: 0.0)
        chunk_size: Not used (kept for compatibility)
        system_prompt: System prompt for the LLM
    
    Returns:
        Conversational RAG chain with history
    """
    
    # Set defaults
    if temperature is None:
        temperature = 0.0
    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant specialized in power electronics design."
    
    with st.spinner(text="Loading ChromaDB and building RAG chain – hang tight! This should take a few seconds."):
        
        try:
            # Initialize HuggingFace Embeddings
            embedding_model = HF_EMBEDDINGS_CLASS(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'} 
            )
            
            # Load the existing Chroma vector store
            db = Chroma(
                persist_directory=database_folder,
                embedding_function=embedding_model,
                collection_name="pe_gpt_knowledge"
            )
            
            retriever = db.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
            
            # Initialize Groq LLM
            llm = ChatGroq(model=llm_model, temperature=temperature)
            
            # Create RAG prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\nCONTEXT:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            # Helper function to format retrieved documents
            def format_docs(docs):
                return "\n\n---\n\n".join(doc.page_content for doc in docs)
            
            # Contextualized question
            contextualized_question = lambda x: x["input"]
            
            # Build RAG chain
            rag_chain = (
                RunnablePassthrough.assign(
                    context=contextualized_question | retriever | format_docs
                )
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Wrap with message history
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )
            
            st.success(f"✅ ChromaDB loaded successfully from {database_folder}")
            
            return conversational_rag_chain
            
        except Exception as e:
            st.error(f"❌ Error loading ChromaDB: {str(e)}")
            st.warning("Ensure you have built the knowledge base using build_kb_hf.py first.")
            return None


# =====================================================================
# CHAT ENGINE CREATION
# =====================================================================

def create_chat_engine(
    llm_model: str,
    system_prompt: str,
    temperature: float = 0.0,
    use_rag: bool = False,
    database_folder: Optional[str] = None,
    session_id: str = "default"
):
    """
    Create a chat engine with or without RAG
    
    Args:
        llm_model: Groq model name
        system_prompt: System prompt for the LLM
        temperature: LLM temperature
        use_rag: Whether to use RAG with ChromaDB
        database_folder: Path to ChromaDB folder (required if use_rag=True)
        session_id: Session ID for conversation history
    
    Returns:
        Chat engine (either RAG chain or simple LLM chain)
    """
    
    if use_rag and database_folder:
        # Create RAG-enabled chat engine with ChromaDB
        conversational_rag_chain = rag_load(
            database_folder=database_folder,
            llm_model=llm_model,
            temperature=temperature,
            system_prompt=system_prompt
        )
        return conversational_rag_chain
    else:
        # Create simple chat engine without RAG
        llm = ChatGroq(
            model=llm_model,
            temperature=temperature
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        # Wrap with message history
        conversational_chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        return conversational_chain


# =====================================================================
# SESSION HISTORY MANAGEMENT
# =====================================================================

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieve or create chat history for a session
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        BaseChatMessageHistory: Chat history for the session
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_msg_history():
    """
    Get the message history in LangChain format from Streamlit session state
    
    Returns:
        List of messages formatted for LangChain
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    messages_history = []
    
    for msg in st.session_state.get("messages", []):
        if msg["role"] == "user":
            messages_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_history.append(AIMessage(content=msg["content"]))
    
    return messages_history


def clear_session_history(session_id: str = "default"):
    """
    Clear chat history for a specific session
    
    Args:
        session_id: Session ID to clear
    """
    if session_id in store:
        store[session_id].clear()
        st.success(f"✅ Chat history cleared for session: {session_id}")


# =====================================================================
# SYSTEM PROMPTS
# =====================================================================

def get_system_prompt(task_type: str = "general") -> str:
    """
    Get system prompt for different task types
    
    Args:
        task_type: Type of task (general, modulation, evaluation, etc.)
    
    Returns:
        System prompt string
    """
    
    prompts = {
        "general": """You are PE-GPT, an expert in power electronics industry, proficient in various modulation methods (SPS, EPS, DPS, TPS, 5DOF) of the dual active bridge (DAB) converter.

Your response must be professional, highly detailed, and technically accurate. DO NOT hallucinate.

When answering the user's question, use the provided context and your expertise.""",
        
        "modulation": """You are PE-GPT, an expert in power electronics modulation strategies for DAB converters.

You understand the trade-offs between different modulation methods:
- SPS: Simple, easy to implement
- DPS/EPS: Good balance of performance and complexity
- TPS: Excellent performance, moderate complexity
- 5DOF: Best performance across all metrics, higher complexity

Provide detailed, technical recommendations based on user requirements.""",
        
        "evaluation": """You are PE-GPT, an expert in evaluating power electronics converter performance.

You can analyze:
- Current stress and RMS values
- Soft-switching ranges (ZVS/ZCS)
- Power losses and efficiency
- Thermal performance

Provide quantitative analysis and comparisons."""
    }
    
    return prompts.get(task_type, prompts["general"])


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def check_groq_api_key() -> bool:
    """
    Check if Groq API key is available
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    api_key = os.environ.get("GROQ_API_KEY") or st.session_state.get("GROQ_API_KEY")
    return api_key is not None and api_key != ""


def format_chat_history(messages: list) -> str:
    """
    Format chat history for display
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        Formatted string
    """
    formatted = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n\n".join(formatted)