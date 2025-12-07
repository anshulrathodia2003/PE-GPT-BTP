import streamlit as st
from dotenv import load_dotenv
import os
from io import BytesIO

# --- RAG/LangChain/LLM Imports ---
# Assumes these packages are installed: 
# google-genai, groq, pypdf, langchain, chromadb, langchain-community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq

# Load environment variables (API Keys will be configured here)
load_dotenv() 

# --- CONFIGURATION ---
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "models/text-embedding-004" # Strongest general-purpose model for embeddings
LLM_MODEL = "llama3-8b-8192" # Fast Groq model

# --- API KEY CHECK ---
# Ensure these variables are read once
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")

# --- Helper Function for Truncation ---
def truncate_chat_name(name, length=25):
    """Truncate chat name for display in the sidebar."""
    if len(name) > length:
        return name[:length-3] + "..."
    return name

# --- RAG Core Functions ---

@st.cache_resource
def get_embedding_function():
    """Initializes and returns the Gemini Embedding function."""
    # Check for the key before initializing
    if not GEMINI_API_KEY:
        st.error("Embedding function unavailable: GEMINI_API_KEY is missing.")
        return None
        
    # Explicitly pass the API key to ensure it is used, bypassing default credential lookup
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, 
        api_key=GEMINI_API_KEY
    )

def create_vector_db(files):
    """Loads documents, splits them, and stores embeddings in ChromaDB."""
    embeddings = get_embedding_function()
    if not embeddings:
        return

    documents = []
    
    # 1. Load Documents
    for uploaded_file in files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Read file content into a byte stream
        bytes_data = uploaded_file.getvalue()
        file_stream = BytesIO(bytes_data)

        if file_extension == 'pdf':
            # Use PyPDFLoader for PDF files
            loader = PyPDFLoader(file_stream)
            documents.extend(loader.load())
        elif file_extension in ['txt', 'md']:
            # Handle text/markdown files directly
            text_content = file_stream.read().decode('utf-8')
            # LangChain Document structure requires 'page_content' and 'metadata'
            documents.append(
                {"page_content": text_content, "metadata": {"source": uploaded_file.name}}
            )
        else:
            # Skip unsupported files, though Streamlit filters them
            continue

    if not documents:
        st.warning("No supported documents found to process.")
        return

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Create Vector Store (ChromaDB)
    # Using the same persistent directory ensures we update/append to the existing DB
    Chroma.from_documents(
        chunks,
        embeddings, # Pass the initialized embeddings function
        persist_directory=CHROMA_PATH
    )
    
    return len(chunks)

def get_rag_response(query: str):
    """Performs retrieval and generates a response using Groq."""
    if not GROQ_API_KEY or not GEMINI_API_KEY:
        return "Error: API keys are missing. Cannot run RAG pipeline.", []
        
    try:
        # Initialize Groq client
        groq_client = Groq(api_key=GROQ_API_KEY)
        embeddings = get_embedding_function()

        if not embeddings:
            return "Error: Embedding function is not available due to missing key.", []

        # 1. Retrieve Context from ChromaDB
        vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        
        # Get the top 4 most relevant documents
        retriever = vector_db.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        
        if not docs:
            # If no context is retrieved (e.g., empty DB), prompt the LLM to answer generally
            context_text = "No document context was retrieved from the database."
            system_instruction = "You are a helpful assistant. Since no context was found, please answer the user's question to the best of your general knowledge."
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
            system_instruction = f"""
            You are a highly efficient RAG assistant powered by Groq and Gemini.
            Answer the user's question based ONLY on the provided context. 
            If the answer is not found in the context, state clearly that the context does not contain the answer.

            CONTEXT:
            ---
            {context_text}
            ---
            """
        
        # 2. Build Prompt for Groq
        prompt_template = f"QUESTION: {query}"

        # 3. Generate Response using Groq LLM
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.1,
        )
        
        return response.choices[0].message.content, docs
        
    except Exception as e:
        return f"An error occurred during RAG generation: {e}", []


# --- 1. Page Configuration (from user's code) ---
st.set_page_config(
    page_title="Hybrid RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Sidebar for Conversation History (from user's code) ---
with st.sidebar:
    
    # --- Top Section (New Chat) ---
    # New Chat Button (Small, Icon, Subtle type)
    if st.button("New Chat", type="secondary", icon=None, use_container_width=True):
        st.session_state.messages = []
        st.info("New chat started.")

    st.markdown("---") # Separation as requested
    
    # --- Middle Section (History) ---
    st.subheader("Chats")
    
    # Placeholder for list of past chats
    if "past_chats" not in st.session_state:
        st.session_state.past_chats = [
            "RAG Intro", 
            "Embedding Test Run", 
            "Groq Speed Demo",
            "A Very Long Chat Name That Needs to Be Truncated Neatly"
        ]

    # Display placeholder chat names
    for i, chat_name in enumerate(st.session_state.past_chats):
        display_name = truncate_chat_name(chat_name)
        
        if st.button(display_name, key=f"chat_{i}", use_container_width=True, icon=None):
            st.info(f"Loaded placeholder conversation: **{chat_name}**")

    # --- Footer Section (Profile/Account) ---
    
    # Use native Streamlit spacer widgets to push the footer content visually down
    for _ in range(12): # Adding significant vertical space
        st.empty() 

    st.markdown("---") 
    
    # Profile/Account Placeholder
    with st.container(border=False):
        # Using a column layout for the icon and text
        col_icon, col_text = st.columns([0.5, 3])
        
        with col_icon:
            # Using your original shortcode as requested
            st.markdown(":bust_in_silhouette:") 
            
        with col_text:
            # Placeholder user info
            st.markdown("**User Name**")
            st.markdown("Free Tier Account")


# --- 3. Main Title and Document Upload (Top Right) ---
st.title("📚 Hybrid RAG Chatbot (Gemini + Chroma + Groq)")

# Use columns to place the upload popover neatly on the right
col1, col2 = st.columns([5, 1])

with col2:
    # A small button that triggers a popover for document upload
    with st.popover("📤 Update KB", use_container_width=True):
        st.markdown("### ⚙️ Knowledge Base Uploader")
        st.markdown("_Upload documents to update the RAG knowledge base._")
        
        # Placeholder for file upload widget
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, TXT, MD)",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        # Button to trigger the embedding and storage process
        if st.button("Build/Update Vector DB", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please select files before updating the database.")
            elif not GEMINI_API_KEY:
                st.error("Cannot proceed: GEMINI_API_KEY is missing.")
            else:
                with st.spinner(f"Processing {len(uploaded_files)} files and building vector store..."):
                    chunk_count = create_vector_db(uploaded_files)
                
                if chunk_count is not None:
                    st.success(f"Knowledge Base successfully updated! Processed {chunk_count} chunks.")


with col1:
    st.caption("Ready to answer questions about your documents using a powerful RAG pipeline.")
    st.markdown("---") # Visual separator

# --- 4. Main Chat Interface (ChatGPT-like structure) ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a friendly initial greeting message
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your Hybrid RAG assistant. Upload your documents to the Knowledge Base (top right) to get started, or ask a general question."})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    # Use the native Streamlit chat elements for the ChatGPT look
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask your question here..."):
    
    if not GROQ_API_KEY or not GEMINI_API_KEY:
        # Display the user message, then the error message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        error_content = "Error: Please set both `GEMINI_API_KEY` and `GROQ_API_KEY` in your environment."
        st.session_state.messages.append({"role": "assistant", "content": error_content})
        with st.chat_message("assistant"): st.markdown(error_content)
        st.stop()
    
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching context and generating response with Groq..."):
            
            rag_response, docs = get_rag_response(prompt)
            st.markdown(rag_response)
            
            # Show sources for transparency
            if docs and "An error occurred" not in rag_response:
                with st.expander("📚 Sources Used"):
                    for i, doc in enumerate(docs):
                        source_name = doc.metadata.get('source', 'Unknown Source')
                        # For PDF pages, try to extract the page number
                        page_number = doc.metadata.get('page', 'N/A')
                        st.markdown(f"**Chunk {i+1}** (Source: `{source_name}`, Page: {page_number}):")
                        # Show a snippet of the chunk content
                        st.code(doc.page_content[:250] + "...", language='text')

            
        # 3. Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": rag_response})

st.markdown("---")
st.caption("RAG functionality is now fully integrated using Gemini Embeddings, ChromaDB, and Groq!")
