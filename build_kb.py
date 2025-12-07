import os
from dotenv import load_dotenv

# --- RAG/LangChain/LLM Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# DEPRECATION FIX: Using the new dedicated package for HuggingFaceEmbeddings
try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Fallback to older community import if the new one isn't installed
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv() 

# --- Configuration ---
KNOWLEDGE_BASE_PATH = "kb" 
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Dependency Check for pypdf ---
try:
    import pypdf
except ImportError:
    print("\n" + "="*80)
    print("❌ DEPENDENCY ERROR:")
    print("The PDF loader requires the 'pypdf' package.")
    print("Please install it: >>> pip install pypdf")
    print("="*80 + "\n")
    exit(1)

# --- Dependency Check for HuggingFace Embeddings ---
try:
    import sentence_transformers
except ImportError:
    print("\n" + "="*80)
    print("❌ DEPENDENCY ERROR:")
    print("The embedding model requires the 'sentence-transformers' package.")
    print("Please install it: >>> pip install sentence-transformers")
    print("="*80 + "\n")
    exit(1)


# --- Setup ---
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)

print("Initializing HuggingFace Embeddings...")
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'} 
)

# Initialize the text splitter
text_processor = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

# --- Initialize Chroma DB ---
print(f"Loading/Creating Chroma DB at: {CHROMA_PATH}")
# When persist_directory is set, Chroma automatically saves upon adding documents.
DOCUMENT_VECTOR_DB = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=EMBEDDING_MODEL,
    collection_name="pe_gpt_knowledge" 
)

# --- Indexing Loop ---
pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_PATH) if f.endswith('.pdf')]

if not pdf_files:
    print(f"⚠️ Warning: No PDFs found in the directory: {KNOWLEDGE_BASE_PATH}")
    print("Please place your PDF documents in the 'kb' folder.")
else:
    print(f"Found {len(pdf_files)} PDFs. Starting indexing...")

for filename in pdf_files:
    file_path = os.path.join(KNOWLEDGE_BASE_PATH, filename)
    print(f"Processing: {filename}")
    
    try:
        # Load, Chunk, and Add to Index
        document_loader = PyPDFLoader(file_path)
        raw_documents = document_loader.load()
        processed_chunks = text_processor.split_documents(raw_documents)
        
        # Add chunks to the vector store (persistence happens here automatically)
        DOCUMENT_VECTOR_DB.add_documents(processed_chunks)
        print(f"  -> Added {len(processed_chunks)} chunks.")

    except Exception as e:
        print(f"  ❌ ERROR processing {filename}: {e}")


# --- Finalize ---
print("✅ All existing documents indexed and saved to persistent Chroma DB.")
print("You can now run the Streamlit app.")
