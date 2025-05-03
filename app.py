import streamlit as st
from pathlib import Path
import csv
import os
import re
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
import base64
import logging # Import the logging module
from PIL import Image
import io

# Import Gemini API
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    ChatGoogleGenerativeAI = None # Placeholder if import fails
    genai = None
    st.error("Google Generative AI libraries not found. Please install them: pip install google-generativeai langchain-google-genai")


# Import embedding and vectorstore components
try:
    import torch # Import torch first to check availability
    from sentence_transformers import SentenceTransformer
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_COMMUNITY_AVAILABLE = True
    # Hint to Streamlit's watcher to ignore torch internals that might cause issues
    # This is a known issue with Streamlit watching large libraries like torch.
    # Using a file watcher ignore pattern in Streamlit config is the preferred way,
    # but this code tries a programmatic approach (less reliable).
    if hasattr(torch, '_C') and hasattr(torch._C, '_get_custom_class_python_wrapper'):
        pass # Avoid risky deletion attempts, config file is better
except ImportError:
    LANGCHAIN_COMMUNITY_AVAILABLE = False
    # Define placeholders if imports fail
    HuggingFaceEmbeddings = None
    FAISS = None
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    st.error("Langchain community components (Embeddings, Vectorstores, Loaders) or Sentence Transformers not found. Please install them: pip install langchain-community sentence-transformers faiss-cpu pypdf torch")


# Import chain and memory components
try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False
    ConversationalRetrievalChain = None
    ConversationBufferMemory = None
    st.error("Core Langchain components (Chains, Memory) not found. Please install them: pip install langchain")


# Import Neo4j components
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    st.error("Neo4j driver not found. Please install it: pip install neo4j")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("python-dotenv not found. Cannot load .env file. Ensure environment variables are set.")


# --- Configuration ---
# Get environment variables with fallback to placeholder values
# IMPORTANT: Replace placeholders with your actual keys/credentials or ensure .env is set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBv-I8Ld-k09Lxu9Yi7HPffZHKXIqGSdHU") # Placeholder
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://1b47920f.databases.neo4j.io") # Placeholder
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j") # Placeholder
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "eCqDfyhDcuGMLzbYfiqL6jsvjH3LIXr86xQGAEKmY8Y") # Placeholder

# Validate required environment variables if libraries are available
if GOOGLE_GENAI_AVAILABLE and (GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or not GEMINI_API_KEY):
    st.error("❗ Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable or update the placeholder in the code.")
if NEO4J_AVAILABLE and (NEO4J_URI == "neo4j+s://YOUR_NEO4J_URI" or NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD" or not NEO4J_URI or not NEO4J_PASSWORD or not NEO4J_USER):
    st.error("❗ Neo4j connection details (URI, User, Password) are missing or incomplete. Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables or update placeholders.")


# Update the NEO4J_AUTH variable to use environment variables
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# Threshold settings
THRESHOLDS = {
    "symptom_extraction": 0.6,
    "disease_matching": 0.5, # Base threshold for KG to identify *a* disease
    "disease_symptom_followup_threshold": 0.8, # Below this confidence for a disease query, trigger symptom confirmation UI
    "knowledge_graph_general": 0.6, # General threshold for KG info (treatments/remedies)
    "medical_relevance": 0.6 # Threshold for medical relevance check
}

# Hardcoded PDF files to use (relative to the script location)
# Ensure these files exist in the same directory as your script or provide full paths.
PDF_DIRECTORY = Path(__file__).parent # Directory where the script is located
HARDCODED_PDF_FILES = [
    PDF_DIRECTORY / "rawdata.pdf",
    # Add more PDF paths here if needed, e.g., PDF_DIRECTORY / "another_doc.pdf"
]

# Configure logging
# Log messages to the console where streamlit is run
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_image_as_base64(file_path):
    """Converts an image file to a base64 string."""
    if not os.path.exists(file_path):
        logger.warning(f"Image file not found at {file_path}")
        # Return a tiny valid base64 image as a fallback (1x1 transparent pixel)
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            # Determine image type for correct data URI
            extension = Path(file_path).suffix.lower()
            mime_type = f"image/{extension[1:]}" if extension else "image/png" # Default to png if no extension
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error encoding image {file_path} to base64: {e}")
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

# Load and encode the image
image_path = Path(__file__).parent / "Zoom My Life.jpg" # Assuming image is in the same directory
icon_base64 = get_image_as_base64(image_path)

# Cache for expensive operations
CACHE = {}

def get_cached(key):
    """Get cached result if it exists"""
    key_str = json.dumps(key, sort_keys=True)
    if key_str in CACHE:
        logger.debug(f"Cache hit for key: {key_str[:50]}...")
        return CACHE[key_str]
    logger.debug(f"Cache miss for key: {key_str[:50]}...")
    return None

def set_cached(key, value):
    """Set cache for a key"""
    key_str = json.dumps(key, sort_keys=True)
    logger.debug(f"Setting cache for key: {key_str[:50]}...")
    CACHE[key_str] = value
    # Limit cache size (simple approach: clear if too large)
    if len(CACHE) > 100: # Adjust cache size limit as needed
        logger.warning("Cache size limit reached, clearing cache.")
        CACHE.clear()
    return value

# --- DocumentChatBot Class ---
class DocumentChatBot:
    def __init__(self):
        logger.info("DocumentChatBot initializing...")
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.vectordb: Optional[FAISS] = None
        self.chat_history: List[Tuple[str, str]] = []
        self.followup_context = {"round": 0} # Tracks LLM follow-up round (0 or 1)

        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.kg_driver = None
        self.kg_connection_ok = False

        # Initialize components only if libraries are available
        if LANGCHAIN_COMMUNITY_AVAILABLE and 'torch' in globals():
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Initializing SentenceTransformer embeddings on device: {device}")
                # Ensure cache directory exists
                cache_dir = Path("./cache")
                cache_dir.mkdir(parents=True, exist_ok=True)

                self.embedding_model = HuggingFaceEmbeddings(
                    model_name='sentence-transformers/all-MiniLM-L6-v2',
                    cache_folder=str(cache_dir), # Pass cache folder path
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': True}
                )
                # Test embedding model
                test_embedding = self.embedding_model.embed_query("test query")
                if not test_embedding or len(test_embedding) == 0:
                    raise ValueError("Test embedding failed (empty result).")
                logger.info("Embedding model initialized and tested successfully.")
            except Exception as e:
                logger.error(f"CRITICAL ERROR: Could not initialize embedding model: {e}", exc_info=True)
                self.embedding_model = None
                st.warning("⚠️ Embedding model initialization failed. RAG features will be unavailable. Check logs.")
        else:
            logger.warning("Langchain community components or torch not available. Skipping embedding model initialization.")

        if NEO4J_AVAILABLE:
            self._init_kg_connection()
        else:
            logger.warning("Neo4j library not available. Skipping KG connection.")
            st.warning("⚠️ Neo4j library not found. Knowledge Graph features will be unavailable.")

        logger.info("DocumentChatBot initialization finished.")


    def _init_kg_connection(self):
        """Attempts to connect to the Neo4j database."""
        if not NEO4J_AVAILABLE:
            logger.error("Neo4j library not available, cannot connect.")
            self.kg_connection_ok = False
            return

        # Check if connection details are placeholders
        if NEO4J_URI == "neo4j+s://YOUR_NEO4J_URI" or NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD":
            logger.error("Neo4j connection details are placeholders. Cannot connect.")
            st.warning("⚠️ Neo4j connection details are placeholders. Knowledge Graph features unavailable.")
            self.kg_connection_ok = False
            return

        logger.info("Attempting to connect to Neo4j...")
        try:
            self.kg_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, connection_timeout=10.0) # Increased timeout
            # Perform a simple query to truly verify connectivity and authentication
            with self.kg_driver.session(database="neo4j") as session: # Ensure default 'neo4j' db if needed
                 session.run("MATCH (n) RETURN count(n) LIMIT 1")
            logger.info("Successfully connected and verified connection to Neo4j.")
            self.kg_connection_ok = True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}. KG features will be unavailable.", exc_info=True)
            self.kg_driver = None
            self.kg_connection_ok = False
            st.warning(f"⚠️ Failed to connect to Neo4j: {e}. Knowledge Graph features unavailable.")


    def create_vectordb(self) -> Tuple[Optional[FAISS], str]:
        """Create vector database from hardcoded PDF documents."""
        if not LANGCHAIN_COMMUNITY_AVAILABLE or self.embedding_model is None:
             msg = "Vector DB creation skipped: Required libraries (Langchain, Embeddings) not available or embeddings failed."
             logger.warning(msg)
             return None, msg

        logger.info("Creating vector database...")
        pdf_files_found = []
        for pdf_path in HARDCODED_PDF_FILES:
            if pdf_path.exists() and pdf_path.is_file():
                pdf_files_found.append(pdf_path)
            else:
                logger.warning(f"PDF file not found or is not a file: {pdf_path}")

        if not pdf_files_found:
            msg = "No valid PDF files found at specified paths. Cannot create vector database."
            logger.warning(msg)
            st.warning(f"⚠️ {msg}")
            return None, msg

        loaders = []
        for pdf_file in pdf_files_found:
            try:
                loaders.append(PyPDFLoader(str(pdf_file)))
                logger.debug(f"Created PDF loader for: {pdf_file}")
            except Exception as e:
                logger.error(f"Error creating loader for {pdf_file}: {e}")

        if not loaders:
             msg = "No valid PDF loaders could be created. Check PDF files."
             logger.warning(msg)
             st.warning(f"⚠️ {msg}")
             return None, msg

        pages = []
        for loader in loaders:
            try:
                loaded_pages = loader.load()
                pages.extend(loaded_pages)
                logger.info(f"Loaded {len(loaded_pages)} pages from {loader.file_path}.")
            except Exception as e:
                logger.error(f"Error loading pages from PDF {loader.file_path}: {e}")

        if not pages:
             msg = "No pages were loaded from the PDF files."
             logger.warning(msg)
             st.warning(f"⚠️ {msg}")
             return None, msg

        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # Slightly increased overlap
            splits = text_splitter.split_documents(pages)
            logger.info(f"Split {len(pages)} pages into {len(splits)} chunks.")
        except Exception as e:
            msg = f"Error splitting documents: {e}"
            logger.error(msg, exc_info=True)
            st.warning(f"⚠️ {msg}")
            return None, msg


        if not splits:
            msg = "No text chunks were created from the PDF pages."
            logger.warning(msg)
            return None, msg

        try:
            logger.info("Creating FAISS vectorstore from documents...")
            vectordb = FAISS.from_documents(splits, self.embedding_model)
            # Test the vectorstore
            _ = vectordb.similarity_search("test", k=1)
            logger.info("FAISS vectorstore created and tested successfully.")
            return vectordb, "Vector database created successfully."
        except Exception as e:
            msg = f"Error creating FAISS vector database: {e}"
            logger.error(msg, exc_info=True)
            st.warning(f"⚠️ {msg}")
            return None, msg


    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        """ Check if the query is relevant to the medical domain using LLM (if available) or keywords."""
        cache_key = {"type": "medical_relevance", "query": query}
        cached = get_cached(cache_key)
        if cached:
            return cached

        medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose", "sick", "doctor", "condition", "illness", "remedy", "medicine", "medication", "therapy"]
        query_lower = query.lower()

        # Keyword check first (quick fallback)
        if any(keyword in query_lower for keyword in medical_keywords):
            # If LLM is not available, rely solely on keywords
            if self.llm is None:
                 logger.info("Medical relevance check (LLM unavailable): Keyword match.")
                 result = (True, "Keyword match")
                 set_cached(cache_key, result)
                 return result
        # If no keywords found and LLM unavailable, assume not medical
        elif self.llm is None:
             logger.info("Medical relevance check (LLM unavailable): No keyword match.")
             result = (False, "LLM unavailable, no keywords found")
             set_cached(cache_key, result)
             return result

        # --- LLM Check (if LLM is available) ---
        medical_relevance_prompt = f'''
        Analyze the user query. Is it related to health, medical conditions, symptoms, treatments, medication, diagnostics, or any other medical or health science topic?
        Consider both explicit medical terms and implicit health concerns. Answer concisely.

        Query: "{query}"

        Return ONLY a JSON object with this format:
        {{
            "is_medical": true_or_false,
            "confidence": float_between_0.0_and_1.0,
            "reasoning": "brief explanation"
        }}
        '''

        try:
            response = self.local_generate(medical_relevance_prompt, max_tokens=150)
            # Use regex to find JSON object robustly
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    is_medical = data.get("is_medical", False)
                    confidence = data.get("confidence", 0.0)
                    reasoning = data.get("reasoning", "")

                    if is_medical and confidence >= THRESHOLDS.get("medical_relevance", 0.6):
                        logger.info(f"Medical relevance check (LLM): YES (Conf: {confidence:.2f})")
                        result = (True, reasoning)
                    else:
                        # If LLM is not confident or says no, double-check with keywords as a safety net
                        if any(keyword in query_lower for keyword in medical_keywords):
                             logger.info(f"Medical relevance check (LLM): NO/Low Conf ({confidence:.2f}), but keyword match found. Treating as medical.")
                             result = (True, f"LLM low confidence ({confidence:.2f}), but keyword match.")
                        else:
                             logger.info(f"Medical relevance check (LLM): NO (Conf: {confidence:.2f})")
                             result = (False, reasoning)

                    set_cached(cache_key, result)
                    return result
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse medical relevance JSON from LLM: {response}. Falling back to keyword check.")
            else:
                 logger.warning(f"LLM response did not contain JSON for medical relevance: {response}. Falling back to keyword check.")

        except Exception as e:
            logger.error(f"Error checking medical relevance with LLM: {e}. Falling back to keyword check.")

        # Fallback to keyword check if LLM fails or doesn't return valid JSON
        if any(keyword in query_lower for keyword in medical_keywords):
            logger.info("Medical relevance check (Fallback): Keyword match.")
            result = (True, "Keyword match after LLM failure")
        else:
            logger.info("Medical relevance check (Fallback): No keyword match.")
            result = (False, "LLM failed and no keywords found")

        set_cached(cache_key, result)
        return result

    def initialize_qa_chain(self) -> Tuple[bool, str]:
        """Initialize the QA chain with LLM, Embeddings, Vector DB, and KG."""
        logger.info("Attempting to initialize/verify QA chain components...")

        # Check if already initialized and components are healthy
        if (self.llm is not None and self.embedding_model is not None and
            self.vectordb is not None and self.qa_chain is not None and
            (self.kg_driver is not None or not NEO4J_AVAILABLE)): # KG optional if lib not installed
             # Quick health check (optional, can add simple tests here if needed)
             try:
                 if self.llm: _ = self.llm.invoke("hello") # Test LLM
                 if self.vectordb: _ = self.vectordb.similarity_search("test", k=1) # Test VDB
                 if self.kg_driver and self.kg_connection_ok: # Test KG if available and connected
                     with self.kg_driver.session(database="neo4j") as s: s.run("RETURN 1")
                 logger.info("QA Chain components appear already initialized and healthy.")
                 success, msg = self._get_init_status_message()
                 return success, msg
             except Exception as health_e:
                  logger.warning(f"Health check failed for existing components: {health_e}. Re-initializing.")
                  # Force re-initialization by clearing components
                  self.llm = None
                  self.vectordb = None
                  self.qa_chain = None
                  # Don't clear embedding model as it's expensive, assume it's ok if initialized once


        llm_init_success = False
        vdb_init_success = False
        chain_init_success = False

        # --- 1. Initialize LLM ---
        if GOOGLE_GENAI_AVAILABLE and self.llm is None:
            if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                logger.warning("Gemini API key not set or invalid. LLM will not be initialized.")
                self.llm = None
            else:
                try:
                    logger.info("Initializing Gemini Flash 1.5...")
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=GEMINI_API_KEY,
                        temperature=0.3,
                        top_p=0.95,
                        top_k=40,
                        convert_system_message_to_human=True
                    )
                    # Test LLM
                    test_response = self.llm.invoke("Hello!")
                    if test_response and hasattr(test_response, 'content') and test_response.content:
                        logger.info("Successfully connected to Gemini Flash 1.5.")
                        llm_init_success = True
                    else:
                        raise ValueError("LLM test invocation failed or returned empty response.")
                except Exception as e:
                    logger.error(f"Failed to initialize or test Gemini Flash 1.5: {e}", exc_info=True)
                    self.llm = None
                    st.warning(f"⚠️ Failed to initialize LLM (Gemini): {e}. Core functionality limited.")
        elif self.llm is not None:
             llm_init_success = True # Already initialized
        else:
             logger.warning("Google GenAI library not available. Skipping LLM initialization.")
             st.warning("⚠️ Google GenAI library not found. LLM features unavailable.")


        # --- 2. Create Vector Database ---
        # Only create if needed and embeddings are available
        if LANGCHAIN_COMMUNITY_AVAILABLE and self.embedding_model is not None and self.vectordb is None:
             self.vectordb, vdb_message = self.create_vectordb()
             if self.vectordb is not None:
                 vdb_init_success = True
                 logger.info(f"Vector DB Initialization: {vdb_message}")
             else:
                 logger.warning(f"Vector DB creation failed: {vdb_message}")
                 # Don't block everything if VDB fails, RAG will just be skipped
        elif self.vectordb is not None:
            vdb_init_success = True # Already initialized
        else:
             logger.warning("Vector DB creation skipped: Libraries or embedding model unavailable, or already failed.")


        # --- 3. Create Retrieval QA Chain ---
        # Requires LLM, VDB, Embeddings, and core Langchain libs
        if (LANGCHAIN_CORE_AVAILABLE and LANGCHAIN_COMMUNITY_AVAILABLE and
            self.llm is not None and self.vectordb is not None and self.qa_chain is None):
            try:
                 logger.info("Creating Conversational Retrieval Chain.")
                 memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    output_key='answer',
                    return_messages=True
                 )
                 self.qa_chain = ConversationalRetrievalChain.from_llm(
                     self.llm,
                     retriever=self.vectordb.as_retriever(search_kwargs={"k": 5}), # Increase k slightly
                     memory=memory,
                     return_source_documents=True,
                     verbose=False, # Set True for debugging langchain steps
                     # Combine docs prompt can be customized if needed
                     # combine_docs_chain_kwargs={"prompt": your_custom_prompt}
                 )
                 # Test the chain lightly
                 # _ = self.qa_chain.invoke({"question": "hello"}) # Can be slow, skip for faster init
                 logger.info("Conversational Retrieval Chain initialized.")
                 chain_init_success = True
            except Exception as e:
                logger.error(f"Failed to create Retrieval Chain: {e}", exc_info=True)
                self.qa_chain = None
                st.warning(f"⚠️ Failed to create RAG chain: {e}. RAG features unavailable.")
        elif self.qa_chain is not None:
             chain_init_success = True # Already initialized
        else:
             logger.warning("Cannot create Retrieval Chain: Required components (LLM, VDB, Langchain libs) not available or not initialized.")

        # --- 4. Verify KG Connection ---
        # Re-check KG connection status if driver exists but status is false
        if NEO4J_AVAILABLE and self.kg_driver and not self.kg_connection_ok:
             logger.warning("Re-checking Neo4j connection...")
             self._init_kg_connection() # Attempt reconnection/verification

        # Determine overall success and message
        overall_success, overall_message = self._get_init_status_message()

        logger.info(f"Initialization Result: Success={overall_success}, Message='{overall_message}'")
        return overall_success, overall_message

    def _get_init_status_message(self) -> Tuple[bool, str]:
        """Helper to generate the status message based on component availability."""
        status_parts = []
        llm_ok = self.llm is not None
        embed_ok = self.embedding_model is not None
        vdb_ok = self.vectordb is not None
        chain_ok = self.qa_chain is not None
        kg_ok = self.kg_connection_ok or not NEO4J_AVAILABLE # KG is ok if connected or library not installed

        status_parts.append(f"LLM: {'OK' if llm_ok else 'Failed'}")
        status_parts.append(f"Embeddings: {'OK' if embed_ok else 'Failed'}")
        status_parts.append(f"Vector DB: {'OK' if vdb_ok else 'Failed'}")
        status_parts.append(f"RAG Chain: {'OK' if chain_ok else 'Failed'}")
        status_parts.append(f"KG: {'OK' if kg_ok else 'Failed'}")

        # Basic function requires LLM. RAG requires LLM+Embed+VDB+Chain. KG requires KG driver.
        # Consider initialization successful if at least the LLM is available for basic chat.
        overall_success = llm_ok

        # Add warning if optional components failed
        warnings = []
        if not chain_ok and (llm_ok and embed_ok and vdb_ok): warnings.append("RAG unavailable (chain init failed)")
        elif not vdb_ok and (llm_ok and embed_ok): warnings.append("RAG unavailable (VDB init failed)")
        elif not embed_ok: warnings.append("RAG unavailable (Embedding init failed)")
        if not kg_ok and NEO4J_AVAILABLE: warnings.append("KG unavailable (Connection failed)")

        status_string = f"Status: {', '.join(status_parts)}"
        if warnings:
            status_string += f" | Warnings: {'; '.join(warnings)}"

        return overall_success, status_string

    def local_generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using the initialized LLM, with fallback."""
        if self.llm is None:
            logger.error("LLM not initialized, cannot generate.")
            return "Error: LLM is not available."

        try:
            logger.debug(f"Generating LLM response (max_tokens={max_tokens}). Prompt starts: {prompt[:100]}...")
            response = self.llm.invoke(prompt, config={"max_output_tokens": max_tokens})
            # Ensure response.content is a string
            content = getattr(response, 'content', '')
            if not isinstance(content, str):
                 content = str(content) # Convert if not string
            logger.debug(f"LLM response received. Length: {len(content)}")
            return content.strip()
        except Exception as e:
            logger.error(f"Error generating with Langchain LLM wrapper: {e}", exc_info=True)
            # Fallback direct generation using genai library if available
            if GOOGLE_GENAI_AVAILABLE and genai:
                try:
                    logger.warning("Falling back to direct genai generation...")
                    # Ensure API key is configured for direct use
                    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
                        genai.configure(api_key=GEMINI_API_KEY)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        # Set generation config for direct API call
                        generation_config = genai.types.GenerationConfig(
                             max_output_tokens=max_tokens,
                             temperature=0.3, # Match temperature if possible
                             top_p=0.95,
                             top_k=40
                        )
                        result = model.generate_content(prompt, generation_config=generation_config)
                        # Check response structure for safety
                        response_text = getattr(result, 'text', '')
                        if not response_text:
                            # Log the full response if text is missing
                            logger.error(f"Direct genai fallback failed: No text in response. Full response: {result}")
                            return "Error: Fallback generation failed (empty response)."
                        logger.info("Direct genai fallback generation successful.")
                        return response_text.strip()
                    else:
                        logger.error("Cannot use direct genai fallback: API key missing.")
                        return "Error: LLM generation failed (API key missing)."
                except Exception as inner_e:
                    logger.error(f"Error in direct genai fallback generation: {inner_e}", exc_info=True)
                    return f"Error: LLM generation failed ({inner_e})."
            else:
                return "Error: LLM generation failed and fallback unavailable."


    def generate_llm_answer(self, query: str, kg_content: Optional[str] = None, rag_content: Optional[str] = None, initial_combined_answer: Optional[str] = None, missing_elements: Optional[List[str]] = None) -> str:
        """ Generates an LLM answer, synthesizing information from KG and RAG (Path 2). """
        logger.info("➡️ LLM Synthesis Step")
        if self.llm is None:
            logger.warning("LLM not initialized. Skipping synthesis.")
            fallback = "I can provide some information based on the search, but cannot synthesize a full answer currently.\n\n"
            if initial_combined_answer: fallback += initial_combined_answer
            elif kg_content: fallback += kg_content + "\n\n"
            elif rag_content: fallback += rag_content
            else: fallback = "I couldn't find information or synthesize an answer due to initialization issues."
            return fallback + "\n\nPlease consult a healthcare professional for definitive advice."


        prompt_parts = [
            "You are DxAI-Agent, a helpful medical AI assistant. Your goal is to provide a comprehensive and safe answer based *only* on the provided information below.",
            f"USER QUESTION: {query}"
        ]

        # Provide context based on what's available
        context_provided = False
        if initial_combined_answer and initial_combined_answer.strip() and "limited specific information" not in initial_combined_answer:
             prompt_parts.append(f"AVAILABLE INFORMATION DRAFT (from Knowledge Graph and Document Search):\n---\n{initial_combined_answer}\n---")
             context_provided = True
        # If no draft, provide raw content if meaningful
        else:
             kg_meaningful = kg_content and kg_content.strip() and "unavailable" not in kg_content and "did not find" not in kg_content
             rag_meaningful = rag_content and rag_content.strip() and "error occurred" not in rag_content and "unavailable" not in rag_content and "couldn't find specific" not in rag_content

             if kg_meaningful:
                  prompt_parts.append(f"AVAILABLE MEDICAL KNOWLEDGE GRAPH INFORMATION:\n---\n{kg_content}\n---")
                  context_provided = True
             if rag_meaningful:
                  prompt_parts.append(f"AVAILABLE RETRIEVED INFORMATION (Document Search):\n---\n{rag_content}\n---")
                  context_provided = True

        if not context_provided:
             prompt_parts.append("No specific relevant information was found from internal knowledge sources. Provide a general, safe response based on common medical knowledge, emphasizing the need for professional consultation.")

        # Instructions for synthesis
        prompt_parts.append("INSTRUCTIONS:")
        prompt_parts.append("1. Synthesize the AVAILABLE INFORMATION (if any) into a helpful, accurate, and comprehensive answer to the USER QUESTION.")
        prompt_parts.append("2. **Do NOT include information beyond what is provided in the AVAILABLE INFORMATION sections.** If no information is provided, state that and recommend professional consultation.")
        prompt_parts.append("3. Structure the answer clearly using markdown (headings, lists).")

        # Focus on missing elements if identified
        if missing_elements:
            missing_desc = []
            if "duration" in missing_elements: missing_desc.append("symptom duration")
            if "severity" in missing_elements: missing_desc.append("symptom severity")
            if "location" in missing_elements: missing_desc.append("symptom location")
            if "frequency" in missing_elements: missing_desc.append("symptom frequency")
            if "onset" in missing_elements: missing_desc.append("symptom onset")
            if missing_desc:
                prompt_parts.append(f"4. If possible using the available information, try to address aspects related to: {', '.join(missing_desc)}.")

        prompt_parts.append("5. **Crucially, include a prominent medical disclaimer** stating this is not a substitute for professional advice and recommending consultation with a healthcare provider.")
        prompt_parts.append("6. If potentially serious symptoms like chest pain or shortness of breath were mentioned or implied, add an explicit warning to seek immediate medical attention.")


        prompt = "\n\n".join(prompt_parts)

        try:
            # Use local_generate with appropriate token limit for synthesis
            response = self.local_generate(prompt, max_tokens=1200)
            # Basic check for disclaimer presence
            if "disclaimer" not in response.lower() and "professional medical advice" not in response.lower():
                 logger.warning("LLM Synthesis missing disclaimer! Appending default.")
                 response += "\n\n**IMPORTANT MEDICAL DISCLAIMER:** This information is for educational purposes only and not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider."
            return response
        except Exception as e:
            logger.error(f"Error generating LLM synthesis answer: {e}")
            return "I encountered an error while synthesizing the final answer. Please consult a healthcare professional for reliable advice."


    def format_kg_diagnosis_with_llm(self, disease_name: str, symptoms_list: List[str], confidence: float) -> str:
        """ Uses LLM to format the KG-identified disease and symptoms for Path 1. """
        logger.info("➡️ LLM Formatting KG Diagnosis Step")
        if self.llm is None:
            logger.warning("LLM not initialized. Skipping KG diagnosis formatting.")
            symptoms_str = ", ".join(symptoms_list) if symptoms_list else "your symptoms"
            return f"Based on {symptoms_str}, **{disease_name}** is a potential condition according to the knowledge graph. Confidence: {confidence:.2f}. This is not a diagnosis; consult a professional."

        symptoms_str = ", ".join(symptoms_list) if symptoms_list else "the symptoms reported"

        prompt = f"""
        You are DxAI-Agent, a medical assistant.
        Task: Concisely explain a potential medical condition based on symptoms found in a knowledge graph.
        Input:
        - Identified Disease: {disease_name}
        - Symptoms Considered: {symptoms_str}
        - Confidence Score (internal): {confidence:.2f}

        Output Requirements:
        1. Write a concise, single-paragraph statement for the user.
        2. Mention the key symptoms considered ({symptoms_str}).
        3. State that these symptoms *might be associated* with the {disease_name} based on the knowledge graph analysis.
        4. **Crucially, include a clear disclaimer:** This is NOT a definitive diagnosis and professional medical evaluation is essential.
        5. **Do NOT** add treatments, remedies, or ask follow-up questions here. Focus only on the potential condition and the disclaimer.

        Example: "Based on symptoms like {symptoms_str}, the knowledge graph suggests a possible association with **{disease_name}**. However, this is not a definitive diagnosis and requires evaluation by a qualified healthcare professional."

        Generate the statement now:
        """
        try:
            response = self.local_generate(prompt, max_tokens=200) # Keep it concise
            # Ensure disclaimer is present
            if "diagnosis" not in response.lower() or "professional" not in response.lower():
                 logger.warning("LLM KG formatter missing disclaimer! Appending default.")
                 response += " This is not a definitive diagnosis and requires professional medical evaluation."
            return response.strip()
        except Exception as e:
            logger.error(f"Error formatting KG diagnosis with LLM: {e}")
            symptoms_fallback = ", ".join(symptoms_list) if symptoms_list else "your symptoms"
            return f"Based on {symptoms_fallback}, the knowledge graph suggests **{disease_name}** as a possibility (Confidence: {confidence:.2f}). This is not a diagnosis. Please consult a healthcare professional."


    def identify_missing_info(self, user_query: str, generated_answer: str, conversation_history: List[Tuple[str, str]]) -> Tuple[bool, List[str]]:
            """ Final check: Identifies if CRITICAL info is missing from the generated answer for safety/completeness, considering history. """
            logger.info("🕵️ Identifying missing info from generated answer (Final Check)...")

            if self.llm is None:
                 logger.warning("LLM not initialized. Cannot perform final completeness check.")
                 return (False, [])

            # Limit history context to avoid exceeding token limits
            history_limit = 6 # Last 3 user/bot pairs
            recent_history = conversation_history[-history_limit:]
            context = "Conversation History (most recent first):\n---\n"
            for i, entry in enumerate(reversed(recent_history)): # Show recent first
                if isinstance(entry, tuple) and len(entry) == 2:
                    user_msg, bot_msg = entry
                    user_msg_str = str(user_msg)[:300] + '...' if user_msg and len(str(user_msg)) > 300 else str(user_msg or '')
                    bot_msg_str = str(bot_msg)[:300] + '...' if bot_msg and len(str(bot_msg)) > 300 else str(bot_msg or '')
                    context += f"User: {user_msg_str}\nAssistant: {bot_msg_str}\n"
                else:
                    context += "[Invalid history entry]\n"
            context += "---\n"

            # Prompt focused on safety and critical gaps, avoiding redundant questions
            MISSING_INFO_PROMPT = f'''
            You are DxAI-Agent's safety supervisor. Your task is to review the LATEST generated answer to the user's query, considering the recent conversation history, and determine if a *single, critical* follow-up question is *absolutely necessary* for safety or basic understanding before concluding.

            Conversation History (Recent):
            {context}

            User's Initial Query in this thread: "{user_query}"
            Latest Generated Answer (Evaluate this): "{generated_answer}"

            **CRITICAL EVALUATION:**
            1. Does the LATEST answer directly address the core intent of the user's query using available info?
            2. Does the LATEST answer contain adequate safety disclaimers?
            3. **Crucially:** Given the symptoms/topic discussed, is there *one single piece* of information (like onset, severity of a key symptom, specific relevant history) that is *critically missing* from the LATEST answer, which prevents giving even a minimally safe/helpful response?
            4. **Check History:** Has a similar critical question *already been asked* by the Assistant in the recent history? If yes, DO NOT ask again.

            **Decision:**
            - If the LATEST answer is reasonably complete for a first pass (includes disclaimers) AND no single *critical* piece of info is missing (or was already asked), then NO follow-up is needed.
            - If ONE critical piece of info IS missing AND hasn't been asked yet, formulate ONE concise follow-up question to ask the user. Focus on the *most* critical gap.

            Return ONLY a JSON object in this exact format:
            {{
                "needs_followup": true_or_false,
                "reasoning": "Brief justification for the decision, citing the evaluation points.",
                "followup_question": "The single, concise follow-up question if needs_followup is true, otherwise null or empty string."
            }}
            '''

            try:
                response = self.local_generate(MISSING_INFO_PROMPT, max_tokens=400).strip()
                logger.debug(f"\nRaw Missing Info Evaluation (Final Check):\n{response}")

                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        data = json.loads(json_str)
                        needs_followup_llm = data.get("needs_followup", False)
                        question = data.get("followup_question", "")
                        reasoning = data.get("reasoning", "No reasoning provided.")

                        # Ensure question is a string and not null/empty if followup is needed
                        if needs_followup_llm and isinstance(question, str) and question.strip():
                            logger.info(f"❓ Critical Information Missing (LLM opinion): Needs follow-up. Question: '{question}'. Reasoning: {reasoning}")
                            return (True, [question.strip()]) # Return True and the question in a list
                        elif needs_followup_llm:
                             logger.warning(f"LLM indicated followup needed but provided no valid question. Treating as no followup. Reasoning: {reasoning}")
                             return (False, [])
                        else:
                             logger.info(f"✅ Final Answer appears sufficient (LLM opinion). Reasoning: {reasoning}")
                             return (False, [])

                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse final missing info JSON from LLM: {response}")
                        return (False, []) # Assume sufficient if JSON parsing fails
                    except Exception as e:
                         logger.error(f"Error processing LLM response structure in identify_missing_info: {e}", exc_info=True)
                         return (False, [])

                else:
                    logger.warning(f"LLM response did not contain expected JSON for missing info: {response}")
                    return (False, [])

            except Exception as e:
                logger.error(f"⚠️ Error during LLM call in identify_missing_info: {e}", exc_info=True)
                return (False, [])


    def knowledge_graph_agent(self, user_query: str, all_symptoms: List[str]) -> Dict[str, Any]:
        """ Knowledge Graph Agent: Identifies diseases, treatments, remedies based on symptoms. """
        logger.info(f"📚 KG Agent: Processing symptoms: {all_symptoms}")

        # Default structure for results
        kg_results: Dict[str, Any] = {
            "extracted_symptoms": all_symptoms,
            "identified_diseases_data": [], # List[Dict{disease, conf, matched_symp, all_kg_symp}]
            "top_disease_confidence": 0.0,
            "kg_matched_symptoms": [], # For top disease
            "kg_treatments": [],
            "kg_treatment_confidence": 0.0,
            "kg_home_remedies": [],
            "kg_remedy_confidence": 0.0,
            "kg_content_diagnosis_data_for_llm": { # Data for LLM formatting (Path 1)
                 "disease_name": "an unidentifiable condition", "symptoms_list": all_symptoms, "confidence": 0.0
            },
            "kg_content_other": "Knowledge Graph information on treatments or remedies is unavailable.", # For synthesis (Path 2)
        }

        if not NEO4J_AVAILABLE or not self.kg_connection_ok or self.kg_driver is None:
             logger.warning("📚 KG Agent: Connection not available. Skipping KG queries.")
             kg_results["kg_content_other"] = "Medical Knowledge Graph is currently unavailable."
             return kg_results

        try:
            # Use the default database specified in the connection URI usually
            with self.kg_driver.session() as session:
                # 1. Identify Diseases from Symptoms
                if all_symptoms:
                    disease_data_from_kg = self._query_disease_from_symptoms_with_session(session, all_symptoms)
                    if disease_data_from_kg:
                        kg_results["identified_diseases_data"] = disease_data_from_kg
                        top_disease_record = disease_data_from_kg[0]
                        top_disease_name = top_disease_record["Disease"]
                        top_disease_conf = top_disease_record["Confidence"]
                        kg_results["top_disease_confidence"] = top_disease_conf
                        kg_results["kg_matched_symptoms"] = top_disease_record.get("MatchedSymptoms", [])
                        logger.info(f"✔️ KG Diseases Identified: Top='{top_disease_name}' (Conf: {top_disease_conf:.3f}), Total found: {len(disease_data_from_kg)}")

                        # Prepare data for LLM formatting (Path 1) - Use top disease
                        kg_results["kg_content_diagnosis_data_for_llm"] = {
                            "disease_name": top_disease_name,
                            "symptoms_list": all_symptoms, # Use all input symptoms for context
                            "confidence": top_disease_conf
                        }

                        # 2. Find Treatments/Remedies for the TOP disease if confidence is sufficient
                        if top_disease_conf >= THRESHOLDS.get("knowledge_graph_general", 0.6):
                            logger.info(f"📚 KG Tasks: Finding Treatments & Remedies for '{top_disease_name}'")
                            treatments, treat_conf = self._query_treatments_with_session(session, top_disease_name)
                            if treatments:
                                kg_results["kg_treatments"] = treatments
                                kg_results["kg_treatment_confidence"] = treat_conf
                                logger.info(f"✔️ KG Treatments found: {len(treatments)} (Avg Conf: {treat_conf:.3f})")

                            remedies, rem_conf = self._query_home_remedies_with_session(session, top_disease_name)
                            if remedies:
                                kg_results["kg_home_remedies"] = remedies
                                kg_results["kg_remedy_confidence"] = rem_conf
                                logger.info(f"✔️ KG Home Remedies found: {len(remedies)} (Avg Conf: {rem_conf:.3f})")
                        else:
                            logger.info(f"📚 KG Tasks: Treatments/Remedies skipped for '{top_disease_name}' (Confidence {top_disease_conf:.3f} below threshold {THRESHOLDS.get('knowledge_graph_general', 0.6)})")
                    else:
                         logger.info("📚 KG Task: No diseases found matching the provided symptoms.")
                         # Keep default kg_content_diagnosis_data_for_llm
                else:
                     logger.info("📚 KG Task: Identify Diseases skipped - No symptoms provided.")


                # Prepare combined Treatments/Remedies content for Path 2 synthesis
                other_parts: List[str] = []
                if kg_results["kg_treatments"]:
                     other_parts.append("## Potential Treatments (from Knowledge Graph)")
                     other_parts.extend([f"- {t}" for t in kg_results["kg_treatments"]])
                     other_parts.append("")
                if kg_results["kg_home_remedies"]:
                     other_parts.append("## Potential Home Remedies (from Knowledge Graph)")
                     other_parts.extend([f"- {r}" for r in kg_results["kg_home_remedies"]])
                     other_parts.append("")

                combined_other_content = "\n".join(other_parts).strip()
                if combined_other_content:
                     kg_results["kg_content_other"] = combined_other_content
                elif kg_results["top_disease_confidence"] > 0: # If disease was found but no treatments/remedies
                     kg_results["kg_content_other"] = f"Knowledge Graph did not find specific treatments or home remedies listed for {kg_results['kg_content_diagnosis_data_for_llm']['disease_name']}."
                else: # No disease found, so no treatments/remedies either
                    kg_results["kg_content_other"] = "Knowledge Graph did not find relevant diseases, treatments, or remedies for the provided symptoms."


                logger.info("📚 Knowledge Graph Agent Finished.")
                return kg_results

        except Exception as e:
            logger.error(f"⚠️ Error within KG Agent: {e}", exc_info=True)
            kg_results["kg_content_other"] = f"An error occurred querying the Knowledge Graph: {e}"
            # Reset potentially partially filled data on error
            kg_results["identified_diseases_data"] = []
            kg_results["top_disease_confidence"] = 0.0
            kg_results["kg_treatments"] = []
            kg_results["kg_home_remedies"] = []
            kg_results["kg_content_diagnosis_data_for_llm"] = {
                 "disease_name": "an error condition", "symptoms_list": all_symptoms, "confidence": 0.0
            }
            return kg_results


    # --- KG Helper Query Methods ---
    # These use the provided session from the agent

    def _query_disease_from_symptoms_with_session(self, session, symptoms: List[str]) -> List[Dict[str, Any]]:
         """ Queries KG for diseases based on symptoms using an existing session. """
         if not symptoms: return []
         symptoms_lower = [s.lower().strip() for s in symptoms if s and s.strip()]
         if not symptoms_lower: return []

         cache_key = {"type": "kg_disease_match", "symptoms": tuple(sorted(symptoms_lower))}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("🧠 Using cached KG disease match.")
             return cached

         # Optimized Cypher Query: Match input symptoms -> diseases -> all symptoms of those diseases
         cypher_query = """
         // 1. Find symptom nodes matching input (case-insensitive)
         UNWIND $symptomNames AS input_symptom_name
         MATCH (s:symptom)
         WHERE toLower(s.Name) = input_symptom_name
         WITH COLLECT(DISTINCT s) AS matched_input_symptom_nodes, $symptomNames AS input_symptom_list

         // If no input symptoms match nodes in KG, return empty
         WHERE size(matched_input_symptom_nodes) > 0

         // 2. Find diseases indicated by ANY of the matched input symptoms
         UNWIND matched_input_symptom_nodes AS matched_s_node
         MATCH (matched_s_node)-[:INDICATES]->(d:disease)
         WITH d, matched_input_symptom_nodes, input_symptom_list
         ORDER BY d.Name // Ensure deterministic grouping
         WITH d, matched_input_symptom_nodes, input_symptom_list
         // Collect distinct diseases
         WITH COLLECT(DISTINCT d) AS candidate_diseases, matched_input_symptom_nodes, input_symptom_list

         // 3. For each candidate disease, find ALL symptoms it indicates in the KG
         UNWIND candidate_diseases AS disease
         OPTIONAL MATCH (disease)<-[:INDICATES]-(all_s:symptom)
         WITH disease, matched_input_symptom_nodes, input_symptom_list, COLLECT(DISTINCT all_s) AS all_kg_symptom_nodes

         // 4. Calculate matching symptoms and confidence
         WITH disease,
              [s_node IN all_kg_symptom_nodes WHERE s_node IN matched_input_symptom_nodes | s_node.Name] AS matched_symptom_names, // Symptoms from input that matched this disease's symptoms
              [s_node IN all_kg_symptom_nodes | s_node.Name] AS all_kg_symptom_names, // All symptoms for this disease in KG
              size(all_kg_symptom_nodes) AS total_kg_symptoms_count,
              size([s_node IN all_kg_symptom_nodes WHERE s_node IN matched_input_symptom_nodes]) AS matched_symptoms_count

         // Confidence: Ratio of (matched input symptoms for *this* disease) / (total symptoms for *this* disease in KG)
         // Penalize slightly if the input had many symptoms not associated with this disease? (Optional, simpler for now)
         WITH disease.Name AS DiseaseName,
              matched_symptom_names AS MatchedSymptoms,
              all_kg_symptom_names AS AllDiseaseSymptomsKG,
              matched_symptoms_count,
              total_kg_symptoms_count,
              CASE
                WHEN total_kg_symptoms_count = 0 THEN 0.0 // Avoid division by zero
                // Boost score slightly if more matched symptoms found (up to a limit)
                ELSE (matched_symptoms_count * 1.0 / total_kg_symptoms_count) * (1.0 + least(matched_symptoms_count, 5) * 0.05) // Small boost for more matches
              END AS confidence_score

         // Filter out diseases with no matched symptoms or below a minimal base threshold
         WHERE matched_symptoms_count > 0 AND confidence_score >= $min_base_threshold

         // Return results ordered by confidence
         RETURN DiseaseName AS Disease,
                round(confidence_score * 1000) / 1000 AS Confidence, // Round confidence
                MatchedSymptoms,
                AllDiseaseSymptomsKG
         ORDER BY Confidence DESC, Disease // Secondary sort for stability
         LIMIT 5 // Limit results
         """
         try:
             params = {"symptomNames": symptoms_lower, "min_base_threshold": THRESHOLDS["disease_matching"] * 0.5} # Use half threshold as absolute minimum
             result = session.run(cypher_query, params)
             records = list(result) # Consume the result iterator fully

             disease_data = [
                  {
                       "Disease": rec["Disease"],
                       "Confidence": float(rec["Confidence"]),
                       "MatchedSymptoms": sorted(rec["MatchedSymptoms"]), # Sort for consistency
                       "AllDiseaseSymptomsKG": sorted(rec["AllDiseaseSymptomsKG"]) # Sort for consistency
                  }
                  for rec in records if rec["Confidence"] >= THRESHOLDS["disease_matching"] # Apply final threshold
             ]

             logger.debug(f"🦠 Executed KG Disease Query for {symptoms_lower}, found {len(disease_data)} results passing threshold.")
             set_cached(cache_key, disease_data) # Cache the final thresholded list
             return disease_data
         except Exception as e:
             logger.error(f"⚠️ Error executing KG query for diseases: {e}", exc_info=True)
             return []


    def _query_treatments_with_session(self, session, disease: str) -> Tuple[List[str], float]:
         """ Queries KG for treatments using an existing session. """
         if not disease: return [], 0.0
         disease_lower = disease.lower().strip()

         cache_key = {"type": "kg_treatment", "disease": disease_lower}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("🧠 Using cached KG treatments.")
             return cached

         # Simplified query, assuming Treatment nodes have Name property
         cypher_query = """
         MATCH (d:disease)-[:TREATED_BY]->(t:treatment)
         WHERE toLower(d.Name) = $diseaseName
         RETURN DISTINCT t.Name as TreatmentName
         ORDER BY TreatmentName // Order alphabetically
         LIMIT 10 // Limit number of treatments shown
         """
         try:
             result = session.run(cypher_query, diseaseName=disease_lower)
             treatments_list = sorted([rec["TreatmentName"] for rec in result if rec["TreatmentName"]]) # Sort and remove nulls

             # Confidence is now based on whether treatments were found for this disease, not relation count
             confidence = 0.8 if treatments_list else 0.0

             logger.debug(f"💊 Executed KG Treatment Query for '{disease_lower}', found {len(treatments_list)} treatments.")
             final_result = (treatments_list, confidence)
             set_cached(cache_key, final_result)
             return final_result
         except Exception as e:
             logger.error(f"⚠️ Error executing KG query for treatments: {e}", exc_info=True)
             return [], 0.0


    def _query_home_remedies_with_session(self, session, disease: str) -> Tuple[List[str], float]:
         """ Queries KG for home remedies using an existing session. """
         if not disease: return [], 0.0
         disease_lower = disease.lower().strip()

         cache_key = {"type": "kg_remedy", "disease": disease_lower}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("🧠 Using cached KG home remedies.")
             return cached

         # Simplified query, assuming homeremedy nodes have Name property
         cypher_query = """
         MATCH (d:disease)-[:HAS_HOMEREMEDY]->(h:homeremedy)
         WHERE toLower(d.Name) = $diseaseName
         RETURN DISTINCT h.Name as RemedyName
         ORDER BY RemedyName // Order alphabetically
         LIMIT 10 // Limit number of remedies shown
         """
         try:
             result = session.run(cypher_query, diseaseName=disease_lower)
             remedies_list = sorted([rec["RemedyName"] for rec in result if rec["RemedyName"]])

             # Confidence based on whether remedies were found
             confidence = 0.75 if remedies_list else 0.0

             logger.debug(f"🏡 Executed KG Remedy Query for '{disease_lower}', found {len(remedies_list)} remedies.")
             final_result = (remedies_list, confidence)
             set_cached(cache_key, final_result)
             return final_result
         except Exception as e:
             logger.error(f"⚠️ Error executing KG query for home remedies: {e}", exc_info=True)
             return [], 0.0


    # --- Symptom Extraction & Query Analysis ---

    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        """ Extract symptoms from user query using LLM (if available) and keywords. """
        cache_key = {"type": "symptom_extraction", "query": user_query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("🧠 Using cached symptom extraction.")
            return cached

        llm_symptoms = []
        llm_avg_confidence = 0.0
        final_confidence = 0.0
        combined_symptoms = []

        # Common keywords (lowercase)
        common_symptom_keywords = {"fever", "cough", "headache", "sore throat", "nausea", "dizziness", "chest pain", "shortness of breath", "difficulty breathing", "fatigue", "body aches", "runny nose", "congestion", "chills", "sweats", "joint pain", "muscle aches", "rash", "swelling", "pain", "ache", "burning", "itching", "numbness", "tingling", "diarrhea", "vomiting", "difficulty swallowing", "stomach ache", "abdominal pain", "back pain", "wheezing", "palpitations"}
        query_lower = user_query.lower()

        # Keyword extraction (always run as baseline/fallback)
        keyword_symptoms = {kw.capitalize() for kw in common_symptom_keywords if kw in query_lower}
        logger.debug(f"Keyword check found: {keyword_symptoms}")

        # LLM Extraction (if LLM available)
        if self.llm:
            SYMPTOM_PROMPT = f'''
            You are a medical expert focused *only* on extracting potential symptoms from text.
            Analyze the following user query and extract all potential medical symptoms or signs mentioned.
            Do NOT infer conditions. Only extract explicit mentions or very close paraphrases of symptoms.
            Assign a confidence score (0.0 to 1.0) for each extracted term being a symptom.

            User Query: "{user_query}"

            Return ONLY a JSON list object containing dictionaries in this format:
            [{{"symptom": "Symptom Name 1", "confidence": 0.95}}, {{"symptom": "Symptom Name 2", "confidence": 0.80}}]
            If no symptoms are found, return an empty list: []
            '''
            try:
                response = self.local_generate(SYMPTOM_PROMPT, max_tokens=300).strip()
                # Try to find JSON list directly
                json_match = re.search(r'(\[[\s\S]*?\])', response) # Non-greedy match for list
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        symptom_data = json.loads(json_str)
                        if isinstance(symptom_data, list):
                            # Process LLM results
                            llm_symptoms_confident = []
                            confidences = []
                            for item in symptom_data:
                                if isinstance(item, dict) and "symptom" in item and "confidence" in item:
                                     symptom_name = str(item["symptom"]).strip()
                                     # Simple normalization (capitalize first letter)
                                     if symptom_name:
                                          symptom_name = symptom_name[0].upper() + symptom_name[1:].lower()
                                     try:
                                         confidence = float(item["confidence"])
                                         if confidence >= THRESHOLDS["symptom_extraction"]:
                                             llm_symptoms_confident.append(symptom_name)
                                             confidences.append(confidence)
                                     except (ValueError, TypeError):
                                          logger.warning(f"Invalid confidence value from LLM: {item.get('confidence')}")
                                else:
                                     logger.warning(f"Invalid item format from LLM symptom extraction: {item}")


                            llm_symptoms = llm_symptoms_confident
                            if confidences:
                                llm_avg_confidence = sum(confidences) / len(confidences)
                            logger.debug(f"🔍 LLM Extracted Symptoms (confident): {llm_symptoms} (Avg Conf: {llm_avg_confidence:.3f})")

                        else:
                            logger.warning(f"LLM returned JSON, but not a list: {json_str}")

                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse symptom JSON from LLM response: {json_str}")
                else:
                    logger.warning(f"Could not find JSON list '[]' in LLM symptom response: {response}")

            except Exception as e:
                logger.error(f"Error in LLM symptom extraction: {e}")
        else:
             logger.info("LLM not available for symptom extraction, relying on keywords.")


        # Combine and determine final confidence
        # Use a set for deduplication, comparing lowercase versions
        combined_symptoms_set = {s.lower() for s in llm_symptoms}
        combined_symptoms_set.update({s.lower() for s in keyword_symptoms})

        # Convert back to original/capitalized form (prefer LLM's if available)
        final_symptoms_map = {s.lower(): s for s in keyword_symptoms} # Start with keywords
        final_symptoms_map.update({s.lower(): s for s in llm_symptoms}) # Overwrite with LLM versions if they exist

        combined_symptoms = sorted([final_symptoms_map[s_lower] for s_lower in combined_symptoms_set])


        # Confidence logic:
        if llm_symptoms: # If LLM found confident symptoms, use its average confidence
            final_confidence = llm_avg_confidence
        elif keyword_symptoms: # If only keywords found something
            final_confidence = 0.45 # Assign a moderate confidence for keyword-only
        else: # Nothing found
            final_confidence = 0.0

        logger.info(f"🔍 Final Extracted Symptoms: {combined_symptoms} (Confidence Score: {final_confidence:.3f})")

        result = (combined_symptoms, final_confidence)
        set_cached(cache_key, result)
        return result


    def is_disease_identification_query(self, query: str, extracted_symptoms: List[str]) -> bool:
        """ Check if query primarily aims to identify a disease from symptoms. """
        query_lower = query.lower()
        has_symptoms = len(extracted_symptoms) > 0

        # Strong indicators: Explicit questions about cause/diagnosis
        disease_keywords = [
            r"what disease", r"what condition", r"what could (this|it) be", r"what might be",
            r"possible disease", r"possible condition", r"diagnose", r"what causes", r"what is causing",
            r"what do i have", r"what could i have", r"identify .* (condition|disease|issue)",
            r"(symptoms|signs) of what", r"what could be wrong"
        ]
        if any(re.search(pattern, query_lower) for pattern in disease_keywords):
            logger.debug(f"Query '{query[:50]}...' identified as disease query (keyword match).")
            return True

        # Moderate indicators: Personal symptom description + implicit question
        personal_phrases = ["i have", "my symptoms are", "i'm experiencing", "feeling", "suffering from"]
        ends_with_question = query.strip().endswith('?')

        if has_symptoms and any(phrase in query_lower for phrase in personal_phrases) and ends_with_question:
             logger.debug(f"Query '{query[:50]}...' identified as disease query (personal symptoms + question mark).")
             return True

        # Weak indicator: List of symptoms only, ending in question mark (less reliable)
        # Check if query is mostly just the extracted symptoms
        symptom_words = set(word for sym in extracted_symptoms for word in sym.lower().split())
        query_words = set(query_lower.replace('?', '').replace('.', '').replace(',', '').split())
        # High overlap between query words and symptom words might indicate a symptom list query
        overlap = len(symptom_words.intersection(query_words))
        # Heuristic: if > 50% of non-common query words are symptom words AND it ends with '?'
        common_words = {'i', 'have', 'a', 'is', 'and', 'the', 'my', 'are', 'what', 'me', 'with'}
        non_common_query_words = query_words - common_words
        if non_common_query_words and has_symptoms and ends_with_question:
            if overlap / len(non_common_query_words) > 0.5:
                logger.debug(f"Query '{query[:50]}...' identified as disease query (symptom list heuristic).")
                return True

        logger.debug(f"Query '{query[:50]}...' NOT identified as primarily a disease identification query.")
        return False


    def identify_missing_elements(self, user_query: str, generated_answer: str) -> List[str]:
        """ Heuristic check for potentially missing high-level concepts (duration, severity etc.) in the answer for LLM synthesis focus. """
        logger.debug("🔍 Identifying high-level potential missing elements for LLM focus...")
        missing = set()
        query_lower = user_query.lower()
        answer_lower = generated_answer.lower()

        # Check only if the query seems personal (implies context like duration/severity might be relevant)
        personal_phrases = ["i have", "my symptoms are", "i'm experiencing", "feeling"]
        is_personal = any(phrase in query_lower for phrase in personal_phrases) or self.is_disease_identification_query(user_query, []) # Basic check if it looks like disease query

        if is_personal:
            # Duration Check (Keywords in Answer)
            duration_kws = {"duration", "days", "weeks", "months", "long", "since", "started", "began", "onset"}
            if not any(kw in answer_lower for kw in duration_kws):
                missing.add("duration/onset")

            # Severity Check (Keywords in Answer)
            severity_kws = {"severity", "mild", "moderate", "severe", "how severe", "intense", "level", "scale"}
            if not any(kw in answer_lower for kw in severity_kws):
                 missing.add("severity")

            # Location Check (if query mentions location-relevant symptoms)
            location_relevant_symptoms = {"pain", "ache", "rash", "swelling", "bruise", "tenderness", "lump", "sore"}
            location_kws = {"location", "where", "area", "side", "chest", "abdomen", "head", "limb", "back"}
            if any(symptom in query_lower for symptom in location_relevant_symptoms):
                 if not any(kw in answer_lower for kw in location_kws):
                      missing.add("location")

            # Frequency Check (if query mentions episodic symptoms)
            frequency_relevant_symptoms = {"pain", "headache", "dizziness", "nausea", "palpitations", "attack", "episode"}
            frequency_kws = {"frequency", "often", "intermittent", "constant", "sporadic", "times", "comes and goes", "episodic"}
            if any(symptom in query_lower for symptom in frequency_relevant_symptoms):
                 if not any(kw in answer_lower for kw in frequency_kws):
                    missing.add("frequency")

        missing_list = sorted(list(missing))
        logger.debug(f"Identified potential missing elements (for LLM focus): {missing_list}")
        return missing_list


    def combine_initial_answer_draft(self, kg_diagnosis_component: Optional[str], kg_content_other: str, rag_content: str) -> str:
         """ Combines KG diagnosis (Path 1), other KG info, and RAG content into a draft for LLM synthesis. """
         logger.info("🧩 Combining KG and RAG results into initial draft...")
         combined_parts: List[str] = []
         has_kg_diag = False
         has_kg_other = False
         has_rag = False

         # Add KG diagnosis component (from Path 1 LLM formatting) if meaningful
         if kg_diagnosis_component and kg_diagnosis_component.strip() and "unidentifiable condition" not in kg_diagnosis_component and "error condition" not in kg_diagnosis_component :
              combined_parts.append(kg_diagnosis_component.strip())
              has_kg_diag = True

         # Add other KG content (treatments/remedies) if meaningful
         kg_other_meaningful = kg_content_other and kg_content_other.strip() and "unavailable" not in kg_content_other and "did not find" not in kg_content_other
         if kg_other_meaningful:
              # Add heading if not already implied by diagnosis component structure
              if not has_kg_diag or not kg_content_other.startswith("##"):
                   # Check if diagnosis part already included treatments/remedies (unlikely with current format)
                   if "treatment" not in (kg_diagnosis_component or "").lower() and "remed" not in (kg_diagnosis_component or "").lower():
                        # Add separator if diagnosis was present
                        if has_kg_diag: combined_parts.append("\n---\n")
                        # Add the content (which should already have markdown headings)
                        combined_parts.append(kg_content_other.strip())
                        has_kg_other = True
                   else: # Diagnosis component seems to cover it, don't add redundantly
                       logger.debug("Skipping kg_content_other as diagnosis component might already cover it.")
              else: # Diagnosis component likely already structured to include other KG info
                   has_kg_other = True # Mark as included


         # Add RAG content if meaningful
         rag_meaningful = rag_content and rag_content.strip() and "error occurred" not in rag_content and "unavailable" not in rag_content and "couldn't find specific" not in rag_content
         if rag_meaningful:
              # Add separator if KG content was added
              if has_kg_diag or has_kg_other:
                   combined_parts.append("\n---\n")
              combined_parts.append("## Information from Document Search\n")
              combined_parts.append(rag_content.strip())
              has_rag = True

         # If nothing meaningful was found from any source
         if not has_kg_diag and not has_kg_other and not has_rag:
               combined_parts.append("I searched the knowledge graph and documents but found limited specific information regarding your query.")

         initial_combined_answer = "\n".join(combined_parts).strip()
         logger.debug(f"Initial combined draft created (KG Diag: {has_kg_diag}, KG Other: {has_kg_other}, RAG: {has_rag}). Length: {len(initial_combined_answer)}")
         return initial_combined_answer

    # --- Main Response Generation Orchestrator ---

    def generate_response(self, user_input: str, user_type: str = "User / Family", confirmed_symptoms: Optional[List[str]] = None, original_query_if_followup: Optional[str] = None) -> Tuple[str, List[str], str, Optional[Dict]]:
        """
        Orchestrates the response generation using KG, RAG, LLM, and follow-up logic.

        Returns: (response_text, sources_list, action_flag, ui_data)
        action_flag: "final_answer", "llm_followup_prompt", "symptom_ui_prompt", "error", "none"
        ui_data: Dict for "symptom_ui_prompt" containing {"symptom_options": Dict, "original_query": str}
        """
        t_start_generate = datetime.now()
        logger.info(f"--- START generate_response --- Input: '{user_input[:50]}...' ---")
        logger.info(f"   Confirmed symptoms: {confirmed_symptoms}, Original query context: {original_query_if_followup[:50] if original_query_if_followup else 'None'}")
        logger.info(f"   Follow-up round: {self.followup_context['round']}, History length: {len(self.chat_history)}")

        # --- Initialization Check ---
        init_ok, init_msg = self.initialize_qa_chain() # Verify/re-init components
        if not init_ok and self.llm is None: # LLM is absolutely critical
             error_message = f"Critical Error: Chat assistant core (LLM) failed to initialize ({init_msg}). Cannot proceed."
             logger.critical(error_message)
             # Add user message to history, but respond with error
             self.chat_history.append((user_input, error_message))
             return error_message, [], "error", None
        elif not init_ok:
            logger.warning(f"Initialization incomplete ({init_msg}). Proceeding with limited features.")
            # UI should show warnings based on init_msg in sidebar

        # --- Determine Core Query and Context ---
        # If processing response to a prompt/UI, use the original query as the main topic
        core_query_for_processing = original_query_if_followup if original_query_if_followup is not None else user_input
        is_response_to_llm_followup = original_query_if_followup is not None and self.followup_context["round"] == 1 and confirmed_symptoms is None
        is_response_to_symptom_ui = confirmed_symptoms is not None

        logger.info(f"   Core query for logic: '{core_query_for_processing[:50]}...'")
        if is_response_to_llm_followup: logger.info("   Context: Response to LLM follow-up.")
        if is_response_to_symptom_ui: logger.info("   Context: Response to Symptom UI.")

        # --- Step 1: Basic Input Validation & Relevance Check ---
        if not core_query_for_processing.strip():
             logger.info("Empty core query. Skipping.")
             return "", [], "none", None

        is_med_query, med_reason = self.is_medical_query(core_query_for_processing)
        if not is_med_query:
            logger.info(f"Query classified as non-medical: '{core_query_for_processing[:50]}...'. Reason: {med_reason}")
            # Provide a polite refusal or a general non-medical answer using LLM
            non_med_prompt = f"The user asked: '{core_query_for_processing}'. This seems unrelated to medical topics. Provide a polite, brief response indicating you are a medical assistant and cannot help with this topic. Do not answer the query directly."
            response_text = self.local_generate(non_med_prompt, max_tokens=100) if self.llm else "I am a medical assistant and can only help with health-related questions."
            self.log_orchestration_decision(core_query_for_processing, "SELECTED_STRATEGY: NON_MEDICAL", 0, 0)
            self.chat_history.append((user_input, response_text)) # Log user input and refusal
            return response_text, [], "final_answer", None


        # --- Step 2: Extract Symptoms ---
        # Combine symptoms from core query, user's response (if follow-up), and confirmed symptoms (if UI response)
        all_symptoms: List[str] = []
        symptom_confidence = 0.0

        # Extract from core query first
        symptoms_core, conf_core = self.extract_symptoms(core_query_for_processing)

        if is_response_to_llm_followup:
             # Also extract from the user's *answer* to the follow-up
             symptoms_response, conf_response = self.extract_symptoms(user_input)
             all_symptoms = list(set(symptoms_core + symptoms_response))
             symptom_confidence = max(conf_core, conf_response) # Simple max confidence
             logger.info(f"Combined symptoms from original query and follow-up response: {all_symptoms}")
        elif is_response_to_symptom_ui:
             # Use confirmed symptoms + any *new* symptoms mentioned in the text box alongside UI
             symptoms_new_text, conf_new_text = self.extract_symptoms(user_input) # user_input might just be original query again if text box empty
             all_symptoms = list(set(confirmed_symptoms + symptoms_new_text))
             symptom_confidence = max(0.9, conf_new_text) # High confidence due to UI confirmation
             logger.info(f"Using symptoms from UI confirmation (+ new text): {all_symptoms}")
        else: # Standard initial query
             all_symptoms = symptoms_core
             symptom_confidence = conf_core
             logger.info(f"Using symptoms extracted from initial query: {all_symptoms}")

        # Filter out empty strings just in case
        all_symptoms = [s for s in all_symptoms if s and s.strip()]


        # --- Step 3: Knowledge Graph Processing ---
        kg_data = self.knowledge_graph_agent(core_query_for_processing, all_symptoms)
        top_disease_confidence = kg_data.get("top_disease_confidence", 0.0)
        kg_diagnosis_data_for_llm = kg_data.get("kg_content_diagnosis_data_for_llm")
        kg_content_other = kg_data.get("kg_content_other", "")
        kg_found_diseases = len(kg_data.get("identified_diseases_data", [])) > 0

        # --- Step 4: Symptom Follow-up UI Trigger (Decision Point 1) ---
        is_diag_query = self.is_disease_identification_query(core_query_for_processing, all_symptoms) # Pass extracted symptoms
        trigger_symptom_ui = False
        symptom_options_for_ui = {}

        # Conditions to trigger UI:
        # - Is a diagnosis-seeking query.
        # - KG found *some* potential diseases.
        # - Confidence is below the UI trigger threshold.
        # - We are NOT currently processing a response *from* the symptom UI.
        # - KG provided symptom associations for the top diseases (needed for checklist).
        # - We haven't asked the single LLM follow-up yet (round 0).
        # - Critical components (LLM, KG) are available.
        if (is_diag_query and kg_found_diseases and
            top_disease_confidence < THRESHOLDS["disease_symptom_followup_threshold"] and
            not is_response_to_symptom_ui and # Don't trigger if already responding to UI
            any(d.get("AllDiseaseSymptomsKG") for d in kg_data.get("identified_diseases_data", [])[:3]) and # Check top 3 for symptoms
            self.followup_context["round"] == 0 and # Only if no LLM follow-up used yet
            self.llm is not None and self.kg_connection_ok):

            logger.info(f"❓ Decision: Trigger Symptom Follow-up UI (Query: '{core_query_for_processing[:50]}...', KG Conf: {top_disease_confidence:.3f})")
            trigger_symptom_ui = True

            # Prepare data for the UI checklist (Top 3-5 diseases with symptoms)
            relevant_diseases = [d for d in kg_data["identified_diseases_data"][:5] if d.get("AllDiseaseSymptomsKG")]
            for disease_data in relevant_diseases:
                 disease_label = f"{disease_data['Disease']} (KG Conf: {disease_data['Confidence']:.2f})"
                 # Get unique, non-empty, string symptoms associated with this disease in KG
                 symptoms = sorted(list(set(str(s).strip() for s in disease_data.get("AllDiseaseSymptomsKG", []) if isinstance(s, str) and str(s).strip())))
                 if symptoms: symptom_options_for_ui[disease_label] = symptoms

            if symptom_options_for_ui: # Only trigger if we have options to show
                follow_up_prompt_text = f"""
                Based on '{core_query_for_processing[:100]}...' and symptoms like {', '.join(all_symptoms) if all_symptoms else 'those mentioned'}, the knowledge graph suggests a few possibilities.

                To help refine this, please check any *additional* symptoms from the list below that you are experiencing. You can also add others in the text box.
                """
                self.log_orchestration_decision(core_query_for_processing, f"SELECTED_STRATEGY: SYMPTOM_UI_FOLLOWUP\nREASONING: Disease query with low KG confidence ({top_disease_confidence:.2f}). Presenting symptom checklist.", top_disease_confidence, 0.0)
                # Return UI trigger
                # Pass original query back so Streamlit can resubmit it with confirmed symptoms
                return follow_up_prompt_text.strip(), [], "symptom_ui_prompt", {"symptom_options": symptom_options_for_ui, "original_query": core_query_for_processing}
            else:
                 logger.warning("Symptom UI trigger condition met, but no valid symptom options found from KG. Skipping UI.")
                 trigger_symptom_ui = False # Abort UI trigger


        # --- Step 5: Path 1 KG Diagnosis Component Generation ---
        # Generate if: (High KG confidence OR responding to symptom UI) AND is a diagnosis query
        path1_kg_diagnosis_component = None
        generate_kg_diag_component = False
        is_high_conf_kg_diagnosis = kg_found_diseases and top_disease_confidence >= THRESHOLDS["disease_symptom_followup_threshold"]

        if is_diag_query and (is_high_conf_kg_diagnosis or is_response_to_symptom_ui):
             if kg_diagnosis_data_for_llm and kg_diagnosis_data_for_llm["disease_name"] not in ["an unidentifiable condition", "an error condition"]:
                  if self.llm:
                      logger.info(f"✅ Path 1: Formatting KG diagnosis component (Disease: '{kg_diagnosis_data_for_llm['disease_name']}', Conf: {kg_diagnosis_data_for_llm['confidence']:.3f})")
                      path1_kg_diagnosis_component = self.format_kg_diagnosis_with_llm(
                          kg_diagnosis_data_for_llm["disease_name"],
                          all_symptoms, # Use all current symptoms for phrasing
                          kg_diagnosis_data_for_llm["confidence"]
                      )
                      generate_kg_diag_component = True
                  else: # Fallback format if LLM failed
                      logger.warning("⚠️ LLM not available for formatting KG diagnosis. Using manual format.")
                      d_name = kg_diagnosis_data_for_llm["disease_name"]
                      s_str = ", ".join(all_symptoms) if all_symptoms else "your symptoms"
                      conf = kg_diagnosis_data_for_llm["confidence"]
                      path1_kg_diagnosis_component = f"Based on {s_str}, the knowledge graph suggests **{d_name}** (Confidence: {conf:.2f}). This is not a diagnosis; consult a professional."
                      generate_kg_diag_component = True
             else: # KG didn't find a specific disease even after UI/high conf check
                 logger.info("Path 1: KG found no specific disease match for diagnosis query.")
                 path1_kg_diagnosis_component = "Based on the symptoms provided, the knowledge graph did not identify a specific matching condition."
                 generate_kg_diag_component = True # Generate this neutral statement


        # --- Step 6: Conclude with KG-Only? (Decision Point 2) ---
        # If we generated a KG diagnosis component (Path 1) AND the original query seems *only* about diagnosis (no treatment/remedy keywords)
        # AND we have an LLM to ensure formatting is good.
        if generate_kg_diag_component and path1_kg_diagnosis_component and self.llm:
            query_lower = core_query_for_processing.lower()
            asks_for_more = any(kw in query_lower for kw in ["treat", "medication", "cure", "what to do", "manage", "remedy", "home", "natural", "relief", "information on", "tell me about"])

            if not asks_for_more:
                 logger.info(f"✅ Decision Point 2: Concluding with KG-only diagnosis answer (High Conf or Post-UI). Query didn't explicitly ask for more details.")
                 final_response_text = path1_kg_diagnosis_component
                 # Add KG source mention
                 all_sources = ["[Source: Medical Knowledge Graph (Diagnosis Suggestion)]"] if self.kg_connection_ok else []
                 self.log_orchestration_decision(core_query_for_processing, f"SELECTED_STRATEGY: KG_DIAGNOSIS_ONLY\nREASONING: Diagnosis query answered with sufficient KG confidence ({top_disease_confidence:.2f}) or post-UI confirmation. Query focus seemed limited to diagnosis.", top_disease_confidence, 0.0)
                 self.chat_history.append((user_input, final_response_text.strip())) # Add user input + final bot answer
                 processing_time = (datetime.now() - t_start_generate).total_seconds()
                 logger.info(f"--- END generate_response (KG_DIAGNOSIS_ONLY) --- Time: {processing_time:.2f}s ---")
                 return final_response_text.strip(), all_sources, "final_answer", None
            else:
                 logger.info("Proceeding to Path 2 (RAG/Synthesis) as query asked for more than just diagnosis, or KG diagnosis component was neutral.")


        # --- Step 7: Path 2 - RAG Processing ---
        rag_content = ""
        rag_source_docs = []
        rag_confidence = 0.0

        if self.qa_chain: # Check if RAG chain is available
            logger.info("📚 Path 2: Processing with RAG...")
            try:
                 # Pass the core query. History is managed by the chain's memory.
                 # Ensure chat history used by memory is reasonably up-to-date.
                 # Note: The memory buffer might include the current user_input if added *before* invoke.
                 # This might be okay, or might need adjustment depending on desired memory behavior.
                 # Let's assume memory adds history *after* a turn completes.
                 # The current user_input is not yet in self.chat_history when invoke is called here.
                 rag_response = self.qa_chain.invoke({"question": core_query_for_processing})

                 rag_content = rag_response.get("answer", "").strip()
                 rag_source_docs = rag_response.get("source_documents", [])

                 # Simple RAG confidence based on finding sources
                 rag_confidence = 0.7 if rag_source_docs else 0.2
                 if rag_content and "couldn't find" not in rag_content.lower() and "don't know" not in rag_content.lower():
                     rag_confidence += 0.2 # Boost if answer seems substantive

                 logger.info(f"📊 RAG Result: Found {len(rag_source_docs)} sources. Confidence estimate: {rag_confidence:.2f}")

            except Exception as e:
                 logger.error(f"⚠️ Error during RAG processing: {e}", exc_info=True)
                 rag_content = "An error occurred while retrieving information from documents."
                 rag_confidence = 0.0
        else:
             logger.warning("RAG chain not available. Skipping RAG processing.")
             rag_content = "Document search is currently unavailable."


        # --- Step 8: Initial Combination & Missing Element ID ---
        initial_combined_answer = self.combine_initial_answer_draft(
             path1_kg_diagnosis_component, # Use the KG component generated in Step 5 (if any)
             kg_content_other, # Use other KG info (treatments/remedies)
             rag_content
        )
        missing_elements_for_llm = self.identify_missing_elements(core_query_for_processing, initial_combined_answer)


        # --- Step 9: LLM Synthesis ---
        final_core_answer = ""
        if self.llm:
            logger.info("➡️ Path 2: Initiating LLM Synthesis Step...")
            final_core_answer = self.generate_llm_answer(
                core_query_for_processing,
                initial_combined_answer=initial_combined_answer, # Pass combined draft
                missing_elements=missing_elements_for_llm
            ).strip()
        else:
            logger.warning("LLM not available for synthesis. Using combined draft directly.")
            final_core_answer = initial_combined_answer.strip()
            # Add mandatory disclaimer if LLM is missing
            final_core_answer += "\n\n**IMPORTANT MEDICAL DISCLAIMER:** LLM synthesis unavailable. This information is combined from sources but lacks final review. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider."


        # --- Step 10: Final Reflection Check for LLM Follow-up ---
        needs_final_followup = False
        missing_questions_list = []
        # Only check if LLM is available AND we haven't used the follow-up round AND we are not processing a response TO the follow-up
        if self.llm is not None and self.followup_context["round"] == 0 and not is_response_to_llm_followup:
            logger.info("🧠 Initiating Final Reflection Check (for LLM Follow-up)...")
            needs_final_followup, missing_questions_list = self.identify_missing_info(
                 core_query_for_processing, final_core_answer, self.chat_history
            )
        else:
             logger.info("Skipping final reflection check (LLM unavailable, follow-up used, or processing follow-up response).")


        # --- Step 11: Final Decision: LLM Follow-up or Final Answer ---
        if needs_final_followup and missing_questions_list:
             logger.info("❓ Decision: Ask the one allowed LLM Follow-up.")
             follow_up_question_text = missing_questions_list[0] # Use the first (and likely only) question

             llm_follow_up_prompt_text = f"Thank you. To provide the most relevant information based on our discussion about '{core_query_for_processing[:50]}...', could you please clarify:\n\n> {follow_up_question_text}"

             self.followup_context["round"] = 1 # Mark follow-up as used for this thread
             self.log_orchestration_decision(core_query_for_processing, f"SELECTED_STRATEGY: LLM_FINAL_FOLLOWUP\nREASONING: Final answer completeness check failed (LLM opinion). Asking critical follow-up (Round 1). Question: {follow_up_question_text}", top_disease_confidence, rag_confidence)

             # Add user message + bot prompt to history now
             self.chat_history.append((user_input, llm_follow_up_prompt_text.strip()))

             processing_time = (datetime.now() - t_start_generate).total_seconds()
             logger.info(f"--- END generate_response (LLM_FINAL_FOLLOWUP) --- Time: {processing_time:.2f}s ---")
             # Return prompt text and action flag
             # Pass original query so next turn knows the context
             return llm_follow_up_prompt_text.strip(), [], "llm_followup_prompt", {"original_query": core_query_for_processing}

        else:
            logger.info("✅ Decision: Provide Final Answer (Synthesis complete or follow-up not needed/used).")
            # --- Step 12: Final Answer Formatting ---
            final_response_text = final_core_answer
            all_sources: List[str] = []

            # Add RAG sources
            source_strings = set() # Use set to avoid duplicates initially
            for doc in rag_source_docs:
                source_name = "Unknown Document"
                page_num = ""
                if hasattr(doc, "metadata"):
                     source_name = doc.metadata.get("source", source_name)
                     # Clean up source path if necessary
                     source_name = os.path.basename(source_name) # Show only filename
                     page_num = f", Page {doc.metadata.get('page', 'N/A')}" if doc.metadata.get('page') is not None else ""
                source_strings.add(f"Document: {source_name}{page_num}")

            # Add KG source mentions if KG contributed meaningfully
            kg_used = (kg_data.get("identified_diseases_data") or
                       (kg_data.get("kg_treatments") and "unavailable" not in kg_content_other) or
                       (kg_data.get("kg_home_remedies") and "unavailable" not in kg_content_other))

            if self.kg_connection_ok and kg_used:
                 source_strings.add("Medical Knowledge Graph")

            all_sources = sorted(list(source_strings))


            # Append references section if sources exist and LLM didn't add one
            ref_header_patterns = [r"##?\s+references", r"##?\s+sources"]
            if all_sources and not any(re.search(p, final_response_text, re.IGNORECASE) for p in ref_header_patterns):
                references_section = "\n\n---\n**Sources:**\n"
                references_section += "\n".join([f"- {src}" for src in all_sources])
                final_response_text += references_section

            # Ensure disclaimer is present (final check)
            disclaimer_patterns = [r"professional medical advice", r"healthcare provider", r"not a substitute", r"disclaimer"]
            if not any(re.search(p, final_response_text, re.IGNORECASE) for p in disclaimer_patterns):
                 logger.warning("Final answer missing disclaimer! Appending default.")
                 final_response_text += "\n\n**IMPORTANT MEDICAL DISCLAIMER:** This information is for educational purposes only and not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider."

            # Log final decision
            self.log_orchestration_decision(core_query_for_processing, "SELECTED_STRATEGY: FINAL_ANSWER\nREASONING: Synthesis complete, follow-up not needed or already used.", top_disease_confidence, rag_confidence)

            # Add final user input and bot answer to chat history
            self.chat_history.append((user_input, final_response_text.strip()))

            processing_time = (datetime.now() - t_start_generate).total_seconds()
            logger.info(f"--- END generate_response (FINAL_ANSWER) --- Time: {processing_time:.2f}s ---")
            # Return final answer
            return final_response_text.strip(), all_sources, "final_answer", None


    # --- Logging and Reset ---

    def log_orchestration_decision(self, query, orchestration_result, kg_confidence, rag_confidence):
        """ Logs the high-level strategy decision for analysis. """
        try:
            strategy = "UNKNOWN"
            reasoning = "Not provided"
            if "SELECTED_STRATEGY:" in orchestration_result:
                strategy = orchestration_result.split("SELECTED_STRATEGY:")[1].split("\n", 1)[0].strip()
            if "REASONING:" in orchestration_result:
                reasoning = orchestration_result.split("REASONING:", 1)[1].strip()
                reasoning = reasoning.split("\n")[0][:200] + ('...' if len(reasoning.split("\n")[0]) > 200 else '') # Limit length

            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query.replace("\n", " ").replace(",", ";")[:200], # Sanitize/limit query
                "strategy": strategy,
                "reasoning": reasoning.replace("\n", " ").replace(",", ";"),
                "kg_confidence": f"{kg_confidence:.3f}",
                "rag_confidence": f"{rag_confidence:.3f}"
            }

            # Log to console
            logger.info(f"📊 Orchestration: Strategy='{strategy}', KG Conf={log_entry['kg_confidence']}, RAG Conf={log_entry['rag_confidence']}, Reasoning='{reasoning}'")

            # Save to CSV
            log_file = Path("orchestration_log.csv")
            file_exists = log_file.is_file()
            with open(log_file, mode='a', newline='', encoding='utf-8') as file:
                fieldnames = ['timestamp', 'query', 'strategy', 'reasoning', 'kg_confidence', 'rag_confidence']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_entry)

        except Exception as e:
            logger.error(f"⚠️ Error logging orchestration decision: {e}")


    def reset_conversation(self):
      """Reset the conversation history and follow-up context"""
      logger.info("🔄 Resetting conversation state.")
      self.chat_history = []
      self.followup_context = {"round": 0}
      # Also reset the memory of the Langchain chain if it exists
      if self.qa_chain and hasattr(self.qa_chain, 'memory'):
           self.qa_chain.memory.clear()
           logger.info("Cleared RAG chain memory.")
      return "Conversation has been reset."


# --- Feedback Functions ---

def save_feedback(log_file, data):
    """Append feedback data to a CSV file."""
    try:
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            # Define fieldnames based on expected keys in data dictionary
            # Ensure all potential keys are listed
            fieldnames = ['timestamp', 'user_type', 'user_query', 'bot_response',
                          'vote_type', 'feedback_text', 'full_history']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            # Ensure all fieldnames exist in the data dict, adding None if missing
            row_data = {field: data.get(field, None) for field in fieldnames}
            writer.writerow(row_data)
        return True
    except Exception as e:
        logger.error(f"Error saving feedback to {log_file}: {e}")
        return False

def vote_message(user_query, bot_response, vote_type, user_type):
    """Handles thumbs up/down feedback."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_data = {
        "timestamp": timestamp,
        "user_type": user_type,
        "user_query": user_query.replace("\n", " "), # Sanitize
        "bot_response": bot_response.replace("\n", " "), # Sanitize
        "vote_type": vote_type,
        "feedback_text": None, # No text for vote
        "full_history": None # History not needed for simple vote
    }
    if save_feedback("feedback_log.csv", feedback_data):
        logger.info(f"Feedback vote '{vote_type}' recorded for query: '{user_query[:50]}...'")
        return f"Feedback ({vote_type}) recorded. Thank you!"
    else:
        return "Error recording feedback."

def submit_feedback(feedback_text, chat_history, user_type):
    """Handles detailed text feedback submission."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
     # Format chat history for CSV (e.g., simple JSON string)
    history_str = json.dumps([{"user": u, "bot": b} for u, b in chat_history], indent=None)

    feedback_data = {
        "timestamp": timestamp,
        "user_type": user_type,
        "user_query": chat_history[-1][0].replace("\n", " ") if chat_history else "N/A", # Last user query
        "bot_response": chat_history[-1][1].replace("\n", " ") if chat_history else "N/A", # Last bot response
        "vote_type": None, # No vote for text feedback
        "feedback_text": feedback_text.replace("\n", " "), # Sanitize
        "full_history": history_str # Include history context
    }
    if save_feedback("feedback_log.csv", feedback_data):
        logger.info(f"Detailed feedback submitted by {user_type}: '{feedback_text[:50]}...'")
        return "Detailed feedback submitted. Thank you!"
    else:
        return "Error submitting feedback."


# --- Streamlit UI ---

def display_symptom_checklist(symptom_options: Dict[str, List[str]], original_query: str):
    """ Streamlit UI for confirming symptoms via checkboxes. """
    st.subheader("Confirm Your Symptoms")
    st.info(f"Based on your query ('{original_query[:50]}...') and initial analysis, please confirm any additional symptoms you are experiencing from the list below to help narrow down possibilities.")

    # Unique key for the form based on original query hash and a timestamp (robust against reruns)
    form_key = f"symptom_form_{abs(hash(original_query))}_{st.session_state.get('form_timestamp', 0)}"
    # Local state within the form to track selections *during* interaction before submit
    local_confirm_key = f"{form_key}_confirmed_local"

    if local_confirm_key not in st.session_state:
        st.session_state[local_confirm_key] = set() # Initialize as set

    # Combine all unique symptoms from all diseases into one list
    all_unique_symptoms = set()
    for symptoms_list in symptom_options.values():
        if isinstance(symptoms_list, list):
            for symptom in symptoms_list:
                 if isinstance(symptom, str) and symptom.strip():
                      all_unique_symptoms.add(symptom.strip().capitalize()) # Capitalize for display consistency

    sorted_all_symptoms = sorted(list(all_unique_symptoms))

    with st.form(key=form_key):
        st.markdown("**Check all symptoms that apply:**")

        if not sorted_all_symptoms:
            st.warning("No specific associated symptoms found in the knowledge graph to confirm.")
        else:
            # Display checkboxes in columns
            num_columns = 4
            cols = st.columns(num_columns)
            for i, symptom in enumerate(sorted_all_symptoms):
                col_idx = i % num_columns
                # Checkbox state reflects the *local* confirmation set for this form instance
                is_checked = symptom in st.session_state[local_confirm_key]
                if cols[col_idx].checkbox(symptom, key=f"{form_key}_{symptom}", value=is_checked):
                     st.session_state[local_confirm_key].add(symptom) # Add to local set if checked
                else:
                     st.session_state[local_confirm_key].discard(symptom) # Remove if unchecked

        st.markdown("**Other Symptoms (optional, comma-separated):**")
        other_symptoms_text = st.text_input("Add any other symptoms here", key=f"{form_key}_other_input")

        # Submit button
        submit_button = st.form_submit_button("Confirm Symptoms and Continue")

        if submit_button:
            # Combine checked symptoms with manually entered ones
            confirmed_set = st.session_state[local_confirm_key].copy() # Start with checked ones
            if other_symptoms_text:
                 other_list = [s.strip().capitalize() for s in other_symptoms_text.split(',') if s.strip()]
                 confirmed_set.update(other_list)

            final_confirmed_list = sorted(list(confirmed_set))
            logger.info(f"Symptom confirmation submitted. Confirmed: {final_confirmed_list}")

            # Store the final list in the main session state to be picked up by the next processing cycle
            st.session_state.confirmed_symptoms_from_ui = final_confirmed_list
            st.session_state.awaiting_symptom_confirmation = False # Turn off UI flag
            # Do NOT clear original_query_for_followup here, generate_response needs it

            # Set the trigger for processing in the next rerun
            st.session_state.input_ready_for_processing = {
                 "text": st.session_state.original_query_for_followup, # Process the original query again
                 "confirmed_symptoms": final_confirmed_list,
                 "original_query_context": st.session_state.original_query_for_followup
            }

            # Clear the form's local state and update timestamp *after* processing submit logic
            del st.session_state[local_confirm_key]
            st.session_state.form_timestamp = datetime.now().timestamp()

            st.rerun() # Trigger rerun to process with confirmed symptoms


# --- Main Streamlit App ---
def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="DxAI-Agent",
        page_icon=icon_base64, # Use base64 icon
        layout="wide"
    )

    # --- Title and Logo ---
    col1, col2 = st.columns([1, 10])
    try:
        logo = Image.open(image_path)
        with col1:
            st.image(logo, width=100)
    except FileNotFoundError:
        with col1: st.markdown(" ") # Placeholder if logo fails
        logger.warning(f"Logo image not found at: {image_path}")
    except Exception as e:
        with col1: st.markdown(" ") # Placeholder if error
        logger.error(f"Error loading logo: {e}")

    with col2:
        st.markdown("# DxAI-Agent")
        st.caption("Your AI Medical Assistant for Information Retrieval")

    # --- Initialize Chatbot in Session State ---
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DocumentChatBot()
        # Initial attempt to initialize components
        with st.spinner("Initializing assistant..."):
            init_ok, init_msg = st.session_state.chatbot.initialize_qa_chain()
            st.session_state.init_status_msg = init_msg # Store status message
            st.session_state.init_failed = not init_ok and st.session_state.chatbot.llm is None # Failed if core LLM failed
            if st.session_state.init_failed:
                st.error(f"Critical Initialization Failed: {init_msg}. Core features unavailable.")
            elif not init_ok:
                 st.warning(f"Initialization Incomplete: {init_msg}. Some features may be limited.")
            else:
                 st.success(f"Initialization Status: {init_msg}") # Show initial status briefly

    # --- Initialize UI State Variables ---
    if 'messages' not in st.session_state: st.session_state.messages = [] # For UI display: List[(content, is_user)]
    if 'awaiting_symptom_confirmation' not in st.session_state: st.session_state.awaiting_symptom_confirmation = False
    if 'symptom_options_for_ui' not in st.session_state: st.session_state.symptom_options_for_ui = {}
    if 'confirmed_symptoms_from_ui' not in st.session_state: st.session_state.confirmed_symptoms_from_ui = None
    if 'original_query_for_followup' not in st.session_state: st.session_state.original_query_for_followup = None
    if 'form_timestamp' not in st.session_state: st.session_state.form_timestamp = datetime.now().timestamp()
    if 'input_ready_for_processing' not in st.session_state: st.session_state.input_ready_for_processing = None
    if 'init_failed' not in st.session_state: st.session_state.init_failed = False
    if 'init_status_msg' not in st.session_state: st.session_state.init_status_msg = "Initializing..."
    if 'current_user_type' not in st.session_state: st.session_state.current_user_type = "User / Family"


    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        st.session_state.current_user_type = st.selectbox(
            "Select User Type:",
            ["User / Family", "Physician"],
            index=["User / Family", "Physician"].index(st.session_state.current_user_type) # Maintain selection
        )
        st.markdown("---")
        st.header("Status")
        if st.session_state.init_failed:
             st.error(f"Status: Initialization Failed!\n{st.session_state.init_status_msg}")
        else:
             st.info(st.session_state.init_status_msg) # Display current status

        st.markdown("---")
        st.info("DxAI-Agent uses AI to answer medical questions based on provided documents and a knowledge graph. Always consult a professional for medical advice.")
        st.markdown(f"LLM Follow-up Count: {st.session_state.chatbot.followup_context['round']} / 1")


    # --- Main Area Tabs ---
    tab1, tab2 = st.tabs(["Chat Interface", "About & Diagnostics"])

    with tab1:
        st.subheader("Example Queries")
        examples = [
            "What are treatments for cough and cold?",
            "I have a headache and sore throat. What could it be?",
            "What home remedies help with flu symptoms?",
            "I have chest pain and shortness of breath. What could i do?"
        ]
        cols = st.columns(len(examples))
        # Disable examples if init failed or symptom UI is active
        examples_disabled = st.session_state.init_failed or st.session_state.awaiting_symptom_confirmation
        for i, col in enumerate(cols):
            if col.button(examples[i], key=f"example_{i}", disabled=examples_disabled, use_container_width=True):
                # Reset state for a new example thread
                st.session_state.chatbot.reset_conversation() # Reset internal state
                st.session_state.messages = [] # Clear UI messages
                st.session_state.awaiting_symptom_confirmation = False
                st.session_state.original_query_for_followup = None
                st.session_state.confirmed_symptoms_from_ui = None

                # Add example to messages and trigger processing
                st.session_state.messages.append((examples[i], True)) # Add user msg to UI
                st.session_state.input_ready_for_processing = {
                    "text": examples[i],
                    "confirmed_symptoms": None,
                    "original_query_context": None
                }
                st.rerun()


        st.divider()

        # --- Chat History Display ---
        for i, (msg_content, is_user) in enumerate(st.session_state.messages):
            with st.chat_message("user" if is_user else "assistant"):
                st.write(msg_content)
                # Add feedback buttons only to non-prompt assistant messages
                if not is_user:
                    # Simple check if it looks like a prompt
                    is_prompt = "could you please clarify" in msg_content.lower() or \
                                "please check any additional symptoms" in msg_content.lower() or \
                                "confirm your symptoms" in msg_content.lower()

                    if not is_prompt:
                         feedback_key_base = f"feedback_{i}_{abs(hash(msg_content))}"
                         vote_cols = st.columns([1, 1, 10]) # Adjust spacing
                         if vote_cols[0].button("👍", key=f"{feedback_key_base}_up", help="Good response"):
                              # Find preceding user message
                              user_q = ""
                              if i > 0 and st.session_state.messages[i-1][1]: user_q = st.session_state.messages[i-1][0]
                              vote_result = vote_message(user_q, msg_content, "thumbs_up", st.session_state.current_user_type)
                              st.toast(vote_result, icon="👍")
                         if vote_cols[1].button("👎", key=f"{feedback_key_base}_down", help="Bad response"):
                              user_q = ""
                              if i > 0 and st.session_state.messages[i-1][1]: user_q = st.session_state.messages[i-1][0]
                              vote_result = vote_message(user_q, msg_content, "thumbs_down", st.session_state.current_user_type)
                              st.toast(vote_result, icon="👎")


        # --- Dynamic Input Area (at the bottom) ---
        # This container ensures elements are grouped correctly at the bottom
        input_container = st.container()

        with input_container:
            # Case 1: Initialization Failed
            if st.session_state.init_failed:
                st.error("Chat assistant failed to initialize. Cannot accept input.")
            # Case 2: Awaiting Symptom Confirmation UI
            elif st.session_state.awaiting_symptom_confirmation:
                if st.session_state.symptom_options_for_ui and st.session_state.original_query_for_followup:
                    display_symptom_checklist(st.session_state.symptom_options_for_ui, st.session_state.original_query_for_followup)
                else:
                    st.warning("Symptom confirmation needed, but data is missing. Please reset.")
                # Disable chat input while form is active
                st.chat_input("Confirm symptoms above...", disabled=True, key="disabled_chat_input_form")
            # Case 3: Standard Chat Input
            else:
                user_query = st.chat_input("Ask your medical question...", key="main_chat_input")
                if user_query:
                    logger.info(f"User input detected: '{user_query[:50]}...'")
                    # Add user message to UI state immediately
                    st.session_state.messages.append((user_query, True))
                    # Prepare for processing this new input
                    st.session_state.input_ready_for_processing = {
                        "text": user_query,
                        "confirmed_symptoms": None, # Not from UI
                        "original_query_context": None # This *is* the original query
                    }
                    # Reset follow-up context for a new query thread
                    st.session_state.chatbot.followup_context = {"round": 0}
                    st.session_state.original_query_for_followup = None # Clear any previous follow-up context
                    st.session_state.confirmed_symptoms_from_ui = None # Clear any lingering confirmed symptoms

                    st.rerun() # Rerun to process the input


        # --- Process Input (if ready) ---
        # This block runs *after* UI elements are drawn and potentially input detected
        if st.session_state.input_ready_for_processing:
            input_data = st.session_state.input_ready_for_processing
            # Clear the flag immediately to prevent reprocessing on next rerun unless explicitly set again
            st.session_state.input_ready_for_processing = None

            logger.info(f"Processing input: Text='{input_data['text'][:50]}...', ConfirmedSymp={input_data['confirmed_symptoms']}, OrigQueryContext='{input_data['original_query_context'][:50] if input_data['original_query_context'] else 'None'}'")

            with st.chat_message("assistant"): # Show thinking indicator in assistant bubble
                with st.spinner("Thinking..."):
                     try:
                         response_text, sources, action_flag, ui_data = st.session_state.chatbot.generate_response(
                              input_data["text"],
                              st.session_state.current_user_type,
                              confirmed_symptoms=input_data["confirmed_symptoms"],
                              original_query_if_followup=input_data["original_query_context"]
                         )
                         logger.info(f"generate_response returned action: {action_flag}")
                     except Exception as e:
                          logger.error(f"Unhandled error during generate_response: {e}", exc_info=True)
                          response_text = f"An unexpected error occurred: {e}"
                          sources = []
                          action_flag = "error"
                          ui_data = None

            # --- Handle Action Flag ---
            rerun_needed = False
            if action_flag == "symptom_ui_prompt":
                st.session_state.awaiting_symptom_confirmation = True
                st.session_state.symptom_options_for_ui = ui_data.get("symptom_options", {})
                st.session_state.original_query_for_followup = ui_data.get("original_query", input_data["text"]) # Store query that triggered UI
                st.session_state.form_timestamp = datetime.now().timestamp() # New form instance
                # Add the prompt message to UI state
                st.session_state.messages.append((response_text, False))
                rerun_needed = True

            elif action_flag == "llm_followup_prompt":
                # Store the original query context for the *next* turn
                st.session_state.original_query_for_followup = ui_data.get("original_query", input_data["text"])
                # Add prompt message to UI state
                st.session_state.messages.append((response_text, False))
                # No UI state change needed here, just display message
                rerun_needed = True


            elif action_flag == "final_answer":
                # Add final answer to UI state
                st.session_state.messages.append((response_text, False))
                # Clear follow-up context as the thread is done or follow-up used
                st.session_state.original_query_for_followup = None
                st.session_state.awaiting_symptom_confirmation = False # Ensure UI is off
                rerun_needed = True

            elif action_flag == "error":
                 # Add error message to UI state
                 st.session_state.messages.append((response_text, False))
                 # Clear follow-up context
                 st.session_state.original_query_for_followup = None
                 st.session_state.awaiting_symptom_confirmation = False
                 rerun_needed = True

            elif action_flag == "none":
                 # No response generated (e.g., empty input)
                 logger.info("Action 'none' received, no UI update needed.")
                 pass # No rerun needed, no message added

            # Rerun Streamlit to display the new message/UI state
            if rerun_needed:
                 st.rerun()


        # --- Reset Button ---
        st.divider()
        if st.button("🔄 Reset Conversation", key="reset_button_main"):
            reset_message = st.session_state.chatbot.reset_conversation()
            st.session_state.messages = []
            st.session_state.awaiting_symptom_confirmation = False
            st.session_state.original_query_for_followup = None
            st.session_state.confirmed_symptoms_from_ui = None
            st.session_state.input_ready_for_processing = None
            st.toast(reset_message, icon="🔄")
            logger.info("Conversation reset by user.")
            st.rerun()

    # --- About Tab ---
    with tab2:
        st.header("About DxAI-Agent")
        st.markdown("""
        This application demonstrates an AI assistant designed to answer medical questions using multiple information sources.

        **Core Components:**
        *   **Language Model (LLM):** Google Gemini Flash 1.5 for understanding queries, extracting information, synthesizing answers, and checking relevance/completeness.
        *   **Knowledge Graph (KG):** Neo4j graph database containing structured medical knowledge (symptoms, diseases, treatments, remedies). Used for direct lookups based on extracted symptoms. *(Requires active Neo4j connection & credentials)*
        *   **Retrieval-Augmented Generation (RAG):** Searches local PDF documents using semantic embeddings (Sentence Transformers) and a vector store (FAISS) to find relevant text passages. These passages augment the LLM's knowledge. *(Requires PDF files, embedding model, and vector store libraries)*
        *   **Orchestration:** Logic that decides which components to use based on the query type, extracted information, and confidence scores. Manages follow-up questions (Symptom UI, LLM prompt) to gather more information when needed.

        **Workflow Summary:**
        1.  **Input & Symptom Extraction:** User query is analyzed, and symptoms are extracted (LLM + keywords).
        2.  **Relevance Check:** Determines if the query is medical.
        3.  **KG Query:** If symptoms are found, the KG is queried for associated diseases, treatments, and remedies.
        4.  **Symptom Confirmation (UI):** If it's a diagnosis query with low KG confidence, a symptom checklist may be shown.
        5.  **RAG Query:** Relevant documents are searched based on the query.
        6.  **Synthesis:** Information from KG and RAG (and potentially the KG diagnosis component) is combined and synthesized by the LLM into a final answer.
        7.  **Reflection & Follow-up:** The final answer is checked for completeness. If critical info is missing (and a follow-up hasn't already been asked), one clarifying question might be posed by the LLM.
        8.  **Output:** The final answer, including sources and disclaimers, is presented.

        **Disclaimer:** This is a demonstration system. **It is NOT a substitute for professional medical advice, diagnosis, or treatment.** Always consult with a qualified healthcare provider for any health concerns. Information provided may not be complete or accurate.
        """)

        st.subheader("Diagnostics & Logs")
        # Display basic info from chatbot instance
        st.text(f"LLM Initialized: {st.session_state.chatbot.llm is not None}")
        st.text(f"Embedding Model Initialized: {st.session_state.chatbot.embedding_model is not None}")
        st.text(f"Vector DB Initialized: {st.session_state.chatbot.vectordb is not None}")
        st.text(f"RAG Chain Initialized: {st.session_state.chatbot.qa_chain is not None}")
        st.text(f"KG Driver Initialized: {st.session_state.chatbot.kg_driver is not None}")
        st.text(f"KG Connection OK: {st.session_state.chatbot.kg_connection_ok}")
        st.text(f"Current Chat History Length: {len(st.session_state.chatbot.chat_history)}")
        st.text(f"Current LLM Follow-up Round: {st.session_state.chatbot.followup_context['round']}")

        # Display logs (consider showing only recent or allowing download)
        st.subheader("Recent Logs (View Console for Full Logs)")
        log_file_orch = Path("orchestration_log.csv")
        log_file_fb = Path("feedback_log.csv")
        if log_file_orch.exists():
            try:
                with open(log_file_orch, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                st.text_area("Orchestration Log (Last 10 entries)", "".join(lines[-10:]), height=200, key="log_orch_area")
            except Exception as e: st.error(f"Error reading orchestration log: {e}")
        else: st.info("Orchestration log not found.")

        if log_file_fb.exists():
            try:
                with open(log_file_fb, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                st.text_area("Feedback Log (Last 10 entries)", "".join(lines[-10:]), height=200, key="log_fb_area")
            except Exception as e: st.error(f"Error reading feedback log: {e}")
        else: st.info("Feedback log not found.")

        # Detailed Feedback Form (copied from Tab 1 for convenience)
        st.divider()
        st.subheader("🩺 Detailed Feedback")
        with st.form("feedback_form_tab2"):
            feedback_text = st.text_area(
                "Enter corrections, improvements, or comments here...",
                height=100, key="feedback_text_tab2"
            )
            submit_feedback_btn = st.form_submit_button("Submit Feedback")

            if submit_feedback_btn and feedback_text:
                # Use the chatbot's internal chat_history for feedback context
                history_for_feedback = st.session_state.chatbot.chat_history
                feedback_result = submit_feedback(feedback_text, history_for_feedback, st.session_state.current_user_type)
                st.success(feedback_result)


# --- Run Main App ---
if __name__ == "__main__":
    # Check if required libraries are installed before running main
    libs_ok = all([GOOGLE_GENAI_AVAILABLE, LANGCHAIN_COMMUNITY_AVAILABLE, LANGCHAIN_CORE_AVAILABLE, NEO4J_AVAILABLE])
    if libs_ok:
        main()
    else:
        st.error("Essential libraries are missing. Please install the required packages mentioned in the error messages above and restart.")
        logger.critical("Application cannot start due to missing essential libraries.")
