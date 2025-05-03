import streamlit as st
from pathlib import Path
import csv
import os
import re
import torch
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
import base64
import logging # Import the logging module
from PIL import Image
import io

# Import Gemini API
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Import embedding and vectorstore components
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import chain and memory components
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Import Neo4j components
from neo4j import GraphDatabase
try:
    import torch
    # Hint to Streamlit's watcher to ignore torch internals that cause issues
    # This might prevent the __path__._path error
    if hasattr(torch, '_C') and hasattr(torch._C, '_get_custom_class_python_wrapper'):
         # Try to disable watching on this specific part if it exists
         # This is a heuristic and might not work on all torch versions
         try:
              del torch._C._get_custom_class_python_wrapper # Remove the attribute temporarily during watcher scan? (Risky)
              # A safer approach might be to configure Streamlit's watcher if possible, but this requires app config files.
              # Let's stick to the config file approach if needed.
              pass # Do nothing risky here.
         except AttributeError:
              pass # Attribute doesn't exist, no need to delete

except ImportError:
     pass
# Configuration
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
# Log messages to the console where streamlit is run
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for module-level logger

logger.info("Starting application setup...")

# Get environment variables with fallback to placeholder values
# IMPORTANT: Replace 'YOUR_GEMINI_API_KEY' with your actual key or ensure .env is set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAifk9Gntw6eYfaZkLOsd9d1-TkfOR1el0")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://1b47920f.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "eCqDfyhDcuGMLzbYfiqL6jsvjH3LIXr86xQGAEKmY8Y")

# Check if placeholder values are being used and log warnings
if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_PLACEHOLDER":
    logger.warning("GEMINI_API_KEY is using a placeholder value. LLM features will likely fail.")
if NEO4J_URI == "neo4j+s://YOUR_NEO4J_URI_PLACEHOLDER":
    logger.warning("NEO4J_URI is using a placeholder value. Knowledge Graph features will likely fail.")
if NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD_PLACEHOLDER":
    logger.warning("NEO4J_PASSWORD is using a placeholder value. Knowledge Graph connection will likely fail.")

# Update the NEO4J_AUTH variable to use environment variables
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)
logger.debug(f"Neo4j Auth configured for user: {NEO4J_USER}")

# Threshold settings
THRESHOLDS = {
    "symptom_extraction": 0.6,
    "disease_matching": 0.5, # Base threshold for KG to identify *a* disease
    "disease_symptom_followup_threshold": 0.8, # Below this confidence for a disease query, trigger symptom confirmation UI
    "knowledge_graph_general": 0.6, # General threshold for KG info (treatments/remedies)
    "medical_relevance": 0.6 # Threshold for medical relevance check
}
logger.info(f"Thresholds set: {THRESHOLDS}")

# Load and convert the image to base64
def get_image_as_base64(file_path):
    """Converts an image file to a base64 string."""
    logger.debug(f"Attempting to load image for base64 encoding: {file_path}")
    if not os.path.exists(file_path):
        logger.warning(f"Image file not found at {file_path}. Returning fallback image.")
        # Return a tiny valid base64 image as a fallback
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            logger.debug(f"Successfully encoded image {file_path} to base64.")
            return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image {file_path} to base64: {e}", exc_info=True)
        # Return a tiny valid base64 image as a fallback
        logger.warning("Returning fallback image due to encoding error.")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


# Option 1: If your image is stored locally
image_path = "Zoom My Life.jpg"  # Update with your actual path
logger.info(f"Setting image path for icon: {image_path}")
icon = get_image_as_base64(image_path)

# Cache for expensive operations
CACHE = {}
logger.info("Initialized empty cache.")

def get_cached(key):
    """Get cached result if it exists"""
    key_str = json.dumps(key, sort_keys=True) # Use JSON dump for complex keys
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
    # logger.debug(f"Value being cached: {str(value)[:100]}...") # Optional: Log value snippet
    return value


# Hardcoded PDF files to use
# Update with your actual local PDF paths
HARDCODED_PDF_FILES = [
    "rawdata.pdf",
    # Add more PDF paths here if needed
]
logger.info(f"Using hardcoded PDF files: {HARDCODED_PDF_FILES}")

# For testing purposes - add more relevant known diseases if possible
# These aren't strictly used in the new logic, but useful for context/dummy data
known_diseases = ["hypertension", "type 2 diabetes mellitus", "respiratory infections", "obesity", "cardiovascular disease", "common cold", "influenza", "strep throat", "anxiety", "acid reflux", "costochondritis", "angina"]
logger.debug(f"Known diseases list (for context/testing): {known_diseases}")


# DocumentChatBot class
class DocumentChatBot:
    def __init__(self):
        logger.info("DocumentChatBot initializing...")
        self.qa_chain: Optional[ConversationalRetrievalChain] = None # Langchain QA chain
        self.vectordb: Optional[FAISS] = None # Langchain Vectorstore
        self.chat_history: List[Tuple[str, str]] = [] # Stores (user_msg, bot_msg) tuples for display and history
        # Tracks if the single allowed LLM follow-up has been asked (0 or 1) for the current thread
        self.followup_context = {"round": 0}
        logger.debug("Initialized chat_history and followup_context.")

        # Initialize embedding model here, handling potential errors
        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        try:
            # Check for CUDA availability and set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing SentenceTransformer embeddings on device: {device}")
            # Use HuggingFaceEmbeddings which wraps SentenceTransformer
            self.embedding_model = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                cache_folder='./cache',
                model_kwargs={'device': device}, # Pass device via model_kwargs for HuggingFaceEmbeddings
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("HuggingFaceEmbeddings wrapper created.")
            # Test the embedding model (important check)
            try:
                logger.debug("Testing embedding model with a sample query...")
                test_embedding = self.embedding_model.embed_query("test query")
                if test_embedding and len(test_embedding) > 0:
                     logger.info("Embedding model initialized and tested successfully.")
                else:
                    # Raise ValueError explicitly for logging purposes
                    raise ValueError("Test embedding was empty or invalid.")
            except Exception as test_e:
                 logger.warning(f"Embedding model test failed: {test_e}. Setting embedding_model to None.", exc_info=True)
                 self.embedding_model = None # Set to None if test fails

        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Could not initialize embedding model: {e}", exc_info=True)
            self.embedding_model = None # Ensure it's None on failure


        self.llm: Optional[ChatGoogleGenerativeAI] = None # LLM for general generation and specific prompts

        # Initialize KG driver connection status
        self.kg_driver = None
        self.kg_connection_ok = False
        self._init_kg_connection() # Attempt connection during init
        logger.info("DocumentChatBot initialization finished.")


    def _init_kg_connection(self):
        """Attempts to connect to the Neo4j database."""
        logger.info(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
        try:
            # Use a small timeout for the connection test
            self.kg_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, connection_timeout=5.0)
            logger.debug("Neo4j driver created. Verifying connectivity...")
            self.kg_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
            self.kg_connection_ok = True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}. KG features will be unavailable.", exc_info=True)
            self.kg_driver = None
            self.kg_connection_ok = False
        logger.debug(f"KG Connection Status: {self.kg_connection_ok}")


    def create_vectordb(self):
            """Create vector database from hardcoded PDF documents."""
            logger.info("Attempting to create vector database...")
            pdf_files = [Path(pdf_file) for pdf_file in HARDCODED_PDF_FILES if Path(pdf_file).exists()]
            logger.debug(f"Found PDF files: {[str(p) for p in pdf_files]}")

            if not pdf_files:
                logger.warning("No PDF files found at the specified paths. Cannot create vector database.")
                return None, "No PDF files found at the specified paths. Cannot create vector database."

            loaders = []
            for pdf_file in pdf_files:
                try:
                    logger.debug(f"Creating PyPDFLoader for: {pdf_file}")
                    loaders.append(PyPDFLoader(str(pdf_file)))
                    logger.debug(f"Successfully created loader for: {pdf_file}")
                except Exception as e:
                    logger.error(f"Error creating loader for {pdf_file}: {e}", exc_info=True)
                    # Continue with other files

            if not loaders:
                 logger.warning("No valid PDF loaders could be created. Cannot proceed with vector database creation.")
                 return None, "No valid PDF loaders could be created."

            pages = []
            logger.info("Loading pages from PDF files...")
            for loader in loaders:
                try:
                    logger.debug(f"Loading pages from {loader.file_path}...")
                    loaded_pages = loader.load()
                    pages.extend(loaded_pages)
                    logger.info(f"Loaded {len(loaded_pages)} pages from {loader.file_path}.")
                except Exception as e:
                    logger.error(f"Error loading pages from PDF {loader.file_path}: {e}", exc_info=True)
                    # Continue loading other files


            if not pages:
                 logger.warning("No pages were loaded from the PDFs. Cannot create vector database.")
                 return None, "No pages were loaded from the PDFs."
            logger.info(f"Total pages loaded: {len(pages)}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # Adjust chunk size if needed
                chunk_overlap=100 # Adjust overlap if needed
            )
            logger.info(f"Splitting documents using RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=100)...")
            splits = text_splitter.split_documents(pages)
            logger.info(f"Split {len(pages)} pages into {len(splits)} document chunks.")

            if not splits:
                logger.warning("No text chunks were created from the PDF pages. Cannot create vector database.")
                return None, "No text chunks were created from the PDF pages."

            # Use the initialized embedding model
            if self.embedding_model is None:
                 logger.warning("Embedding model is not initialized. Cannot create vector database.")
                 return None, "Embedding model is not initialized. Cannot create vector database."

            try:
                logger.info("Creating FAISS vectorstore from document chunks using the embedding model...")
                # Use the real FAISS
                vectordb = FAISS.from_documents(splits, self.embedding_model)
                logger.info("FAISS vectorstore created successfully.")
                return vectordb, "Vector database created successfully."
            except Exception as e:
                logger.error(f"Error creating FAISS vector database: {e}", exc_info=True)
                return None, f"Failed to create vector database: {str(e)}"


    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        """
        Check if the query is relevant to the medical domain using LLM.
        Returns a tuple of (is_relevant, reason)
        """
        logger.debug(f"Checking medical relevance for query: '{query[:100]}...'")
        cache_key = {"type": "medical_relevance", "query": query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug(f"Returning cached medical relevance: {cached}")
            return cached

        if self.llm is None:
             logger.warning("LLM not initialized. Using fallback keyword check for medical relevance.")
             # Fallback if LLM is not available - simple keyword check
             medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose", "sick", "doctor"]
             if any(keyword in query.lower() for keyword in medical_keywords):
                  logger.debug("Fallback: Found medical keyword.")
                  return (True, "Fallback heuristic match")
             logger.debug("Fallback: No medical keywords found.")
             return (False, "Fallback: LLM not available and no keywords found")


        # Use local_generate which wraps self.llm
        medical_relevance_prompt = '''
        Analyze the user query. Is it related to health, medical conditions, symptoms, treatments, medication, diagnostics, or any other medical or health science topic?
        Consider both explicit medical terms and implicit health concerns. Answer concisely.

        Query: "{}"

        Return ONLY a JSON object with this format:
        {{
            "is_medical": true,
            "confidence": 0.0,
            "reasoning": "brief explanation"
        }}
        '''.format(query)
        logger.debug(f"Medical relevance prompt (first 100 chars): {medical_relevance_prompt[:100]}...")

        try:
            logger.info("Calling LLM for medical relevance check...")
            response = self.local_generate(medical_relevance_prompt, max_tokens=150) # Reduced max_tokens for speed
            logger.debug(f"LLM response for medical relevance: {response}")
            json_match = re.search(r'\{[\s\S]*\}', response)

            if json_match:
                json_str = json_match.group(0)
                logger.debug(f"Found JSON in LLM response: {json_str}")
                try:
                    data = json.loads(json_str)
                    is_medical = data.get("is_medical", False)
                    confidence = data.get("confidence", 0.0)
                    reasoning = data.get("reasoning", "")
                    threshold = THRESHOLDS.get("medical_relevance", 0.6)
                    logger.debug(f"Parsed relevance data: is_medical={is_medical}, confidence={confidence}, threshold={threshold}")

                    # Use a confidence threshold
                    if is_medical and confidence >= threshold:
                        logger.info(f"Query deemed medically relevant (confidence {confidence} >= {threshold}).")
                        result = (True, "")
                    else:
                        logger.info(f"Query deemed NOT medically relevant (is_medical={is_medical} or confidence {confidence} < {threshold}). Reasoning: {reasoning}")
                        result = (False, reasoning)

                    set_cached(cache_key, result)
                    return result
                except json.JSONDecodeError as json_e:
                    logger.warning(f"Could not parse medical relevance JSON from LLM response: {json_e}. Falling back to keywords.")
                    # Fallback if JSON parsing fails
                    medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose"]
                    if any(keyword in query.lower() for keyword in medical_keywords):
                         logger.debug("Fallback (JSON parse failed): Found medical keyword.")
                         return (True, "Fallback heuristic match")
                    logger.debug("Fallback (JSON parse failed): No medical keywords found.")
                    return (False, "Fallback: JSON parsing failed and no keywords found")
            else:
                logger.warning("Could not find JSON object in LLM response for medical relevance. Falling back to keywords.")
                # Fallback if no JSON found
                medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose"]
                if any(keyword in query.lower() for keyword in medical_keywords):
                     logger.debug("Fallback (JSON not found): Found medical keyword.")
                     return (True, "Fallback heuristic match")
                logger.debug("Fallback (JSON not found): No medical keywords found.")
                return (False, "Fallback: JSON parsing failed and no keywords found")


        except Exception as e:
            logger.error(f"Error checking medical relevance via LLM: {e}", exc_info=True)
            # Fallback if LLM call fails
            logger.warning("Falling back to keyword check due to LLM error.")
            medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose"]
            if any(keyword in query.lower() for keyword in medical_keywords):
                 logger.debug("Fallback (LLM error): Found medical keyword.")
                 return (True, "Fallback heuristic match")
            logger.debug("Fallback (LLM error): No medical keywords found.")

        # Default fallback - treat as non-medical if everything else fails
        logger.warning("Medical relevance check defaulted to False after multiple failures.")
        return (False, "Default: Check failed")


    def initialize_qa_chain(self):
        """Initialize the QA chain with Gemini Flash 1.5 and vector database."""
        # Check if already initialized to prevent redundant work
        if self.qa_chain is not None and self.llm is not None and self.vectordb is not None and self.embedding_model is not None:
            logger.info("QA Chain and critical components are already initialized.")
            # Generate status message based on current state
            status_parts = []
            if self.llm is not None: status_parts.append("LLM OK")
            else: status_parts.append("LLM Failed")
            if self.embedding_model is not None: status_parts.append("Embeddings OK")
            else: status_parts.append("Embeddings Failed")
            if self.vectordb is not None: status_parts.append("Vector DB OK")
            else: status_parts.append("Vector DB Failed")
            if self.qa_chain is not None: status_parts.append("RAG Chain OK")
            else: status_parts.append("RAG Chain Failed")
            if self.kg_connection_ok: status_parts.append("KG OK")
            else: status_parts.append("KG Failed")
            overall_message = f"Initialization Status: {', '.join(status_parts)}."
            overall_success = self.llm is not None and self.qa_chain is not None # RAG requires both
            return overall_success, overall_message

        logger.info("Attempting to initialize QA chain and components...")
        # This method sets up self.llm, self.embedding_model, self.vectordb, and self.qa_chain
        # self.embedding_model is already initialized in __init__ (check status)
        # self.kg_driver is initialized in __init__ (check status)

        llm_init_success = False
        llm_init_message = "LLM initialization skipped."

        # --- 1. Initialize LLM ---
        if self.llm is None: # Only initialize if not already done
            logger.info("Initializing LLM component...")
            if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_PLACEHOLDER":
                logger.warning("Gemini API key not set or is placeholder. LLM will not be initialized.")
                self.llm = None
                llm_init_message = "Gemini API key not found or invalid."
            else:
                try:
                    logger.info("Initializing Gemini Flash 1.5 via ChatGoogleGenerativeAI...")
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=GEMINI_API_KEY,
                        temperature=0.3, # Adjust temperature
                        top_p=0.95,
                        top_k=40,
                        convert_system_message_to_human=True
                    )
                    # Test the model (important check)
                    try:
                         logger.debug("Testing Gemini LLM connection...")
                         test_response = self.llm.invoke("Hello, are you ready?")
                         if test_response and test_response.content:
                             logger.info(f"Gemini LLM test successful. Response snippet: {test_response.content[:50]}...")
                             llm_init_success = True
                             llm_init_message = "Gemini Flash 1.5 initialized successfully."
                         else:
                            # Raise specific error if response is empty
                            raise ValueError("LLM test response was empty or invalid.")
                    except Exception as test_e:
                         logger.warning(f"Initial Gemini test failed: {test_e}. LLM set to None.", exc_info=True)
                         self.llm = None # Set back to None if test fails
                         llm_init_success = False
                         llm_init_message = f"Gemini LLM test failed: {test_e}"

                except Exception as e:
                    logger.error(f"Failed to initialize Gemini Flash 1.5: {e}", exc_info=True)
                    self.llm = None # Ensure LLM is None on failure
                    llm_init_success = False
                    llm_init_message = f"Failed to initialize Gemini Flash 1.5: {str(e)}"
        else:
            logger.info("LLM component already initialized.")
            llm_init_success = True
            llm_init_message = "Gemini Flash 1.5 was already initialized."


        # --- 2. Create Vector Database (Requires Embedding Model) ---
        vdb_message = "Vector database initialization skipped."
        if self.vectordb is None: # Only initialize if not already done
            logger.info("Initializing Vector DB component...")
            if self.embedding_model is None:
                 logger.warning("Embedding model not initialized. Cannot create vector database.")
                 self.vectordb = None
                 vdb_message = "Embedding model not initialized."
            else:
                 # Create the vector database
                 logger.info("Calling create_vectordb method...")
                 self.vectordb, vdb_message = self.create_vectordb()
                 if self.vectordb is None:
                      logger.warning(f"Vector DB creation failed: {vdb_message}")
                 else:
                      logger.info(f"Vector database initialization result: {vdb_message}")
        else:
            logger.info("Vector DB component already initialized.")
            vdb_message = "Vector DB was already initialized."


        # --- 3. Create Retrieval QA Chain ---
        chain_message = "Retrieval chain initialization skipped."
        if self.qa_chain is None: # Only initialize if not already done
            logger.info("Initializing RAG Chain component...")
            try:
                 # The real ConversationalRetrievalChain requires a functional LLM and a retriever from a vectorstore
                 if self.llm is not None and self.vectordb is not None:
                      logger.info("Creating Real Conversational Retrieval Chain (LLM and Vector DB available).")
                      memory = ConversationBufferMemory(
                         memory_key="chat_history", # This key must match the chain's memory key
                         output_key='answer', # Output key from the chain
                         return_messages=True # Keep messages in list format
                      )
                      logger.debug("ConversationBufferMemory created.")

                      self.qa_chain = ConversationalRetrievalChain.from_llm(
                          self.llm, # Pass the real LLM
                          retriever=self.vectordb.as_retriever(search_kwargs={"k": 5}), # Pass real retriever
                          chain_type="stuff", # Use the 'stuff' chain type
                          memory=memory, # Pass memory
                          return_source_documents=True, # Ensure source documents are returned
                          verbose=False, # Set to True for detailed logs from Langchain
                      )
                      logger.info("Real Conversational Retrieval Chain initialized successfully.")
                      chain_message = "Real Conversational Retrieval Chain initialized."
                 else:
                      missing_comps = []
                      if self.llm is None: missing_comps.append("LLM")
                      if self.vectordb is None: missing_comps.append("Vector DB")
                      logger.warning(f"Cannot create real Retrieval Chain: Missing components: {', '.join(missing_comps)}.")
                      self.qa_chain = None # Ensure qa_chain is None if creation fails
                      chain_message = f"Retrieval chain requires {', '.join(missing_comps)}."


            except Exception as e:
                logger.error(f"Failed to create Retrieval Chain: {e}", exc_info=True)
                # Set qa_chain to None to indicate failure
                self.qa_chain = None
                chain_message = f"Failed to create Retrieval Chain: {str(e)}"
        else:
             logger.info("RAG Chain component already initialized.")
             chain_message = "RAG Chain was already initialized."


        # Determine overall success message
        status_parts = []
        if self.llm is not None: status_parts.append("LLM OK")
        else: status_parts.append("LLM Failed")

        if self.embedding_model is not None: status_parts.append("Embeddings OK")
        else: status_parts.append("Embeddings Failed")

        if self.vectordb is not None: status_parts.append("Vector DB OK")
        else: status_parts.append("Vector DB Failed")

        if self.qa_chain is not None: status_parts.append("RAG Chain OK")
        else: status_parts.append("RAG Chain Failed")

        if self.kg_connection_ok: status_parts.append("KG OK")
        else: status_parts.append("KG Failed")

        overall_message = f"Initialization Status: {', '.join(status_parts)}." # Combine init messages

        # Return success status based on whether critical components are available for basic function
        # Basic function requires LLM. RAG requires LLM+VDB+Chain. KG requires KG driver.
        # Let's define success as LLM + RAG chain being available for full RAG functionality.
        overall_success = self.llm is not None and self.qa_chain is not None


        logger.info(f"QA Chain Initialization Result: Success={overall_success}, Message='{overall_message}'")
        return overall_success, overall_message

    def local_generate(self, prompt, max_tokens=500):
        """Generate text using Gemini Flash 1.5"""
        logger.debug(f"local_generate called with max_tokens={max_tokens}. Prompt starts with: '{prompt[:100]}...'")
        if self.llm is None:
            logger.error("LLM is not initialized, cannot generate text.")
            raise ValueError("LLM is not initialized")

        try:
            logger.info("Invoking Langchain LLM (ChatGoogleGenerativeAI)...")
            response = self.llm.invoke(prompt)
            logger.info("LLM invocation successful.")
            logger.debug(f"LLM response content snippet: {str(response.content)[:100]}...")
            return response.content
        except Exception as e:
            logger.error(f"Error generating with Langchain LLM: {e}", exc_info=True)
            # Fallback direct generation using genai
            logger.warning("Attempting fallback generation using direct google.generativeai API...")
            try:
                # Ensure genai is configured if not done globally
                if not genai.is_configured():
                     logger.debug("Configuring google.generativeai...")
                     genai.configure(api_key=GEMINI_API_KEY)

                model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Invoking direct genai model...")
                result = model.generate_content(prompt)
                logger.info("Direct genai invocation successful.")
                logger.debug(f"Direct genai response text snippet: {str(result.text)[:100]}...")
                return result.text
            except Exception as inner_e:
                logger.error(f"Error in fallback generation using direct genai: {inner_e}", exc_info=True)
                return f"Error generating response. Please try again. Details: {inner_e}"


    def generate_llm_answer(self, query: str, kg_content: Optional[str] = None, rag_content: Optional[str] = None, initial_combined_answer: Optional[str] = None, missing_elements: Optional[List[str]] = None) -> str:
        """
        Generates an LLM answer, synthesizing information from KG and RAG,
        and potentially focusing on missing elements identified.
        This is the core synthesis step in Path 2.
        """
        logger.info("➡️ Entering LLM Synthesis Step")
        logger.debug(f"   Query: '{query[:100]}...'")
        logger.debug(f"   KG Content available: {kg_content is not None and len(kg_content)>0}")
        logger.debug(f"   RAG Content available: {rag_content is not None and len(rag_content)>0}")
        logger.debug(f"   Initial Combined Answer available: {initial_combined_answer is not None and len(initial_combined_answer)>0}")
        logger.debug(f"   Missing Elements for focus: {missing_elements}")


        if self.llm is None:
            logger.warning("LLM not initialized. Skipping synthesis. Returning fallback message.")
            # Fallback response if LLM isn't available
            return "I'm currently unable to synthesize a complete answer as my core processing unit is offline. Please consult a healthcare professional."


        prompt_parts = [
            "You are a helpful medical AI assistant providing a comprehensive answer based on the provided information.",
            f"USER QUESTION: {query}"
        ]

        # Provide the initial combined answer draft to the LLM
        helpful_draft_found = False
        if initial_combined_answer and initial_combined_answer.strip() and \
           initial_combined_answer.strip() != "I found limited specific information regarding your query from my knowledge sources.":
             prompt_parts.append(f"Available Information Draft (from Knowledge Graph and Document Search):\n---\n{initial_combined_answer}\n---")
             helpful_draft_found = True
             logger.debug("Added initial combined answer draft to synthesis prompt.")
        # If no helpful draft, provide raw content if available and meaningful
        elif kg_content or rag_content:
            logger.debug("Initial draft was empty or unhelpful. Adding raw KG/RAG content if available.")
            kg_added = False
            if kg_content and kg_content.strip() and \
               kg_content.strip() not in ["Knowledge Graph information on treatments or remedies is unavailable.",
                                         "Knowledge Graph did not find specific relevant information on treatments or remedies."]:
                 prompt_parts.append(f"Available Medical Knowledge Graph Information:\n---\n{kg_content}\n---")
                 helpful_draft_found = True
                 kg_added = True
                 logger.debug("Added raw KG content to synthesis prompt.")
            rag_added = False
            if rag_content and rag_content.strip() and \
               rag_content.strip() not in ["An error occurred while retrieving information from documents.",
                                        "Document search is currently unavailable.",
                                        "I searched my documents but couldn't find specific information for that."]:
                 prompt_parts.append(f"Available Retrieved Information (Document Search):\n---\n{rag_content}\n---")
                 helpful_draft_found = True
                 rag_added = True
                 logger.debug("Added raw RAG content to synthesis prompt.")

            if not kg_added and not rag_added:
                 # If raw sources were also unhelpful, explicitly state limited info
                 prompt_parts.append("No specific relevant information was found from knowledge sources.")
                 logger.debug("Added 'no specific info found' message to synthesis prompt.")
        else:
             # If no context or draft, rely on LLM's general knowledge but with caution
             prompt_parts.append("No specific information was found from knowledge sources. Please provide a general, safe response based on your medical knowledge.")
             logger.debug("No draft or raw content provided. Added 'no specific info found' message to synthesis prompt.")


        prompt_parts.append("Please synthesize the available information (if any) to provide a helpful, accurate, and comprehensive answer to the USER QUESTION.")

        if missing_elements:
            logger.debug(f"Adding focus instructions for missing elements: {missing_elements}")
            # Refine missing elements list to be more descriptive for the LLM prompt
            missing_desc = []
            if "duration" in missing_elements: missing_desc.append("how long the symptoms have lasted")
            if "severity" in missing_elements: missing_desc.append("how severe the symptoms are")
            if "location" in missing_elements: missing_desc.append("the location of symptoms (e.g., pain location)")
            if "frequency" in missing_elements: missing_desc.append("how often symptoms occur")
            if "onset" in missing_elements: missing_desc.append("when the symptoms started")
            # Add other specific element descriptions

            if missing_desc:
                focus_text = "Ensure your answer addresses the user's question and attempts to incorporate details related to: " + ", ".join(missing_desc) + ". If this information isn't available in the provided context, acknowledge that appropriately."
                prompt_parts.append(focus_text)
                logger.debug(f"Added focus text: {focus_text}")

        prompt_parts.append("Include appropriate medical disclaimers about consulting healthcare professionals for diagnosis and treatment.")
        prompt_parts.append("Format your answer clearly and concisely using markdown.")


        prompt = "\n\n".join(prompt_parts)
        logger.debug(f"Final synthesis prompt (first 300 chars): {prompt[:300]}...")

        try:
            # Use the local_generate method for this specific LLM call
            # Increase max_tokens for synthesis, but avoid excessively large responses
            logger.info("Calling LLM for synthesis...")
            response = self.local_generate(prompt, max_tokens=1200) # Adjusted max_tokens
            logger.info("LLM synthesis successful.")
            logger.debug(f"Synthesized answer snippet: {response.strip()[:100]}...")
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating LLM synthesis answer: {e}", exc_info=True)
            return "I'm sorry, but I encountered an issue while trying to put together a complete answer to your question. Please consult a healthcare professional for personalized advice."


    def format_kg_diagnosis_with_llm(self, disease_name: str, symptoms_list: List[str], confidence: float) -> str:
        """
        Uses LLM to format the KG-identified disease and symptoms into a user-friendly statement for Path 1.
        """
        logger.info("➡️ Entering LLM Formatting for KG Diagnosis Step")
        logger.debug(f"   Disease: {disease_name}, Symptoms: {symptoms_list}, Confidence: {confidence:.2f}")

        if self.llm is None:
            logger.warning("LLM not initialized. Skipping KG diagnosis formatting. Using manual fallback.")
            fallback_symptoms_str = ", ".join(symptoms_list) if symptoms_list else "your symptoms"
            formatted_text = f"Based on {fallback_symptoms_str}, **{disease_name}** is a potential condition. This assessment is based on patterns in my knowledge base (confidence score: {confidence:.2f}). This is not a definitive diagnosis and requires professional medical evaluation."
            logger.debug(f"Returning manual fallback format: {formatted_text}")
            return formatted_text


        # Use the symptoms list that was *used to query KG* for this diagnosis
        symptoms_str = ", ".join(symptoms_list) if symptoms_list else "the symptoms you've reported"

        prompt = f"""
        You are a medical assistant tasked with explaining a potential medical condition based on symptoms derived from a knowledge graph query.
        Given a likely disease identified based on the reported symptoms, write a concise, single-paragraph statement for a user.
        The statement should mention the key symptoms provided ({symptoms_str}) and state that they MIGHT BE associated with the identified disease ({disease_name}). Use cautious language like "potential condition", "might be associated with", or "could suggest".
        Crucially, include a clear disclaimer that this is NOT a definitive diagnosis and professional medical advice is necessary. Mentioning the confidence score ({confidence:.2f}) is optional but can be included subtly if it helps convey uncertainty (e.g., "based on available patterns").
        Do NOT add references, bullet points for treatments/remedies, or other detailed information here. Keep it focused on the possibility of the condition based on the input symptoms.

        Identified disease: {disease_name}
        Symptoms considered: {symptoms_str}
        Confidence score (for internal context, do not explicitly state unless helpful): {confidence:.2f}

        Example format: "Based on the symptoms you've reported, like {symptoms_str}, these might be associated with {disease_name}, which is a potential condition identified from my knowledge base patterns. However, this is not a definitive diagnosis and requires professional medical evaluation."

        Write the statement now:
        """
        logger.debug(f"KG Diagnosis formatting prompt (first 300 chars): {prompt[:300]}...")
        try:
            logger.info("Calling LLM for KG diagnosis formatting...")
            response = self.local_generate(prompt, max_tokens=300) # Keep it concise
            logger.info("LLM KG diagnosis formatting successful.")
            logger.debug(f"Formatted KG diagnosis: {response.strip()}")
            return response.strip()
        except Exception as e:
            logger.error(f"Error formatting KG diagnosis with LLM: {e}", exc_info=True)
            # Fallback manual format if LLM call fails
            logger.warning("Using manual fallback format due to LLM error.")
            fallback_symptoms_str = ", ".join(symptoms_list) if symptoms_list else "your symptoms"
            formatted_text = f"Based on {fallback_symptoms_str}, **{disease_name}** is a potential condition identified from my knowledge base (confidence: {confidence:.2f}). This is not a definitive diagnosis and requires professional medical evaluation."
            logger.debug(f"Returning manual fallback format: {formatted_text}")
            return formatted_text


    def identify_missing_info(self, user_query: str, generated_answer: str, conversation_history: List[Tuple[str, str]]) -> Tuple[bool, List[str]]:
            """
            Identifies what CRITICAL medical information is still missing from the GENERATED ANSWER
            relative to the USER QUERY, using conversation context.
            This is used for the FINAL completeness check in Path 2.
            """
            logger.info("🕵️ Identifying missing info from generated answer (Final Check)...")
            logger.debug(f"   Query: '{user_query[:100]}...'")
            logger.debug(f"   Generated Answer Snippet: '{generated_answer[:100]}...'")
            logger.debug(f"   Conversation History Length: {len(conversation_history)}")

            if self.llm is None:
                 logger.warning("LLM not initialized. Cannot perform final completeness check. Assuming answer is complete.")
                 return (False, []) # Cannot check completeness without LLM


            # Convert conversation history to a string for context
            # Include a few recent exchanges for better understanding
            context = ""
            history_limit = 6 # Include last 3 exchanges (user+bot)
            recent_history = conversation_history[-history_limit:]
            logger.debug(f"Processing last {len(recent_history)} history entries for context.")
            for i, entry in enumerate(recent_history):
                # Ensure the entry is a tuple of length 2
                if isinstance(entry, tuple) and len(entry) == 2:
                    user_msg, bot_msg = entry

                    # Safely get string representation of user_msg
                    user_msg_str = str(user_msg) if user_msg is not None else "[Empty User Message]"
                    context += f"User: {user_msg_str}\n"
                    logger.debug(f"   History Context: Added User: {user_msg_str[:80]}...")

                    # Safely get string representation of bot_msg before formatting
                    if isinstance(bot_msg, str):
                        truncated_bot_msg = bot_msg[:300] + "..." if len(bot_msg) > 300 else bot_msg
                        context += f"Assistant: {truncated_bot_msg}\n"
                        logger.debug(f"   History Context: Added Assistant: {truncated_bot_msg[:80]}...")
                    elif bot_msg is not None:
                        # Log if it's not a string but not None (unexpected type)
                        logger.warning(f"Unexpected type in chat_history bot message at history index {-len(recent_history)+i}. Type: {type(bot_msg)}. Value: {str(bot_msg)[:50]}... Appending placeholder.")
                        context += f"Assistant: [Non-string response of type {type(bot_msg)}]\n"
                    else: # bot_msg is None
                        logger.debug(f"   History Context: Bot message at index {-len(recent_history)+i} is None. Skipping.")

                else:
                    # Log if an entry in history is not a tuple of length 2 or not a tuple at all
                    logger.warning(f"Unexpected format in chat_history entry at index {-len(recent_history)+i}. Entry: {entry}. Skipping entry.")
                    context += f"[Invalid history entry at index {-len(recent_history)+i}]\n"


            # The `generate_response` function will check `self.followup_context["round"]`
            # This function just needs to determine *if* a follow-up is logically required based on completeness.

            MISSING_INFO_PROMPT = '''
            You are a meticulous medical AI assistant reviewing a conversation to ensure the latest answer is safe and sufficiently addresses the user's core question, given the context.
            Your goal is to determine if ONE critical piece of information is *still missing* from the *latest generated answer* that PREVENTS it from being minimally helpful or safe, requiring a clarification from the user.

            Conversation history (includes latest answer):
            ---
            {context}
            ---

            USER'S INITIAL QUESTION (for core intent): "{user_query}"
            LATEST GENERATED ANSWER (evaluate this): "{generated_answer}"

            **CRITICAL EVALUATION & FOLLOW-UP DECISION:**
            1.  **Core Question:** Does the LATEST ANSWER fundamentally address the user's INITIAL QUESTION, considering all information shared in the history?
            2.  **Safety:** Does the LATEST ANSWER include necessary safety disclaimers (e.g., consult a doctor)?
            3.  **Critical Gaps:** Given the topic (symptoms, conditions discussed), is there any glaring omission of *critical safety information* or context (like symptom duration/severity/onset if highly relevant and unaddressed) in the LATEST ANSWER?
            4.  **History Review:** Did the Assistant *already ask* for this specific critical information in a previous turn within the provided history? Avoid asking again if a clear attempt was already made.
            5.  **Necessity:** Is asking for this *one* piece of missing information *absolutely necessary* to provide a significantly safer or more relevant response? Or is the current answer sufficient, albeit potentially general?

            **Output:** Based ONLY on the above evaluation, return a JSON object indicating if a follow-up is needed. If `needs_followup` is true, provide *exactly one* specific, concise question to ask the user to fill the single most critical gap identified. Do not ask if the history shows the question was already attempted.

            Return ONLY the JSON object:
            {{
                "needs_followup": true/false,
                "reasoning": "Briefly explain the decision based on the evaluation points (especially 3, 4, 5). e.g., 'Answer lacks critical symptom duration, not asked before.', or 'Answer is sufficient, covers core query with disclaimers.'",
                "missing_info_questions": [
                    {{"question": "Your single, specific follow-up question here."}}
                ]
            }}
            (Include the "missing_info_questions" array *only* if "needs_followup" is true, and limit it to *one* question.)
            '''.format(
                context=context,
                user_query=user_query,
                generated_answer=generated_answer
            )
            logger.debug(f"Final missing info check prompt (first 300 chars): {MISSING_INFO_PROMPT[:300]}...")

            try:
                # Use local_generate for this LLM call
                logger.info("Calling LLM for final completeness check...")
                response = self.local_generate(MISSING_INFO_PROMPT, max_tokens=500).strip()
                logger.debug(f"LLM response for final check:\n{response}")

                # Attempt to parse JSON
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    logger.debug(f"Found JSON in final check response: {json_str}")
                    try:
                        data = json.loads(json_str)
                        needs_followup_llm = data.get("needs_followup", False) # LLM's opinion
                        missing_info_questions_raw = data.get("missing_info_questions", [])
                        # Safely extract the question text
                        missing_info_questions = [item["question"] for item in missing_info_questions_raw if isinstance(item, dict) and "question" in item and item["question"]]
                        reasoning = data.get("reasoning", "No reasoning provided.")
                        logger.debug(f"Parsed final check data: needs_followup={needs_followup_llm}, questions={missing_info_questions}, reasoning='{reasoning}'")

                        if needs_followup_llm and missing_info_questions:
                             logger.info(f"❓ Final Check: LLM determined follow-up needed. Question: '{missing_info_questions[0]}'. Reasoning: {reasoning}")
                             return (True, missing_info_questions) # Return True and questions

                        else:
                             if needs_followup_llm:
                                 logger.warning(f"Final Check: LLM indicated followup needed but provided no valid question. Treating as no followup needed. Reasoning: {reasoning}")
                             else:
                                 logger.info(f"✅ Final Check: LLM determined answer is sufficient or followup not needed. Reasoning: {reasoning}")
                             return (False, []) # Return False and no questions

                    except json.JSONDecodeError as json_e:
                        logger.warning(f"Could not parse final missing info JSON from LLM response: {json_e}. Assuming answer is complete.")
                        return (False, [])
                    except Exception as e:
                         logger.error(f"Error processing LLM response structure in identify_missing_info: {e}", exc_info=True)
                         logger.warning("Assuming answer is complete due to structure processing error.")
                         return (False, []) # Fallback on structure error

                else:
                    logger.warning("LLM response for final check did not contain expected JSON format. Assuming answer is complete.")
                    return (False, [])

            except Exception as e:
                logger.error(f"⚠️ Error during LLM call in identify_missing_info: {e}", exc_info=True)
                logger.warning("Assuming answer is complete due to LLM call error during final check.")
                return (False, [])


    def knowledge_graph_agent(self, user_query: str, all_symptoms: List[str]) -> Dict[str, Any]:
        """
        Knowledge Graph Agent - Extracts symptoms (done before calling),
        identifies diseases, and finds treatments/remedies.
        Returns a dictionary of KG results.
        Updated to return symptom associations for the UI step.
        """
        logger.info("📚 Knowledge Graph Agent Initiated")
        logger.debug(f"   Query Context (for potential future use): '{user_query[:100]}...'")
        logger.debug(f"   Input Symptoms for KG Query: {all_symptoms}")

        # Initialize results structure
        kg_results: Dict[str, Any] = {
            "extracted_symptoms": all_symptoms, # Symptoms used for KG query
            "identified_diseases_data": [], # List of {disease, conf, matched_symp, all_kg_symp} - used internally & for UI step
            "top_disease_confidence": 0.0, # Confidence of the highest match
            "kg_matched_symptoms": [], # Symptoms from input that matched for the top disease
            "kg_treatments": [],
            "kg_treatment_confidence": 0.0,
            "kg_home_remedies": [],
            "kg_remedy_confidence": 0.0,
            "kg_content_diagnosis_data_for_llm": { # Default data for LLM formatting fallback
                 "disease_name": "an unidentifiable condition",
                 "symptoms_list": all_symptoms,
                 "confidence": 0.0
            },
            "kg_content_other": "Medical Knowledge Graph information on treatments or remedies is unavailable.", # Default message for other content part
        }
        logger.debug("Initialized KG results structure.")

        if not self.kg_connection_ok or self.kg_driver is None:
             logger.warning("📚 KG Agent: Connection not OK or driver is None. Skipping KG queries.")
             kg_results["kg_content_other"] = "Medical Knowledge Graph is currently unavailable."
             # KG results remain empty/default
             return kg_results

        try:
            logger.debug("Attempting to get Neo4j session...")
            with self.kg_driver.session() as session:
                logger.debug("Neo4j session obtained.")
                # Task 1: Identify Diseases from Symptoms
                if all_symptoms:
                    logger.info(f"📚 KG Task: Identifying Diseases from symptoms: {all_symptoms}")
                    disease_data_from_kg: List[Dict[str, Any]] = self._query_disease_from_symptoms_with_session(session, all_symptoms)
                    logger.debug(f"Raw disease data from KG query: {disease_data_from_kg}")

                    if disease_data_from_kg:
                        # Store the raw data for internal use and potential UI
                        kg_results["identified_diseases_data"] = disease_data_from_kg

                        # Get data for the top disease
                        top_disease_record = disease_data_from_kg[0]
                        top_disease_name = top_disease_record["Disease"]
                        top_disease_conf = top_disease_record["Confidence"]
                        kg_results["top_disease_confidence"] = top_disease_conf
                        # Use matched symptoms from KG result (should be in KG's case)
                        kg_results["kg_matched_symptoms"] = top_disease_record.get("MatchedSymptoms", [])

                        logger.info(f"✔️ KG: Diseases Identified (Top 5): {[(d['Disease'], d['Confidence']) for d in disease_data_from_kg]}")
                        logger.info(f"✔️ KG: Top Disease: {top_disease_name}, Confidence: {top_disease_conf:.4f}")
                        logger.debug(f"   Matched Symptoms for Top Disease: {kg_results['kg_matched_symptoms']}")
                        logger.debug(f"   All KG Symptoms for Top Disease: {top_disease_record.get('AllDiseaseSymptomsKG', [])}")


                        # Task 2 & 3: Find Treatments/Remedies (if a primary disease was identified with decent confidence)
                        general_threshold = THRESHOLDS.get("knowledge_graph_general", 0.6)
                        logger.debug(f"Checking if top disease confidence ({top_disease_conf:.4f}) >= general KG threshold ({general_threshold}) for treatment/remedy query.")
                        if top_disease_conf >= general_threshold:
                            logger.info(f"📚 KG Tasks: Finding Treatments & Remedies for Top Disease: {top_disease_name}")

                            kg_treatments, kg_treatment_confidence = self._query_treatments_with_session(session, top_disease_name)
                            kg_results["kg_treatments"] = kg_treatments
                            kg_results["kg_treatment_confidence"] = kg_treatment_confidence
                            logger.info(f"✔️ KG: Treatments found for {top_disease_name}: {kg_treatments} (Avg Confidence: {kg_treatment_confidence:.4f})")

                            kg_remedies, kg_remedy_confidence = self._query_home_remedies_with_session(session, top_disease_name)
                            kg_results["kg_home_remedies"] = kg_remedies
                            kg_results["kg_remedy_confidence"] = kg_remedy_confidence
                            logger.info(f"✔️ KG: Home Remedies found for {top_disease_name}: {kg_remedies} (Avg Confidence: {kg_remedy_confidence:.4f})")
                        else:
                            logger.info(f"📚 KG Tasks: Treatments/Remedies query skipped for {top_disease_name} - Top disease confidence below threshold.")
                    else:
                        logger.info("📚 KG Task: Identify Diseases - No diseases found matching the input symptoms.")

                else:
                     logger.info("📚 KG Task: Identify Diseases skipped - No symptoms provided.")


                # Prepare data needed for the LLM formatting step if Path 1 is chosen
                # This data should be prepared even if no diseases were found, for the fallback phrasing
                logger.debug("Preparing KG data for potential LLM formatting (Path 1)...")
                kg_results["kg_content_diagnosis_data_for_llm"] = {
                      "disease_name": kg_results["identified_diseases_data"][0]["Disease"] if kg_results["identified_diseases_data"] else "an unidentifiable condition", # Use top disease or fallback
                      "symptoms_list": all_symptoms, # Use all input/confirmed symptoms for phrasing
                      "confidence": kg_results["top_disease_confidence"] # Use top confidence or 0.0
                }
                logger.debug(f"   Data for LLM diagnosis formatting: {kg_results['kg_content_diagnosis_data_for_llm']}")


                # Prepare Other KG content part (treatments/remedies) for Path 2 combination
                logger.debug("Preparing KG content (treatments/remedies) for potential Path 2 combination...")
                other_parts: List[str] = []
                if kg_results["kg_treatments"]:
                     other_parts.append("## Recommended Treatments (from Knowledge Graph)")
                     for treatment in kg_results["kg_treatments"]:
                          other_parts.append(f"- {treatment}")
                     other_parts.append("") # Add empty line for separation
                     logger.debug(f"Added {len(kg_results['kg_treatments'])} treatments to KG other content.")

                if kg_results["kg_home_remedies"]:
                     other_parts.append("## Home Remedies (from Knowledge Graph)")
                     for remedy in kg_results["kg_home_remedies"]:
                          remedy_text = remedy # Assume it's already a string
                          # Add source if available? KG typically doesn't have per-remedy sources unless modeled
                          other_parts.append(f"- {remedy_text}")
                     other_parts.append("") # Add empty line for separation
                     logger.debug(f"Added {len(kg_results['kg_home_remedies'])} remedies to KG other content.")

                kg_results["kg_content_other"] = "\n".join(other_parts).strip()
                # Only set default message if no treatments *or* remedies were actually found
                if not kg_results["kg_content_other"] and not kg_results["kg_treatments"] and not kg_results["kg_home_remedies"]:
                    # Check if any disease was found - tailor message slightly
                    if kg_results["identified_diseases_data"]:
                        top_disease_name_for_msg = kg_results["identified_diseases_data"][0]["Disease"]
                        kg_results["kg_content_other"] = f"Medical Knowledge Graph did not find specific treatments or remedies associated with {top_disease_name_for_msg}."
                        logger.debug(f"Setting KG other content: No treatments/remedies found for {top_disease_name_for_msg}.")
                    else:
                        kg_results["kg_content_other"] = "Medical Knowledge Graph did not find specific relevant information on treatments or remedies for the provided symptoms."
                        logger.debug("Setting KG other content: No treatments/remedies found (no disease identified either).")

                else:
                    logger.debug(f"Final KG other content (snippet): {kg_results['kg_content_other'][:100]}...")


                logger.info("📚 Knowledge Graph Agent Finished successfully.")
                return kg_results

        except Exception as e:
            logger.error(f"⚠️ Error within KG Agent execution: {e}", exc_info=True) # Log traceback
            # Populate with error/empty info on failure
            kg_results["kg_content_diagnosis_data_for_llm"] = {
                 "disease_name": "an unidentifiable condition",
                 "symptoms_list": all_symptoms,
                 "confidence": 0.0
            } # Still provide data for LLM formatting fallback
            kg_results["kg_content_other"] = f"An error occurred while querying the Medical Knowledge Graph. Some information may be missing. Details: {str(e)}"
            kg_results["top_disease_confidence"] = 0.0
            logger.warning("KG Agent finished with errors.")
            return kg_results


    # Helper methods to query KG with a session (reduces repetitive session handling)
    # These methods are called *by* the kg_agent
    def _query_disease_from_symptoms_with_session(self, session, symptoms: List[str]) -> List[Dict[str, Any]]:
         """Queries KG for diseases based on symptoms using an existing session."""
         logger.debug(f"Executing KG query for diseases with symptoms: {symptoms}")
         if not symptoms:
              logger.debug("No symptoms provided, skipping KG disease query.")
              return []

         # Use a cache key based on the sorted list of symptoms
         symptoms_lower_sorted = tuple(sorted([s.lower() for s in symptoms if s]))
         cache_key = {"type": "disease_matching_v2", "symptoms": symptoms_lower_sorted}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("🧠 Using cached disease match (v2).")
             return cached

         cypher_query = """
         UNWIND $symptomNames AS input_symptom_name
         MATCH (s:symptom) WHERE toLower(s.Name) = toLower(input_symptom_name) // Case-insensitive match
         MATCH (s)-[:INDICATES]->(d:disease)
         WITH d, COLLECT(DISTINCT s.Name) AS matched_symptoms_from_input // Symptoms from the input that were found and matched

         // Now, for these potential diseases, find ALL symptoms they indicate in the KG
         OPTIONAL MATCH (d)<-[:INDICATES]-(all_s:symptom)
         WITH d, matched_symptoms_from_input, COLLECT(DISTINCT all_s.Name) AS all_disease_symptoms_in_kg, size(COLLECT(DISTINCT all_s)) AS total_disease_symptoms_count, size(matched_symptoms_from_input) AS matching_symptoms_count

         // Calculate confidence based on input symptoms matching KG symptoms for the disease
         WITH d.Name AS Disease, matched_symptoms_from_input, all_disease_symptoms_in_kg,
              // Improved confidence: Jaccard index (intersection / union size)
              // Union size = total_disease_symptoms_count + size($symptomNames) - matching_symptoms_count
              // Avoid division by zero if no symptoms in KG or input
              CASE WHEN total_disease_symptoms_count = 0 OR size($symptomNames) = 0 THEN 0.0
                   ELSE (matching_symptoms_count * 1.0 / (total_disease_symptoms_count + size($symptomNames) - matching_symptoms_count))
              END AS confidence_score
                WHERE matching_symptoms_count > 0 // Only return diseases with at least one matching symptom from input
         RETURN Disease, confidence_score AS Confidence, matched_symptoms_from_input AS MatchedSymptoms, all_disease_symptoms_in_kg AS AllDiseaseSymptomsKG
         ORDER BY confidence_score DESC, matching_symptoms_count DESC // Prioritize higher confidence, then more matches
         LIMIT 5 // Limit potential diseases shown for performance/relevance
         """
         logger.debug(f"Running Cypher query for disease matching (limit 5)...")
         # logger.debug(f"Cypher Query (Disease): {cypher_query}") # Optional: Log full query

         try:
              # Pass symptom names ensuring they are strings and lowercase
              symptom_names_param = [s.lower() for s in symptoms if isinstance(s, str) and s]
              logger.debug(f"   Parameters: symptomNames = {symptom_names_param}")
              result = session.run(cypher_query, symptomNames=symptom_names_param) # Pass parameter
              records = list(result) # Consume the result iterator
              logger.debug(f"KG query executed. Found {len(records)} raw records.")

              disease_data = []
              for rec in records:
                   disease_data.append({
                        "Disease": rec["Disease"],
                        "Confidence": float(rec["Confidence"]),
                        "MatchedSymptoms": rec["MatchedSymptoms"], # List of symptom strings (KG case)
                        "AllDiseaseSymptomsKG": rec["AllDiseaseSymptomsKG"] # List of symptom strings (KG case)
                   })

              logger.debug(f"🦠 Processed KG Disease Query, returning {len(disease_data)} results.")
              set_cached(cache_key, disease_data)
              return disease_data

         except Exception as e:
              logger.error(f"⚠️ Error executing KG query for diseases: {e}", exc_info=True)
              return [] # Return empty list on failure


    def _query_treatments_with_session(self, session, disease: str) -> Tuple[List[str], float]:
         """Queries KG for treatments using an existing session."""
         logger.debug(f"Executing KG query for treatments for disease: {disease}")
         if not disease:
              logger.debug("No disease name provided, skipping KG treatment query.")
              return [], 0.0

         disease_lower = disease.lower()
         cache_key = {"type": "treatment_query_kg", "disease": disease_lower}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("🧠 Using cached KG treatments.")
             return cached

         cypher_query = """
         MATCH (d:disease)-[r:TREATED_BY]->(t:treatment)
         WHERE toLower(d.Name) = toLower($diseaseName)
         WITH t, COUNT(r) as rel_count // Count relationships for potential weighting (though simple confidence used here)
         RETURN t.Name as Treatment,
                // Simple confidence based on existence, could be refined
                CASE WHEN rel_count > 0 THEN 0.75
                     ELSE 0.0
                END as Confidence // Simplified confidence
         ORDER BY t.Name // Order alphabetically for consistency
         """ # Use parameter $diseaseName
         logger.debug(f"Running Cypher query for treatments...")
         # logger.debug(f"Cypher Query (Treatment): {cypher_query}") # Optional: Log full query

         try:
              logger.debug(f"   Parameters: diseaseName = {disease}")
              result = session.run(cypher_query, diseaseName=disease) # Pass parameter
              records = list(result)
              logger.debug(f"KG query executed. Found {len(records)} raw treatment records.")

              treatments_list: List[str] = []
              avg_confidence = 0.0

              if records:
                   # Ensure confidence is treated as float
                   treatments = [(rec["Treatment"], float(rec.get("Confidence", 0.0))) for rec in records]
                   treatments_list = [t[0] for t in treatments if t[0]] # Filter out None/empty names
                   valid_confidences = [t[1] for t in treatments if t[0]] # Get confidences for valid treatments
                   avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0

              logger.debug(f"💊 Processed KG Treatment Query for {disease}, found {len(treatments_list)} treatments. Avg Confidence: {avg_confidence:.4f}")
              result_tuple = (treatments_list, avg_confidence)
              set_cached(cache_key, result_tuple)
              return result_tuple
         except Exception as e:
              logger.error(f"⚠️ Error executing KG query for treatments: {e}", exc_info=True)
              return [], 0.0


    def _query_home_remedies_with_session(self, session, disease: str) -> Tuple[List[str], float]:
         """Queries KG for home remedies using an existing session."""
         logger.debug(f"Executing KG query for home remedies for disease: {disease}")
         if not disease:
             logger.debug("No disease name provided, skipping KG home remedy query.")
             return [], 0.0

         disease_lower = disease.lower()
         cache_key = {"type": "remedy_query_kg", "disease": disease_lower}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("🧠 Using cached KG home remedies.")
             return cached

         cypher_query = """
         MATCH (d:disease)-[r:HAS_HOMEREMEDY]->(h:homeremedy)
         WHERE toLower(d.Name) = toLower($diseaseName)
         WITH h, COUNT(r) as rel_count
         RETURN h.Name as HomeRemedy,
                CASE WHEN rel_count > 0 THEN 0.70 // Simplified confidence
                     ELSE 0.0
                END as Confidence
         ORDER BY h.Name // Order alphabetically
         """ # Use parameter $diseaseName
         logger.debug(f"Running Cypher query for home remedies...")
         # logger.debug(f"Cypher Query (Remedy): {cypher_query}") # Optional: Log full query

         try:
             logger.debug(f"   Parameters: diseaseName = {disease}")
             result = session.run(cypher_query, diseaseName=disease) # Pass parameter
             records = list(result)
             logger.debug(f"KG query executed. Found {len(records)} raw home remedy records.")

             remedies_list: List[str] = []
             avg_confidence = 0.0

             if records:
                 remedies = [(rec["HomeRemedy"], float(rec.get("Confidence", 0.0))) for rec in records]
                 remedies_list = [r[0] for r in remedies if r[0]] # Filter out None/empty names
                 valid_confidences = [r[1] for r in remedies if r[0]]
                 avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0

             logger.debug(f"🏡 Processed KG Remedy Query for {disease}, found {len(remedies_list)} remedies. Avg Confidence: {avg_confidence:.4f}")
             result_tuple = (remedies_list, avg_confidence)
             set_cached(cache_key, result_tuple)
             return result_tuple
         except Exception as e:
             logger.error(f"⚠️ Error executing KG query for home remedies: {e}", exc_info=True)
             return [], 0.0


    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        """Extract symptoms from user query with confidence scores using LLM."""
        logger.debug(f"Attempting symptom extraction for query: '{user_query[:100]}...'")
        cache_key = {"type": "symptom_extraction", "query": user_query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("🧠 Using cached symptom extraction.")
            return cached

        llm_symptoms = []
        llm_avg_confidence = 0.0
        extraction_threshold = THRESHOLDS["symptom_extraction"]

        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform LLM symptom extraction. Proceeding with keyword fallback only.")
             # Fallback handled below
        else:
            # Use LLM for extraction first
            # Use local_generate which wraps self.llm
            SYMPTOM_PROMPT = '''
            You are a medical assistant specialized in symptom identification from text.
            Carefully analyze the user query below and extract **only** the specific medical symptoms or signs mentioned.
            For each potential symptom extracted, assign a confidence score (0.0 to 1.0) reflecting your certainty that it is indeed a symptom being described. Be conservative if unsure.
            **Crucially, return your answer ONLY in the following JSON format:**
            ```json
            {{
              "extracted_symptoms": [
                {{"symptom": "Symptom Name 1", "confidence": 0.95}},
                {{"symptom": "Symptom Name 2", "confidence": 0.70}}
              ]
            }}
            ```
            If no symptoms are found, return an empty list: `{{"extracted_symptoms": []}}`.

            User Query: "{}"
            '''.format(user_query)
            logger.debug(f"Symptom extraction prompt (first 150 chars): {SYMPTOM_PROMPT[:150]}...")

            try:
                # Use local_generate for the LLM call
                logger.info("Calling LLM for symptom extraction...")
                response = self.local_generate(SYMPTOM_PROMPT, max_tokens=500).strip()
                logger.debug(f"Raw Symptom Extraction Response:\n{response}")

                # Enhanced JSON parsing: look for ```json ... ``` block first, then standalone {}
                json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", response)
                if not json_match:
                    json_match = re.search(r"\{[\s\S]*\}", response) # Fallback to any JSON structure

                if json_match:
                    json_str = json_match.group(1) if len(json_match.groups()) == 1 else json_match.group(0)
                    logger.debug(f"Found JSON for symptoms: {json_str}")
                    try:
                        data = json.loads(json_str)
                        symptom_data = data.get("extracted_symptoms", [])

                        if isinstance(symptom_data, list):
                            # Filter symptoms based on confidence threshold
                            llm_symptoms_confident = []
                            all_confidences = []
                            for item in symptom_data:
                                if isinstance(item, dict) and "symptom" in item and "confidence" in item:
                                    symptom_name = item["symptom"].strip()
                                    confidence = float(item.get("confidence", 0))
                                    all_confidences.append(confidence)
                                    if symptom_name and confidence >= extraction_threshold:
                                        llm_symptoms_confident.append(symptom_name) # Keep original casing from LLM
                                    else:
                                        logger.debug(f"LLM symptom '{symptom_name}' ignored (confidence {confidence} < {extraction_threshold} or empty name).")
                                else:
                                    logger.warning(f"Invalid item format in symptom_data list: {item}")


                            # Calculate average confidence for all valid symptoms returned by LLM before thresholding
                            llm_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

                            llm_symptoms = llm_symptoms_confident # Use the thresholded list
                            logger.info(f"🔍 LLM Extracted Symptoms (>= {extraction_threshold} confidence): {llm_symptoms} (Avg raw confidence: {llm_avg_confidence:.4f})")
                        else:
                            logger.warning("LLM response JSON 'extracted_symptoms' key did not contain a list.")

                    except json.JSONDecodeError as json_e:
                        logger.warning(f"Could not parse symptom JSON from LLM response: {json_e}")
                    except ValueError as val_e:
                        logger.warning(f"Error converting confidence to float in symptom extraction: {val_e}")
                    except Exception as ex:
                         logger.error(f"Unexpected error processing LLM symptom JSON: {ex}", exc_info=True)
                else:
                     logger.warning("Could not find valid JSON ('```json...```' or '{...}') in LLM symptom response.")

            except Exception as e:
                logger.error(f"Error in LLM symptom extraction call: {e}", exc_info=True)

        # Fallback/Enhancement with Keyword Matching
        logger.debug("Applying keyword matching for symptoms as fallback/enhancement...")
        fallback_symptoms_from_keywords = []
        common_symptom_keywords = ["fever", "cough", "headache", "sore throat", "nausea", "dizziness", "chest pain", "shortness of breath", "fatigue", "body aches", "runny nose", "congestion", "chills", "sweats", "joint pain", "muscle aches", "rash", "swelling", "pain", "ache", "burning", "itching", "numbness", "tingling", "diarrhea", "vomiting", "difficulty breathing", "difficulty swallowing", "abdominal pain", "back pain", "bloating", "constipation", "ear pain", "eye pain", "loss of appetite", "loss of smell", "loss of taste", "palpitations", "weight loss", "weight gain", "wheezing"] # Expanded list
        query_lower = user_query.lower()

        for symptom_keyword in common_symptom_keywords:
            # Simple keyword presence check - use word boundaries for more precision? (e.g., r'\b' + symptom + r'\b')
            # Let's stick to simple substring for now to catch variations like "headaches"
            if symptom_keyword in query_lower:
                # Capitalize first letter for consistency, rest lowercase
                formatted_symptom = symptom_keyword.capitalize()
                fallback_symptoms_from_keywords.append(formatted_symptom)
                logger.debug(f"Keyword match found: '{formatted_symptom}'")

        # Combine LLM and keyword results
        # Use lowercase for deduplication, then restore original case prefering LLM's if available
        llm_symptoms_lower = {s.lower(): s for s in llm_symptoms}
        keyword_symptoms_lower = {s.lower(): s for s in fallback_symptoms_from_keywords}

        combined_symptoms_lower = set(llm_symptoms_lower.keys()) | set(keyword_symptoms_lower.keys())

        final_symptoms_list = []
        for s_lower in combined_symptoms_lower:
            # Prefer LLM casing if it exists, otherwise use keyword casing
            final_symptoms_list.append(llm_symptoms_lower.get(s_lower, keyword_symptoms_lower.get(s_lower)))

        # Assign final confidence
        # If LLM extraction yielded confident results, use its average raw confidence.
        # Otherwise, if keywords found anything, use a lower fixed confidence.
        # If neither found anything, confidence is 0.
        if llm_symptoms:
            final_confidence = llm_avg_confidence
            logger.debug(f"Using LLM avg confidence ({final_confidence:.4f}) as final confidence.")
        elif final_symptoms_list: # Keywords found something, LLM didn't (or failed)
            final_confidence = 0.4 # Lower confidence for keyword-only or combined-if-LLM-failed
            logger.debug(f"Using fallback confidence ({final_confidence:.4f}) as final confidence (LLM failed/empty, keywords found).")
        else: # Nothing found
            final_confidence = 0.0
            logger.debug("No symptoms found by LLM or keywords. Final confidence is 0.0.")

        logger.info(f"🔍 Final Extracted Symptoms: {final_symptoms_list} (Confidence: {final_confidence:.4f})")

        result = (final_symptoms_list, final_confidence)
        set_cached(cache_key, result)
        return result


    def is_disease_identification_query(self, query: str) -> bool:
        """Improved check for queries primarily focused on identifying a disease from symptoms."""
        logger.debug(f"Checking if query is for disease identification: '{query[:100]}...'")
        query_lower = query.lower()

        # Keywords/patterns that strongly suggest disease identification
        disease_keywords = [
            r"what disease", r"what condition", r"what could be causing",
            r"what might be causing", r"possible disease", r"possible condition",
            r"diagnosis", r"diagnose", r"what causes", r"what is causing",
            r"what do i have", r"what do they have", r"could this be", r"is it possible i have",
            r"what's wrong with me", r"what does this mean", r"identify (?:a )?(?:condition|disease)", # Added more phrases
            r"symptoms.*mean", r"what does .* symptom.* indicate", r"what is .* symptom of",
            r"what about .* symptoms", r"what could be .* (?:illness|sickness)",
            r"why do i feel", r"what explains these symptoms" # Added more
        ]

        # Check for symptom mentions (using a broader keyword list or rely on symptom extraction)
        # Rely on successful symptom extraction as a strong indicator
        # Use a cached version if available for performance, otherwise extract again
        logger.debug("Checking for symptoms within the query for disease ID check...")
        extracted_symptoms, _ = self.extract_symptoms(query) # This uses caching internally
        has_symptoms = len(extracted_symptoms) > 0
        logger.debug(f"   Symptoms found during check: {has_symptoms} ({extracted_symptoms})")


        # Check for disease identification intent using regex patterns
        is_asking_for_disease = any(re.search(pattern, query_lower) for pattern in disease_keywords)
        logger.debug(f"   Asking for disease based on keywords: {is_asking_for_disease}")

        # It's a disease identification query if it explicitly asks for one AND symptoms were found,
        # or if it's a personal query structure combined with symptoms found,
        # or if it's a query that looks like a list of symptoms and asks implicitly.
        is_symptom_query_pattern_v2 = re.search(r"i have .* (?:and|with) .*\.? what could (?:it|they|this) be", query_lower) is not None
        is_personal_symptoms_query = (
             ("i have" in query_lower or "my symptoms are" in query_lower or "i'm experiencing" in query_lower or "these are my symptoms" in query_lower or "my health issue is" in query_lower or "i feel" in query_lower) and has_symptoms
        )
        # Check if the query is primarily a list of symptoms ending with a question mark
        is_symptom_list_query = False
        if has_symptoms and query_lower.strip().endswith('?'):
            # Check if a significant portion of the query words are related to the extracted symptoms
            query_words = query_lower.split()
            symptom_related_words = 0
            if query_words: # Avoid division by zero
                for word in query_words:
                    # Check if the word is part of any extracted symptom (lowercase comparison)
                    if any(s_lower in word for s_lower in [s.lower() for s in extracted_symptoms]):
                        symptom_related_words += 1
                symptom_word_ratio = symptom_related_words / len(query_words)
                if symptom_word_ratio > 0.3: # Heuristic threshold
                     is_symptom_list_query = True
                logger.debug(f"   Symptom list query check: ends with '?', has symptoms, symptom word ratio={symptom_word_ratio:.2f} -> {is_symptom_list_query}")


        # Combine conditions:
        # 1. Explicit ask + has symptoms
        # 2. Specific pattern "I have X and Y what could it be"
        # 3. Personal phrasing + has symptoms
        # 4. Looks like a list of symptoms ending in '?'
        is_disease_query = (is_asking_for_disease and has_symptoms) or \
                           is_symptom_query_pattern_v2 or \
                           is_personal_symptoms_query or \
                           is_symptom_list_query

        logger.info(f"Query ('{query[:50]}...') determined as disease identification query: {is_disease_query}")
        return is_disease_query


    def identify_missing_elements(self, user_query: str, generated_answer: str) -> List[str]:
        """
        Identifies high-level concepts (like duration, severity, specific history points)
        that *might* be missing from the answer relative to the query's potential intent.
        This is a simpler check than identify_missing_info.
        Used *before* LLM synthesis to tell the LLM what to focus on.
        """
        logger.debug("🔍 Identifying high-level potential missing elements for LLM synthesis focus...")
        logger.debug(f"   Query: '{user_query[:100]}...'")
        logger.debug(f"   Generated Answer Snippet: '{generated_answer[:100]}...'")
        missing = set()
        query_lower = user_query.lower()
        answer_lower = generated_answer.lower()

        # Simple rule-based checks based on common medical info needs
        # Check if query mentions personal symptoms ("i have", "my symptoms") or looks like a personal case
        # Re-use the more robust check here
        is_personal_symptom_query = self.is_disease_identification_query(user_query)
        logger.debug(f"   Is personal symptom query (for missing elements check): {is_personal_symptom_query}")

        if is_personal_symptom_query:
            # Look for common ways duration is mentioned in the answer
            duration_keywords_in_answer = [" duration", "days", "weeks", "months", "how long", "since", "for x time", "lasted", "started"] # Added 'lasted', 'started' overlaps onset
            if not any(kw in answer_lower for kw in duration_keywords_in_answer):
                logger.debug("   Missing element identified: duration")
                missing.add("duration")

            # Look for common ways severity is mentioned in the answer
            severity_keywords_in_answer = [" severity", "mild", "moderate", "severe", "how severe", "intense", "level of pain", "scale of", "intensity"] # Added more terms
            if not any(kw in answer_lower for kw in severity_keywords_in_answer):
                 logger.debug("   Missing element identified: severity")
                 missing.add("severity")

            # Location: only check if query implies localization
            localizable_symptoms_in_query = any(symptom in query_lower for symptom in ["pain", "ache", "rash", "swelling", "bruise", "tenderness", "lump", "lesion", "numbness", "tingling"])
            if localizable_symptoms_in_query:
                location_keywords_in_answer = [" location", "where", "area", "on the left", "on the right", "in the chest", "in the abdomen", "radiating", "localized", "site", "region"] # Example
                if not any(kw in answer_lower for kw in location_keywords_in_answer):
                      logger.debug("   Missing element identified: location (query mentioned localizable symptom)")
                      missing.add("location")

            # Frequency: only check if query implies episodic nature
            episodic_symptoms_in_query = any(symptom in query_lower for symptom in ["pain", "headache", "dizziness", "nausea", "palpitations", "seizure", "attack", "episode"])
            if episodic_symptoms_in_query:
                 frequency_keywords_in_answer = [" frequency", "how often", "intermittent", "constant", "sporadic", "comes and goes", "episodes", "times per day", "times per week"]
                 if not any(kw in answer_lower for kw in frequency_keywords_in_answer):
                    logger.debug("   Missing element identified: frequency (query mentioned episodic symptom)")
                    missing.add("frequency")

            # Onset: generally useful for personal queries
            onset_keywords_in_answer = [" onset", "started", "began", "when it happened", "first noticed"]
            # Onset is generally useful for any personal symptom query
            if not any(kw in answer_lower for kw in onset_keywords_in_answer):
                 logger.debug("   Missing element identified: onset")
                 missing.add("onset")

        # This is a heuristic, not a definitive check like identify_missing_info
        missing_list = sorted(list(missing)) # Sort for consistent logging
        logger.debug(f"Identified potential missing elements (for LLM focus): {missing_list}")
        return missing_list


    def combine_initial_answer_draft(self, kg_diagnosis_component: Optional[str], kg_content_other: str, rag_content: str) -> str:
         """Combines the Path 1 KG diagnosis component (if any), other KG content, and RAG content."""
         logger.info("Merging KG (diagnosis & other) and RAG results into initial draft...")
         combined_parts: List[str] = []

         # Add KG diagnosis component if generated and not empty
         if kg_diagnosis_component and kg_diagnosis_component.strip():
              logger.debug("Adding KG diagnosis component to draft.")
              combined_parts.append(kg_diagnosis_component.strip())

         # Add other KG content (treatments/remedies) if found and is not the default empty/error message
         kg_other_content_is_meaningful = False
         if kg_content_other and kg_content_other.strip():
             # Check against known non-meaningful messages
             non_meaningful_kg_other = [
                 "Medical Knowledge Graph information on treatments or remedies is unavailable.",
                 "Knowledge Graph did not find specific relevant information on treatments or remedies.",
                 "Medical Knowledge Graph is currently unavailable.", # Added connection failure case
                 # Added potential error message patterns
                 "An error occurred while querying the Medical Knowledge Graph"
             ]
             if not any(msg in kg_content_other for msg in non_meaningful_kg_other):
                 kg_other_content_is_meaningful = True

         if kg_other_content_is_meaningful:
              logger.debug("Adding other KG content (treatments/remedies) to draft.")
              # Add separation if diagnosis was already added
              if combined_parts:
                   combined_parts.append("\n\n" + kg_content_other.strip())
              else:
                   combined_parts.append(kg_content_other.strip())
         else:
              logger.debug("Skipping other KG content (empty, unavailable, or error message).")


         # Add RAG content if found and is not the default empty/error message
         rag_content_is_meaningful = False
         if rag_content and rag_content.strip():
             non_meaningful_rag = [
                 "An error occurred while retrieving information from documents.",
                 "Document search is currently unavailable.",
                 "I searched my documents but couldn't find specific information for that."
             ]
             if not any(msg in rag_content for msg in non_meaningful_rag):
                  rag_content_is_meaningful = True

         if rag_content_is_meaningful:
              logger.debug("Adding RAG content to draft.")
              # Add header and separation if other content was already added
              if combined_parts:
                   combined_parts.append("\n\n## Additional Information from Document Search\n")
                   combined_parts.append(rag_content.strip())
              else:
                   combined_parts.append(rag_content.strip())
         elif not combined_parts and not kg_other_content_is_meaningful: # Only add limited info message if absolutely nothing else was added
               logger.debug("Adding 'limited specific information' message as neither KG nor RAG provided meaningful content.")
               combined_parts.append("I found limited specific information regarding your query from my knowledge sources.")
         else:
               logger.debug("Skipping RAG content (empty, unavailable, or error message).")


         initial_combined_answer = "\n".join(combined_parts).strip()
         logger.debug(f"Initial combined answer draft created (length: {len(initial_combined_answer)}). Snippet: '{initial_combined_answer[:100]}...'")
         return initial_combined_answer


    # --- Main Response Generation Function (Orchestrator) ---

    # Updated signature to accept confirmed_symptoms and original_query_if_followup
    # Returns: (response_text, sources_list, action_flag, ui_data)
    # action_flag: "final_answer", "llm_followup_prompt", "symptom_ui_prompt", "none" (no response/action needed)
    # ui_data: None or Dict { "symptom_options": {disease_label: [symptoms]}, "original_query": str } for "symptom_ui_prompt"

    def generate_response(self, user_input: str, user_type: str = "User / Family", confirmed_symptoms: Optional[List[str]] = None, original_query_if_followup: Optional[str] = None) -> Tuple[str, List[str], str, Optional[Dict]]:
        """
        Generate response using orchestration based on Path 1 / Path 2 logic.

        Args:
            user_input: The current input from the user (could be original query or response to prompt/UI).
            user_type: The type of user ("User / Family" or "Physician").
            confirmed_symptoms: List of symptoms selected by the user from the UI, if this turn is
                                a response to a symptom confirmation prompt. None otherwise.
            original_query_if_followup: The original query that triggered a symptom UI or LLM prompt,
                                        passed back when the user responds to the prompt/UI. None otherwise.

        Returns:
            A tuple containing:
            - response_text (str): The message to display to the user (answer or prompt).
            - sources_list (List[str]): List of source references (only for final answers).
            - action_flag (str): Indicates what the UI should do ("final_answer", "llm_followup_prompt", "symptom_ui_prompt", "none").
            - ui_data (Optional[Dict]): Additional data needed by the UI if action_flag requires it (e.g., symptom options).
        """
        orchestration_start_time = datetime.now()
        logger.info(f"--- Starting generate_response ---")
        logger.info(f"   User Input: '{user_input[:100]}...'")
        logger.info(f"   User Type: {user_type}")
        logger.info(f"   Confirmed Symptoms from UI: {confirmed_symptoms}")
        logger.info(f"   Original Query if Follow-up: '{original_query_if_followup[:100] if original_query_if_followup else None}...']")
        logger.info(f"   Current followup_context: {self.followup_context}")
        logger.info(f"   Current chat_history length: {len(self.chat_history)}")


        # Determine the core query being processed in this turn
        # If confirmed_symptoms is provided, the "real" query is the one that triggered the UI (`original_query_if_followup`).
        # If original_query_if_followup is provided (and confirmed_symptoms is None), this implies a response to an LLM prompt, so the original query is the context.
        # Otherwise, the current user_input is the start of a new thread.
        if confirmed_symptoms is not None and original_query_if_followup:
            core_query_for_processing = original_query_if_followup
            logger.info(f"   Processing based on symptom confirmation for original query: '{core_query_for_processing[:100]}...'")
        elif original_query_if_followup is not None:
             core_query_for_processing = original_query_if_followup
             logger.info(f"   Processing response ('{user_input[:100]}...') to LLM follow-up for original query: '{core_query_for_processing[:100]}...'")
        else:
            core_query_for_processing = user_input
            logger.info(f"   Processing new input as core query: '{core_query_for_processing[:100]}...'")


        if not core_query_for_processing.strip():
             logger.info("Core query for processing is empty. Skipping generation.")
             return "Please provide some input.", [], "none", None # Return a simple message if input is blank


        # --- Initialization Check ---
        # Define critical components needed for *any* reasonable response (LLM)
        # and for full functionality (LLM, Embeddings, VectorDB, QA Chain, KG connection)
        llm_ready = self.llm is not None
        embeddings_ready = self.embedding_model is not None
        vdb_ready = self.vectordb is not None
        rag_chain_ready = self.qa_chain is not None
        kg_ready = self.kg_connection_ok

        # Check if critical components are missing. Attempt re-init if needed.
        if not llm_ready or not rag_chain_ready: # RAG chain implies VDB+Embeddings+LLM were needed for its init
            logger.warning("Chatbot critical components (LLM or RAG Chain) not ready. Attempting re-initialization...")
            # Pass current state for more informative message
            init_check_needed = True
            if init_check_needed:
                 success, message = self.initialize_qa_chain() # Re-attempt initialization
                 if not success:
                     # If LLM *still* failed after re-attempt, we likely can't proceed meaningfully.
                     if self.llm is None:
                          error_message = f"CRITICAL ERROR: Assistant failed to initialize core LLM component ({message}). Cannot process requests."
                          logger.critical(error_message)
                          self.log_orchestration_decision(core_query_for_processing, f"SELECTED_STRATEGY: INIT_ERROR\nREASONING: Critical LLM initialization failed: {message}", 0.0, 0.0, datetime.now() - orchestration_start_time)
                          return error_message, [], "final_answer", None
                     else:
                          # LLM is OK, but RAG chain might have failed (e.g., VDB error)
                          warning_message = f"Warning: Assistant initialization incomplete ({message}). Proceeding with potentially limited features (e.g., no document search)."
                          logger.warning(warning_message)
                          # Proceed, but RAG steps below will be skipped gracefully.
                 else:
                      logger.info(f"Re-initialization attempt successful: {message}")
        else:
             logger.info("Chatbot critical components appear initialized.")


        # --- Step 0.1: Handle User Response to Prior LLM Follow-up ---
        # Check if this input is a response to the single allowed LLM follow-up prompt.
        # This is indicated by `original_query_if_followup` being present AND `self.followup_context["round"] == 1` AND `confirmed_symptoms` being None.
        is_response_to_llm_followup = (original_query_if_followup is not None and
                                       self.followup_context["round"] == 1 and
                                       confirmed_symptoms is None)
        if is_response_to_llm_followup:
             logger.info(f"Detected response to LLM follow-up (round {self.followup_context['round']}). Processing response '{user_input[:50]}...' in context of original query '{original_query_if_followup[:50]}...'.")
             # `core_query_for_processing` is already set correctly above.
             # The user's *response* text (`user_input`) will be added to history later.
             # RAG chain memory will pick up the full history including the prompt and the response.


        # --- Step 2: Extract Symptoms ---
        # Combine symptoms:
        # 1. From the core query being processed.
        # 2. From confirmed symptoms (if provided via UI).
        # 3. If it's a response to an LLM follow-up, also extract from the *user's response text* (`user_input`).
        logger.info("--- Step 2: Extracting Symptoms ---")
        symptom_confidence = 0.0
        all_symptoms_set = set() # Use a set for automatic deduplication

        # Extract from core query first
        logger.debug(f"Extracting symptoms from core query: '{core_query_for_processing[:100]}...'")
        extracted_symptoms_from_core_query, extracted_conf_core = self.extract_symptoms(core_query_for_processing)
        logger.debug(f"   Core query symptoms: {extracted_symptoms_from_core_query} (Conf: {extracted_conf_core:.2f})")
        all_symptoms_set.update([s.lower() for s in extracted_symptoms_from_core_query])
        symptom_confidence = max(symptom_confidence, extracted_conf_core)

        # Add confirmed symptoms if available
        if confirmed_symptoms is not None:
            logger.info(f"Adding confirmed symptoms from UI: {confirmed_symptoms}")
            all_symptoms_set.update([s.lower() for s in confirmed_symptoms])
            # Boost confidence significantly if user confirmed via UI
            symptom_confidence = max(symptom_confidence, 0.9) # Assume high confidence
            logger.debug(f"   Symptoms after adding confirmed: {all_symptoms_set} (Conf boosted to: {symptom_confidence:.2f})")

        # Extract from user's response text if it's a response to LLM follow-up
        if is_response_to_llm_followup:
            logger.debug(f"Extracting symptoms from LLM follow-up response text: '{user_input[:100]}...'")
            extracted_symptoms_from_response, response_conf = self.extract_symptoms(user_input)
            logger.debug(f"   Response text symptoms: {extracted_symptoms_from_response} (Conf: {response_conf:.2f})")
            all_symptoms_set.update([s.lower() for s in extracted_symptoms_from_response])
            symptom_confidence = max(symptom_confidence, response_conf) # Update overall confidence
            logger.debug(f"   Symptoms after adding response text: {all_symptoms_set} (Overall Conf: {symptom_confidence:.2f})")

        # Final list of symptoms (restore preferred casing if possible - simplistic approach: capitalize first letter)
        all_symptoms: List[str] = sorted([s.capitalize() for s in all_symptoms_set])
        logger.info(f"Final combined symptoms for processing: {all_symptoms} (Overall Confidence Estimate: {symptom_confidence:.2f})")


        # --- Step 3: KG Processing ---
        logger.info("--- Step 3: Processing with Knowledge Graph ---")
        t_start_kg = datetime.now()
        kg_data = {}
        top_disease_confidence = 0.0
        kg_diagnosis_data_for_llm = None
        kg_content_other = "Knowledge Graph processing skipped or failed." # Default pessimistic message

        if self.kg_connection_ok:
            # Pass the core query as context to KG agent if needed internally (optional)
            kg_data = self.knowledge_graph_agent(core_query_for_processing, all_symptoms)
            top_disease_confidence = kg_data.get("top_disease_confidence", 0.0)
            kg_diagnosis_data_for_llm = kg_data.get("kg_content_diagnosis_data_for_llm") # Data for LLM Path 1 formatting
            kg_content_other = kg_data.get("kg_content_other", "") # Treatments/Remedies
            kg_duration = (datetime.now() - t_start_kg).total_seconds()
            logger.info(f"📊 KG Processing finished. Top Disease Confidence: {top_disease_confidence:.4f} (took {kg_duration:.2f}s)")
        else:
             logger.warning("Skipping KG processing as connection is not OK.")
             kg_content_other = "Medical Knowledge Graph is currently unavailable."


        # --- Step 4: Path 1 - Diagnosis Focus & Symptom Follow-up UI (Decision Point 1) ---
        logger.info("--- Step 4: Evaluating Path 1 (Diagnosis Focus / Symptom UI) ---")
        # Condition checks for triggering Symptom Follow-up UI:
        followup_threshold = THRESHOLDS["disease_symptom_followup_threshold"]
        is_disease_query = self.is_disease_identification_query(core_query_for_processing)
        kg_found_diseases = len(kg_data.get("identified_diseases_data", [])) > 0
        # Check if KG returned potential symptoms for top diseases (needed for UI)
        kg_provided_symptom_options = any(d.get("AllDiseaseSymptomsKG") for d in kg_data.get("identified_diseases_data", [])[:5])

        logger.debug(f"   Conditions for Symptom UI:")
        logger.debug(f"      Is disease query? {is_disease_query}")
        logger.debug(f"      KG found diseases? {kg_found_diseases}")
        logger.debug(f"      Top KG confidence ({top_disease_confidence:.4f}) < Followup threshold ({followup_threshold})? {top_disease_confidence < followup_threshold}")
        logger.debug(f"      Is this the first pass (confirmed_symptoms is None)? {confirmed_symptoms is None}")
        logger.debug(f"      KG provided symptom options for UI? {kg_provided_symptom_options}")
        logger.debug(f"      LLM follow-up round == 0? {self.followup_context['round'] == 0}")
        logger.debug(f"      LLM initialized? {self.llm is not None}")
        logger.debug(f"      KG connection OK? {self.kg_connection_ok}")

        # Trigger Symptom UI?
        trigger_symptom_ui = (
           is_disease_query and
           kg_found_diseases and
           top_disease_confidence < followup_threshold and
           confirmed_symptoms is None and # IMPORTANT: Don't trigger if we *just processed* confirmed symptoms
           kg_provided_symptom_options and
           self.followup_context["round"] == 0 and # Don't trigger UI if LLM followup already used
           self.llm is not None and # Needed to process UI response later
           self.kg_connection_ok # Needed to get symptom options
        )

        if trigger_symptom_ui:
            logger.info(f"❓ Decision: Triggering Symptom Follow-up UI (Disease query, Low KG Conf: {top_disease_confidence:.4f})")
            # Prepare data for the UI checklist
            symptom_options_for_ui: Dict[str, List[str]] = {}
            # Get symptoms associated with top N diseases (e.g., top 3-5) for the UI
            relevant_diseases_for_ui = [d for d in kg_data["identified_diseases_data"][:5] if d.get("AllDiseaseSymptomsKG")]
            logger.debug(f"   Preparing UI options from {len(relevant_diseases_for_ui)} diseases.")
            for disease_data in relevant_diseases_for_ui:
                 disease_label = f"{disease_data['Disease']} (Confidence: {disease_data['Confidence']:.2f})"
                 # Ensure symptom names are strings and unique within the disease list, sorted
                 symptoms_list = sorted(list(set(str(s).strip() for s in disease_data.get("AllDiseaseSymptomsKG", []) if isinstance(s, str) and s.strip())))
                 if symptoms_list:
                    symptom_options_for_ui[disease_label] = symptoms_list
                    logger.debug(f"      Added {len(symptoms_list)} symptoms for '{disease_label}' to UI options.")

            if not symptom_options_for_ui:
                 logger.warning("Conditions met for Symptom UI, but KG did not provide valid symptom options. Proceeding to Path 2 instead.")
                 # Fall through to Path 2 logic if no options can be generated
            else:
                follow_up_prompt_text = f"""
Okay, based on '{all_symptoms}' and your query '{core_query_for_processing}', my knowledge graph suggests a few possibilities, but I need a bit more detail to refine the answer.

Could you please confirm which of the following symptoms, often associated with these possibilities, you are also experiencing? Checking the relevant boxes will help me provide a more focused response.
                """
                # Log decision
                self.log_orchestration_decision(
                    core_query_for_processing,
                    f"SELECTED_STRATEGY: SYMPTOM_UI_FOLLOWUP\nREASONING: Disease query with low KG confidence ({top_disease_confidence:.2f} < {followup_threshold}). Presenting symptom checklist.",
                    top_disease_confidence,
                    0.0, # RAG skipped
                    datetime.now() - orchestration_start_time
                )

                # Return data indicating UI action is needed
                logger.info("Returning Symptom UI prompt and data.")
                return follow_up_prompt_text.strip(), [], "symptom_ui_prompt", {"symptom_options": symptom_options_for_ui, "original_query": core_query_for_processing}

        else:
             logger.info("Decision: Not triggering Symptom Follow-up UI (Conditions not met).")


        # --- Step 5: Path 1 - Direct KG Diagnosis Component (if high confidence or after symptom confirmation) ---
        logger.info("--- Step 5: Evaluating Path 1 (Generate KG Diagnosis Component) ---")
        path1_kg_diagnosis_component = None
        # Conditions: Is disease query? AND (KG Conf >= threshold OR we just processed confirmed symptoms) AND KG actually found diseases
        is_high_conf_kg_diagnosis = is_disease_query and kg_found_diseases and top_disease_confidence >= followup_threshold
        is_post_symptom_confirmation = confirmed_symptoms is not None and kg_found_diseases # Make sure KG still found diseases after confirmation

        generate_kg_component = (is_high_conf_kg_diagnosis or is_post_symptom_confirmation)

        logger.debug(f"   Conditions for Generating KG Diagnosis Component:")
        logger.debug(f"      Is high confidence diagnosis? {is_high_conf_kg_diagnosis}")
        logger.debug(f"      Is post symptom confirmation? {is_post_symptom_confirmation}")
        logger.debug(f"      Generate KG Component? {generate_kg_component}")

        if generate_kg_component and kg_diagnosis_data_for_llm:
            if self.llm is not None: # LLM needed for formatting
                 logger.info(f"✅ Path 1: Generating KG diagnosis component using LLM (KG Conf: {top_disease_confidence:.4f}, Source: {'High Conf' if is_high_conf_kg_diagnosis else 'Post-Confirmation'}).")

                 # Use LLM to format the KG diagnosis into a user-friendly statement
                 # Pass all available symptoms (extracted + confirmed) to the formatter for phrasing
                 path1_kg_diagnosis_component = self.format_kg_diagnosis_with_llm(
                      kg_diagnosis_data_for_llm["disease_name"],
                      all_symptoms, # Use the combined list of symptoms for phrasing
                      kg_diagnosis_data_for_llm["confidence"] # Use the actual confidence calculated by KG agent
                  )
                 logger.info(f"   --- KG Diagnosis Component Generated (LLM) ---")
                 logger.debug(f"      Component Content (LLM): {path1_kg_diagnosis_component[:150]}...")

            else:
                 # Fallback to manual formatting if LLM is not available but KG found data
                 logger.warning("⚠️ LLM not available for formatting KG diagnosis. Using manual format.")
                 disease_name = kg_diagnosis_data_for_llm["disease_name"]
                 symptoms_str = ", ".join(all_symptoms) if all_symptoms else "your symptoms"
                 path1_kg_diagnosis_component = f"Based on {symptoms_str}, **{disease_name}** seems like a potential condition according to my knowledge base (confidence score: {kg_diagnosis_data_for_llm['confidence']:.2f}). Please remember, this is not a definitive diagnosis and requires professional medical evaluation."
                 logger.info(f"   --- KG Diagnosis Component Generated (Manual) ---")
                 logger.debug(f"      Component Content (Manual): {path1_kg_diagnosis_component[:150]}...")

        elif generate_kg_component: # Condition met, but kg_diagnosis_data_for_llm was missing (shouldn't happen if kg_found_diseases is true)
             logger.warning("⚠️ generate_kg_component=True but kg_diagnosis_data_for_llm is missing. Cannot generate component.")
        elif is_disease_query and not kg_found_diseases:
             logger.info("⚠️ Path 1: Is disease query, but KG found no matching diseases. Generating placeholder component.")
             path1_kg_diagnosis_component = "Based on the symptoms provided, I couldn't identify a specific matching medical condition in my knowledge base at this time."
        else:
            logger.info("Path 1: Not generating KG diagnosis component (not a disease query or conditions not met).")


        # --- NEW: Decision Point 2 - Conclude with KG-only answer for high-confidence diagnosis query ---
        logger.info("--- Decision Point 2: Evaluate KG-Only Answer ---")
        # Only consider if:
        # 1. A KG diagnosis component was successfully generated.
        # 2. LLM is available (to ensure quality/disclaimers).
        # 3. It meets the high-confidence OR post-confirmation criteria.
        # 4. The original query didn't explicitly ask for treatments/remedies.

        conclude_with_kg_only = False
        if path1_kg_diagnosis_component and self.llm is not None and (is_high_conf_kg_diagnosis or is_post_symptom_confirmation):
             logger.debug("   Checking conditions for KG-only conclusion...")
             query_lower = core_query_for_processing.lower()
             asks_for_treatment = any(kw in query_lower for kw in ["treat", "medication", "cure", "what to do", "how to manage", "resolve", "therapy", "prescription", "medicine"])
             asks_for_remedy = any(kw in query_lower for kw in ["remedy", "home", "natural", "relief", "alternative", "self-care"])
             logger.debug(f"      Original query asks for treatment? {asks_for_treatment}")
             logger.debug(f"      Original query asks for remedy? {asks_for_remedy}")

             # Conclude with KG-only if diagnosis component exists, LLM is ready, criteria met, AND query didn't ask for more.
             if not asks_for_treatment and not asks_for_remedy:
                 logger.info(f"✅ Decision: Concluding with KG-only answer. Meets criteria (High Conf/Post-Confirm) and query didn't explicitly request treatment/remedy.")
                 conclude_with_kg_only = True

                 # The formatted diagnosis component includes the disclaimer via format_kg_diagnosis_with_llm
                 final_response_text = path1_kg_diagnosis_component

                 # Collect KG sources for this diagnosis component
                 all_sources: List[str] = []
                 if self.kg_connection_ok:
                      source_detail = kg_diagnosis_data_for_llm['disease_name'] if kg_diagnosis_data_for_llm else "Diagnosis Data"
                      all_sources.append(f"[Source: Medical Knowledge Graph ({source_detail})]")
                 logger.debug(f"   Sources for KG-only answer: {all_sources}")

                 # Log the orchestration decision
                 self.log_orchestration_decision(
                     core_query_for_processing,
                     f"SELECTED_STRATEGY: KG_DIAGNOSIS_ONLY\nREASONING: High KG confidence ({top_disease_confidence:.2f}) or post-confirmation, and query focused on diagnosis.",
                     top_disease_confidence,
                     0.0, # RAG was skipped
                     datetime.now() - orchestration_start_time
                 )

                 # Add to chat history *before* returning
                 # Use the user_input (what the user actually typed this turn) and the final formatted response
                 logger.info(f"Adding KG-only response for user input '{user_input[:50]}...' to chat history.")
                 self.chat_history.append((user_input, final_response_text.strip()))


                 logger.info("Returning KG-only Final Answer.")
                 # Return the formatted KG diagnosis answer as the final answer
                 return final_response_text.strip(), all_sources, "final_answer", None
             else:
                 logger.info("Decision: Proceeding to Path 2 (KG Diagnosis generated, but query asked for treatments/remedies).")
        else:
             logger.info("Decision: Not concluding with KG-only answer (conditions not met). Proceeding to Path 2.")


        # --- Step 6: Path 2 - RAG Processing ---
        logger.info("--- Step 6: Processing with RAG ---")
        t_start_rag = datetime.now()

        rag_content = ""
        rag_source_docs = []
        rag_confidence = 0.0

        # Only attempt RAG if the QA Chain is initialized (implies LLM, VDB, Embeddings are ready)
        if self.qa_chain is not None:
            try:
                 # The qa_chain handles chat history internally via its memory.
                 # We pass the *core query* to the RAG chain. If it's a response to an LLM follow-up,
                 # the memory should contain the previous turns including the bot's prompt and the user's response (`user_input`).
                 # The question driving the retrieval should be the original intent (`core_query_for_processing`).
                 logger.info(f"Invoking RAG chain with question: '{core_query_for_processing[:100]}...'")
                 # The memory automatically includes the latest user_input if it was added to history,
                 # but the question parameter drives the retrieval.
                 rag_response = self.qa_chain.invoke({"question": core_query_for_processing})
                 logger.debug(f"RAG chain response received: {str(rag_response)[:200]}...") # Log snippet of raw response

                 rag_content = rag_response.get("answer", "").strip()
                 # Clean up potential prefixes or unwanted phrases from RAG LLM step
                 if rag_content.startswith("Helpful Answer:"):
                      rag_content = rag_content.split("Helpful Answer:", 1)[-1].strip()
                      logger.debug("Removed 'Helpful Answer:' prefix from RAG content.")
                 elif not rag_content:
                      logger.warning("RAG chain returned an empty answer.")
                      rag_content = "I searched my documents but couldn't find specific information for that query."


                 # Extract RAG sources
                 rag_source_docs = rag_response.get("source_documents", [])
                 if rag_source_docs:
                      # Calculate RAG confidence (simplified)
                      # Base score 0.3, adds up to 0.7 based on up to 5 docs retrieved
                      num_docs = len(rag_source_docs)
                      rag_confidence = 0.3 + min(num_docs, 5) / 5.0 * 0.4
                      logger.info(f"RAG chain retrieved {num_docs} source documents.")
                      logger.debug(f"   Sample source doc metadata: {rag_source_docs[0].metadata if rag_source_docs else 'N/A'}")
                 else:
                      logger.info("RAG chain did not retrieve any source documents.")
                      rag_confidence = 0.0 # No confidence if no sources

                 rag_duration = (datetime.now() - t_start_rag).total_seconds()
                 logger.info(f"📊 RAG Processing finished. Confidence: {rag_confidence:.4f} (took {rag_duration:.2f}s)")

            except Exception as e:
                 logger.error(f"⚠️ Error during RAG processing: {e}", exc_info=True) # Log traceback
                 rag_content = "An error occurred while retrieving information from documents."
                 rag_source_docs = []
                 rag_confidence = 0.0
        else:
             logger.warning("Warning: RAG chain not initialized. Skipping RAG processing.")
             rag_content = "Document search is currently unavailable." # Indicate RAG skipped


        # --- Step 7: Initial Combination of Path 1 Component and RAG ---
        logger.info("--- Step 7: Combining KG/RAG into Initial Draft ---")
        initial_combined_answer = self.combine_initial_answer_draft(
             path1_kg_diagnosis_component, # May be None or a string
             kg_content_other, # Treatments/Remedies string (or unavailable msg)
             rag_content # RAG answer string (or unavailable msg)
        )
        logger.debug(f"Initial Combined Answer Draft (First 150 chars): {initial_combined_answer[:150]}...")

        # --- Step 8: Identify Missing Elements for LLM Focus ---
        logger.info("--- Step 8: Identifying Missing Elements for LLM Synthesis Focus ---")
        # Evaluate the *initial combined answer* to guide the LLM synthesis
        missing_elements_for_llm = self.identify_missing_elements(core_query_for_processing, initial_combined_answer)
        logger.debug(f"Missing elements identified for LLM focus: {missing_elements_for_llm}")


        # --- Step 9: LLM Synthesis ---
        logger.info("--- Step 9: LLM Synthesis of Combined Information ---")
        llm_synthesized_answer = ""
        if self.llm is not None: # Only attempt synthesis if LLM is available
            # Provide the initial combined answer draft and the original query to the synthesis LLM
            logger.info("Calling LLM to synthesize KG/RAG draft...")
            llm_synthesized_answer = self.generate_llm_answer(
                core_query_for_processing, # Original user query context
                # Pass raw KG/RAG content if draft is minimal? No, pass the draft.
                initial_combined_answer=initial_combined_answer, # The combined draft
                missing_elements=missing_elements_for_llm # Tell LLM to focus on these
            ).strip()
            logger.info("LLM synthesis complete.")
            logger.debug(f"LLM Synthesized Answer (First 150 chars): {llm_synthesized_answer[:150]}...")
        else:
            logger.warning("LLM not initialized. Skipping synthesis step. Using initial combined answer draft directly.")
            llm_synthesized_answer = initial_combined_answer.strip() # Use draft directly as fallback
            if not llm_synthesized_answer: # Handle case where draft was also empty
                 llm_synthesized_answer = "I apologize, but I encountered difficulties processing your request and retrieving information. Please consult a healthcare professional."
                 logger.warning("Both LLM synthesis and initial draft failed. Using generic error message.")

        final_core_answer = llm_synthesized_answer # The LLM synthesis (or draft) is the core final answer


        # --- Step 10: Final Reflection Check for LLM Follow-up (Decision Point 3) ---
        logger.info("--- Step 10: Final Reflection Check for LLM Follow-up ---")
        needs_final_followup_llm_opinion = False
        missing_questions_list = []
        trigger_llm_followup = False

        # Check conditions for asking the ONE allowed LLM follow-up
        # 1. LLM is available.
        # 2. We are NOT currently processing a response to the LLM follow-up (round == 0).
        # 3. The final answer isn't just an error message.
        can_check_completeness = (self.llm is not None and
                                  self.followup_context["round"] == 0 and
                                  "error occurred" not in final_core_answer.lower() and # Avoid follow-up on errors
                                  "unable to synthesize" not in final_core_answer.lower())

        logger.debug(f"   Conditions for Final Reflection Check:")
        logger.debug(f"      LLM available? {self.llm is not None}")
        logger.debug(f"      Follow-up round == 0? {self.followup_context['round'] == 0}")
        logger.debug(f"      Answer is not error? {'error occurred' not in final_core_answer.lower() and 'unable to synthesize' not in final_core_answer.lower()}")
        logger.debug(f"      --> Can check completeness? {can_check_completeness}")


        if can_check_completeness:
            logger.info("Performing final reflection check with LLM...")
            needs_final_followup_llm_opinion, missing_questions_list = self.identify_missing_info(
                 core_query_for_processing, final_core_answer, self.chat_history # Pass current chat history state
            )
            logger.info(f"   LLM opinion on follow-up need: {needs_final_followup_llm_opinion}")
            logger.info(f"   LLM suggested questions: {missing_questions_list}")

            # Decide if the ONE allowed LLM follow-up should be asked now.
            if needs_final_followup_llm_opinion and missing_questions_list:
                 logger.info("Final reflection suggests critical info missing. Triggering the one allowed LLM follow-up.")
                 trigger_llm_followup = True
            else:
                 logger.info("Final reflection indicates answer is sufficient or no specific question provided.")
        else:
             logger.info("Skipping final reflection check (conditions not met).")


        # --- Step 11: Final LLM Follow-up Decision / Return Final Answer ---
        logger.info("--- Step 11: Final Decision (LLM Follow-up or Final Answer) ---")
        if trigger_llm_followup:
             logger.info("❓ Decision: Asking the one allowed LLM follow-up question.")
             # Construct the final LLM follow-up prompt
             follow_up_question_text = missing_questions_list[0] # Use the first question recommended

             llm_follow_up_prompt_text = f"""
Thank you for the information. To give you the most relevant and safe response possible, could you please clarify one more thing regarding your query ('{core_query_for_processing[:50]}...')?

{follow_up_question_text}

Your answer will help me refine the information.
             """
             # Mark that the one LLM follow-up has been asked for this thread
             self.followup_context["round"] = 1
             logger.info(f"   Updated followup_context: {self.followup_context}")

             # Log decision
             self.log_orchestration_decision(
                 core_query_for_processing,
                 f"SELECTED_STRATEGY: LLM_FINAL_FOLLOWUP\nREASONING: Final answer completeness check failed (LLM opinion). Asking for critical missing info: '{follow_up_question_text}'.",
                 top_disease_confidence, # Use specific KG conf if available
                 rag_confidence,
                 datetime.now() - orchestration_start_time
             )

             # Return prompt text and action flag
             # Do not add prompt to chat_history here; UI adds user msg + bot prompt together.
             logger.info("Returning LLM Follow-up prompt.")
             # Ensure the original query is passed back for context when user replies
             # Note: generate_response caller (Streamlit UI) needs to store this prompt's original query context
             # The 'prompt' variable here holds the user's *current* input, which might be a response to a *previous* prompt.
             # We need the original query that *led to this follow-up*. This is `core_query_for_processing`.
             # Let's return it in ui_data for clarity, though the main loop might handle it.
             return llm_follow_up_prompt_text.strip(), [], "llm_followup_prompt", {"original_query": core_query_for_processing}

        else:
            # This is the end of the processing path, return the final answer.
            logger.info("✅ Decision: Providing Final Answer.")

            # Collect all sources (RAG docs + KG mentions)
            logger.debug("Collecting and formatting sources...")
            all_sources: List[str] = []
            # Add RAG sources
            if rag_source_docs:
                logger.debug(f"Processing {len(rag_source_docs)} RAG source documents.")
                for i, doc in enumerate(rag_source_docs):
                    source_str = f"[Doc {i+1}]"
                    page_info = ""
                    doc_name = "Unknown Document"
                    if hasattr(doc, "metadata"):
                         if "source" in doc.metadata:
                            doc_name = os.path.basename(doc.metadata["source"]) # Get filename
                         if "page" in doc.metadata:
                             try:
                                 # Add 1 to page number if it's zero-indexed
                                 page_num_display = int(doc.metadata["page"]) + 1
                                 page_info = f", Page {page_num_display}"
                             except ValueError:
                                 page_info = f", Page {doc.metadata['page']}" # Keep as string if not int
                    # Include snippet
                    snippet = doc.page_content[:80].replace('\n', ' ').strip() + '...' if doc.page_content else ''
                    all_sources.append(f"{source_str} {doc_name}{page_info}: {snippet}")
            else:
                logger.debug("No RAG source documents to add.")


            # Add KG source mentions if KG contributed specific sections and connection was ok
            if self.kg_connection_ok and (kg_data.get("identified_diseases_data") or kg_data.get("kg_treatments") or kg_data.get("kg_home_remedies")):
                 kg_parts_mentioned = []
                 if kg_data.get("identified_diseases_data"): kg_parts_mentioned.append("Diagnosis Data")
                 if kg_data.get("kg_treatments"): kg_parts_mentioned.append("Treatment Data")
                 if kg_data.get("kg_home_remedies"): kg_parts_mentioned.append("Home Remedy Data")

                 if kg_parts_mentioned:
                    kg_source_str = f"[Knowledge Base] Medical KG ({', '.join(kg_parts_mentioned)})"
                    all_sources.append(kg_source_str)
                    logger.debug(f"Added KG source mention: {kg_source_str}")
                 # No need for fallback KG mention if no specific data used.
            elif self.kg_connection_ok:
                 logger.debug("KG connection was OK, but no specific KG data used in final answer components.")
            else:
                 logger.debug("KG connection was not OK, skipping KG source mention.")


            # Deduplicate and clean up source strings for final display
            all_sources_unique = sorted(list(set(s.strip() for s in all_sources if s.strip())))
            logger.debug(f"Final unique sources for display ({len(all_sources_unique)}): {all_sources_unique}")


            final_response_text = final_core_answer

            # Add references section if sources exist and section not already included by LLM
            has_reference_section_pattern = re.compile(r"##?\s*(?:References|Sources|Knowledge Base)[:\n]", re.IGNORECASE | re.DOTALL)
            has_reference_section = has_reference_section_pattern.search(final_response_text) is not None
            logger.debug(f"Checking for existing references section: {has_reference_section}")

            if not has_reference_section and all_sources_unique:
                logger.debug("Adding references section to the response.")
                references_section = "\n\n---\n**References:**\n"
                # Use bullet points for sources
                for src_display in all_sources_unique:
                     references_section += f"\n* {src_display}"

                final_response_text += references_section
            elif not all_sources_unique:
                 logger.debug("No sources to add, skipping references section.")
            else:
                 logger.debug("References section likely already present in LLM output.")


            # Add standard medical disclaimer if not already present
            has_disclaimer_pattern = re.compile(r"substitute for professional medical advice|always consult with a qualified healthcare provider", re.IGNORECASE | re.DOTALL)
            has_disclaimer = has_disclaimer_pattern.search(final_response_text) is not None
            logger.debug(f"Checking for existing medical disclaimer: {has_disclaimer}")

            if not has_disclaimer:
                logger.debug("Adding standard medical disclaimer.")
                disclaimer = "\n\n---\n**IMPORTANT MEDICAL DISCLAIMER:**\nThis information is intended for informational purposes only and does not constitute medical advice. It is not a substitute for professional medical advice, diagnosis, or treatment. "
                # Add specific warning for chest pain/shortness of breath if relevant to query or results
                query_or_symptoms_lower = (core_query_for_processing + ' '.join(all_symptoms)).lower()
                mentions_serious_symptoms = any(symptom in query_or_symptoms_lower for symptom in ["chest pain", "shortness of breath", "difficulty breathing", "severe pain", "sudden weakness", "vision loss"]) or \
                                           any(any(disease_kw.lower() in d['Disease'].lower() for disease_kw in ["heart attack", "stroke", "pulmonary embolism", "angina"]) for d in kg_data.get("identified_diseases_data", []))

                if mentions_serious_symptoms:
                     logger.debug("Adding urgent care warning to disclaimer.")
                     disclaimer += "**Seek immediate medical attention if you experience severe symptoms** such as chest pain, shortness of breath, sudden weakness or numbness, difficulty speaking, or vision changes, as these could indicate a serious condition. "

                disclaimer += "Always consult with a qualified healthcare provider regarding any medical condition or health concerns. Never disregard professional medical advice or delay in seeking it because of something you have read or heard here."
                final_response_text += disclaimer
            else:
                 logger.debug("Standard medical disclaimer likely already present.")

            # Log the final orchestration decision
            self.log_orchestration_decision(
                core_query_for_processing,
                f"SELECTED_STRATEGY: FINAL_ANSWER\nREASONING: Path 2 completed. Answer synthesized and final checks passed (or single LLM follow-up already used/skipped).",
                top_disease_confidence,
                rag_confidence,
                datetime.now() - orchestration_start_time
            )

            # Add the conversation turn to the chat history *before* returning
            # Use the user_input (what the user *actually* typed this turn) and the final formatted response
            logger.info(f"Adding final response for user input '{user_input[:50]}...' to chat history.")
            self.chat_history.append((user_input, final_response_text.strip()))

            logger.info("Returning Final Answer and sources.")
            # Return response text, sources list, and action flag
            source_strings_for_display = all_sources_unique # Use the cleaned unique list
            return final_response_text.strip(), source_strings_for_display, "final_answer", None


    # --- Existing and Adjusted Logging Functions ---

    def log_response_metrics(self, metrics):
        """Log response generation metrics to CSV for analysis."""
        # This function seems deprecated in favor of log_orchestration_decision
        # Keeping it here but commenting out the direct call for now.
        # If needed, call this from generate_response before returning a final_answer.
        logger.debug("log_response_metrics called (currently not used for primary logging).")
        # try:
        #     log_file = "response_metrics.csv"
        #     file_exists = os.path.isfile(log_file)
        #     logger.debug(f"Logging response metrics to {log_file}")

        #     metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     # Sanitize query for CSV
        #     metrics["query"] = metrics.get("query", "N/A").replace("\n", " ").replace(",", ";")

        #     # Ensure all expected keys exist
        #     fieldnames = ['timestamp', 'query', 'kg_confidence', 'rag_confidence',
        #                   'strategy', 'response_length', 'processing_time', 'source_count']
        #     row_data = {field: metrics.get(field, "N/A") for field in fieldnames}


        #     with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        #         writer = csv.DictWriter(file, fieldnames=fieldnames)

        #         if not file_exists:
        #             logger.info(f"Creating response metrics log file: {log_file}")
        #             writer.writeheader()

        #         writer.writerow(row_data)
        #     logger.debug("Response metrics logged successfully.")

        # except Exception as e:
        #     logger.error(f"⚠️ Error logging response metrics: {e}", exc_info=True)

    def log_orchestration_decision(self, query, orchestration_result, kg_confidence, rag_confidence, processing_time_delta):
        """
        Analyzes and logs the orchestration decision for monitoring and improvement.
        Extracts the strategy and reasoning from the orchestration result.
        Also logs processing time.
        """
        logger.debug("Logging orchestration decision...")
        try:
            # Extract strategy and reasoning
            strategy = "UNKNOWN"
            reasoning = "Not provided"

            if "SELECTED_STRATEGY:" in orchestration_result:
                # Extract strategy robustly, handling potential multiline strategy definitions if any
                strategy_match = re.search(r"SELECTED_STRATEGY:\s*([^\n]+)", orchestration_result)
                if strategy_match:
                    strategy = strategy_match.group(1).strip()
                logger.debug(f"   Extracted Strategy: {strategy}")

            if "REASONING:" in orchestration_result:
                # Capture everything between REASONING: and the end of the string or next potential section marker
                reasoning_match = re.search(r"REASONING:(.*)", orchestration_result, re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                logger.debug(f"   Extracted Reasoning (raw): {reasoning[:150]}...")

            # Truncate reasoning if too long for CSV/logs
            max_reasoning_len = 250
            if len(reasoning) > max_reasoning_len:
                 reasoning = reasoning[:max_reasoning_len] + "..."
                 logger.debug(f"   Truncated Reasoning: {reasoning}")

            # Format processing time
            processing_time_sec = processing_time_delta.total_seconds()

            # Log the decision details clearly
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query.replace("\n", " ").replace(",", ";"), # Sanitize query
                "strategy": strategy,
                "reasoning": reasoning.replace("\n", " ").replace(",", ";"), # Sanitize reasoning
                "kg_confidence": f"{kg_confidence:.4f}",
                "rag_confidence": f"{rag_confidence:.4f}",
                "processing_time_sec": f"{processing_time_sec:.2f}"
            }

            # Print concise logging information to console
            logger.info(f"📊 Orchestration Decision: Strategy='{strategy}', KG_Conf={log_entry['kg_confidence']}, RAG_Conf={log_entry['rag_confidence']}, Time={log_entry['processing_time_sec']}s")
            logger.info(f"   Reasoning: {log_entry['reasoning']}")
            logger.debug(f"   Full Log Entry Data: {log_entry}")

            # Save to CSV file for analysis
            log_file = "orchestration_log.csv"
            file_exists = os.path.isfile(log_file)
            logger.debug(f"Saving orchestration decision to {log_file}")

            with open(log_file, mode='a', newline='', encoding='utf-8') as file:
                fieldnames = ['timestamp', 'query', 'strategy', 'reasoning', 'kg_confidence', 'rag_confidence', 'processing_time_sec']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                if not file_exists:
                    logger.info(f"Creating orchestration log file: {log_file}")
                    writer.writeheader()

                writer.writerow(log_entry)
            logger.debug("Orchestration decision saved to CSV.")

            return strategy # Return strategy for potential use elsewhere

        except Exception as e:
            logger.error(f"⚠️ Error logging orchestration decision: {e}", exc_info=True)
            return "ERROR_LOGGING"


    def reset_conversation(self):
      """Reset the conversation history and follow-up context"""
      logger.info("🔄 Resetting conversation state...")
      self.chat_history = []
      self.followup_context = {"round": 0} # Reset round counter
      # Clear the cache? Optional, might be useful
      # global CACHE
      # CACHE = {}
      # logger.info("Cache cleared during conversation reset.")
      logger.info("Conversation history and follow-up context reset.")
      return "Conversation has been reset."


# --- Streamlit UI Components and Logic ---

# Helper function to display symptom checklist in Streamlit
def display_symptom_checklist(symptom_options: Dict[str, List[str]], original_query: str):
    """
    Streamlit UI component to display symptom checkboxes from a combined list.

    Args:
        symptom_options: A dictionary where keys are disease labels (string) and
                         values are lists of associated symptoms (List[str]).
                         The function will combine symptoms from all lists.
        original_query: The user's original query that triggered the UI.
    """
    logger.info("Displaying symptom checklist UI.")
    st.subheader("Confirm Your Symptoms")
    st.info(f"To help refine the possibilities based on your query ('{original_query[:50]}...'), please check any relevant symptoms below.")

    # Use a unique key for the form based on the original query and a timestamp stored in session state
    # This ensures the form state persists correctly across reruns for *this specific* checklist interaction.
    form_key = f"symptom_confirmation_form_{abs(hash(original_query))}_{st.session_state.get('form_timestamp', 'default_ts')}"
    logger.debug(f"Using form key for symptom checklist: {form_key}")

    # Initialize a *form-specific* local set in session state to store confirmed symptoms during the current form interaction
    local_confirmed_symptoms_key = f'{form_key}_confirmed_symptoms_local'
    if local_confirmed_symptoms_key not in st.session_state:
        st.session_state[local_confirmed_symptoms_key] = set()
        logger.debug(f"Initialized NEW local symptom set for form {form_key}")
    else:
         logger.debug(f"Using EXISTING local symptom set for form {form_key} with {len(st.session_state[local_confirmed_symptoms_key])} items.")


    # --- Combine all symptoms into a single unique, sorted list ---
    logger.debug("Combining and sorting symptom options for UI display...")
    all_unique_symptoms = set()
    for disease_label, symptoms_list in symptom_options.items():
        # Ensure symptoms_list is iterable and contains strings before adding to the set
        if isinstance(symptoms_list, list):
            for symptom in symptoms_list:
                 if isinstance(symptom, str) and symptom.strip():
                      # Add stripped symptom, keep original case for display via dictionary later
                      all_unique_symptoms.add(symptom.strip().lower()) # Store lowercase for uniqueness check
        else:
             logger.warning(f"Symptoms list for disease '{disease_label}' is not a list: {symptoms_list}")

    # Create a mapping from lower case back to original case (preferring the first one encountered)
    symptom_display_map = {}
    for symptoms_list in symptom_options.values():
         if isinstance(symptoms_list, list):
              for symptom in symptoms_list:
                   if isinstance(symptom, str) and symptom.strip():
                        s_lower = symptom.strip().lower()
                        if s_lower not in symptom_display_map:
                             symptom_display_map[s_lower] = symptom.strip() # Store first original casing

    # Sort the unique lowercase symptoms alphabetically for consistent display order
    sorted_lower_symptoms = sorted(list(all_unique_symptoms))
    logger.debug(f"Total unique symptoms to display: {len(sorted_lower_symptoms)}")
    # --- End Combining Logic ---


    with st.form(form_key):
        st.markdown("**Please check all symptoms that apply to you:**")

        if not sorted_lower_symptoms:
            st.warning("No specific associated symptoms were found in the knowledge graph to display. You can still add symptoms below.")
            logger.warning("No symptom options available to display in checklist.")
        else:
            # Arrange checkboxes in columns
            num_columns = 4 # Adjust as needed
            cols = st.columns(num_columns)
            logger.debug(f"Displaying {len(sorted_lower_symptoms)} symptoms in {num_columns} columns.")
            for i, s_lower in enumerate(sorted_lower_symptoms):
                col = cols[i % num_columns]
                # Get the display name using the original casing
                symptom_display_name = symptom_display_map.get(s_lower, s_lower.capitalize()) # Fallback to capitalized lower
                # Use a unique key for each checkbox based on form and the symptom's lowercase name
                checkbox_key = f"{form_key}_checkbox_{s_lower}"
                # Check if this symptom (lowercase) was previously selected in this form render cycle
                initial_state = s_lower in st.session_state[local_confirmed_symptoms_key]

                # Create the checkbox
                is_checked = col.checkbox(symptom_display_name, key=checkbox_key, value=initial_state)

                # Update the local set based on checkbox state
                if is_checked:
                    # Add the lowercase symptom to the local set if checked
                    if s_lower not in st.session_state[local_confirmed_symptoms_key]:
                         logger.debug(f"Checkbox '{symptom_display_name}' CHECKED, adding '{s_lower}' to local set.")
                         st.session_state[local_confirmed_symptoms_key].add(s_lower)
                else:
                    # Remove from the local set if unchecked
                    if s_lower in st.session_state[local_confirmed_symptoms_key]:
                         logger.debug(f"Checkbox '{symptom_display_name}' UNCHECKED, removing '{s_lower}' from local set.")
                         st.session_state[local_confirmed_symptoms_key].discard(s_lower)
            # --- End Display Logic ---


        # Add an "Other" symptom input field
        st.markdown("**Other Symptoms (comma-separated):**")
        other_symptoms_text_key = f"{form_key}_other_symptoms_input"
        other_symptoms_text = st.text_input("Enter any additional symptoms here...", key=other_symptoms_text_key)


        # Submit button
        submit_button = st.form_submit_button("Confirm and Continue")

        if submit_button:
            logger.info(f"Symptom confirmation form '{form_key}' submitted.")
            final_confirmed_set = set(st.session_state[local_confirmed_symptoms_key]) # Copy the set

            # Process "Other" symptoms text input *on submission*
            if other_symptoms_text:
                 other_symptoms_list = [s.strip().lower() for s in other_symptoms_text.split(',') if s.strip()]
                 if other_symptoms_list:
                      logger.debug(f"Adding other symptoms from text input: {other_symptoms_list}")
                      final_confirmed_set.update(other_symptoms_list)

            final_confirmed_symptoms_list = sorted(list(final_confirmed_set))
            logger.info(f"Final confirmed symptoms after submission: {final_confirmed_symptoms_list}")

            # Store the final confirmed symptoms into the *main* session state variable for generate_response
            # Use the display casing (capitalized) for consistency with other symptom extraction path
            st.session_state.confirmed_symptoms_from_ui = [s.capitalize() for s in final_confirmed_symptoms_list]

            # Clear the UI state flags *after* submitting and processing the form, before the rerun.
            st.session_state.awaiting_symptom_confirmation = False
            logger.debug("Set awaiting_symptom_confirmation to False.")
            # Keep original_query_for_followup; generate_response needs it.

            # Set a new form timestamp for the next potential checklist interaction (if any)
            st.session_state.form_timestamp = datetime.now().timestamp()
            logger.debug(f"Updated form_timestamp to {st.session_state.form_timestamp}")

            # Clear the form-specific local symptom set to avoid carry-over if the same form key reappears
            del st.session_state[local_confirmed_symptoms_key]
            logger.debug(f"Cleared local symptom set state variable: {local_confirmed_symptoms_key}")


            # Set the original query into session state to trigger its processing in the next rerun
            st.session_state.input_ready_for_processing = {
                "text": original_query, # Use original query text
                "confirmed_symptoms": st.session_state.confirmed_symptoms_from_ui, # Pass the list we just finalized
                "original_query_context": original_query # Pass original query as context
            }
            logger.info("Set input_ready_for_processing flag with confirmed symptoms. Triggering rerun.")
            st.rerun() # Trigger rerun to process the confirmed symptoms


# Create chatbot instance using session state
def get_chatbot_instance():
    """Gets or creates the chatbot instance in session state."""
    if 'chatbot' not in st.session_state:
        logger.info("Creating new DocumentChatBot instance in session state.")
        st.session_state.chatbot = DocumentChatBot()
        # Trigger initialization immediately after creation
        with st.spinner("Initializing chat assistant..."):
             logger.info("Calling initialize_qa_chain for the new chatbot instance...")
             success, init_message = st.session_state.chatbot.initialize_qa_chain()
             st.session_state.init_status_message = init_message # Store message
             if not success:
                  logger.error(f"Initial chatbot initialization failed: {init_message}")
                  st.error(f"Initialization failed: {init_message}")
                  st.session_state.init_failed = True
             else:
                  logger.info(f"Initial chatbot initialization successful: {init_message}")
                  # Don't show success message here, show status in sidebar
                  st.session_state.init_failed = False
    return st.session_state.chatbot

# Function for feedback logging (Example - adjust as needed)
def vote_message(user_query, bot_response, vote_type, user_role):
    """Logs user feedback (thumbs up/down) to a CSV file."""
    log_file = "feedback_votes.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "user_role": user_role,
        "vote_type": vote_type,
        "user_query": user_query.replace("\n", " ").replace(",", ";"),
        "bot_response": bot_response.replace("\n", " ").replace(",", ";"),
    }
    logger.info(f"Received feedback vote: {vote_type} for query '{user_query[:50]}...'")
    try:
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            fieldnames = ["timestamp", "user_role", "vote_type", "user_query", "bot_response"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
                logger.info(f"Created feedback votes log file: {log_file}")
            writer.writerow(log_entry)
        logger.debug("Feedback vote logged to CSV.")
        return f"Feedback ({vote_type}) submitted. Thank you!"
    except Exception as e:
        logger.error(f"Error logging feedback vote to {log_file}: {e}", exc_info=True)
        return "Error submitting feedback."

def submit_feedback(feedback_text, chat_history_for_feedback, user_role):
    """Logs detailed text feedback to a CSV file."""
    log_file = "detailed_feedback.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Format chat history for logging
    history_str = ""
    for i, (user_msg, bot_msg) in enumerate(chat_history_for_feedback):
        history_str += f"Turn {i+1} User: {str(user_msg).replace(',', ';').replace(chr(10), ' ')}\n" # Replace newline char 10 too
        history_str += f"Turn {i+1} Bot: {str(bot_msg).replace(',', ';').replace(chr(10), ' ')}\n"

    log_entry = {
        "timestamp": timestamp,
        "user_role": user_role,
        "feedback_text": feedback_text.replace("\n", " ").replace(",", ";"),
        "chat_history": history_str.replace("\n", " || ").replace(",", ";"), # Sanitize history string
    }
    logger.info(f"Received detailed feedback from {user_role}: '{feedback_text[:50]}...'")
    try:
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            fieldnames = ["timestamp", "user_role", "feedback_text", "chat_history"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
                logger.info(f"Created detailed feedback log file: {log_file}")
            writer.writerow(log_entry)
        logger.debug("Detailed feedback logged to CSV.")
        return "Detailed feedback submitted. Thank you!"
    except Exception as e:
        logger.error(f"Error logging detailed feedback to {log_file}: {e}", exc_info=True)
        return "Error submitting detailed feedback."


# --- Main Streamlit App Function ---
def main():
    logger.info("--- Starting Streamlit main() ---")
    # Set page title and favicon
    try:
        st.set_page_config(
            page_title="DxAI-Agent",
            page_icon=f"data:image/png;base64,{icon}", # Use base64 encoded icon
            layout="wide"
        )
        logger.debug("Streamlit page config set.")
    except Exception as e:
        logger.error(f"Error setting page config: {e}", exc_info=True) # Log potential errors
        st.set_page_config(page_title="DxAI-Agent", layout="wide") # Fallback config

    # Initialize chatbot instance via getter function (handles creation and initial init)
    chatbot = get_chatbot_instance()
    logger.debug(f"Chatbot instance obtained. LLM Ready: {chatbot.llm is not None}, RAG Chain Ready: {chatbot.qa_chain is not None}, KG Ready: {chatbot.kg_connection_ok}")

    # --- Initialize UI State Variables ---
    # These variables control how the UI behaves in the current rerun cycle
    if 'messages' not in st.session_state:
        st.session_state.messages = [] # List of (content, is_user) tuples for UI display
        logger.debug("Initialized 'messages' in session state.")
    if 'awaiting_symptom_confirmation' not in st.session_state:
        st.session_state.awaiting_symptom_confirmation = False
        logger.debug("Initialized 'awaiting_symptom_confirmation' in session state.")
    if 'symptom_options_for_ui' not in st.session_state:
        st.session_state.symptom_options_for_ui = {}
        logger.debug("Initialized 'symptom_options_for_ui' in session state.")
    # confirmed_symptoms_from_ui is set by the checklist form, check presence later
    if 'original_query_for_followup' not in st.session_state:
        st.session_state.original_query_for_followup = ""
        logger.debug("Initialized 'original_query_for_followup' in session state.")
    if 'init_failed' not in st.session_state:
         st.session_state.init_failed = False # Flag to track initialization status
         logger.debug("Initialized 'init_failed' in session state.")
    if 'init_status_message' not in st.session_state:
         st.session_state.init_status_message = "Initializing..."
         logger.debug("Initialized 'init_status_message' in session state.")
    # Timestamp for symptom confirmation form key uniqueness
    if 'form_timestamp' not in st.session_state:
         st.session_state.form_timestamp = datetime.now().timestamp()
         logger.debug("Initialized 'form_timestamp' in session state.")
    # Flag/Data structure to hold input that needs processing in the next rerun cycle
    if 'input_ready_for_processing' not in st.session_state:
         st.session_state.input_ready_for_processing = None
         logger.debug("Initialized 'input_ready_for_processing' in session state.")


    # Title and description
    try:
        logger.debug(f"Attempting to open logo image: {image_path}")
        logo = Image.open(image_path)
        col1, col2 = st.columns([1, 10])
        with col1:
            st.image(logo, width=100)  # Adjust width as needed
        with col2:
            st.markdown("<h1 style='margin-top: 20px;'>DxAI-Agent</h1>", unsafe_allow_html=True) # Adjust margin
        logger.debug("Displayed logo and title.")
    except FileNotFoundError:
         logger.warning(f"Logo file not found at {image_path}. Displaying text title only.")
         st.markdown("# DxAI-Agent") # Fallback title if logo not found
    except Exception as e:
         logger.error(f"Error displaying logo: {e}", exc_info=True)
         st.markdown("# DxAI-Agent")


    # User type selection dropdown in sidebar
    logger.debug("Setting up sidebar.")
    user_type = st.sidebar.selectbox(
        "Who is asking?",
        ["User / Family", "Physician"],
        index=0,
        key="user_type_selector" # Add key for stability
    )
    st.sidebar.info("DxAI-Agent helps answer medical questions using its knowledge base.")
    # Display LLM follow-up attempt count from the chatbot instance
    # Ensure chatbot exists before accessing its attributes
    if chatbot:
        st.sidebar.markdown(f"LLM Final Follow-up Attempts: {chatbot.followup_context['round']} / 1")
    else:
        st.sidebar.markdown("LLM Final Follow-up Attempts: N/A")

    # Display initialization status in sidebar using stored message
    st.sidebar.subheader("System Status")
    if st.session_state.get('init_failed', False):
         st.sidebar.error(f"Initialization Failed: {st.session_state.get('init_status_message', 'Check logs.')}")
    else:
         st.sidebar.success(f"{st.session_state.get('init_status_message', 'Ready.')}")
         # Optionally add more detail if needed
         # status_details = []
         # if chatbot and chatbot.llm: status_details.append("LLM: OK") else: status_details.append("LLM: Failed/NA")
         # if chatbot and chatbot.qa_chain: status_details.append("RAG: OK") else: status_details.append("RAG: Failed/NA")
         # if chatbot and chatbot.kg_connection_ok: status_details.append("KG: OK") else: status_details.append("KG: Failed/NA")
         # st.sidebar.caption(", ".join(status_details))

    logger.debug("Sidebar setup complete.")

    # Tabs
    tab1, tab2 = st.tabs(["Chat", "About"])
    logger.debug("Created Chat and About tabs.")

    with tab1:
        logger.debug("Entering Chat tab.")
        # Examples section
        st.subheader("Try these examples")
        examples = [
            "What are treatments for cough and cold?",
            "I have a headache and sore throat. What could it be?", # Example that might trigger Path 1 / low conf
            "What home remedies help with flu symptoms?",
            "I have chest pain and shortness of breath. What could i do?" # Example that should trigger symptom UI/warning
        ]

        cols = st.columns(len(examples))
        # Disable examples if initialization failed OR if awaiting symptom confirmation
        examples_disabled = st.session_state.get('init_failed', False) or st.session_state.get('awaiting_symptom_confirmation', False)
        logger.debug(f"Examples disabled: {examples_disabled} (init_failed={st.session_state.get('init_failed', False)}, awaiting_symptoms={st.session_state.get('awaiting_symptom_confirmation', False)})")

        for i, col in enumerate(cols):
            example_key = f"example_{i}"
            if col.button(examples[i], key=example_key, disabled=examples_disabled):
                logger.info(f"Example button clicked: '{examples[i]}'")
                # Clear relevant UI state for a new thread initiated by example
                st.session_state.messages.append((examples[i], True)) # Add example as user message for display
                st.session_state.awaiting_symptom_confirmation = False
                st.session_state.symptom_options_for_ui = {}
                # confirmed_symptoms_from_ui should be None already or cleared if needed
                st.session_state.original_query_for_followup = "" # Clear any pending follow-up context
                st.session_state.form_timestamp = datetime.now().timestamp() # Reset form timestamp
                # Reset the bot's internal follow-up counter if starting fresh
                if chatbot:
                    chatbot.followup_context = {"round": 0}
                    logger.debug("Reset bot's internal follow-up context for example.")

                # Set the input into the processing flag to be picked up later in the script
                st.session_state.input_ready_for_processing = {
                    "text": examples[i],
                    "confirmed_symptoms": None,
                    "original_query_context": None
                }
                logger.info("Set input_ready_for_processing flag for example. Triggering rerun.")
                st.rerun() # Trigger rerun


        # --- Chat Messages Display ---
        st.divider()
        st.subheader("Conversation")
        # Create a container for chat messages for better scrolling/layout if needed
        chat_container = st.container() # Adjust height if needed: height=500
        with chat_container:
            logger.debug(f"Displaying {len(st.session_state.messages)} messages from session state.")
            # Iterate through messages state for display
            for i, (msg_content, is_user) in enumerate(st.session_state.messages):
                role = "user" if is_user else "assistant"
                with st.chat_message(role):
                    st.write(msg_content) # Use st.write for markdown rendering
                    logger.debug(f"Displayed message {i+1} ({role}): '{str(msg_content)[:80]}...'")

                    # Add feedback buttons only for assistant messages that are final answers
                    if not is_user:
                        # Heuristic check if it's a prompt/UI trigger vs a final answer
                        # It's likely a prompt if awaiting symptom confirmation OR if original_query_for_followup is set (meaning last bot message was likely a prompt)
                        # Or check for common prompt phrases.
                        is_prompt_or_ui_trigger = False
                        if st.session_state.get('awaiting_symptom_confirmation', False) and i == len(st.session_state.messages) - 1:
                             is_prompt_or_ui_trigger = True # Last message when awaiting UI is the prompt
                             logger.debug(f"Message {i+1} identified as symptom UI prompt.")
                        elif st.session_state.get('original_query_for_followup', "") != "" and i == len(st.session_state.messages) - 1:
                             is_prompt_or_ui_trigger = True # Last message when original_query is set is likely LLM prompt
                             logger.debug(f"Message {i+1} identified as likely LLM follow-up prompt.")
                        elif isinstance(msg_content, str):
                             prompt_phrases = ["could you please clarify", "please confirm which", "enter additional symptoms", "to help me answer better"]
                             if any(phrase in msg_content.lower() for phrase in prompt_phrases):
                                  is_prompt_or_ui_trigger = True
                                  logger.debug(f"Message {i+1} identified as prompt based on phrasing.")


                        if not is_prompt_or_ui_trigger:
                            logger.debug(f"Adding feedback buttons for assistant message {i+1}.")
                            # Add feedback buttons using unique keys per message
                            feedback_key_up = f"thumbs_up_{i}_{abs(hash(str(msg_content)))}"
                            feedback_key_down = f"thumbs_down_{i}_{abs(hash(str(msg_content)))}"

                            # Use columns for compact layout
                            b_col1, b_col2, b_col_spacer = st.columns([1, 1, 10])
                            with b_col1:
                                 if st.button("👍", key=feedback_key_up, help="Mark as helpful"):
                                     # Find the preceding user message for context
                                     user_msg_for_feedback = "[User query not found]"
                                     if i > 0 and st.session_state.messages[i-1][1] is True: # Check if previous message was user
                                         user_msg_for_feedback = st.session_state.messages[i-1][0]

                                     feedback_result = vote_message(
                                         user_msg_for_feedback, msg_content,
                                         "thumbs_up", user_type
                                     )
                                     st.toast(feedback_result, icon="👍")
                            with b_col2:
                                 if st.button("👎", key=feedback_key_down, help="Mark as not helpful"):
                                    user_msg_for_feedback = "[User query not found]"
                                    if i > 0 and st.session_state.messages[i-1][1] is True:
                                        user_msg_for_feedback = st.session_state.messages[i-1][0]
                                    feedback_result = vote_message(
                                        user_msg_for_feedback, msg_content,
                                        "thumbs_down", user_type
                                    )
                                    st.toast(feedback_result, icon="👎")
                        else:
                             logger.debug(f"Skipping feedback buttons for assistant message {i+1} (identified as prompt/UI trigger).")


        # --- Input Area and Processing Logic ---
        st.divider()
        logger.debug("Setting up input area.")

        # Conditional rendering of symptom checklist vs chat input
        if st.session_state.get('init_failed', False):
             st.error("Chat assistant failed to initialize. Cannot accept input. Please check logs and configuration.")
             logger.warning("Chat input disabled due to initialization failure.")
        elif st.session_state.get('awaiting_symptom_confirmation', False):
            logger.info("UI State: Awaiting symptom confirmation. Displaying checklist.")
            # Display the symptom checklist UI (includes form and submit button)
            display_symptom_checklist(
                st.session_state.symptom_options_for_ui,
                st.session_state.original_query_for_followup
            )
            # Hide the standard chat input while checklist is active
            st.chat_input("Confirm symptoms above...", disabled=True, key="disabled_chat_input_while_form_active")
            logger.debug("Disabled standard chat input while symptom form is active.")

        else:
            # Display the standard chat input
            logger.debug("UI State: Ready for standard chat input.")
            user_query = st.chat_input("Ask your medical question...", disabled=st.session_state.get('init_failed', False), key="main_chat_input")

            # --- Logic to Detect Chat Input Submission ---
            if user_query:
                logger.info(f"Detected chat input submission: '{user_query[:100]}...'")
                # Add user message to state immediately for display
                st.session_state.messages.append((user_query, True))
                logger.debug("Added user query to session state messages.")

                # Clear any previous follow-up state for a brand new query thread
                st.session_state.awaiting_symptom_confirmation = False
                st.session_state.original_query_for_followup = ""
                st.session_state.confirmed_symptoms_from_ui = None # Ensure this is clear
                st.session_state.form_timestamp = datetime.now().timestamp() # New timestamp for potential future forms
                if chatbot:
                     chatbot.followup_context = {"round": 0} # Reset bot's internal follow-up round
                logger.debug("Cleared follow-up/UI state for new query thread.")

                # Set the input into the processing flag
                st.session_state.input_ready_for_processing = {
                    "text": user_query,
                    "confirmed_symptoms": None, # Not from symptom form
                    "original_query_context": None # Not a follow-up response yet
                }
                logger.info("Set input_ready_for_processing flag for chat input. Triggering rerun.")
                st.rerun()


        # --- Process Input if Flag is Set ---
        # This block runs *after* chat input or form submission potentially sets the flag in the *previous* rerun.
        if 'input_ready_for_processing' in st.session_state and st.session_state.input_ready_for_processing is not None:
            input_data = st.session_state.input_ready_for_processing
            # Clear the flag immediately to prevent reprocessing in the same run or accidental loops
            st.session_state.input_ready_for_processing = None
            logger.info("Detected 'input_ready_for_processing' flag. Starting response generation.")
            logger.debug(f"   Input data for generation: {input_data}")

            prompt = input_data["text"]
            confirmed_symps = input_data["confirmed_symptoms"]
            original_query_context = input_data["original_query_context"] # This tells generate_response if it's processing a follow-up

            # Call the chatbot's generate_response function
            logger.info(f"Calling chatbot.generate_response with user_type='{user_type}'...")
            with st.spinner("Thinking..."):
                 try:
                     response_text, sources, action_flag, ui_data = chatbot.generate_response(
                          prompt, user_type,
                          confirmed_symptoms=confirmed_symps,
                          original_query_if_followup=original_query_context # Pass context
                     )
                     logger.info(f"chatbot.generate_response finished. Action Flag: {action_flag}")
                 except Exception as gen_e:
                      logger.error(f"CRITICAL ERROR during chatbot.generate_response: {gen_e}", exc_info=True)
                      response_text = f"Sorry, I encountered an unexpected error while processing your request ({type(gen_e).__name__}). Please try rephrasing or contact support if the issue persists."
                      sources = []
                      action_flag = "final_answer" # Treat as final answer to display error
                      ui_data = None


            # --- Process the action flag returned by generate_response ---
            if action_flag == "symptom_ui_prompt":
                 logger.info("Action: 'symptom_ui_prompt'. Setting state for UI display.")
                 # Update UI state to show the symptom checklist next rerun
                 st.session_state.awaiting_symptom_confirmation = True
                 st.session_state.symptom_options_for_ui = ui_data.get("symptom_options", {})
                 st.session_state.original_query_for_followup = ui_data.get("original_query", prompt) # Store query that triggered UI
                 st.session_state.form_timestamp = datetime.now().timestamp() # Set new timestamp for the form key

                 # Add the prompt message for the UI to messages list if not already there
                 if not st.session_state.messages or st.session_state.messages[-1] != (response_text, False):
                      st.session_state.messages.append((response_text, False))
                      logger.debug("Added symptom UI prompt message to state.")
                 st.rerun() # Rerun to display the form

            elif action_flag == "llm_followup_prompt":
                 logger.info("Action: 'llm_followup_prompt'. Adding prompt to messages.")
                 # Add the LLM prompt message to messages list if not already there
                 if not st.session_state.messages or st.session_state.messages[-1] != (response_text, False):
                    st.session_state.messages.append((response_text, False))
                    logger.debug("Added LLM follow-up prompt message to state.")

                 # Store the original query context returned by generate_response (or use the core query processed)
                 # This context is crucial for the *next* call to generate_response.
                 if ui_data and "original_query" in ui_data:
                    st.session_state.original_query_for_followup = ui_data["original_query"]
                 else:
                    logger.warning("LLM followup action flag set, but ui_data did not contain 'original_query'. Using current prompt as fallback context.")
                    st.session_state.original_query_for_followup = prompt
                 logger.info(f"Stored original_query_for_followup: '{st.session_state.original_query_for_followup[:50]}...'")
                 st.rerun() # Rerun to display the prompt message


            elif action_flag == "final_answer":
                 logger.info("Action: 'final_answer'. Adding response to messages.")
                 # Add the final answer message to messages list if not already there
                 if not st.session_state.messages or st.session_state.messages[-1] != (response_text, False):
                       st.session_state.messages.append((response_text, False))
                       logger.debug("Added final answer message to state.")

                 # Clear original_query_for_followup as the thread is concluded or follow-up limit reached
                 st.session_state.original_query_for_followup = ""
                 # Clear symptom UI specific state too just in case
                 st.session_state.awaiting_symptom_confirmation = False
                 st.session_state.symptom_options_for_ui = {}
                 logger.debug("Cleared follow-up/UI state after final answer.")
                 st.rerun() # Rerun to display the final answer and feedback buttons


            elif action_flag == "none":
                 logger.info("Action: 'none'. No UI update needed from generate_response result.")
                 # No action needed, do not add to messages, do not rerun unless necessary for other reasons
                 pass

        # --- Reset Conversation Button ---
        st.divider()
        if st.button("Reset Conversation", key="reset_conversation_button_chat"):
            logger.info("Reset Conversation button clicked.")
            if chatbot:
                reset_message = chatbot.reset_conversation() # Resets internal history and followup_context
                logger.info(f"Chatbot reset method returned: {reset_message}")
            st.session_state.messages = [] # Clear UI messages
            # Also reset UI state variables
            st.session_state.awaiting_symptom_confirmation = False
            st.session_state.symptom_options_for_ui = {}
            # confirmed_symptoms_from_ui should be cleared if it existed
            if 'confirmed_symptoms_from_ui' in st.session_state:
                del st.session_state['confirmed_symptoms_from_ui']
            st.session_state.original_query_for_followup = ""
            st.session_state.input_ready_for_processing = None # Clear any pending input
            st.session_state.form_timestamp = datetime.now().timestamp() # Reset form timestamp
            logger.info("Cleared UI state variables for reset.")
            st.toast("Conversation Reset!", icon="🔄")
            st.rerun()


        # Physician feedback section
        st.divider()
        st.subheader("🩺 Detailed Feedback (Optional)")
        with st.form("feedback_form"):
            feedback_text = st.text_area(
                "Enter corrections, improvements, or comments about the conversation here...",
                height=100,
                key="detailed_feedback_text"
            )
            submit_feedback_btn = st.form_submit_button("Submit Detailed Feedback")

            if submit_feedback_btn and feedback_text:
                logger.info("Detailed feedback form submitted.")
                # Use the chatbot's internal chat_history for feedback context
                history_for_feedback = chatbot.chat_history if chatbot else []
                logger.debug(f"Submitting feedback with {len(history_for_feedback)} history turns.")
                feedback_result = submit_feedback(feedback_text, history_for_feedback, user_type)
                st.success(feedback_result)
            elif submit_feedback_btn:
                st.warning("Please enter some feedback before submitting.")


    with tab2:
        logger.debug("Entering About tab.")
        st.markdown("""
        ## About DxAI-Agent

        This application acts as a medical information assistant, leveraging several AI techniques to process your queries.

        **Core Components:**
        *   **Symptom Extraction:** Uses Language Models (LLMs) and keyword matching to identify medical symptoms in your text. (Requires LLM)
        *   **Knowledge Graph (KG):** Queries a Neo4j graph database containing medical relationships (symptoms, diseases, treatments, remedies). (Requires Neo4j Connection)
        *   **Retrieval-Augmented Generation (RAG):** Searches local PDF documents using vector embeddings and summarizes relevant findings with an LLM. (Requires PDFs, Embedding Model, Vector DB, LLM)
        *   **LLM Synthesis & Reasoning:** Employs an LLM (Gemini Flash 1.5) to combine information from KG and RAG, generate coherent answers, format responses, and perform reasoning steps like completeness checks. (Requires LLM)

        **Processing Workflow:**

        1.  **Input & Symptom Extraction:** Your query is received, and symptoms are extracted.
        2.  **Knowledge Graph Query:** The KG is queried using the extracted symptoms to find potential diseases, treatments, and remedies.
        3.  **Path Decision:**
            *   **Symptom Confirmation (Path 1 Trigger):** If your query seems to be asking for a diagnosis, KG finds potential diseases but with low confidence, *and* this is the first clarification step, you may be asked to confirm associated symptoms via a checklist. This helps refine the KG results.
            *   **Direct Processing (Path 2 / Path 1 Direct):** If symptom confirmation isn't needed (e.g., query isn't diagnosis-focused, KG confidence is high, or confirmation was already done), the process continues.
        4.  **KG Diagnosis Component (Path 1 Output):** If a high-confidence disease is found (either initially or after confirmation), a formatted statement about this potential condition is prepared.
        5.  **RAG Document Search (Path 2):** The system searches relevant documents based on your original query and conversation history.
        6.  **Information Synthesis:** Information from the KG (diagnosis component, treatments, remedies) and RAG search results are combined. An LLM synthesizes this into a comprehensive draft answer, attempting to address your query and fill potential gaps (like duration, severity if applicable).
        7.  **Final Completeness Check:** The synthesized answer is checked by an LLM for critical missing information relative to your original query and conversation history.
        8.  **LLM Follow-up (Optional):** If the check reveals a *single critical* piece of missing information, *and* no follow-up question has been asked in this specific conversation thread yet, the system will ask one targeted clarification question.
        9.  **Final Output:** The final answer (either after synthesis or the LLM follow-up prompt) is displayed, including sources/references and a medical disclaimer.

        **Disclaimers:**
        *   This tool is for informational purposes only. **It is not a substitute for professional medical advice, diagnosis, or treatment.**
        *   Always consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment.
        *   System availability depends on the successful initialization of its components (LLM, KG, Embeddings, Document Access). Status is shown in the sidebar.
        """)
        logger.debug("Displayed About tab content.")

    logger.info("--- Finished Streamlit main() execution cycle ---")


# Add a main guard for execution
if __name__ == "__main__":
    logger.info("Script executed directly. Calling main().")
    # When running with `streamlit run app.py`, streamlit calls main()
    main()
