import streamlit as st
from pathlib import Path
import csv
import os
import re
import torch # Import torch for compatibility, ignore watcher errors
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
import base64
import logging
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

# Import chain and memory components (Note: ConversationalRetrievalChain isn't directly used in this specific workflow logic, but its components are)
# from langchain.chains import ConversationalRetrievalChain # Not directly used in the core workflow
# from langchain.memory import ConversationBufferMemory # Not directly used in the core workflow

# Import Neo4j components
from neo4j import GraphDatabase

# Attempt basic torch import handling. Note: Streamlit watcher issues with torch
# might still occur regardless of this block, requiring external configuration (.streamlit/config.toml).
# This block also ensures numpy is used explicitly where needed to avoid torch dependency in some operations.
try:
    import torch
except ImportError:
     pass # Torch is not strictly required if you only use CPU embeddings and don't use torch directly elsewhere

try:
    import numpy as np
except ImportError:
    # Provide a dummy numpy if not installed, affecting only non-essential float checks
    class DummyNumpy:
        def floating(self, *args, **kwargs):
            return float
    np = DummyNumpy()
    logging.warning("NumPy not found, some float checks might be less robust.")


# Configuration
from dotenv import load_dotenv
load_dotenv()

# !!! WARNING: Replace these with your actual credentials or ensure .env is configured correctly !!!
# It is strongly recommended to use environment variables loaded by dotenv instead of hardcoding.
# If using environment variables, remove the default "YOUR_..." values.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY") # Use a clear placeholder
NEO4J_URI = os.getenv("NEO4J_URI", "YOUR_NEO4J_URI") # Use a clear placeholder
NEO4J_USER = os.getenv("NEO4J_USER", "YOUR_NEO4J_USER") # Use a clear placeholder
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "YOUR_NEO4J_PASSWORD") # Use a clear placeholder
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# Configure logging *before* potential errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Provide informative errors if required environment variables are missing
# Check for actual placeholder values as well
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    logger.critical("GEMINI_API_KEY environment variable is not set or is the placeholder value. LLM initialization will fail.")
    # Set to None so LLM initialization fails gracefully
    GEMINI_API_KEY = None
else:
    # Basic check if it looks like a key (minimum length)
    if len(GEMINI_API_KEY) < 20: # API keys are typically much longer than this
         logger.warning("GEMINI_API_KEY appears short, possibly invalid. LLM initialization may fail.")


if not NEO4J_URI or NEO4J_URI == "YOUR_NEO4J_URI" or not NEO4J_USER or NEO4J_USER == "YOUR_NEO4J_USER" or not NEO4J_PASSWORD or NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD":
     logger.critical("NEO4J environment variables (URI, USER, PASSWORD) are not fully set or are placeholder values. KG connection will fail.")
     # Set to None so KG connection fails gracefully
     NEO4J_URI = None
     NEO4J_USER = None
     NEO4J_PASSWORD = None


# Threshold settings
THRESHOLDS = {
    "symptom_extraction": 0.6,
    "disease_matching": 0.5, # Base threshold for KG to identify *a* disease
    "disease_symptom_followup_threshold": 0.8, # Below this confidence for a disease query, trigger symptom confirmation UI
    "kg_context_selection": 0.6, # Threshold for KG confidence to be included in context sent to LLM (for symptom queries)
    "rag_context_selection": 0.7, # Threshold for RAG confidence to be included in context sent to LLM (for both symptom and non-symptom queries)
    "medical_relevance": 0.6, # Threshold for medical relevance check
    "high_kg_context_only": 0.8 # New threshold for KG-only path
}

# Load and convert the image to base64 for favicon
def get_image_as_base64(file_path):
    logger.debug(f"Attempting to load image: {file_path}")
    # Check if file exists and is not a directory
    if not Path(file_path).is_file():
        logger.warning(f"Image file not found or is not a file at {file_path}")
        # Return base64 for a tiny transparent pixel instead of black
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    try:
        with open(file_path, "rb") as image_file:
            b64_string = base64.b64encode(image_file.read()).decode()
            logger.debug("Image loaded and encoded to base64.")
            return b64_string
    except Exception as e:
        logger.error(f"Error encoding image {file_path} to base64: {e}")
        # Return base64 for a tiny transparent pixel
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

image_path = "Zoom My Life.jpg" # Update with your actual logo path
# Load favicon outside of main function to ensure it runs early
try:
    icon = get_image_as_base64(image_path)
except Exception as e:
    logger.error(f"Failed to load favicon: {e}")
    # Use a default transparent pixel favicon on failure
    icon = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


# Cache for expensive operations
CACHE = {}

def get_cached(key):
    # Use a more robust key type conversion for consistent hashing
    try:
        key_str = json.dumps(key, sort_keys=True)
    except (TypeError, json.JSONDecodeError) as e:
        logger.warning(f"Could not JSON dump cache key, using string conversion: {key} - {e}")
        key_str = str(key) # Fallback to string conversion if json.dumps fails

    if key_str in CACHE:
        # logger.debug(f"Cache hit for key: {key_str[:50]}...") # Suppress debug spam
        return CACHE[key_str]
    # logger.debug(f"Cache miss for key: {key_str[:50]}...") # Suppress debug spam
    return None

def set_cached(key, value):
    # Use a more robust key type conversion for consistent hashing
    try:
        key_str = json.dumps(key, sort_keys=True)
    except (TypeError, json.JSONDecodeError) as e:
        logger.warning(f"Could not JSON dump cache key, using string conversion for setting: {key} - {e}")
        key_str = str(key) # Fallback to string conversion

    # logger.debug(f"Setting cache for key: {key_str[:50]}...") # Suppress debug spam
    CACHE[key_str] = value
    return value

# Hardcoded PDF files to use
HARDCODED_PDF_FILES = [
    "rawdata.pdf",
    # Add more PDF paths here if needed
]

def get_system_prompt(user_type):
    # Ensure user_type is one of the expected values
    if user_type not in ["physician", "family", "User / Family"]:
        logger.warning(f"Unknown user type '{user_type}', defaulting to 'family'.")
        user_type = "family"
        
    # Map "User / Family" to "family" for internal logic consistency
    internal_user_type = "family" if user_type == "User / Family" else user_type

    base_prompt = "You are MediAssist, an AI assistant specialized in medical information. "
    if internal_user_type == "physician":
        return base_prompt + (
            "Respond using professional medical terminology and consider offering differential diagnoses "
            "when appropriate. Provide detailed clinical insights and evidence-based recommendations. "
            "Use medical jargon freely, assuming high medical literacy. Cite specific guidelines or studies "
            "when possible. Structure your responses with clear clinical reasoning."
        )
    else:  # family user (including "User / Family")
        return base_prompt + (
            "Respond using clear, accessible language appropriate for someone without medical training. "
            "Explain medical terms when you use them. If the query describes a potentially urgent medical "
            "situation, explicitly identify it as a triage situation and provide clear guidance on "
            "appropriate next steps (e.g., 'This situation requires immediate medical attention' or "
            "'This can be managed at home with the following care steps...').\n\n"
            "For triage situations, explicitly state the level of urgency using one of these categories:\n"
            "1. Emergency (Call 911/Emergency Services immediately)\n"
            "2. Urgent Care (See a doctor within 24 hours)\n"
            "3. Primary Care (Schedule a regular appointment)\n"
            "4. Self-care (Can be managed at home with the following steps...)\n\n"
            "Always prioritize patient safety in your recommendations."
        )

# Helper functions for feedback (logging to CSV)
def vote_message(user_message: str, bot_message: str, vote: str, user_type: str):
    logger.info(f"Logging vote: {vote} for user_type: {user_type}")
    try:
        feedback_file = "feedback_log.csv"
        file_exists = os.path.isfile(feedback_file)

        with open(feedback_file, mode='a', newline='', encoding='utf-8') as file:
            fieldnames = ['timestamp', 'user_type', 'user_message', 'bot_message', 'vote']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Sanitize bot message to remove appended info like disclaimer or pathway
            sanitized_bot_msg = bot_message.split("IMPORTANT MEDICAL DISCLAIMER:", 1)[0].strip()
            # Also remove the pathway info line at the very end using the span pattern
            sanitized_bot_msg = re.sub(r"\n\n<span style='font-size: 0.8em; color: grey;'>.*</span>$", "", sanitized_bot_msg).strip()
            # Replace internal | delimiter used in history display if present
            sanitized_bot_msg = sanitized_bot_msg.replace('||', '')

            # Sanitize user message to replace internal | delimiter
            sanitized_user_msg = user_message.replace('||', '')


            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'user_type': user_type,
                'user_message': sanitized_user_msg.replace("\n", " "), # Replace newlines for CSV
                'bot_message': sanitized_bot_msg.replace("\n", " "), # Replace newlines for CSV
                'vote': vote
            })
        logger.info(f"Feedback '{vote}' logged successfully.")
        return f"Feedback '{vote}' logged successfully!"
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")
        return "Failed to log feedback."

def submit_feedback(feedback_text: str, conversation_history: List[Tuple[str, str]], user_type: str):
    logger.info(f"Logging detailed feedback for user_type: {user_type}")
    try:
        feedback_file = "detailed_feedback_log.csv"
        file_exists = os.path.isfile(feedback_file)

        with open(feedback_file, mode='a', newline='\n', encoding='utf-8') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(['Timestamp', 'User Type', 'Feedback', 'Conversation History'])

            # Sanitize history: remove disclaimers, references, pathway info, and the | delimiter used in history string
            history_string_parts = []
            for u, b in conversation_history:
                sanitized_b = b.split("IMPORTANT MEDICAL DISCLAIMER:", 1)[0].strip()
                sanitized_b = re.sub(r"\n\n<span style='font-size: 0.8em; color: grey;'>.*</span>$", "", sanitized_b).strip()
                history_string_parts.append(f"User: {u.replace('||', '')} | Bot: {sanitized_b.replace('||', '')}") # Replace || in user msg too

            # Use a less common delimiter like "~~~" for conversation turns in the log
            history_string = " ~~~ ".join(history_string_parts)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_type,
                feedback_text.replace('\n', ' ').replace('||', ''), # Sanitize feedback text
                history_string
            ])
        logger.info("Detailed feedback submitted successfully.")
        return "Detailed feedback submitted successfully!"
    except Exception as e:
        logger.error(f"Error submitting detailed feedback: {e}")
        return "Failed to submit detailed feedback."


# DocumentChatBot class - Contains all backend logic
class DocumentChatBot:
    def __init__(self):
        logger.info("DocumentChatBot initializing...")
        self.qa_chain: Optional[Any] = None # Use Any as type hint is not strictly ConversationalRetrievalChain
        self.vectordb: Optional[FAISS] = None
        # Removed chat_history from instance, using st.session_state
        self.followup_context = {"round": 0} # Unused, can be removed

        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        try:
            # Check for CUDA availability and set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu' if 'torch' in globals() else 'cpu' # Check if torch is imported
            logger.info(f"Initializing SentenceTransformer embeddings on device: {device}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name='pritamdeka/S-PubMedBert-MS-MARCO',
                cache_folder='./cache',
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            # Test the embedding model
            try:
                # Use a simple query string directly
                test_embedding = self.embedding_model.embed_query("test query")
                if test_embedding is not None and len(test_embedding) > 0:
                     logger.info("Embedding model initialized and tested successfully.")
                else:
                    # Even if test returns empty, the model object exists. Log warning but don't fail init yet.
                    logger.warning("Test embedding was empty, but embedding model object exists.")
            except Exception as test_e:
                 logger.warning(f"Embedding model test failed: {test_e}. Setting embedding_model to None.")
                 self.embedding_model = None # Set to None if test fails

        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Could not initialize embedding model: {e}")
            self.embedding_model = None # Ensure it's None on failure

        self.llm: Optional[ChatGoogleGenerativeAI] = None

        self.kg_driver = None
        self.kg_connection_ok = False
        self._init_kg_connection() # Attempt KG connection during init


    def _init_kg_connection(self):
        logger.info("Attempting to connect to Neo4j...")
        # Check if credentials are set before attempting connection and are not placeholders
        if not NEO4J_URI or NEO4J_URI == "YOUR_NEO4J_URI" or \
           not NEO4J_USER or NEO4J_USER == "YOUR_NEO4J_USER" or \
           not NEO4J_PASSWORD or NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD":
             logger.error("Neo4j credentials missing or are placeholder values. Cannot connect.")
             self.kg_driver = None
             self.kg_connection_ok = False
             return

        try:
            # The timeout parameter was removed from verify_connectivity, which was the error source.
            # The connection_timeout parameter in GraphDatabase.driver is correct.
            self.kg_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, connection_timeout=10.0) # Increased timeout slightly
            self.kg_driver.verify_connectivity() # This should now work without the 'timeout' parameter
            logger.info("Successfully connected to Neo4j.")
            self.kg_connection_ok = True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}. KG features will be unavailable.")
            self.kg_driver = None
            self.kg_connection_ok = False

    # Removed the simple generate_response function as it's not used in the complex flow.
    # The complex flow's generation steps call local_generate internally.

    def enhance_with_triage_detection(self, query: str, response_content: str, user_type: str) -> str:
        """Enhance family user responses with triage situation detection."""
        # Only enhance if user is a family user
        if user_type != "User / Family" and user_type != "family":
            logger.debug(f"Triage detection skipped for user type: {user_type}")
            return response_content

        # Avoid re-triaging if triage info is already explicitly in the response
        # Check for categories explicitly mentioned or the TRIAGE ASSESSMENT header
        if "TRIAGE ASSESSMENT:" in response_content.upper() or \
           any(cat in response_content for cat in ["1. Emergency", "2. Urgent Care", "3. Primary Care", "4. Self-care"]):
             logger.debug("Triage detection skipped, response already contains triage info.")
             return response_content

        logger.debug(f"Attempting triage detection for family user. Query: {query[:50]}..., Response: {response_content[:100]}...")

        triage_prompt = (
            f"You are an AI assistant analyzing a medical conversation. "
            f"Determine if the user's query, given the assistant's response, represents a triage situation.\n\n"
            f"QUERY: {query}\n\n"
            f"ASSISTANT RESPONSE CONTENT: {response_content}\n\n"
            "If this interaction suggests a need for urgency or specific care steps, classify it according to these categories based on the *most critical* implied level:\n"
            "1. Emergency (Call 911/Emergency Services immediately)\n"
            "2. Urgent Care (See a doctor within 24 hours)\n"
            "3. Primary Care (Schedule a regular appointment)\n"
            "4. Self-care (Can be managed at home)\n\n"
            "Provide ONLY the triage category number and title (e.g., '1. Emergency') followed by a brief, one-sentence explanation of *why* this category is suggested based on the *query and response*. "
            "For example: '1. Emergency: The symptoms described require immediate medical attention.'\n"
            "If it's clearly NOT a triage situation requiring specific immediate action or classification, respond with ONLY 'NO_TRIAGE_NEEDED'."
            "Ensure your response is very concise (max 50 words total). Avoid conversational filler."
        )

        cache_key = {"type": "triage_detection", "query": query, "response": response_content}
        cached = get_cached(cache_key)
        if cached is not None: # Cache can store "NO_TRIAGE_NEEDED" or the assessment string
             logger.debug("Triage detection from cache.")
             triage_text = cached
             if triage_text != "NO_TRIAGE_NEEDED":
                  enhanced_response = f"{response_content}\n\n**TRIAGE ASSESSMENT:**\n{triage_text}"
                  logger.debug("Triage detection: Enhanced response with cached triage info.")
                  return enhanced_response
             logger.debug("Triage detection: Cached NO_TRIAGE_NEEDED.")
             return response_content


        try:
            # Use a small token limit for this specific quick check
            triage_analysis = self.local_generate(triage_prompt, max_tokens=100)
            triage_text = triage_analysis.strip()
            logger.debug(f"Triage detection LLM raw response: {triage_text}")

            set_cached(cache_key, triage_text) # Cache the result (either assessment or NO_TRIAGE_NEEDED)

            if "NO_TRIAGE_NEEDED" not in triage_text:
                enhanced_response = f"{response_content}\n\n**TRIAGE ASSESSMENT:**\n{triage_text}"
                logger.info("Triage detection: Enhanced response with assessment.")
                return enhanced_response
            logger.debug("Triage detection: NO_TRIAGE_NEEDED detected.")
            return response_content
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"Error during triage detection LLM call: {e}")
            # Don't fail the whole response, just skip triage enhancement
            return response_content
        except Exception as e:
            logger.error(f"Unexpected error during triage detection: {e}", exc_info=True)
            # Don't fail the whole response, just skip triage enhancement
            return response_content


    def create_vectordb(self):
        logger.info("Creating vector database...")
        # Filter for existing files
        pdf_files = [Path(pdf_file) for pdf_file in HARDCODED_PDF_FILES if Path(pdf_file).is_file()] # Use is_file()
        if not pdf_files:
            logger.warning("No PDF files found. Cannot create vector database.")
            return None, "No PDF files found."

        loaders = []
        for pdf_file in pdf_files:
            try:
                loaders.append(PyPDFLoader(str(pdf_file)))
                logger.debug(f"Created loader for {pdf_file}")
            except Exception as e:
                logger.error(f"Error creating loader for {pdf_file}: {e}")

        if not loaders:
             logger.warning("No valid PDF loaders could be created.")
             return None, "No valid PDF loaders could be created."

        pages = []
        for loader in loaders:
            try:
                # Use loader.load() which returns a list of Document objects
                loaded_pages = loader.load()
                pages.extend(loaded_pages)
                # Use loader.file_path (public attribute)
                logger.info(f"Loaded {len(loaded_pages)} pages from {loader.file_path}.")
            except Exception as e:
                 # Use loader.file_path
                logger.error(f"Error loading pages from PDF {loader.file_path}: {e}")

        if not pages:
             logger.warning("No pages were loaded from the PDFs.")
             return None, "No pages were loaded from the PDFs."

        logger.info(f"Total pages loaded: {len(pages)}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(pages)
        logger.info(f"Split {len(pages)} pages into {len(splits)} chunks.")

        if not splits:
            logger.warning("No text chunks were created from the PDF pages.")
            return None, "No text chunks were created."

        if self.embedding_model is None:
             logger.warning("Embedding model is not initialized. Cannot create vector database.")
             return None, "Embedding model not initialized."

        try:
            logger.info("Creating FAISS vectorstore...")
            # Correct signature: FAISS.from_documents expects documents and embeddings
            vectordb = FAISS.from_documents(splits, self.embedding_model)
            logger.info("FAISS vectorstore created.")
            return vectordb, "Vector database created successfully."
        except Exception as e:
            logger.error(f"Error creating FAISS vector database: {e}")
            return None, f"Failed to create vector database: {str(e)}"

    def initialize_qa_chain(self):
        logger.info("Initializing QA chain components (LLM, Vector DB)...")
        llm_init_message = "LLM initialization skipped."
        # Use the globally checked GEMINI_API_KEY
        if GEMINI_API_KEY is None or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
            logger.warning("Gemini API key not set or invalid. LLM will not be initialized.")
            self.llm = None
            llm_init_message = "Gemini API key not found or invalid."
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.3,
                    top_p=0.95,
                    top_k=40,
                    # Streamlit runs sequentially, convert_system_message_to_human=True can help
                    # ensure prompts work well with the stateless nature of Streamlit's display.
                    convert_system_message_to_human=True
                )
                # Attempt a quick test invoke
                try:
                    # Use a short, harmless query
                    test_response = self.llm.invoke("Hello, are you ready?", config={"max_output_tokens": 10})
                    if test_response and test_response.content: # Check for None response and content
                        logger.info("Successfully connected to Gemini Flash 1.5")
                        llm_init_message = "Gemini Flash 1.5 initialized."
                    else:
                       # Even if test returns empty, the LLM object exists. Log warning but don't fail init yet.
                       logger.warning("Initial LLM test response was empty or None.")
                       llm_init_message = "Gemini Flash 1.5 initialized, but test response was empty." # Keep LLM object


                except Exception as test_e:
                     logger.warning(f"Initial Gemini test failed: {test_e}. Setting LLM to None.")
                     self.llm = None # Set back to None if test fails
                     llm_init_message = f"Gemini LLM test failed: {test_e}"

            except Exception as e:
                logger.error(f"Failed to initialize Gemini Flash 1.5: {e}")
                self.llm = None
                llm_init_message = f"Failed to initialize Gemini Flash 1.5: {str(e)}"

        vdb_message = "Vector database initialization skipped."
        # Only create VDB if embedding model is initialized
        if self.embedding_model is None:
             vdb_message = "Embedding model not initialized."
             self.vectordb = None
        else:
             # Check if PDF files exist before attempting creation
             if any(Path(pdf_file).is_file() for pdf_file in HARDCODED_PDF_FILES): # Use is_file()
                self.vectordb, vdb_message = self.create_vectordb()
                if self.vectordb is not None:
                     logger.info("Vector DB successfully created/loaded.")
                else:
                     logger.warning(f"Vector DB creation failed: {vdb_message}")
             else:
                  vdb_message = "No PDF files found to create vector database."
                  self.vectordb = None
                  logger.warning(vdb_message)


        status_parts = []
        if self.llm is not None: status_parts.append("LLM OK")
        else: status_parts.append("LLM Failed")
        if self.embedding_model is not None: status_parts.append("Embeddings OK")
        else: status_parts.append("Embeddings Failed")
        if self.vectordb is not None: status_parts.append("Vector DB OK")
        else: status_parts.append("Vector DB Failed")
        if self.kg_connection_ok: status_parts.append("KG OK")
        else: status_parts.append("KG Failed")


        overall_message = f"{llm_init_message}. {vdb_message}. KG Status: {'Connected' if self.kg_connection_ok else 'Failed'}. Overall: {', '.join(status_parts)}."
        overall_success = self.llm is not None # LLM is minimal requirement for any useful response

        logger.info(f"Initialization Result: Success={overall_success}, Message='{overall_message}'")
        return overall_success, overall_message

    def local_generate(self, prompt, max_tokens=500):
        # logger.debug(f"Attempting LLM generation with prompt (first 100 chars): {prompt[:100]}...") # Suppress spam
        # Ensure LLM is accessible, raise informative error if not
        if self.llm is None:
             logger.critical("LLM is not initialized. Generation failed.")
             raise ValueError("LLM is not initialized. Cannot generate response.")

        try:
            # Using invoke with config for max_tokens
            response = self.llm.invoke(prompt, config={"max_output_tokens": max_tokens})
            # logger.debug("LLM generation successful.") # Suppress spam
            if response and response.content:
                 return response.content
            else:
                 logger.warning("LLM invoke returned empty or None response.")
                 # Treat empty response as failure to trigger error handling downstream
                 raise ValueError("LLM returned empty response.")

        except Exception as e:
            logger.error(f"Error during Gemini LLM generation: {e}. Generation failed.")
            # Re-raise the exception to be caught by the caller (e.g., process_user_query)
            # which can then handle it gracefully.
            raise ValueError(f"LLM generation failed: {e}") from e


    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        logger.debug(f"Checking medical relevance for query: {query[:50]}...")
        cache_key = {"type": "medical_relevance", "query": query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("Medical relevance check from cache.")
            return cached

        # Fallback if LLM is not available
        if self.llm is None:
             logger.warning("LLM not initialized. Falling back to keyword medical relevance check.")
             medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose", "medicine", "doctor", "hospital", "clinic", "condition", "illness", "sick", "diagnosed", "test result", "appointment", "prescription", "therapy", "injury", "wound", "allergy", "vaccine", "mental health", "wellness"]
             result = (any(keyword in query.lower() for keyword in medical_keywords), "Fallback heuristic match (LLM unavailable)")
             logger.debug(f"Medical relevance fallback result: {result}")
             return set_cached(cache_key, result)

        medical_relevance_prompt = f'''
        Analyze the user query. Is it directly related to health, medical conditions, symptoms, treatments, medication, diagnostics, hospital visits, appointments, mental health, wellness, or any other medical or health science topic?
        Return ONLY a JSON object formatted as: {{"is_medical": boolean, "confidence": float, "reasoning": "brief explanation"}}. Ensure confidence is a number between 0.0 and 1.0.
        Query: "{query}"
        '''
        try:
            response = self.local_generate(medical_relevance_prompt, max_tokens=150)
            # logger.debug(f"Medical relevance LLM raw response: {response}") # Suppress spam
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    # Safely get is_medical (default to False), confidence (default to 0.0 and convert/clamp), and reasoning
                    is_medical_llm = data.get("is_medical", False) # LLM's boolean output
                    confidence_raw = data.get("confidence", 0.0)
                    confidence = max(0.0, min(1.0, float(confidence_raw))) # Clamp confidence to [0, 1]
                    reasoning = data.get("reasoning", "")

                    # Final decision based on LLM's boolean AND confidence threshold
                    final_is_medical = is_medical_llm and confidence >= THRESHOLDS.get("medical_relevance", 0.6)

                    result = (final_is_medical, reasoning)
                    logger.info(f"Medical relevance check result: {result}")
                    return set_cached(cache_key, result)
                except (json.JSONDecodeError, ValueError) as e: # Catch ValueError from float conversion too
                    logger.warning(f"Could not parse or process medical relevance JSON from LLM response: {e}.")
            else:
                logger.warning("No JSON found in medical relevance response.")

        except ValueError as e: # Catch ValueError from local_generate if LLM fails
             logger.error(f"Error during medical relevance LLM call: {e}")

        # Fallback if LLM call/parsing fails
        medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose", "medicine", "doctor", "hospital", "clinic", "condition", "illness", "sick", "diagnosed", "test result", "appointment", "prescription", "therapy", "injury", "wound", "allergy", "vaccine", "mental health", "wellness"]
        result = (any(keyword in query.lower() for keyword in medical_keywords), "Fallback heuristic match (LLM failed)")
        logger.debug(f"Medical relevance final fallback result: {result}")
        return set_cached(cache_key, result)


    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        logger.debug(f"Attempting symptom extraction for query: {user_query[:50]}...")
        cache_key = {"type": "symptom_extraction", "query": user_query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("Symptom extraction from cache.")
            # Cache stores lowercase symptoms, ensure we return list of strings
            return list(cached[0]), cached[1]

        # Expanded common symptom keywords for fallback
        common_symptom_keywords = ["fever", "cough", "headache", "sore throat", "nausea", "dizziness", "chest pain", "shortness of breath", "fatigue", "body aches", "runny nose", "congestion", "chills", "sweats", "joint pain", "muscle aches", "rash", "swelling", "pain", "ache", "burning", "itching", "numbness", "tingling", "diarrhea", "vomiting", "difficulty breathing", "difficulty swallowing", "weakness", "loss of taste", "loss of smell", "skin discoloration", "blurred vision", "ringing in ears", "loss of appetite", "weight loss", "tired", "sick", "blood", "bleeding", "bruising", "lump", "sore", "ulcer", "cramp", "spasm", "paralysis", "vision loss", "hearing loss", "confusion", "memory loss", "tremor", "seizure", "fainting", "vertigo", "anxiety", "depression", "irritability", "insomnia", "sleep changes", "mood changes", "urinating", "bladder", "bowel", "stool", "constipation", "gas", "bloating", "heart racing", "palpitations", "dizzy spells"]
        query_lower = user_query.lower()
        fallback_symptoms_lower = list(set([s.strip().lower() for s in common_symptom_keywords if s.strip().lower() in query_lower])) # Use set for uniqueness, ensure lowercase, check against lower query

        # Fallback if LLM is not available
        if self.llm is None:
             logger.warning("LLM not initialized. Falling back to keyword symptom extraction.")
             result = (fallback_symptoms_lower, 0.4) # Assign a low confidence for heuristic extraction
             logger.info(f"Symptom extraction fallback result: {result}")
             return set_cached(cache_key, result)

        SYMPTOM_PROMPT = f'''
        Extract all potential medical symptoms mentioned in the following user query.
        A symptom is a subjective manifestation of disease. Be specific. Examples: "severe headache", "loss of appetite", "tingling in left arm", "fatigue".
        For each symptom, assign a confidence score between 0.0 and 1.0 indicating how certain you are that it is a symptom rather than general text.
        Return ONLY a JSON object formatted as: {{"Extracted Symptoms": [{{"symptom": "symptom1", "confidence": 0.9}}, {{"symptom": "symptom2", "confidence": 0.8}}, ...]}}
        If no symptoms are found or identified with sufficient confidence (e.g., just a greeting), return: {{"Extracted Symptoms": []}}

        User Query: "{user_query}"
        '''
        llm_symptoms_lower = []
        llm_avg_confidence = 0.0
        llm_found_any = False # Flag to track if LLM returned *any* symptoms before filtering
        try:
            response = self.local_generate(SYMPTOM_PROMPT, max_tokens=500).strip()
            # logger.debug(f"Symptom extraction LLM raw response: {response}") # Suppress spam
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    symptom_data = data.get("Extracted Symptoms", []) # Get list safely

                    if symptom_data:
                         llm_found_any = True # LLM returned something in the list

                    # Filter by threshold AND ensure data is structured correctly
                    llm_symptoms_confident_items = [
                         item for item in symptom_data
                         if isinstance(item, dict) and "symptom" in item and isinstance(item.get("symptom"), str) and item.get("confidence", 0) >= THRESHOLDS.get("symptom_extraction", 0.6)
                    ]
                    llm_symptoms_lower = sorted(list(set([item["symptom"].strip().lower() for item in llm_symptoms_confident_items if item.get("symptom")]))) # Ensure lowercase, unique, and non-empty symptom names

                    # Calculate average confidence only for items with valid confidence scores (regardless of threshold)
                    valid_confidences = [item.get("confidence", 0) for item in symptom_data if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float))]
                    llm_avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0

                    logger.debug(f"LLM extracted {len(llm_symptoms_lower)} confident symptoms above threshold {THRESHOLDS.get('symptom_extraction', 0.6)}. Avg LLM confidence (all returned): {llm_avg_confidence:.4f}")
                except json.JSONDecodeError:
                    logger.warning("Could not parse symptom JSON from LLM response")
                except Exception as parse_e:
                     logger.warning(f"Error processing symptom data structure: {parse_e}")
            else:
                 logger.warning("Could not find JSON object in LLM response for symptom extraction.")

        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"Error during symptom extraction LLM call: {e}")
            llm_found_any = False # LLM call failed
            # If LLM call fails, combine only fallback symptoms
            combined_symptoms_lower = list(set(fallback_symptoms_lower))
            final_confidence = 0.4 if combined_symptoms_lower else 0.0 # Low confidence if only heuristic finds something
            logger.info(f"Final extracted symptoms (LLM failed): {combined_symptoms_lower} (Confidence: {final_confidence:.4f})")
            result = (combined_symptoms_lower, final_confidence)
            return set_cached(cache_key, result)


        # Combine unique symptoms from LLM (above threshold) and fallback (case-insensitive comparison)
        # Start with LLM symptoms, then add fallback symptoms that weren't already in the confident LLM list
        combined_symptom_set_lower = set(llm_symptoms_lower)
        combined_symptom_set_lower.update(fallback_symptoms_lower) # add fallback if not already in LLM set
        final_symptoms_lower = sorted(list(combined_symptom_set_lower))


        # Calculate final confidence.
        # If LLM found *any* symptoms (llm_found_any is True), use its average confidence.
        # If LLM found *nothing* (llm_found_any is False) but fallback found symptoms, use fallback confidence (0.4).
        # If neither found symptoms, confidence is 0.0.
        if llm_found_any:
             final_confidence = llm_avg_confidence
        elif fallback_symptoms_lower:
             final_confidence = 0.4
        else:
             final_confidence = 0.0

        final_confidence = max(0.0, min(1.0, final_confidence)) # Ensure final confidence is clamped


        result = (final_symptoms_lower, final_confidence) # Return lowercase symptoms for consistency
        logger.info(f"Final extracted symptoms: {final_symptoms_lower} (Confidence: {final_confidence:.4f})")
        return set_cached(cache_key, result)


    def is_symptom_related_query(self, query: str) -> bool:
        # This function is primarily used to *decide* whether to use the KG path at all,
        # and whether the query *might* lead to a symptom confirmation UI.
        # It doesn't need a separate LLM call if extract_symptoms is reliable.
        # We can simplify this: if extract_symptoms finds confident symptoms OR if keywords suggest health,
        # consider it symptom related.

        logger.debug(f"Checking if query is symptom-related heuristic: {query[:50]}...")
        if not query or not query.strip():
            logger.debug("Query is empty or whitespace, not symptom related.")
            return False
        cache_key = {"type": "symptom_query_detection_heuristic", "query": query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("Symptom query detection from cache (heuristic).")
            return cached

        # Run symptom extraction first - this is needed anyway for the KG path
        # We only need the result to check if extraction was confident/successful
        # Note: extract_symptoms already handles its own caching
        extracted_symptoms, symptom_confidence = self.extract_symptoms(query)

        # Check 1: Is the symptom extraction confident and did it find symptoms?
        if extracted_symptoms and symptom_confidence >= THRESHOLDS.get("symptom_extraction", 0.6):
            logger.debug("Query determined symptom-related based on confident extraction.")
            return set_cached(cache_key, True)

        # Check 2: Basic keyword matching (fallback if extraction isn't highly confident or found nothing confident)
        health_keywords = ["symptom", "pain", "sick", "health", "disease", "condition", "diagnosis", "feel", "experiencing", "ache", "diagnosed", "my body", "feeling", "when i", "have had", "problems with", "signs of"]
        result = any(keyword in query.lower() for keyword in health_keywords)
        logger.debug(f"Symptom query detection heuristic result (fallback): {result}")
        return set_cached(cache_key, result)


    def knowledge_graph_agent(self, user_query: str, symptoms_for_kg: List[str]) -> Dict[str, Any]:
        logger.info("📚 Knowledge Graph Agent Initiated for query: %s...", user_query[:50])
        logger.debug("Input symptoms for KG: %s", symptoms_for_kg)

        kg_results: Dict[str, Any] = {
            "extracted_symptoms": symptoms_for_kg, # Use the list passed into the function
            "identified_diseases_data": [],
            "top_disease_confidence": 0.0,
            "kg_matched_symptoms": [], # Symptoms from the input list that matched KG symptoms for the top disease
            "kg_treatments": [],
            "kg_treatment_confidence": 0.0,
            "kg_content_diagnosis_data_for_llm": {
                 "disease_name": "an unidentifiable condition",
                 "symptoms_list": symptoms_for_kg, # Use the list passed into the function, reflecting the user's input/confirmation
                 "confidence": 0.0
            },
            "kg_content_other": "Medical Knowledge Graph information is unavailable.",
        }

        # Check for valid symptoms input
        valid_symptom_names = [s.strip() for s in symptoms_for_kg if isinstance(s, str) and s.strip()]
        if not valid_symptom_names:
             logger.info("📚 KG Agent: No valid symptoms provided. Skipping KG queries.")
             kg_results["kg_content_other"] = "No valid symptoms provided to query the Knowledge Graph."
             logger.info("📚 Knowledge Graph Agent Finished (No Symptoms).")
             return kg_results


        if not self.kg_connection_ok or self.kg_driver is None:
             logger.warning("📚 KG Agent: Connection not OK. Skipping KG queries.")
             kg_results["kg_content_other"] = "Medical Knowledge Graph is currently unavailable."
             logger.info("📚 Knowledge Graph Agent Finished (Connection Error).")
             return kg_results

        try:
            logger.debug("Attempting to acquire Neo4j session.")
            # Use a timeout for the session acquisition itself
            with self.kg_driver.session(connection_acquisition_timeout=10.0) as session: # Add acquisition timeout
                logger.debug("Neo4j session acquired.")

                logger.info("📚 KG Task: Identify Diseases from symptoms: %s", valid_symptom_names)
                # Call the helper to query diseases
                # Pass the valid symptoms list
                disease_data_from_kg: List[Dict[str, Any]] = self._query_disease_from_symptoms_with_session(session, valid_symptom_names)
                logger.debug("KG disease query returned %d results.", len(disease_data_from_kg))

                if disease_data_from_kg:
                    # Ensure confidence scores are float before sorting
                    for item in disease_data_from_kg:
                        item['Confidence'] = float(item.get('Confidence', 0.0))

                    # Sort again defensively by confidence in case the query didn't fully sort (shouldn't happen with ORDER BY, but safe)
                    sorted_disease_data = sorted(disease_data_from_kg, key=lambda x: x.get("Confidence", 0.0), reverse=True)
                    kg_results["identified_diseases_data"] = sorted_disease_data

                    # The first record is the top one after sorting
                    top_disease_record = sorted_disease_data[0]
                    # Ensure keys match what _query_disease_from_symptoms_with_session returns ('Disease', 'Confidence', 'MatchedSymptoms', 'AllDiseaseSymptomsKG')
                    top_disease_name = top_disease_record.get("Disease")
                    top_disease_conf = top_disease_record.get("Confidence", 0.0)
                    kg_results["top_disease_confidence"] = top_disease_conf
                    # Store the matched symptoms from the KG query result for the top disease
                    kg_results["kg_matched_symptoms"] = list(top_disease_record.get("MatchedSymptoms", [])) # Ensure list
                    # Store ALL symptoms for the top disease from KG (useful for UI suggestion)
                    kg_results["all_disease_symptoms_kg_for_top_disease"] = list(top_disease_record.get("AllDiseaseSymptomsKG", [])) # Ensure list


                    logger.info("✔️ Diseases Identified: %s (Top Confidence: %.4f)", [(d.get('Disease'), d.get('Confidence')) for d in sorted_disease_data], top_disease_conf)

                    # Only query treatments if the top disease confidence meets the basic matching threshold
                    if top_disease_conf >= THRESHOLDS.get("disease_matching", 0.5) and top_disease_name:
                        logger.info("📚 KG Task: Find Treatments for %s", top_disease_name)
                        # Call the helper to query treatments
                        kg_results["kg_treatments"], kg_results["kg_treatment_confidence"] = self._query_treatments_with_session(session, top_disease_name)
                        logger.info("✔️ Treatments found: %s (Confidence: %.4f)", kg_results['kg_treatments'], kg_results['kg_treatment_confidence'])
                    else:
                        logger.info("📚 KG Task: Treatments skipped - Top disease confidence below basic matching threshold (%.4f < %.4f) or no top disease name.", top_disease_conf, THRESHOLDS.get("disease_matching", 0.5))
                else:
                     logger.info("📚 KG Task: Identify Diseases skipped - No diseases found for provided symptoms.")


                # Prepare data for LLM phrasing, even if confidence is low or no diseases found
                # Use the top disease found by the KG query, or a placeholder
                top_disease_name_for_llm = kg_results["identified_diseases_data"][0].get("Disease", "an unidentifiable condition") if kg_results["identified_diseases_data"] else "an unidentifiable condition"

                # Ensure symptoms_list reflects the input symptoms list
                kg_results["kg_content_diagnosis_data_for_llm"] = {
                      "disease_name": top_disease_name_for_llm,
                      "symptoms_list": valid_symptom_names, # Use the list passed in, reflecting the user's input/confirmation
                      "confidence": kg_results["top_disease_confidence"]
                }
                logger.debug("KG diagnosis data for LLM: %s", kg_results["kg_content_diagnosis_data_for_llm"])


                # Format other KG content (treatments)
                logger.debug("Formatting other KG content (treatments).")
                other_parts: List[str] = []
                if kg_results["kg_treatments"]:
                     other_parts.append("## Recommended Treatments (from KG)")
                     other_parts.extend([f"- {t}" for t in kg_results["kg_treatments"]])
                     other_parts.append("")

                kg_results["kg_content_other"] = "\n".join(other_parts).strip()
                if not kg_results["kg_content_other"]:
                    kg_results["kg_content_other"] = "Medical Knowledge Graph did not find specific relevant information on treatments."
                logger.debug("Formatted kg_content_other: %s", kg_results["kg_content_other"][:100])

                logger.info("📚 Knowledge Graph Agent Finished successfully.")
                return kg_results

        except Exception as e:
            logger.error("⚠️ Error within KG Agent: %s", e, exc_info=True)
            # Ensure fallback data is set on failure
            kg_results["kg_content_diagnosis_data_for_llm"] = {
                 "disease_name": "an unidentifiable condition",
                 "symptoms_list": valid_symptom_names, # Use the input symptom list even on error
                 "confidence": 0.0
            }
            kg_results["kg_content_other"] = f"An error occurred while querying the Medical Knowledge Graph: {str(e)}"
            kg_results["top_disease_confidence"] = 0.0 # Ensure confidence is 0 on failure
            kg_results["identified_diseases_data"] = [] # Clear potential partial results
            kg_results["kg_matched_symptoms"] = []
            kg_results["kg_treatments"] = []
            kg_results["kg_treatment_confidence"] = 0.0
            kg_results["all_disease_symptoms_kg_for_top_disease"] = []

            logger.info("📚 Knowledge Graph Agent Finished (Error).")
            return kg_results

    def _query_disease_from_symptoms_with_session(self, session, symptoms: List[str]) -> List[Dict[str, Any]]:
        logger.debug("Querying KG for diseases based on symptoms: %s", symptoms)
        # Ensure symptoms are valid strings and not empty
        valid_symptom_names = [s.strip() for s in symptoms if isinstance(s, str) and s.strip()]
        if not valid_symptom_names:
            logger.debug("No valid symptoms provided for KG disease query.")
            return []

        # Use a tuple of sorted lowercase symptoms for consistent cache key
        cache_key = {"type": "disease_matching_v2", "symptoms": tuple(sorted([s.lower() for s in valid_symptom_names]))}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("Disease match query from cache.")
            # Cache stores data in a format suitable for direct use, ensure list conversions
            cached_data = list(cached) # Ensure it's a mutable list copy
            for item in cached_data:
                # Ensure list types for MatchedSymptoms and AllDiseaseSymptomsKG
                if 'MatchedSymptoms' in item:
                    item['MatchedSymptoms'] = list(item['MatchedSymptoms'])
                if 'AllDiseaseSymptomsKG' in item:
                    item['AllDiseaseSymptomsKG'] = list(item['AllDiseaseSymptomsKG'])
                # Ensure confidence is float
                if 'Confidence' in item:
                    item['Confidence'] = float(item['Confidence'])
            return cached_data

        # Ensure symptom names are lowercase for the Cypher query parameter
        symptom_names_lower = [s.lower() for s in valid_symptom_names]

        # Cypher query to find diseases based on input symptoms and calculate confidence
        # Confidence is the ratio of matching input symptoms (found in KG for the disease) to the total number of symptoms for the disease in the KG.
        cypher_query = """
        // Use input_symptom_name_lower for matching against lowercase KG symptom names
        UNWIND $symptomNamesLower AS input_symptom_name_lower
        MATCH (s:symptom)
        WHERE toLower(s.Name) = input_symptom_name_lower
        MATCH (s)-[:INDICATES]->(d:disease)
        // Collect the original-cased names of matching symptoms from the KG node, ensure unique within this list
        WITH d, COLLECT(DISTINCT s.Name) AS matched_symptoms_from_input_in_kg_case
        // Optional match to collect ALL symptoms for this disease in the KG
        OPTIONAL MATCH (d)<-[:INDICATES]-(all_s:symptom)
        WITH d, matched_symptoms_from_input_in_kg_case,
             COLLECT(DISTINCT all_s.Name) AS all_disease_symptoms_in_kg, // Collect all symptom names linked to this disease in KG
             size(COLLECT(DISTINCT all_s)) AS total_disease_symptoms_count, // Count of ALL symptoms linked to this disease in KG
             size(matched_symptoms_from_input_in_kg_case) AS matching_symptoms_count // Count of input symptoms found linked to the disease
        // Ensure we only return diseases for which at least one input symptom was found linked
        WHERE matching_symptoms_count > 0
        // Calculate confidence: ratio of input symptoms found linked to the disease (matching_symptoms_count)
        // to the total number of symptoms linked to that disease in the KG (total_disease_symptoms_count).
        // Handle division by zero if a disease has no symptoms linked in KG (shouldn't happen with WHERE clause but defensive)
        WITH d.Name AS Disease,
             CASE WHEN total_disease_symptoms_count = 0 THEN 0.0 ELSE matching_symptoms_count * 1.0 / total_disease_symptoms_count END AS confidence_score,
             matched_symptoms_from_input_in_kg_case AS MatchedSymptoms,
             all_disease_symptoms_in_kg AS AllDiseaseSymptomsKG // Include all symptoms linked to the disease in KG
        ORDER BY confidence_score DESC
        LIMIT 5
        RETURN Disease, confidence_score, MatchedSymptoms, AllDiseaseSymptomsKG
        """
        try:
            logger.debug("Executing Cypher query for diseases with symptoms: %s", symptom_names_lower)
            # Pass the lowercase symptom names to the query parameter
            result = session.run(cypher_query, symptomNamesLower=symptom_names_lower)
            records = list(result)

            # Process Neo4j Record objects into dictionaries
            disease_data = []
            for rec in records:
                try:
                    # Convert Neo4j Record values, handling potential None or missing fields defensively
                    disease_name = rec.get("Disease")
                    confidence = float(rec.get("confidence_score", 0.0)) # Field name matches Cypher query
                    matched_symptoms = list(rec.get("MatchedSymptoms", [])) # Convert to list
                    all_disease_symptoms = list(rec.get("AllDiseaseSymptomsKG", [])) # Convert to list

                    if disease_name: # Only add if disease name is valid
                        disease_data.append({
                            "Disease": disease_name,
                            "Confidence": confidence,
                            "MatchedSymptoms": matched_symptoms,
                            "AllDiseaseSymptomsKG": all_disease_symptoms
                        })
                except Exception as rec_e:
                    logger.warning(f"Error processing Neo4j disease record: {rec}, error: {rec_e}")

            # Sort by confidence again just in case, although ORDER BY in Cypher should handle it
            disease_data = sorted(disease_data, key=lambda x: x.get('Confidence', 0.0), reverse=True)

            logger.debug("🦠 Executed KG Disease Query, found %d results.", len(disease_data))
            # Cache the result before returning. Ensure symptoms lists are tuples for cache key consistency
            cache_data = []
            for d in disease_data:
                cache_data.append({
                    "Disease": d["Disease"],
                    "Confidence": d["Confidence"],
                    "MatchedSymptoms": tuple(d["MatchedSymptoms"]),  # Store as tuple
                    "AllDiseaseSymptomsKG": tuple(d["AllDiseaseSymptomsKG"])  # Store as tuple
                })
            set_cached(cache_key, cache_data)
            return disease_data

        except Exception as e:
            logger.error(f"Error querying diseases from symptoms: {e}")
            return []



    def _query_treatments_with_session(self, session, disease: str) -> Tuple[List[str], float]:
        logger.debug("Querying KG for treatments for disease: %s", disease)
        if not disease or not isinstance(disease, str) or not disease.strip():
            logger.debug("No valid disease provided for KG treatments query.")
            return [], 0.0

        disease_name_lower = disease.strip().lower()
        cache_key = {"type": "treatment_query_kg", "disease": disease_name_lower}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("KG treatments query from cache.")
            return cached

        cypher_query = """
        MATCH (d:disease)-[r:TREATED_BY]->(t:treatment)
        WHERE toLower(d.Name) = $diseaseNameLower
        RETURN t.Name as Treatment,
               // Simple confidence heuristic based on number of TREATMENTS found for this disease
               // More sophisticated heuristics could count symptom links to the disease or treatment efficacy ratings
               // CASE WHEN COUNT(t) > 1 THEN 0.8 WHEN COUNT(t) = 1 THEN 0.6 ELSE 0.0 END as Confidence // Adjusted heuristic based on treatment count
               // Let's keep a simple fixed confidence for now if any treatments found, as the model doesn't provide efficacy
               1.0 AS Confidence // Assign max confidence if treatment link exists - represents confidence in the *link*, not efficacy
        ORDER BY Treatment // Order alphabetically by treatment name
        """

        try:
            logger.debug("Executing Cypher query for treatments for disease: %s", disease_name_lower)
            result = session.run(cypher_query, diseaseNameLower=disease_name_lower)
            records = list(result)

            # Process Neo4j Record objects
            treatments_list = []
            # Keep track if any treatments were found to assign a confidence
            found_treatments = False
            for rec in records:
                try:
                    # Access the Record object using dictionary-style notation
                    treatment_name = rec.get("Treatment") # Use .get for safety
                    if treatment_name and isinstance(treatment_name, str) and treatment_name.strip():
                        treatments_list.append(treatment_name.strip())
                        found_treatments = True
                except Exception as e:
                    logger.warning(f"Error processing treatment record: {rec}, error: {e}")

            # Ensure treatments are unique and sorted
            treatments_list = sorted(list(set(treatments_list)))

            # Assign confidence based on whether *any* treatments were found
            avg_confidence = 1.0 if found_treatments else 0.0 # 1.0 if any treatments found, 0.0 otherwise

            logger.debug("💊 Executed KG Treatment Query for %s, found %d treatments.", disease, len(treatments_list))
            return set_cached(cache_key, (treatments_list, avg_confidence))

        except Exception as e:
            logger.error("⚠️ Error executing KG query for treatments: %s", e, exc_info=True)
            return [], 0.0

    def retrieve_rag_context(self, query: str) -> Tuple[List[str], float]:
        logger.info(f"📄 RAG Retrieval Initiated for query: {query[:50]}...")
        RAG_THRESHOLD_FOR_SELECTION = THRESHOLDS.get("rag_context_selection", 0.7) # This threshold will be used by select_context, not here

        # Define the number of top results to retrieve
        k = 5 # Increased from 3 to 5 for potentially broader context
        logger.debug(f"Performing vector search for query: {query[:50]}... to retrieve top {k} documents with scores.")


        # Cache key includes the query and the number of results (k).
        cache_key = {"type": "rag_retrieval_topk_chunks_and_scores", "query": query, "k": k}
        cached = get_cached(cache_key)
        if cached:
             logger.debug(f"RAG retrieval (top {k} chunks and scores) from cache.")
             # Return the cached data directly
             # Ensure cached chunks are treated as list of strings, score as float
             return list(cached['chunks']), float(cached['avg_score'])


        # Check if the vector database and embedding model are initialized
        if self.vectordb is None or self.embedding_model is None:
            logger.warning("📄 RAG Retrieval: Vector database or Embedding model not initialized. Skipping RAG retrieval.")
            return [], 0.0 # Return empty results and 0 confidence

        try:
            # Call the vector database's method to get the top k documents and their scores.
            # This method typically returns (Document, score) pairs.
            # The score is often distance (like cosine distance), where lower is better.
            retrieved_docs_with_scores = self.vectordb.similarity_search_with_score(query, k=k)
            logger.debug(f"📄 RAG: Retrieved {len(retrieved_docs_with_scores)} potential documents from vector DB.")

            top_k_chunks_content: List[str] = [] # List to store text content of the top K chunks
            top_k_similarity_scores: List[float] = [] # List to store calculated similarity scores of the top K chunks

            # Process the retrieved documents
            logger.debug(f"--- Processing Retrieved Chunks for Query: {query[:50]}... ---")
            for i, (doc, score) in enumerate(retrieved_docs_with_scores):
                # Ensure the score is a valid numeric type. Skip if not.
                if not isinstance(score, (int, float)):
                     logger.warning(f"Received unexpected non-numeric raw score type ({type(score)}) for a retrieved chunk. Skipping this chunk for scoring/context.")
                     continue

                # Calculate similarity score from distance. Assume cosine distance where lower score is better.
                # Similarity = 1 - Distance. Clamp to [0, 1].
                # Ensure score is not negative before calculating 1 - score
                raw_score_float = float(score)
                if raw_score_float < 0:
                     logger.warning(f"Received unexpected negative raw score ({raw_score_float:.4f}). Clamping to 0 for similarity calculation.")
                     raw_score_float = 0.0

                similarity_score = max(0.0, min(1.0, 1 - raw_score_float))
                # logger.debug(f"Calculated similarity score: {similarity_score:.4f} (from raw score: {raw_score_float:.4f})") # Suppress spam

                # Add the chunk content and its calculated similarity score to the lists
                # We add ALL top K chunks here based on the vector search result.
                if doc and doc.page_content: # Ensure document and content exist
                    top_k_chunks_content.append(doc.page_content)
                    top_k_similarity_scores.append(similarity_score)
                    logger.debug(f"Chunk {i+1}: Score = {similarity_score:.4f}. Content snippet: {doc.page_content[:100]}...") # Log each processed chunk

            logger.debug(f"--- End Processing Retrieved Chunks ---")


            # Calculate the overall RAG confidence score (S_RAG)
            # This is the average of the similarity scores of the *top K* chunks that were successfully processed.
            # If no chunks were processed (e.g., k=0 or all scores were invalid), S_RAG is 0.0
            srag = sum(top_k_similarity_scores) / len(top_k_similarity_scores) if top_k_similarity_scores else 0.0

            logger.info(f"📄 RAG Retrieval Finished. Processed {len(top_k_chunks_content)} chunks. Overall S_RAG (Avg of Processed Top K): {srag:.4f}")

            # Store the result in the cache before returning
            # Cache stores the content of the top K chunks and their average score
            result_data_for_cache = {'chunks': top_k_chunks_content, 'avg_score': srag}
            set_cached(cache_key, result_data_for_cache)

            # Return the list of the top K text chunks and their average similarity score
            return top_k_chunks_content, srag

        except Exception as e:
            # Catch any other unexpected errors during retrieval
            logger.error(f"⚠️ Error during RAG retrieval: {e}", exc_info=True)
            # Return empty results and 0 confidence on error
            return [], 0.0

    def select_context(self,
                       kg_results: Dict[str, Any],
                       s_kg: float,
                       rag_chunks: List[str],
                       s_rag: float,
                       is_symptom_query: bool
                      ) -> Optional[Dict[str, Any]]:
        """
        Selects the most relevant context (KG, RAG, or both) to pass to the LLM based on query type, confidence scores,
        and a special rule to use KG only if its confidence is very high for symptom queries.
        Returns a dictionary containing selected context components or None if no context is selected.
        """
        logger.info("📦 Context Selection Initiated. Symptom Query: %s, S_KG: %.4f, S_RAG: %.4f", is_symptom_query, s_kg, s_rag)
        kg_threshold = THRESHOLDS.get("kg_context_selection", 0.6)
        rag_threshold = THRESHOLDS.get("rag_context_selection", 0.7)
        high_kg_only_threshold = THRESHOLDS.get("high_kg_context_only", 0.8) # Get the new threshold

        logger.debug("Context selection thresholds: Standard KG > %s, RAG > %s. High KG Only Threshold > %s", kg_threshold, rag_threshold, high_kg_only_threshold)

        selected_context: Dict[str, Any] = {} # Start with an empty dictionary

        # --- Apply the NEW Rule: KG Only if s_kg > high_kg_only_threshold for symptom queries ---
        # This special rule takes precedence if met.
        # Ensure kg_results is not empty before considering the high KG rule
        if is_symptom_query and s_kg > high_kg_only_threshold and kg_results and kg_results.get("identified_diseases_data"):
             # Check if KG results are actually available and meaningful (diseases found) before selecting
             logger.info("📦 Applying High KG Confidence Rule (%.4f > %.4f). Selecting KG ONLY, ignoring RAG for this turn.", s_kg, high_kg_only_threshold)
             # In this scenario, we ONLY select KG
             selected_context["kg"] = kg_results
             # RAG is explicitly excluded here
        else:
             logger.debug("High KG Confidence Rule NOT met (is_symptom_query=%s, s_kg=%.4f) or kg_results empty. Applying standard selection logic.", is_symptom_query, s_kg)
             # Fall through to the standard logic


        # --- If the NEW Rule was NOT met, fall back to standard selection logic ---
        # This 'else' block covers:
        # 1. Not a symptom query (is_symptom_query is False)
        # 2. Symptom query, but s_kg is <= high_kg_only_threshold
        # 3. Symptom query, s_kg > high_kg_only_threshold, BUT kg_results was empty or no diseases found

        if not selected_context: # Only execute standard logic if no context was selected by the high KG rule
            # Decide based on query type and standard thresholds
            if is_symptom_query:
                logger.debug("Processing symptom query (standard logic).")
                # KG is selected if its standard threshold is met AND meaningful results exist
                if s_kg >= kg_threshold and kg_results and kg_results.get("identified_diseases_data"):
                    selected_context["kg"] = kg_results
                    logger.debug("KG meets standard threshold (%.4f >= %.4f) and has results. KG selected.", s_kg, kg_threshold)
                elif not (kg_results and kg_results.get("identified_diseases_data")):
                     logger.debug("KG results empty or no diseases found. KG not selected (even if s_kg > 0).")
                else:
                     logger.debug("KG does NOT meet standard threshold (%.4f < %.4f). KG not selected.", s_kg, kg_threshold)

                # RAG is considered for symptom queries if its threshold is met and chunks exist
                if s_rag >= rag_threshold and rag_chunks:
                    selected_context["rag"] = rag_chunks
                    logger.debug("RAG meets threshold (%.4f >= %.4f) and chunks exist. RAG selected.", s_rag, rag_threshold)
                elif not rag_chunks:
                     logger.debug("RAG chunks empty. RAG not selected.")
                else:
                     logger.debug("RAG does NOT meet threshold (%.4f < %.4f). RAG not selected.", s_rag, rag_threshold)

            else: # Not a symptom query
                logger.debug("Processing non-symptom query context selection logic (RAG only).")
                # For non-symptom queries, only RAG is considered if its threshold is met and there are chunks
                if s_rag >= rag_threshold and rag_chunks:
                    selected_context["rag"] = rag_chunks
                    logger.debug("Non-symptom query, RAG meets threshold (%.4f >= %.4f) and chunks exist. RAG selected.", s_rag, rag_threshold)
                elif not rag_chunks:
                     logger.debug("Non-symptom query, RAG chunks empty. RAG not selected.")
                else:
                     logger.debug("Non-symptom query, RAG does NOT meet threshold (%.4f < %.4f). RAG not selected.", s_rag, rag_threshold)


        # --- Final Check: Return None if no context source was selected ---
        if not selected_context:
             logger.info("📦 Context Selection Final: No context source met thresholds.")
             return None

        context_parts = []
        if "kg" in selected_context: context_parts.append("KG")
        if "rag" in selected_context: context_parts.append("RAG")
        logger.info("📦 Context Selection Final: Includes: %s.", ', '.join(context_parts))
        return selected_context


    def generate_initial_answer(self, query: str, selected_context: Optional[Dict[str, Any]], user_type: str) -> str:
        logger.info("🧠 Initial Answer Generation Initiated")
        # Cache key includes query, user type, and a hash of the context structure/content
        cache_key = {"type": "initial_answer", "query": query, "user_type": user_type, "context_hash": abs(hash(json.dumps(selected_context, sort_keys=True)))}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Initial answer from cache.")
             return cached

        base_prompt_instructions = get_system_prompt(user_type) # Use the system prompt for the user type
        context_info_for_prompt = ""
        context_type_description = ""
        prompt_for_initial_answer = "" # Initialize here

        if selected_context is None or not selected_context:
            logger.info("🧠 Initial Answer Generation: No external context available.")
            context_info_for_prompt = "" # No external context provided
            context_type_description = (
                "You have not been provided with any specific external medical knowledge or document snippets for this query.\n"
                "Therefore, generate only a minimal placeholder answer that indicates lack of specific information.\n"
                "Do NOT attempt to answer the user query using your general knowledge in this step.\n"
                "Do NOT mention external documents or knowledge graphs.\n"
                "Your placeholder should be concise, like 'No specific relevant information was found.'"
            )
            prompt_for_initial_answer = (
                f"{base_prompt_instructions.strip()}\n\n"
                f"{context_type_description.strip()}\n\n"
                f"User Query: \"{query}\"\n\n"
                "Minimal Placeholder Answer:\n"
                )
        else:
            context_types = []
            if 'kg' in selected_context: context_types.append('KG')
            if 'rag' in selected_context: context_types.append('RAG')
            logger.debug("Generating initial answer with context type: %s", " + ".join(context_types))

            context_type_description = "Based on the following information,"
            context_parts_for_prompt: List[str] = []

            if "kg" in selected_context:
                kg_data = selected_context.get("kg", {})
                kg_info_str = "Knowledge Graph Information:\n"
                diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                if diag_data and diag_data.get("confidence", 0) > 0.0:
                     disease_name = diag_data.get("disease_name", "an unidentifiable condition")
                     confidence = diag_data.get("confidence", 0.0)
                     if confidence > THRESHOLDS.get("high_kg_context_only", 0.8): # Use highest threshold for strongest phrasing
                          kg_info_str += f"- **Highly Probable Condition:** {disease_name} (KG Confidence: {confidence:.2f})\n"
                     elif confidence > THRESHOLDS.get("kg_context_selection", 0.6): # Use standard threshold for moderate phrasing
                          kg_info_str += f"- **Potential Condition:** {disease_name} (KG Confidence: {confidence:.2f})\n"
                     elif confidence > THRESHOLDS.get("disease_matching", 0.5): # Use basic match threshold for weaker phrasing
                          kg_info_str += f"- **Possible Condition:** {disease_name} (KG Confidence: {confidence:.2f})\n"
                     else:
                          kg_info_str += f"- Possible Condition based on limited match: {disease_name} (KG Confidence: {confidence:.2f})\n"


                     # Add symptoms from the KG that match the input symptoms for the top disease
                     kg_matched_symptoms = kg_data.get('kg_matched_symptoms', [])
                     if kg_matched_symptoms:
                         kg_info_str += f"- Relevant Symptoms (matched in KG): {', '.join(kg_matched_symptoms)}\n"

                     # Decide whether to show all symptoms for the disease in KG based on user type or confidence
                     # Let's show all symptoms linked in KG if confidence is reasonably high or for physician users
                     all_disease_symptoms_kg = kg_data.get('all_disease_symptoms_kg_for_top_disease', [])
                     if all_disease_symptoms_kg and (confidence > THRESHOLDS.get("disease_matching", 0.5) or user_type == "Physician"):
                         # Filter out symptoms already explicitly mentioned as matched to avoid redundancy
                         distinct_all_symptoms = sorted(list(set(all_disease_symptoms_kg))) # Ensure unique and sorted
                         # Optionally filter out matched symptoms if they are redundant - but LLM can handle it better in phrasing.
                         # kg_info_str += f"- Known symptoms for this condition in KG: {', '.join(distinct_all_symptoms)}\n" # Decided against adding this to prompt explicitly unless needed


                other_kg_content = kg_data.get("kg_content_other")
                # Only add other KG content if it contains more than just the "did not find specific relevant information" placeholder
                if other_kg_content and other_kg_content.strip() and "Medical Knowledge Graph did not find" not in other_kg_content:
                      kg_info_str += "\n" + other_kg_content

                # Only add KG section if it has more than just the header or if a disease with confidence > 0 was found
                if len(kg_info_str.splitlines()) > 1 or (diag_data and diag_data.get("confidence", 0) > 0.0):
                     context_parts_for_prompt.append(kg_info_str.strip()) # Add stripped KG info
                     logger.debug("Added KG context to prompt.")
                else:
                     logger.debug("KG context was empty or only header/placeholder, not added to prompt.")


            if "rag" in selected_context:
                rag_chunks = selected_context.get("rag", [])
                if rag_chunks:
                    # Limit the number of chunks passed to the LLM to manage context length
                    rag_chunks_limited = rag_chunks[:3] # Use top 3 chunks
                    rag_info_str = "Relevant Passages from Documents:\n---\n" + "\n---\n".join(rag_chunks_limited) + "\n---"
                    context_parts_for_prompt.append(rag_info_str)
                    logger.debug(f"Added {len(rag_chunks_limited)} RAG chunks to prompt.")
                else:
                     logger.debug("RAG context was empty, not added to prompt.")


            if context_parts_for_prompt:
                 context_info_for_prompt = "\n\n".join(context_parts_for_prompt)
                 # Refine context description based on source combinations
                 if "kg" in selected_context and "rag" in selected_context:
                      context_type_description = "Based on the following structured medical knowledge and relevant passages from medical documents, synthesize a comprehensive answer."
                 elif "kg" in selected_context:
                      context_type_description = "Based on the following information from a medical knowledge graph, answer the user query. Only use the information provided here where possible. Do not refer to external documents unless absolutely necessary for general medical facts."
                 elif "rag" in selected_context:
                      context_type_description = "Based on the following relevant passages from medical documents, answer the user query. Only use the information provided here where possible. Do not refer to a knowledge graph unless absolutely necessary for general medical facts."
            else:
                 # This case happens if selected_context was not None but contained empty lists/dicts or confidence was too low
                 logger.warning("Selected context was passed but resulted in empty context_parts_for_prompt.")
                 context_info_for_prompt = ""
                 context_type_description = (
                    "No effectively usable information was found in external knowledge sources that met confidence thresholds.\n"
                    "Therefore, generate only a minimal placeholder answer that indicates lack of specific external information.\n"
                    "Do NOT attempt to answer the user query using your general knowledge in this step.\n"
                    "Do NOT mention external documents or knowledge graphs.\n"
                     "Your placeholder should be concise, like 'No specific relevant information was found.'"
                 )


            # Use the prompt construction for cases with selected context (even if context_parts_for_prompt ended up empty)
            prompt_for_initial_answer = f"""
{base_prompt_instructions.strip()}
{context_type_description.strip()}

{context_info_for_prompt}

User Query: {query}

Answer:
"""

        # logger.debug("Initial Answer Generation Prompt: %s...", prompt_for_initial_answer[:min(len(prompt_for_initial_answer), 1500)]) # Suppress spam
        try:
            # Use the constructed prompt_for_initial_answer
            initial_answer = self.local_generate(prompt_for_initial_answer, max_tokens=1000)
            logger.info("🧠 Initial Answer Generated successfully.")
            logger.debug("Initial Answer: %s...", initial_answer[:100])

            # Define the expected placeholder text fragments
            placeholder_fragments = ["no specific relevant information was found", "lack of specific information"]
            initial_answer_lower = initial_answer.lower()
            is_placeholder = not initial_answer.strip() or any(frag in initial_answer_lower for frag in placeholder_fragments)

            # Check if context was provided but LLM returned a placeholder (instruction following failure)
            if (selected_context is not None and context_parts_for_prompt) and is_placeholder:
                 logger.warning("Initial answer generated placeholder text despite having selected context. Possible LLM instruction following issue.")
                 # Force a consistent placeholder that reflection expects
                 initial_answer = "No specific relevant information was found in external knowledge sources."
                 logger.warning(f"Overriding unexpected placeholder answer: {initial_answer}")

            # Check if context was NONE or effectively empty, but LLM did *not* return a placeholder (instruction following failure)
            if (selected_context is None or not context_parts_for_prompt) and not is_placeholder:
                 logger.warning("Initial answer generated content despite instruction to provide placeholder when no/empty context. LLM instruction following issue.")
                 # Force the placeholder to ensure reflection triggers supplementary as intended.
                 initial_answer = "No specific relevant information was found in external knowledge sources."
                 logger.warning(f"Overriding LLM instruction following error with placeholder: {initial_answer}")


            return set_cached(cache_key, initial_answer) # Cache and return
        except ValueError as e:
            logger.error(f"⚠️ Error during initial answer generation: %s", e, exc_info=True)
            # Propagate the error to the caller
            raise ValueError("Sorry, I encountered an error while trying to generate an initial answer.") from e


    def reflect_on_answer(self,
                          query: str,
                          initial_answer: str,
                          selected_context: Optional[Dict[str, Any]]
                         ) -> Tuple[str, Optional[str]]:
        logger.info("🔍 Reflection and Evaluation Initiated")
        # Cache key includes query, initial answer, and a hash of the context (formatted for prompt)
        context_for_reflection_prompt = self._format_context_for_reflection(selected_context)
        cache_key = {"type": "reflection", "query": query, "initial_answer": initial_answer, "context_for_reflection_hash": abs(hash(context_for_reflection_prompt))}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Reflection from cache.")
             return cached

        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform reflection.")
             return ('incomplete', 'Reflection LLM is unavailable.')

        # Use a consistent placeholder text check
        placeholder_check_fragment = "no specific relevant information was found"

        reflection_prompt = f'''
        You are an evaluation agent. Review the 'Initial Answer' for its completeness in fully addressing the 'User Query', *considering the provided 'Context' (if any)*.

        First, check if the 'Initial Answer' is a minimal placeholder response indicating a lack of specific external information. It might contain phrases like "{placeholder_check_fragment}".
        If the 'Initial Answer' is a minimal placeholder, the evaluation is **incomplete**. In this case, the 'missing_information' is the topic of the original 'User Query'.

        If the 'Initial Answer' is NOT a minimal placeholder, then evaluate its completeness using the provided 'Context'. Assess if it answered all parts of the 'User Query' and effectively used the relevant information from the 'Context'. If incomplete, identify *exactly* what specific information is missing or incomplete *from the perspective of the User Query and the provided Context*.

        Return ONLY a JSON object: {{"evaluation": "complete" or "incomplete", "missing_information": "Description of what is missing or empty string if complete"}}
        User Query: "{query}"
        Context:
        {context_for_reflection_prompt}
        Initial Answer:
        "{initial_answer}"
        '''

        # logger.debug("Reflection Prompt: %s...", reflection_prompt[:min(len(reflection_prompt), 1500)]) # Suppress spam
        try:
            response = self.local_generate(reflection_prompt, max_tokens=500)
            # logger.debug("Reflection LLM raw response: %s", response) # Suppress spam
            json_match = re.search(r'\{[\s\S]*\}', response)
            evaluation_result = 'incomplete' # Default if parsing fails
            missing_info_description = "Could not parse reflection response or missing information was not provided by evaluator."

            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    evaluation_result = data.get("evaluation", "incomplete").lower() # Default to incomplete
                    missing_info_description = data.get("missing_information", "").strip()

                    if evaluation_result == 'complete':
                        missing_info_description = None # Explicitly None for complete
                        logger.info("🔍 Reflection Result: Complete.")
                    else:
                         # If evaluation is 'incomplete' but no description was provided
                         if not missing_info_description:
                              missing_info_description = f"Answer incomplete, but specific missing info not detailed by evaluator for query '{query[:50]}...'"
                         logger.warning("🔍 Reflection Result: Incomplete. Missing Info: %s...", missing_info_description[:500])
                except json.JSONDecodeError as e:
                    logger.error(f"⚠️ Reflection: Could not parse JSON from LLM response: {e}")
                    missing_info_description = f"Reflection JSON parse error: {e}"
                except Exception as e:
                    logger.error(f"⚠️ Reflection: Error processing JSON data: {e}")
                    missing_info_description = f"Reflection JSON data error: {e}"
            else:
                logger.warning("⚠️ Reflection: No JSON object found in LLM response.")
                # Use default incomplete/description

            result = (evaluation_result, missing_info_description)
            return set_cached(cache_key, result) # Cache and return
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"⚠️ Error during reflection process: {e}", exc_info=True)
            # Re-raise the exception to be caught by process_user_query
            raise ValueError(f"An error occurred during reflection process: {e}") from e
        except Exception as e: # Catch any other unexpected errors in reflection
            logger.error(f"Unexpected error during reflection: {e}", exc_info=True)
            # Re-raise other exceptions to be caught by process_user_query
            raise RuntimeError(f"An unexpected error occurred during reflection: {e}") from e


    def _format_context_for_reflection(self, selected_context: Optional[Dict[str, Any]]) -> str:
        """Helper to format selected context into a string for the reflection prompt."""
        context_str_parts: List[str] = []
        if selected_context is not None:
             if "kg" in selected_context:
                  kg_data = selected_context.get("kg", {})
                  kg_info_str = "Knowledge Graph Info:\n"
                  diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                  if diag_data and diag_data.get("disease_name") and diag_data.get("confidence", 0) > 0.0:
                       kg_info_str += f"  Potential Condition: {diag_data['disease_name']} (Confidence: {diag_data.get('confidence', 0.0):.2f})\n"
                       # Add matched symptoms for context
                       matched_symps = kg_data.get("kg_matched_symptoms", [])
                       if matched_symps:
                            kg_info_str += f"  Matched Symptoms (input found in KG): {', '.join(matched_symps)}\n"
                       # Optionally add all symptoms for the disease in KG
                       all_symps_kg = kg_data.get("all_disease_symptoms_kg_for_top_disease", [])
                       if all_symps_kg:
                           # Only add if it provides significantly more info than matched symptoms
                           if len(all_symps_kg) > len(matched_symps):
                               kg_info_str += f"  All associated symptoms in KG: {', '.join(all_symps_kg)}\n"

                  if kg_data.get("kg_treatments"):
                       kg_info_str += f"  Treatments: {', '.join(kg_data['kg_treatments'])}\n"
                  # Add other KG content if significant
                  other_kg_content = kg_data.get("kg_content_other")
                  # Only add if it's more than just the default placeholder
                  if other_kg_content and other_kg_content.strip() and "Medical Knowledge Graph did not find" not in other_kg_content:
                      # Limit other content length for reflection prompt
                      kg_info_str += "\n" + other_kg_content[:300] + ("..." if len(other_kg_content) > 300 else "")

                  if len(kg_info_str.splitlines()) > 1: # Only include if it has more than the header
                      context_str_parts.append(kg_info_str.strip())
                  else:
                       logger.debug("KG context formatting resulted in empty or only header string for reflection.")


             if "rag" in selected_context:
                  rag_chunks = selected_context.get("rag", [])
                  if rag_chunks:
                       # Limit RAG chunks for reflection prompt brevity
                       # Ensure chunk content is string before slicing
                       valid_chunks = [c for c in rag_chunks if isinstance(c, str)]
                       if valid_chunks:
                            rag_info_str = "Relevant Passages:\n---\n" + "\n---\n".join(valid_chunks[:3]) + "\n---" # Use top 3 chunks
                            context_str_parts.append(rag_info_str)
                       else:
                            logger.debug("RAG context formatting: chunks list was not strings.")
                  else:
                       logger.debug("RAG context formatting: chunks list was empty.")

        if context_str_parts:
             return "\n\n".join(context_str_parts)
        return "None"


    def get_supplementary_answer(self, query: str, missing_info_description: str, user_type: str) -> str:
        logger.info(f"🌐 External Agent (Gap Filling) Initiated. Missing Info: {missing_info_description[:100]}...")
        # Cache key based on the hash of the combination including user type
        cache_key = {"type": "supplementary_answer", "missing_info_hash": abs(hash(missing_info_description)), "query_hash": abs(hash(query)), "user_type": user_type}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Supplementary answer from cache.")
             return cached

        # Fallback if LLM is not available
        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform supplementary generation.")
             return "\n\n-- Additional Information --\nSupplementary information could not be generated because the AI model is unavailable."

        base_prompt_instructions = get_system_prompt(user_type) # Use the system prompt for the user type


        # Refined prompt for supplementary info
        supplementary_prompt = f'''
        {base_prompt_instructions.strip()}
        You are acting as an external agent to provide *only* specific missing details to supplement a previous incomplete answer.
        Based on the user's original medical query and a description of information missing from a previous answer, provide *only* the missing information needed to complete the original query.
        Do NOT restate the original query or the information already provided in the previous answer.
        Focus precisely on the gap described.
        Ensure your response is appropriate for the specified user type (physician or family).
        **You MUST include evidence or source attribution for any medical claims where possible.** Provide links (URLs), names of reliable sources [Source Name], or indicate if it's general medical knowledge [General Medical Knowledge]. Use markdown for formatting.
        Original User Query (for context to understand the scope): "{query}"
        Information Missing from Previous Answer: "{missing_info_description}"
        Provide ONLY the supplementary information addressing the missing part, including evidence/sources. Start directly with the information. If you cannot find specific information for the gap, state this concisely.
        '''
        try:
            # Generate the supplementary answer
            supplementary_answer = self.local_generate(supplementary_prompt, max_tokens=750)

            # Post-process to ensure it doesn't start with conversational filler
            # Removed overly aggressive removal, let LLM handle phrasing, but strip whitespace
            formatted_supplementary_answer = supplementary_answer.strip()

            # If the generated answer is empty or only whitespace after stripping
            if not formatted_supplementary_answer:
                 formatted_supplementary_answer = "The AI could not find specific additional information."


            logger.info("🌐 Supplementary Answer Generated successfully.")
            # Prefix to make it clear this is additional info, but *after* caching the raw generated part
            final_supplementary_text = "\n\n-- Additional Information --\n" + formatted_supplementary_answer
            return set_cached(cache_key, final_supplementary_text) # Cache and return

        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"⚠️ Error during supplementary answer generation: {e}", exc_info=True)
            error_msg = f"Sorry, I could not find supplementary information due to an error: {e}"
            # Cache the error message for this specific lookup
            final_supplementary_text = f"\n\n-- Additional Information --\n{error_msg}"
            set_cached(cache_key, final_supplementary_text)
            return final_supplementary_text

    def collate_answers(self, initial_answer: str, supplementary_answer: str, user_type: str) -> str:
        logger.info("✨ Final Answer Collation Initiated")
        # Cache key based on hash of inputs and user type
        cache_key = {"type": "final_collation", "initial_answer_hash": abs(hash(initial_answer)), "supplementary_answer_hash": abs(hash(supplementary_answer)), "user_type": user_type}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Final collation from cache.")
             return cached

        # Fallback if LLM is not available - just concatenate with a separator
        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform final answer collation.")
             return f"{initial_answer}\n\n{supplementary_answer}"

        # Check if supplementary answer is just the error/no-info placeholder string structure
        # The supplementary function is designed to return "\n\n-- Additional Information --\n" followed by content.
        # If the content part is just the placeholder or empty after stripping, don't collate, just append.
        supp_content_after_header = supplementary_answer.split("-- Additional Information --\n", 1)[-1].strip()
        if not supp_content_after_header or "could not find specific additional information" in supp_content_after_header.lower() or "error occurred while trying to find additional information" in supp_content_after_header.lower():
             logger.debug("Supplementary answer appears to be empty, placeholder, or error. Returning initial answer with supplementary info appended directly.")
             # Ensure initial answer is stripped before appending supplementary answer
             return initial_answer.strip() + supplementary_answer.strip()


        base_prompt_instructions = get_system_prompt(user_type) # Use the system prompt for the user type

        collation_prompt = f'''
        {base_prompt_instructions.strip()}
        You are a medical communicator tasked with combining information from two sources into a single, coherent final response appropriate for the specified user type.
        Combine the following two parts:
        1. An 'Initial Answer' to a medical query.
        2. 'Supplementary Information' that addresses gaps identified in the initial answer.

        Create a single, fluent, and easy-to-understand final answer. Ensure a natural flow.
        Remove any redundancy between the two parts.
        Preserve all factual medical information and any source attributions or links provided in either part. Do NOT add new sources or make claims not supported by the provided parts.
        Format the final response clearly using markdown (e.g., headings, lists) if appropriate.
        Do NOT include the medical disclaimer or source pathway note in the answer text itself; these will be added separately.

        Initial Answer Part:
        "{initial_answer}"

        Supplementary Information Part:
        "{supplementary_answer}"

        Provide ONLY the combined, final answer content. Start directly with the answer.
        '''
        try:
            # Generate the combined answer content
            combined_answer_content = self.local_generate(collation_prompt, max_tokens=1500) # Allow more tokens for combined answer

            logger.info("✨ Final Answer Collated successfully.")
            return set_cached(cache_key, combined_answer_content) # Cache and return
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"⚠️ Error during final answer collation: {e}", exc_info=True)
            # If collation fails, provide a combined answer with an error notice
            error_message = f"\n\n-- Collation Failed --\nAn error occurred while finalizing the answer ({e}). The information below is the initial answer followed by supplementary information, presented uncollated.\n\n"
            final_collated_text = initial_answer.strip() + error_message + supplementary_answer.strip()
            # Cache the error message for this specific combination
            set_cached(cache_key, final_collated_text)
            return final_collated_text


    def reset_conversation(self):
        """Resets the internal state of the chatbot instance (currently minimal)."""
        logger.info("🔄 Resetting chatbot internal state.")
        # Clear any instance variables that hold conversation context if they were added
        # self.followup_context = {"round": 0} # Example, currently unused
        # If you add any other state variables to the chatbot instance that track
        # conversation specifics, clear them here.


    # The Main Orchestrator Function
    # Returns (response_text, ui_action, ui_payload)
    # ui_action: "display_final_answer", "show_symptom_ui", "none"
    # ui_payload: Depends on ui_action, e.g., symptom options for "show_symptom_ui"
    def process_user_query(self,
                           user_query: str,
                           user_type: str,
                           confirmed_symptoms: Optional[List[str]] = None,
                           original_query_if_followup: Optional[str] = None # This parameter is not strictly needed here, kept for clarity if the caller provides it
                           ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        logger.info("--- Processing User Query: '%s' ---", user_query[:100])
        logger.info("User Type: %s, Confirmed Symptoms Received: %s", user_type, len(confirmed_symptoms) if confirmed_symptoms is not None else 'None')

        processed_query = user_query # Query text used for retrieval and generation
        current_symptoms_for_retrieval: List[str] = [] # Symptoms used for KG and symptom UI check
        is_symptom_query = False # Flag for symptom path
        medical_check_ok = False # Assume false initially

        # --- Step 1: Handle Symptom Confirmation Rerun Logic ---
        # This block is triggered if the UI submits confirmed symptoms (confirmed_symptoms is not None).
        if confirmed_symptoms is not None:
             logger.info("--- Step 1: Handling Symptom Confirmation Rerun ---")
             # In a rerun triggered by symptom confirmation, the 'user_query' param holds the original query text.
             # This text is used for RAG search and general query context in the LLM.
             processed_query = user_query
             if not processed_query or not isinstance(processed_query, str):
                  logger.error("Original query text (user_query param) is invalid during symptom rerun.")
                  return "An error occurred during symptom confirmation processing.", "display_final_answer", None

             # Assume it's still a symptom query if we were in the symptom confirmation flow
             is_symptom_query = True
             # Use the confirmed symptoms as the list for KG in this rerun, ensure lowercase and valid
             current_symptoms_for_retrieval = sorted(list(set([s.strip().lower() for s in confirmed_symptoms if isinstance(s, str) and s.strip()])))
             logger.info("Reprocessing '%s...' with confirmed symptoms: %s", processed_query[:50], current_symptoms_for_retrieval)
             medical_check_ok = True # Assume valid as we were in a medical flow
             # Skip initial medical check and symptom extraction (Steps 2 & 3)

        else: # Standard initial query processing (confirmed_symptoms is None)
             logger.info("--- Step 1: Initial Query Processing ---")
             # --- Step 2: Guardrail Check ---
             logger.info("--- Step 2: Guardrail Check ---")
             is_medical, medical_reason = self.is_medical_query(processed_query)
             if not is_medical:
                 logger.warning("Query flagged as non-medical: %s. Workflow terminates.", medical_reason)
                 response = f"I can only answer medical-related questions. Please rephrase your query. ({medical_reason})"
                 # No pathway info needed for this type of response
                 return response, "display_final_answer", None
             logger.info("Query passed medical guardrail.")
             medical_check_ok = True

             # --- Step 3: Symptom Processing ---
             logger.info("--- Step 3: Symptom Processing ---")
             # extracted_symptoms will be lowercase list, sorted and unique
             extracted_symptoms, symptom_extraction_confidence = self.extract_symptoms(processed_query)
             current_symptoms_for_retrieval = extracted_symptoms
             is_symptom_query = self.is_symptom_related_query(processed_query) # Uses heuristic based on extraction + keywords
             logger.info("Initial Symptom Extraction: %s (Confidence: %.4f)", current_symptoms_for_retrieval, symptom_extraction_confidence)
             logger.info("Is Symptom Query: %s", is_symptom_query)


        # --- Proceed only if medical check is OK ---
        if medical_check_ok:
             # --- Step 4: Context Retrieval (KG & RAG) ---
             logger.info("--- Step 4: Context Retrieval ---")
             kg_results: Dict[str, Any] = {} # Initialize empty
             s_kg = 0.0
             rag_chunks = []
             s_rag = 0.0

             # KG Retrieval (if query is symptom-related AND we have symptoms to search with)
             # Use current_symptoms_for_retrieval (either initial extraction or UI confirmed)
             if is_symptom_query and current_symptoms_for_retrieval:
                  logger.info("Triggering KG Pipeline with %d symptoms.", len(current_symptoms_for_retrieval))
                  # Pass the *current* list of symptoms (initial extracted OR confirmed UI list)
                  # The KG agent handles its own internal validation of the symptoms list
                  kg_results = self.knowledge_graph_agent(processed_query, current_symptoms_for_retrieval)
                  s_kg = kg_results.get("top_disease_confidence", 0.0)
                  logger.info("KG Pipeline finished. S_KG: %.4f", s_kg)
             elif is_symptom_query and not current_symptoms_for_retrieval:
                  logger.warning("Query identified as symptom-related but no symptoms extracted/confirmed. Skipping KG.")
             else:
                  logger.info("Query not identified as symptom-related. Skipping KG.")


             # RAG Retrieval (for all medical queries if VDB and embeddings are available)
             # Use processed_query for the RAG search
             if self.vectordb is not None and self.embedding_model is not None:
                 logger.info("Triggering RAG Pipeline.")
                 rag_chunks, s_rag = self.retrieve_rag_context(processed_query)
                 logger.info("RAG Pipeline finished. S_RAG: %.4f", s_rag)
             else:
                 logger.warning("Vector DB or Embedding Model not available. Skipping RAG.")


             # --- Step 5: Symptom Confirmation UI Decision Point (only for initial query path) ---
             # Check if this is the *initial* query processing (not a rerun after UI) AND it's a symptom query
             # AND KG successfully found *any* diseases (confidence > 0.0) AND KG confidence is below UI trigger threshold
             # AND there are potential *new* symptoms from KG to suggest.
             if confirmed_symptoms is None and is_symptom_query:
                  logger.info("--- Step 5: Symptom Confirmation UI Check ---")
                  ui_trigger_threshold = THRESHOLDS.get("disease_symptom_followup_threshold", 0.8)
                  kg_found_diseases = len(kg_results.get("identified_diseases_data", [])) > 0
                  initial_symptoms_lower = set([s.strip().lower() for s in current_symptoms_for_retrieval if isinstance(s, str)])

                  # Condition to show UI:
                  # 1. It's a symptom query based on initial analysis.
                  # 2. KG successfully identified at least one potential disease (confidence > 0).
                  # 3. The top KG disease confidence is below the specific UI trigger threshold.
                  # 4. There are *new* symptoms suggested by the KG that weren't in the user's original input.
                  if kg_found_diseases and s_kg > 0.0 and s_kg < ui_trigger_threshold:
                       logger.info(f"KG confidence (%.4f) below UI trigger threshold (%.4f) and diseases found (%d). Checking for new symptoms for UI.", s_kg, ui_trigger_threshold, len(kg_results.get('identified_diseases_data', [])))

                       symptom_options_for_ui: Dict[str, List[str]] = {}

                       # Use the top disease found by KG
                       top_disease_data = kg_results.get("identified_diseases_data", [])[0] if kg_results.get("identified_diseases_data") else None
                       if top_disease_data:
                            disease_name = top_disease_data.get("Disease", "Unknown")
                            # Get *all* symptoms associated with this disease in the KG from the KG results
                            # The KG query already returned this list ('AllDiseaseSymptomsKG') for the top disease
                            all_disease_symptoms_kg_for_top_disease = kg_results.get("all_disease_symptoms_kg_for_top_disease", [])
                            # Filter out symptoms the user already gave (case-insensitive) and ensure uniqueness, keep original case
                            # Create a set of lowercase symptoms from the KG list
                            all_disease_symptoms_kg_lower_set = set(s.strip().lower() for s in all_disease_symptoms_kg_for_top_disease if isinstance(s, str) and s.strip())

                            # Identify suggested symptoms: symptoms in the KG set that are NOT in the initial user symptoms set
                            suggested_symptoms_lower = sorted(list(all_disease_symptoms_kg_lower_set - initial_symptoms_lower))

                            # Find the original casing for the suggested symptoms from the full KG list
                            suggested_symptoms_original_case = sorted(list(set(
                                s for s in all_disease_symptoms_kg_for_top_disease
                                if isinstance(s, str) and s.strip().lower() in suggested_symptoms_lower
                            )))


                            if suggested_symptoms_original_case:
                                 symptom_options_for_ui[disease_name] = suggested_symptoms_original_case


                       if symptom_options_for_ui:
                            total_suggested_symps = sum(len(v) for v in symptom_options_for_ui.values())
                            logger.info(f"Prepared {total_suggested_symps} potential NEW symptoms for UI checklist across {len(symptom_options_for_ui)} diseases.")
                            prompt_text = "Based on your symptoms, I've identified a few potential conditions. To help me provide the most accurate information, could you please confirm any additional symptoms you are experiencing from the list below?"
                            ui_payload = {
                                 "symptom_options": symptom_options_for_ui,
                                 "original_query": processed_query # Store the query text that triggered this UI
                             }
                            logger.info("Returning UI action: show_symptom_ui")
                            # When triggering UI, we don't generate a final answer yet, so no pathway info is appended here.
                            return prompt_text, "show_symptom_ui", ui_payload

                       # If KG confidence below threshold, but no diseases found or no NEW symptoms to suggest,
                       # proceed to context selection and generation.
                       logger.info("KG confidence below threshold, but no NEW symptoms were found to suggest for UI (or no diseases found > 0 confidence). Proceeding to answer generation based on available info.")


             # --- Step 6: Context Selection Logic ---
             # This step runs if no symptom UI was shown (either not symptom query, high confidence, or rerun from UI)
             logger.info("--- Step 6: Context Selection Logic ---")
             # Pass the results from Step 4
             selected_context = self.select_context(kg_results, s_kg, rag_chunks, s_rag, is_symptom_query)

             # --- Determine Pathway Information (Preliminary) ---
             # Determine which external sources were SELECTED for the *initial* answer generation
             # This list will be finalized later
             initial_context_sources_used: List[str] = []
             if selected_context is not None:
                 if "kg" in selected_context:
                     initial_context_sources_used.append("Knowledge Graph")
                 if "rag" in selected_context:
                     initial_context_sources_used.append("Documents (RAG)")

             # --- Step 7: Initial Answer Generation ---
             logger.info("--- Step 7: Initial Answer Generation ---")
             try:
                 # Pass the current user type to generate_initial_answer
                 initial_answer = self.generate_initial_answer(processed_query, selected_context, user_type)
             except ValueError as e: # Catch specific LLM generation errors
                 logger.error(f"Initial answer generation failed: {e}")
                 # Return error message directly, bypass reflection etc.
                 # Indicate the sources attempted for initial generation
                 pathway_info = ", ".join(initial_context_sources_used) if initial_context_sources_used else "LLM (General Knowledge for Initial Phrasing)"
                 error_pathway_info = pathway_info + " (Initial Generation Failed)"
                 error_response = f"Sorry, I could not generate an initial answer due to an error: {e}\n\n<span style='font-size: 0.8em; color: grey;'>*Sources attempted for this response: {error_pathway_info}*</span>"
                 return error_response, "display_final_answer", None


             # --- Step 8: Reflection and Evaluation ---
             logger.info("--- Step 8: Reflection and Evaluation ---")
             reflection_failed = False
             missing_info_description = None # Initialize to None
             try:
                 evaluation_result, missing_info_description = self.reflect_on_answer(
                      processed_query, initial_answer, selected_context
                 )
                 logger.info(f"Reflection Result: {evaluation_result}")
             except ValueError as e: # Catch specific LLM reflection errors
                  logger.error(f"Reflection step failed: {e}")
                  reflection_failed = True
                  evaluation_result = 'incomplete' # Assume incomplete if reflection fails
                  missing_info_description = f"Reflection failed ({e}). Attempting to find general supplementary information."
                  logger.warning(f"Proceeding with supplementary step despite reflection failure.")
             except Exception as e: # Catch any other unexpected errors during reflection
                  logger.error(f"Unexpected error during reflection: {e}", exc_info=True)
                  reflection_failed = True
                  evaluation_result = 'incomplete'
                  missing_info_description = f"Reflection failed unexpectedly ({e}). Attempting general supplementary info."
                  logger.warning(f"Proceeding with supplementary step despite unexpected reflection failure.")


             # --- Step 9: Conditional External Agent & Collation ---
             supplementary_step_triggered = False
             supplementary_answer = "" # Initialize empty

             # Trigger supplementary step if reflection is incomplete OR reflection failed
             if evaluation_result == 'incomplete':
                  logger.warning("--- Step 9: Reflection Incomplete (or failed). Triggering supplementary pipeline. ---")
                  # Use the missing_info_description from reflection or the fallback error message
                  description_for_supplementary = missing_info_description if missing_info_description is not None else f"Gap filling needed for query: {processed_query[:50]}..."
                  logger.debug(f"Missing Info Description (for supplementary): {description_for_supplementary[:100]}...")
                  supplementary_step_triggered = True # Flag that supplementary step is happening

                  # --- Step 10: External Agent (Gap Filling) ---
                  logger.info("--- Step 10: External Agent (Gap Filling) ---")
                  try:
                       # Pass the current user type to get_supplementary_answer
                       supplementary_answer = self.get_supplementary_answer(processed_query, description_for_supplementary, user_type)
                  except ValueError as e:
                       logger.error(f"Supplementary answer generation failed: {e}")
                       # Ensure supplementary_answer is a string indicating failure, formatted correctly
                       supplementary_answer = f"\n\n-- Additional Information --\nSorry, I could not find supplementary information due to an error: {e}"
                       logger.warning("Proceeding to collation with supplementary error message.")


                  # --- Step 11: Final Answer Collation ---
                  logger.info("--- Step 11: Final Answer Collation ---")
                  # Check if supplementary answer is meaningful (not just the initial header or error)
                  # The get_supplementary_answer function is designed to return a string starting with "-- Additional Information --\n"
                  # even if empty or failed. We check the content *after* this header.
                  supp_content_after_header = supplementary_answer.split("-- Additional Information --\n", 1)[-1].strip()

                  # Only attempt collation if there's actual supplementary content beyond the header/placeholder
                  if supp_content_after_header and "could not find specific additional information" not in supp_content_after_header.lower() and "error occurred while trying to find additional information" not in supp_content_after_header.lower():
                       try:
                            # Pass the current user type to collate_answers
                            final_answer_content = self.collate_answers(initial_answer, supplementary_answer, user_type)
                       except ValueError as e:
                            logger.error(f"Final answer collation failed: {e}")
                            # If collation fails, just combine with a separator and add error message
                            final_answer_content = f"{initial_answer.strip()}\n\n-- Collation Failed --\nAn error occurred while finalizing the answer ({e}).\n\n{supplementary_answer.strip()}"
                            logger.warning("Proceeding with simple answer concatenation due to collation failure.")
                  else:
                      # Supplementary answer was empty, placeholder, or error. Just append it directly.
                      logger.debug("Supplementary answer was empty, placeholder, or error. Skipping collation.")
                      final_answer_content = initial_answer.strip() + supplementary_answer.strip() # Append directly


             else: # Reflection result was 'complete'
                  logger.info("--- Step 9: Reflection Complete. Skipping supplementary pipeline. ---")
                  final_answer_content = initial_answer # Use the initial answer as the final content


             # --- Step 12: Triage Enhancement (for Family Users) ---
             logger.info("--- Step 12: Applying Triage Enhancement ---")
             # Apply triage enhancement to the final answer content
             final_answer_content_with_triage = self.enhance_with_triage_detection(processed_query, final_answer_content, user_type)


             # --- Final Assembly: Determine Final Pathway Info, Add Disclaimer and Pathway Note ---

             # Determine the final list of sources used based on what contributed to the answer
             final_pathway_parts = initial_context_sources_used # Start with the external sources used for initial context

             # Add 'LLM (General Knowledge)' to the pathway if:
             # 1. The supplementary step was triggered (LLM used for gap filling)
             # OR
             # 2. No external context was used at all for the initial step (LLM was the sole source for initial phrasing)
             # OR
             # 3. Reflection failed (implying LLM was used in a critical step)
             if supplementary_step_triggered or not initial_context_sources_used or reflection_failed:
                  final_pathway_parts.append("LLM (General Knowledge)") # Indicate LLM was used beyond just context phrasing if it filled gaps, was sole initial source, or in reflection

             # Add 'Reflection Agent' if reflection step ran (even if it failed)
             if not reflection_failed: # Only add if reflection ran successfully (even if incomplete)
                  final_pathway_parts.append("Reflection Agent")
             elif reflection_failed:
                  final_pathway_parts.append("Reflection Agent (Failed)")


             # Ensure uniqueness and order
             final_pathway_parts = sorted(list(set(final_pathway_parts)))

             if not final_pathway_parts: # Should not happen if logic is correct, but defensive
                 pathway_info = "Unknown Pathway"
             else:
                 pathway_info = ", ".join(final_pathway_parts)


             # Add the medical disclaimer consistently at the end of the main content
             disclaimer = "\n\nIMPORTANT MEDICAL DISCLAIMER: This information is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read here."

             # Add the pathway information using HTML for smaller, grey text below the disclaimer
             # Ensure the pathway info is also stripped for cleanliness
             pathway_note = f"<span style='font-size: 0.8em; color: grey;'>*Sources used for this response: {pathway_info.strip()}*</span>"

             # Combine content, disclaimer, and pathway note. Use the content *with* triage enhancement.
             final_response_text = f"{final_answer_content_with_triage.strip()}{disclaimer}\n\n{pathway_note}"

             # --- Return Final Result ---
             logger.info("--- Workflow Finished ---")
             # Return the final text including pathway info, action to display it, and no payload
             return final_response_text, "display_final_answer", None
        else:
             # This case should ideally not be reachable due to the initial medical_check_ok guard
             logger.error("Reached unexpected state: medical_check_ok was False but processing continued.")
             # Return a generic error message with pathway set to "Internal Error"
             error_pathway_info = "Internal Error"
             error_response = f"An internal error occurred during processing.\n\n<span style='font-size: 0.8em; color: grey;'>*Sources attempted for this response: {error_pathway_info}*</span>"
             return error_response, "display_final_answer", None


# Streamlit UI Component - Symptom Checklist Form
def display_symptom_checklist(symptom_options: Dict[str, List[str]], original_query: str):
    logger.debug("Rendering symptom checklist UI.")
    st.subheader("Confirm Your Symptoms")
    st.info(f"Based on your query: '{original_query}' and initial analysis, please confirm any additional symptoms you are experiencing from the list below:")

    # Use a form key that includes the original query hash and a timestamp for uniqueness across different queries and resets
    # Add user_type to the key to avoid collisions if user switches type while UI is active
    user_type_for_key = st.session_state.get("user_type_select", "family")
    form_key_base = f"symptom_confirmation_form_{abs(hash(original_query))}_{user_type_for_key}"
    form_key = f"{form_key_base}_{st.session_state.get('form_timestamp', datetime.now().timestamp())}"

    # Define session state keys specific to this form instance
    # Store selected symptoms (lowercase set) and text input value
    local_confirmed_symptoms_state_key = f'{form_key}_selected_symptoms_local_set'
    other_symptoms_text_key = f"{form_key}_other_symptoms_input"


    # Initialize the local set if it's not in session state for this form key.
    # It should be initialized as an empty set when this form is first rendered in a session/rerun cycle for a *new* form key.
    if local_confirmed_symptoms_state_key not in st.session_state:
        logger.debug(f"Initializing local symptom set state variable: {local_confirmed_symptoms_state_key}")
        st.session_state[local_confirmed_symptoms_state_key] = set()
        # Also clear any previous text input state for a brand new form
        if other_symptoms_text_key in st.session_state:
             del st.session_state[other_symptoms_text_key]
             logger.debug(f"Cleared old text input state: {other_symptoms_text_key}")

    else:
        logger.debug(f"Using existing local symptom set for form {form_key} with {len(st.session_state.get(local_confirmed_symptoms_state_key, set()))} items.")
        # If we are using an existing local set, ensure we re-add any symptoms from the text input
        # that might have been present on the previous rerun before the form was submitted.
        # This ensures typed symptoms persist visually and are included in the set state tracking.
        current_other_symptoms_text = st.session_state.get(other_symptoms_text_key, "")
        if current_other_symptoms_text:
             typed_symptoms_lower = [s.strip().lower() for s in current_other_symptoms_text.split(',') if s.strip()]
             if typed_symptoms_lower:
                  # Add typed symptoms to the local set as lowercase, without removing existing ones
                  st.session_state[local_confirmed_symptoms_state_key].update(typed_symptoms_lower)
                  logger.debug(f"Re-added typed symptoms from state to local set: {typed_symptoms_lower}")


    # Collect all unique suggested symptoms in their original casing
    all_unique_suggested_symptoms_original_case = sorted(list(set(
        s.strip() for symptoms_list in symptom_options.values()
        if isinstance(symptoms_list, list) for s in symptoms_list
        if isinstance(s, str) and s.strip()
    )))
    logger.debug(f"Total unique suggested symptoms for UI: {len(all_unique_suggested_symptoms_original_case)}")


    with st.form(form_key):
        st.markdown("Please check all symptoms that apply to you from the list below:")
        if not all_unique_suggested_symptoms_original_case:
            st.info("No specific additional symptoms were found for potential conditions to suggest.")
        else:
            # Display symptoms in columns (adjust column count as needed)
            num_cols = min(4, len(all_unique_suggested_symptoms_original_case)) # Max 4 columns
            cols = st.columns(num_cols) if num_cols > 0 else [st] # Use a single column if no symptoms

            for i, symptom in enumerate(all_unique_suggested_symptoms_original_case):
                col = cols[i % num_cols] if num_cols > 0 else cols[0]
                # Checkbox key unique to the form and symptom
                checkbox_key = f"{form_key}_checkbox_{abs(hash(symptom))}" # Hash symptom name for key robustness
                symptom_lower = symptom.strip().lower()

                # Check the initial state from the local set (case-insensitive check)
                # Use .get with default set for safety
                initial_state = symptom_lower in st.session_state.get(local_confirmed_symptoms_state_key, set())

                # Display the checkbox with original capitalization, but state tracking uses lowercase
                # Use a simple callback function to update the state immediately when a box is clicked
                def update_local_symptom_set(symptom_lower_cb, is_checked_cb):
                    # Ensure the set exists
                    if local_confirmed_symptoms_state_key not in st.session_state:
                        st.session_state[local_confirmed_symptoms_state_key] = set()
                        logger.warning(f"Local symptom set state was missing during update, re-initialized for {local_confirmed_symptoms_state_key}")

                    if is_checked_cb:
                         st.session_state[local_confirmed_symptoms_state_key].add(symptom_lower_cb)
                         logger.debug(f"Added '{symptom_lower_cb}' to local set. Current size: {len(st.session_state[local_confirmed_symptoms_state_key])}")
                    else:
                         st.session_state[local_confirmed_symptom_state_key].discard(symptom_lower_cb)
                         logger.debug(f"Discarded '{symptom_lower_cb}' from local set. Current size: {len(st.session_state[local_confirmed_symptoms_state_key])}")


                col.checkbox(
                     symptom,
                     key=checkbox_key,
                     value=initial_state,
                     on_change=update_local_symptom_set, # Use callback
                     args=(symptom_lower, not initial_state) # Pass symptom_lower and the *new* checked state
                )


        st.markdown("**Other Symptoms (if any):**")
        # Get current value from state if it exists, otherwise empty string
        other_symptoms_initial_value = st.session_state.get(other_symptoms_text_key, "")
        other_symptoms_text = st.text_input(
            "Enter additional symptoms here (comma-separated)",
            key=other_symptoms_text_key,
            value=other_symptoms_initial_value
        )

        # Update the local symptom set with any symptoms typed into the text input *immediately* on each rerun
        # This ensures the local set reflects the current text input value before form submission is checked.
        # This is different from the checkbox callback; the text input value needs to be processed every rerun.
        if other_symptoms_text_key in st.session_state:
             # Get previously stored text input value to find symptoms to potentially remove from the set
             old_other_symptoms_text = st.session_state.get(f"{other_symptoms_text_key}_previous_value", "")
             old_typed_symptoms_lower = set([s.strip().lower() for s in old_other_symptoms_text.split(',') if s.strip()])

             # Get current typed symptoms
             current_typed_symptoms_lower = set([s.strip().lower() for s in other_symptoms_text.split(',') if s.strip()])

             # Symptoms to remove are those in the old set but not in the current set
             symptoms_to_remove = old_typed_symptoms_lower - current_typed_symptoms_lower
             if symptoms_to_remove:
                 logger.debug(f"Removing typed symptoms from local set: {list(symptoms_to_remove)}")
                 st.session_state[local_confirmed_symptoms_state_key].difference_update(symptoms_to_remove)

             # Symptoms to add are those in the current set but not in the old set
             symptoms_to_add = current_typed_symptoms_lower - old_typed_symptoms_lower
             if symptoms_to_add:
                  logger.debug(f"Adding typed symptoms to local set: {list(symptoms_to_add)}")
                  st.session_state[local_confirmed_symptoms_state_key].update(symptoms_to_add)

             # Store the current value as the previous value for the next rerun
             st.session_state[f"{other_symptoms_text_key}_previous_value"] = other_symptoms_text

        # --- Form Submission ---
        submit_button = st.form_submit_button("Confirm and Continue")
        if submit_button:
            logger.info(f"Symptom confirmation form submitted for query: '{original_query[:50]}...'.")

            # At this point, the local set (st.session_state[local_confirmed_symptoms_state_key])
            # contains the combined set of confirmed symptoms (checkboxes + text input).
            # Pass this set back to the main processing logic as a sorted list.
            final_confirmed_symptoms_list = sorted(list(st.session_state.get(local_confirmed_symptoms_state_key, set())))
            st.session_state.confirmed_symptoms_from_ui = final_confirmed_symptoms_list
            logger.info(f"Set st.session_state.confirmed_symptoms_from_ui with {len(st.session_state.confirmed_symptoms_from_ui)} symptoms: {st.session_state.confirmed_symptoms_from_ui}")

            # Reset the UI state back to 'input' so the chat input appears on the next rerun, hiding the form.
            st.session_state.ui_state = {"step": "input", "payload": None}
            logger.debug("Set ui_state back to 'input' after form submission.")

            # Clear the local form state variables explicitely for cleanliness and to prevent persistence across forms.
            if local_confirmed_symptoms_state_key in st.session_state:
                 del st.session_state[local_confirmed_symptoms_state_key]
                 logger.debug(f"Cleared {local_confirmed_symptoms_state_key} from state.")
            if other_symptoms_text_key in st.session_state:
                 del st.session_state[other_symptoms_text_key]
                 logger.debug(f"Cleared {other_symptoms_text_key} from state.")
            if f"{other_symptoms_text_key}_previous_value" in st.session_state:
                del st.session_state[f"{other_symptoms_text_key}_previous_value"]
                logger.debug(f"Cleared {other_symptoms_text_key}_previous_value from state.")


            # The rerun is triggered automatically by st.form_submit_button

def create_user_type_selector():
    # Use a key for the selectbox to make its state persistent and unique
    selected_type = st.sidebar.selectbox(
        "Who is asking?",
        ["User / Family", "Physician"],
        index=["User / Family", "Physician"].index(st.session_state.get("user_type_select", "User / Family")), # Maintain state
        key="user_type_select",
        help="Select whether you are a family member or a physician. This affects the AI's response style."
    )

    # Check if the user type has changed
    # If it's the very first run and user_type_select was just initialized, this check won't trigger a reset immediately.
    # The first actual change from the initial value will trigger the reset.
    if st.session_state.get("last_user_type") is not None and selected_type != st.session_state.last_user_type:
        logger.info(f"User type changed from '{st.session_state.last_user_type}' to '{selected_type}'. Triggering conversation reset.")
        # Set a flag to indicate reset is needed
        st.session_state.reset_requested = True
    else:
        # Store the current user type for the next comparison
        st.session_state.last_user_type = selected_type
        # Ensure the reset flag is not lingering if type didn't change
        if 'reset_requested' in st.session_state:
             del st.session_state.reset_requested


# --- Main Streamlit App Function ---
def main():
    # --- App Configuration and Header ---
    logger.info("--- Streamlit App Start ---")
    try:
        st.set_page_config(
            page_title="DxAI-Agent",
            page_icon=f"data:image/png;base64,{icon}", # Assumes 'icon' is globally defined
            layout="wide"
        )
        logger.info("Page config set.")
    except Exception as e:
        logger.error(f"Error setting page config: {e}")
        st.set_page_config(page_title="DxAI-Agent", layout="wide") # Fallback
        logger.warning("Using fallback page config.")

    # --- User Type Selector (run early to set state) ---
    # Call this function near the start of your Streamlit app, before chat UI
    create_user_type_selector()
    current_user_type = st.session_state.get("user_type_select", "User / Family") # Get current user type


    # --- Handle Reset if requested by User Type Change ---
    if st.session_state.get('reset_requested', False):
         logger.info("Executing conversation reset due to user type change.")
         # Perform the reset logic
         if st.session_state.get('chatbot') is not None:
            st.session_state.chatbot.reset_conversation() # Resets backend internal state
            logger.debug("Chatbot instance conversation state reset.")
         else:
             logger.warning("Chatbot instance is None during reset, no backend state to clear.")

         # Reset ALL relevant Streamlit session state variables for a clean slate
         st.session_state.messages = []
         st.session_state.ui_state = {"step": "input", "payload": None}
         st.session_state.processing_input_payload = None
         st.session_state.form_timestamp = datetime.now().timestamp() # Reset timestamp for new forms
         # Clear symptom specific state variables on reset
         if 'confirmed_symptoms_from_ui' in st.session_state:
             del st.session_state.confirmed_symptoms_from_ui
             logger.debug("Cleared confirmed_symptoms_from_ui state on reset.")
         if 'original_query_for_symptom_rerun' in st.session_state:
             del st.session_state.original_query_for_symptom_rerun
             logger.debug("Cleared original_query_for_symptom_rerun state on reset.")
         # Clear any local form state variables if they exist (defensive)
         keys_to_delete = [k for k in st.session_state.keys() if k.startswith("symptom_confirmation_form_")]
         for k in keys_to_delete:
              if k in st.session_state: # Check existence before deleting
                  del st.session_state[k]
                  logger.debug(f"Cleared form-specific state variable: {k}")

         # Clear the reset flag
         del st.session_state.reset_requested
         logger.debug("Reset flag cleared. Triggering rerun.")
         st.rerun() # Trigger rerun to apply the reset


    try:
        # Assumes 'image_path' is globally defined
        logo = Image.open(image_path)
        col1, col2 = st.columns([1, 10])
        with col1: st.image(logo, width=100)
        with col2: st.markdown("# DxAI-Agent")
        logger.debug("Logo and title displayed.")
    except FileNotFoundError:
        logger.warning(f"Logo image not found at {image_path}. Displaying title only.")
        st.markdown("# DxAI-Agent")
    except Exception as e:
         logger.error(f"Error displaying logo: {e}")
         st.markdown("# DxAI-Agent")


    # --- Initialization of Chatbot Instance and Status ---
    # This block runs ONLY on the very first script execution in a session.
    # It ensures the chatbot instance and its initialization status are stored.
    # Use a flag to avoid re-running initialization logic on every rerun.
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False # Flag to indicate initialization is pending or failed
        st.session_state.chatbot = None # Initialize chatbot instance placeholder
        st.session_state.init_status = (False, "Initialization not started.") # Initialize status

        logger.info("Starting chatbot instance and backend setup...")
        try:
             with st.spinner("Initializing chat assistant (LLM, KG, Documents)..."):
                  st.session_state.chatbot = DocumentChatBot() # Create the chatbot instance
                  # Perform the initialization that sets up LLM, VDB, KG connection
                  success, init_message = st.session_state.chatbot.initialize_qa_chain()
                  # Store the final status
                  st.session_state.init_status = (success, init_message)
                  st.session_state.chatbot_initialized = success # Set flag based on success
             logger.info(f"Chatbot initialization complete. Status: {st.session_state.init_status}")
        except Exception as e:
             logger.critical(f"CRITICAL UNCAUGHT ERROR DURING INITIALIZATION: {e}", exc_info=True)
             st.session_state.init_status = (False, f"Critical initialization error: {e}") # Store failure status
             st.session_state.chatbot = None # Ensure chatbot is None if init fails
             st.session_state.chatbot_initialized = False # Ensure flag is False


    # --- Retrieve Initialization Status for Subsequent Reruns ---
    # This happens on every rerun to know if the bot is usable.
    init_success, init_msg = st.session_state.get('init_status', (False, "Initialization status not set."))
    logger.debug(f"Current init status (retrieved from state): Success={init_success}, Msg='{init_msg}'")

    # Determine if interaction is enabled based on init success and chatbot instance existing
    # This guards against trying to use the bot if init failed or the instance wasn't created
    # Also check the chatbot_initialized flag
    is_interaction_enabled = st.session_state.get('chatbot_initialized', False) and st.session_state.get('chatbot') is not None


    # --- UI State Variable Initializations (rest of session state) ---
    # These ensure essential state variables exist on the first run of the script.
    # Controls what the main input/action area shows: "input" or "confirm_symptoms"
    if 'ui_state' not in st.session_state:
        logger.info("Initializing ui_state.")
        st.session_state.ui_state = {"step": "input", "payload": None}

    # Messages for display in the chat history
    if 'messages' not in st.session_state:
        logger.info("Initializing messages state.")
        st.session_state.messages = [] # List of (content, is_user) tuples for UI display

    # Variable to hold the input data that needs processing by process_user_query
    # This is the trigger for the backend call. Set by user input or form submission logic.
    # Cleared BEFORE calling process_user_query.
    if 'processing_input_payload' not in st.session_state:
         logger.info("Initializing processing_input_payload state.")
         st.session_state.processing_input_payload = None # Dict like {"query": ..., "confirmed_symptoms": ...}

    # Variable to store the symptoms confirmed by the UI form, detected by the main loop.
    # Set by display_symptom_checklist on form submit. Cleared by the main loop AFTER
    # it sets the processing_input_payload for the symptom rerun.
    if 'confirmed_symptoms_from_ui' not in st.session_state:
         logger.info("Initializing confirmed_symptoms_from_ui state.")
         st.session_state.confirmed_symptoms_from_ui = None

    # Variable to store the original query text that triggered the symptom UI.
    # This is needed because the main chat input is disabled when the UI is shown.
    # Set by main() when process_user_query returns "show_symptom_ui".
    # Cleared by main() AFTER the symptom rerun processing (triggered by confirmed_symptoms_from_ui) completes
    # and the ui_state is set back to "input".
    if 'original_query_for_symptom_rerun' not in st.session_state:
         logger.info("Initializing original_query_for_symptom_rerun state.")
         st.session_state.original_query_for_symptom_rerun = None

    # Add a timestamp for the symptom confirmation form key to ensure uniqueness across reruns,
    # especially if the same original query triggers the UI multiple times (e.g., after reset).
    if 'form_timestamp' not in st.session_state:
         logger.info("Initializing form_timestamp state.")
         st.session_state.form_timestamp = datetime.now().timestamp()


    # Display initialization status in sidebar
    if not is_interaction_enabled:
         st.sidebar.error(f"Initialization Failed: {init_msg}")
         logger.error(f"Initialization failed. Message: {init_msg}")
    else:
         st.sidebar.success(f"Initialization Status: {init_msg}")
         logger.info("Chatbot initialized successfully.")


    # --- Main Content Tabs ---
    tab1, tab2 = st.tabs(["Chat", "About"])

    with tab1:
        # --- Chat Messages Display ---
        # Display messages from state. Add feedback buttons to the last assistant message
        # Ensure messages are rendered in order they were added
        for i, (msg_content, is_user) in enumerate(st.session_state.messages):
            if is_user:
                with st.chat_message("user"): st.write(msg_content)
            else:
                with st.chat_message("assistant"):
                    # Use st.markdown with unsafe_allow_html=True to render the pathway info span
                    st.markdown(msg_content, unsafe_allow_html=True)
                    # Add feedback buttons only to final answers (when input is enabled again)
                    # A message is considered a final answer display if it's the last message shown
                    # and the UI is back to the 'input' state, meaning the processing for this turn is complete.
                    # Check if the last message in session state *is* this message being displayed.
                    is_last_displayed_message = (i == len(st.session_state.messages) - 1)

                    # Only show feedback if it's the last message AND the UI is in input state
                    # AND interaction is generally enabled.
                    if is_last_displayed_message and st.session_state.ui_state["step"] == "input" and is_interaction_enabled:
                        col = st.container() # Use a container to manage layout
                        with col:
                            # Ensure unique keys for feedback buttons per message index and a hash of content
                            # Using index + hash reduces key collisions, especially if message content is identical.
                            # Sanitize msg_content before hashing to ignore pathway/disclaimer differences
                            sanitized_msg_content = msg_content.split("IMPORTANT MEDICAL DISCLAIMER:", 1)[0].strip()
                            sanitized_msg_content = re.sub(r"\n\n<span style='font-size: 0.8em; color: grey;'>.*</span>$", "", sanitized_msg_content).strip()

                            msg_hash = abs(hash(sanitized_msg_content))
                            feedback_key_up = f"thumbs_up_{i}_{msg_hash}"
                            feedback_key_down = f"thumbs_down_{i}_{msg_hash}"
                            b1, b2, _ = st.columns([0.05, 0.05, 0.9]) # Small columns for buttons, large for spacing

                            # Find the preceding user message for logging
                            user_msg_content = ""
                            # Iterate backwards from the current message index to find the most recent user message
                            for j in range(i - 1, -1, -1):
                                if st.session_state.messages[j][1] is True: # Check if it's a user message
                                    user_msg_content = st.session_state.messages[j][0]
                                    break # Found the last user message before this assistant message

                            with b1:
                                # Check if feedback for this message index+hash has already been given in this session
                                if f'feedback_{feedback_key_up}' not in st.session_state and f'feedback_{feedback_key_down}' not in st.session_state:
                                     if st.button("👍", key=feedback_key_up, help="Rate this response positively"):
                                          logger.info(f"Thumbs Up feedback for user query: '{user_msg_content[:50]}...'")
                                          # Call the global vote_message function
                                          vote_message(user_msg_content, msg_content, "thumbs_up", current_user_type) # Use current user type
                                          st.session_state[f'feedback_{feedback_key_up}'] = True # Mark as given
                                          st.toast("Feedback recorded: Thumbs Up!")
                                          st.rerun() # Rerun to grey out the button
                                elif f'feedback_{feedback_key_up}' in st.session_state:
                                     st.button("👍", key=feedback_key_up, disabled=True, help="You already rated this.") # Grey out if already given

                            with b2:
                                if f'feedback_{feedback_key_up}' not in st.session_state and f'feedback_{feedback_key_down}' not in st.session_state:
                                     if st.button("👎", key=feedback_key_down, help="Rate this response negatively"):
                                          logger.info(f"Thumbs Down feedback for user query: '{user_msg_content[:50]}...'")
                                          # Call the global vote_message function
                                          vote_message(user_msg_content, msg_content, "thumbs_down", current_user_type) # Use current user type
                                          st.session_state[f'feedback_{feedback_key_down}'] = True # Mark as given
                                          st.toast("Feedback recorded: Thumbs Down!")
                                          st.rerun() # Rerun to grey out the button
                                elif f'feedback_{feedback_key_down}' in st.session_state:
                                     st.button("👎", key=feedback_key_down, disabled=True, help="You already rated this.") # Grey out if already given


        input_area_container = st.container() # Use a container to hold the dynamic input area
        st.write("  \n" * 2) # Add some space below chat

        # --- Main Input Area Conditional Rendering ---
        # This block determines whether to show the chat input or the symptom checklist based on ui_state.
        with input_area_container:
            if not is_interaction_enabled:
                 st.error("Chat assistant failed to initialize. Please check the logs and configuration.")
                 # Display a disabled chat input to show the user where it *would* be
                 st.chat_input("Initializing...", disabled=True, key="init_failed_disabled_input")

            elif st.session_state.ui_state["step"] == "confirm_symptoms":
                logger.debug("UI state is 'confirm_symptoms', displaying checklist.")
                # Retrieve payload safely
                ui_payload = st.session_state.ui_state.get("payload")

                # Critical Check: If we are in confirm_symptoms state, payload MUST be valid.
                if ui_payload is None or not isinstance(ui_payload, dict) or "symptom_options" not in ui_payload or "original_query" not in ui_payload:
                     logger.error("UI State is 'confirm_symptoms' but payload is missing or invalid. Resetting UI state and showing error.")
                     # Add an error message to the chat history
                     st.session_state.messages.append(("Sorry, an error occurred trying to show the symptom checklist. Please try your query again.", False))
                     st.session_state.ui_state = {"step": "input", "payload": None} # Reset UI state
                     # Clean up potentially half-set symptom state variables
                     if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                     if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                     st.rerun() # Trigger rerun to display error and fix state
                     return # Stop processing this rerun

                # Display the symptom checklist UI using the payload data
                display_symptom_checklist( # Assumes display_symptom_checklist exists globally
                     ui_payload.get("symptom_options", {}), # Provide default empty dict
                     ui_payload.get("original_query", "") # Provide default empty string
                )
                # Disable the main chat input while the symptom form is active
                st.chat_input("Confirm symptoms above...", disabled=True, key="disabled_chat_input")

            elif st.session_state.ui_state["step"] == "input":
                logger.debug("UI state is 'input', displaying chat input.")
                # Display the main chat input, enabled only if initialization was successful
                user_query = st.chat_input("Ask your medical question...", disabled=not is_interaction_enabled, key="main_chat_input")

                # If the user submits a new query via the chat input
                if user_query and is_interaction_enabled:
                    logger.info(f"Detected new chat input: '{user_query[:50]}...'. Triggering processing.")
                    # Add user message to state immediately for display in this rerun
                    st.session_state.messages.append((user_query, True))
                    # Reset backend state for a brand new conversation thread starting with this query
                    # This is crucial for a fresh start on a new user query.
                    if st.session_state.get('chatbot') is not None:
                         st.session_state.chatbot.reset_conversation()
                         logger.debug("Chatbot instance conversation state reset.")
                    else:
                         logger.warning("Chatbot instance is None, cannot call reset_conversation.")

                    # Reset UI form/symptom state variables for a new thread.
                    # These need to be cleared to ensure previous symptom interactions don't interfere.
                    st.session_state.form_timestamp = datetime.now().timestamp() # Ensures a new form key if UI is triggered again
                    if 'confirmed_symptoms_from_ui' in st.session_state:
                        del st.session_state.confirmed_symptoms_from_ui
                        logger.debug("Cleared confirmed_symptoms_from_ui state on new input.")
                    # Clear the dedicated rerun variable when starting a *new* input thread.
                    if 'original_query_for_symptom_rerun' in st.session_state:
                        del st.session_state.original_query_for_symptom_rerun
                        logger.debug("Cleared original_query_for_symptom_rerun state on new input.")
                    # Also clear feedback state for previous messages on a new query
                    keys_to_delete_feedback = [k for k in st.session_state.keys() if k.startswith("feedback_thumbs_")]
                    for k in keys_to_delete_feedback:
                        del st.session_state[k]
                    logger.debug("Cleared previous feedback state on new input.")


                    # Set the input to be processed in the *next* rerun.
                    # This payload tells the processing block below *what* to process.
                    st.session_state.processing_input_payload = {
                        "query": user_query, # The new query text
                        "confirmed_symptoms": None, # No confirmed symptoms for an initial query
                        # original_query_if_followup is None for initial query
                    }
                    logger.debug("Set processing_input_payload for new chat input.")
                    st.rerun() # Trigger rerun to process the input (the block below will now run)

        # --- Check for Symptom Form Submission and Trigger Processing ---
        # This block runs *after* the symptom form (if active) might have been submitted in this rerun.
        # The display_symptom_checklist function sets st.session_state.confirmed_symptoms_from_ui
        # and updates ui_state to {"step": "input", ...} on submit.
        # We check for confirmed_symptoms_from_ui to detect the form submission.
        if 'confirmed_symptoms_from_ui' in st.session_state and st.session_state.confirmed_symptoms_from_ui is not None:
             logger.info("Detected symptom confirmation form submission via state ('confirmed_symptoms_from_ui' is set). Preparing processing payload.")

             # Get the confirmed symptoms list from the state variable set by the form
             confirmed_symps_to_pass = st.session_state.confirmed_symptoms_from_ui

             # Retrieve the original query from the dedicated state variable set when the UI was initially shown.
             # This is CRUCIAL for the symptom rerun processing as the main input was disabled.
             original_query_to_pass = st.session_state.get('original_query_for_symptom_rerun')

             # Critical Check: If confirmed_symptoms_from_ui is set, original_query_for_symptom_rerun MUST also be set and valid.
             # This catches errors where the state gets corrupted.
             if not original_query_to_pass or not isinstance(original_query_to_pass, str):
                  logger.critical("confirmed_symptoms_from_ui set, but original_query_for_symptom_rerun is missing or invalid! Cannot re-process symptom confirmation.")
                  # Add an error message to the chat history
                  st.session_state.messages.append(("Sorry, an internal error occurred during symptom confirmation processing. Please try your original query again.", False))
                  # Clean up the state variables that indicate form submission/rerun attempt
                  if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                  if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                  st.session_state.ui_state = {"step": "input", "payload": None} # Reset UI state
                  st.rerun() # Trigger rerun to display error and clear state
                  return # Stop processing this rerun


             # Clear the temporary state variable used by the form submission after we've read it.
             # This prevents this block from triggering on subsequent reruns until a new form is submitted.
             del st.session_state.confirmed_symptoms_from_ui
             logger.debug("Cleared confirmed_symptoms_from_ui state.")

             # Note: We DO NOT clear original_query_for_symptom_rerun here yet.
             # It will be cleared *after* the rerun processing is finished (in the processing_input_payload block below)
             # and the ui_state is set back to "input".

             # Set the input into processing_input_payload to trigger the backend call in the next block.
             st.session_state.processing_input_payload = {
                  "query": original_query_to_pass, # Use the stored original query text for retrieval/generation context
                  "confirmed_symptoms": confirmed_symps_to_pass, # Pass the confirmed symptoms from the UI list
                  # original_query_if_followup parameter will be set inside process_user_query itself
             }
             logger.info("Set processing_input_payload for symptom confirmation rerun.")
             st.rerun() # Trigger rerun to process the confirmed symptoms

        # --- Call the Backend Orchestrator if Processing is Needed ---
        # This is the SINGLE point where process_user_query is called based on the state.
        # It runs if a new chat input was detected OR if a symptom form was just submitted.
        if st.session_state.processing_input_payload is not None:
            logger.info("Detected processing_input_payload. Calling chatbot.process_user_query.")
            input_data = st.session_state.processing_input_payload

            # Clear the processing flag immediately so this block doesn't run again
            # until a *new* input or form submission sets it.
            st.session_state.processing_input_payload = None
            logger.debug("Cleared processing_input_payload state.")

            # Ensure chatbot instance exists and is initialized before calling its method
            if st.session_state.get('chatbot') is None or not st.session_state.get('chatbot_initialized', False):
                 logger.critical("Attempted to call process_user_query, but chatbot instance is None or not initialized.")
                 st.session_state.messages.append(("Sorry, the chat assistant is not initialized properly.", False))
                 st.session_state.ui_state = {"step": "input", "payload": None} # Reset UI state
                 # Clean up symptom rerun state if it was active before this error
                 if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                 if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                 st.rerun()
                 return # Stop processing


            # Extract the input data from the payload
            # query_to_process will be the *new* user query for initial inputs,
            # and the *original* query text for symptom confirmation reruns.
            query_to_process = input_data.get("query", "")
            confirmed_symps = input_data.get("confirmed_symptoms") # Will be None for initial inputs


            # Prevent processing if query_to_process is empty (e.g., from a bug)
            if not query_to_process:
                 logger.error("processing_input_payload contained empty query. Skipping processing.")
                 st.session_state.messages.append(("Sorry, I received an empty query for processing.", False))
                 st.session_state.ui_state = {"step": "input", "payload": None} # Reset UI state
                 # Clean up symptom rerun state if applicable (it would only be set if confirmed_symps is not None)
                 if confirmed_symps is not None and 'original_query_for_symptom_rerun' in st.session_state:
                     del st.session_state.original_query_for_symptom_rerun
                     logger.debug("Cleared original_query_for_symptom_rerun state due to empty query.")
                 if confirmed_symps is not None and 'confirmed_symptoms_from_ui' in st.session_state:
                     del st.session_state.confirmed_symptoms_from_ui
                     logger.debug("Cleared confirmed_symptoms_from_ui state due to empty query.")
                 st.rerun()
                 return # Stop processing


            with st.spinner("Thinking..."):
                 try:
                     # Call the chatbot's main processing function
                     # Pass the current user type from the sidebar selector
                     response_text, ui_action, ui_payload = st.session_state.chatbot.process_user_query(
                          query_to_process, current_user_type, # Pass user type
                          confirmed_symptoms=confirmed_symps # Pass confirmed symptoms if available (from form submit)
                          # original_query_if_followup is derived internally in process_user_query now
                     )

                     logger.info(f"process_user_query returned ui_action: {ui_action}")

                     if ui_action == "display_final_answer":
                          logger.info("UI Action: display_final_answer. Adding message.")
                          st.session_state.messages.append((response_text, False))
                          # Reset UI state back to input, clearing symptom UI if it was active
                          st.session_state.ui_state = {"step": "input", "payload": None}
                          logger.debug("UI state set to 'input'.")

                          # If this rerun came from symptom UI (indicated by confirmed_symps being not None),
                          # clear the original_query_for_symptom_rerun state variable now that the thread is complete.
                          if confirmed_symps is not None and 'original_query_for_symptom_rerun' in st.session_state:
                              del st.session_state.original_query_for_symptom_rerun
                              logger.debug("Cleared original_query_for_symptom_rerun after symptom rerun finished.")
                          # Clear confirmed_symptoms_from_ui state if it was set
                          if confirmed_symps is not None and 'confirmed_symptoms_from_ui' in st.session_state:
                              del st.session_state.confirmed_symptoms_from_ui
                              logger.debug("Cleared confirmed_symptoms_from_ui after symptom rerun finished.")


                     elif ui_action == "show_symptom_ui":
                          logger.info("UI Action: show_symptom_ui. Adding prompt message.")
                          # Add the prompt message returned by process_user_query
                          st.session_state.messages.append((response_text, False))
                          # Update UI state to show the symptom checklist on the next rerun
                          st.session_state.ui_state = {"step": "confirm_symptoms", "payload": ui_payload}
                          # Set a new timestamp for the form key to ensure a fresh form rendering
                          st.session_state.form_timestamp = datetime.now().timestamp()
                          logger.debug("UI state set to 'confirm_symptom_ui'.")

                          # Store the original query text from the payload that *triggered* the UI display.
                          # This is essential for the rerun when the form submits.
                          original_query_to_store = ui_payload.get("original_query")
                          if original_query_to_store:
                               st.session_state.original_query_for_symptom_rerun = original_query_to_store
                               logger.debug(f"Stored original_query_for_symptom_rerun: '{original_query_to_store[:50]}...'")
                          else:
                               logger.error("UI payload for 'show_symptom_ui' missing 'original_query'. Cannot store for rerun.")
                               # Attempt to reset state and show an error if original query is missing here
                               st.session_state.messages.append(("Sorry, an error occurred preparing the symptom checklist. Please try your query again.", False))
                               st.session_state.ui_state = {"step": "input", "payload": None}
                               if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                               if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun # Clear potentially bad state


                     elif ui_action == "none":
                          logger.info("UI Action: none. No message added.")
                          # This action implies internal state was updated but no immediate UI change needed.
                          # Ensure UI state is 'input' if it wasn't changed internally.
                          if st.session_state.ui_state["step"] != "confirm_symptoms": # Don't override if it's already set to show the form
                               st.session_state.ui_state = {"step": "input", "payload": None}
                          # Clean up symptom rerun state if this 'none' action somehow concludes a rerun path
                          # (Less likely, but defensive)
                          if confirmed_symps is not None and 'original_query_for_symptom_rerun' in st.session_state:
                              del st.session_state.original_query_for_symptom_rerun
                              logger.debug("Cleared original_query_for_symptom_rerun after 'none' action.")
                          if confirmed_symps is not None and 'confirmed_symptoms_from_ui' in st.session_state:
                              del st.session_state.confirmed_symptoms_from_ui
                              logger.debug("Cleared confirmed_symptoms_from_ui after 'none' action.")

                     else:
                          logger.error("Unknown ui_action returned: %s. Defaulting to input state.", ui_action)
                          st.session_state.messages.append((f"An internal error occurred (Unknown UI action: {ui_action}).", False))
                          st.session_state.ui_state = {"step": "input", "payload": None}
                          # Clean up symptom rerun state on error
                          if confirmed_symps is not None and 'original_query_for_symptom_rerun' in st.session_state:
                              del st.session_state.original_query_for_symptom_rerun
                              logger.debug("Cleared original_query_for_symptom_rerun after unknown UI action error.")
                          if confirmed_symps is not None and 'confirmed_symptoms_from_ui' in st.session_state:
                              del st.session_state.confirmed_symptoms_from_ui
                              logger.debug("Cleared confirmed_symptoms_from_ui after unknown UI action error.")


                 except ValueError as e: # Catch ValueErrors raised by local_generate and its callers
                     logger.error(f"LLM/Processing Error during chatbot execution: {e}", exc_info=True)
                     # Add error message including pathway info attempt
                     error_pathway_info = "Error during AI processing"
                     # Attempt to derive pathway info based on available context if not already set
                     # Note: pathway_info is determined *at the very end* of process_user_query,
                     # so it won't be available in the scope of these catch blocks unless passed in.
                     # Default to a generic error pathway.

                     error_response = f"Sorry, an AI processing error occurred: {e}\n\n<span style='font-size: 0.8em; color: grey;'>*Sources attempted for this response: {error_pathway_info}*</span>"
                     st.session_state.messages.append((error_response, False))

                     # Reset UI state to input on error
                     st.session_state.ui_state = {"step": "input", "payload": None}
                     # Clean up symptom rerun state on error
                     if confirmed_symps is not None and 'original_query_for_symptom_rerun' in st.session_state:
                          del st.session_state.original_query_for_symptom_rerun
                          logger.debug("Cleared original_query_for_symptom_rerun after ValueError.")
                     if confirmed_symps is not None and 'confirmed_symptoms_from_ui' in st.session_state:
                          del st.session_state.confirmed_symptoms_from_ui
                          logger.debug("Cleared confirmed_symptoms_from_ui after ValueError.")


                 except Exception as e: # Catch any other unexpected errors
                     logger.error(f"Unexpected Error during chatbot process_user_query execution: {e}", exc_info=True)
                     # Add error message including pathway info attempt
                     error_pathway_info = "Unexpected error during processing"
                     # Same as above, default to generic error pathway info

                     error_response = f"Sorry, an unexpected error occurred: {e}\n\n<span style='font-size: 0.8em; color: grey;'>*Sources attempted for this response: {error_pathway_info}*</span>"
                     st.session_state.messages.append((error_response, False))
                     # Reset UI state to input on error
                     st.session_state.ui_state = {"step": "input", "payload": None}
                     # Clean up symptom rerun state on error
                     if confirmed_symps is not None and 'original_query_for_symptom_rerun' in st.session_state:
                          del st.session_state.original_query_for_symptom_rerun
                          logger.debug("Cleared original_query_for_symptom_rerun after unexpected error.")
                     if confirmed_symps is not None and 'confirmed_symptoms_from_ui' in st.session_state:
                          del st.session_state.confirmed_symptoms_from_ui
                          logger.debug("Cleared confirmed_symptoms_from_ui after unexpected error.")


            # Always trigger a rerun after processing a payload.
            # This ensures the UI updates based on the new st.session_state.ui_state.
            logger.debug("Triggering rerun after processing_input_payload.")
            st.rerun()

        st.divider()
        # Ensure reset button is disabled if init failed
        if st.button("Reset Conversation", key="reset_conversation_button_main", disabled=not is_interaction_enabled, help="Clear the current conversation history."):
            logger.info("Conversation reset triggered by user.")
            # Ensure chatbot instance exists before resetting its state
            if st.session_state.get('chatbot') is not None:
                st.session_state.chatbot.reset_conversation() # Resets backend internal state
                logger.debug("Chatbot instance conversation state reset.")
            else:
                 logger.warning("Chatbot instance is None, cannot call reset_conversation.")

            # Reset ALL relevant Streamlit session state variables for a clean slate
            st.session_state.messages = []
            st.session_state.ui_state = {"step": "input", "payload": None}
            st.session_state.processing_input_payload = None
            st.session_state.form_timestamp = datetime.now().timestamp() # Reset timestamp for new forms
            # Clear symptom specific state variables on reset
            if 'confirmed_symptoms_from_ui' in st.session_state:
                del st.session_state.confirmed_symptoms_from_ui
                logger.debug("Cleared confirmed_symptoms_from_ui state on reset.")
            if 'original_query_for_symptom_rerun' in st.session_state:
                del st.session_state.original_query_for_symptom_rerun
                logger.debug("Cleared original_query_for_symptom_rerun state on reset.")
            # Clear any local form state variables if they exist (defensive)
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith("symptom_confirmation_form_")]
            for k in keys_to_delete:
                 if k in st.session_state: # Check existence before deleting
                     del st.session_state[k]
                     logger.debug(f"Cleared form-specific state variable: {k}")
            # Clear feedback state on reset
            keys_to_delete_feedback = [k for k in st.session_state.keys() if k.startswith("feedback_thumbs_")]
            for k in keys_to_delete_feedback:
                del st.session_state[k]
            logger.debug("Cleared feedback state on reset.")


            logger.debug("Triggering rerun after reset.")
            st.rerun()

        st.divider()
        st.subheader("🩺 Detailed Feedback")
        # Disable feedback form if init failed
        # Added unique key for feedback form
        with st.form(key="detailed_feedback_form", clear_on_submit=True):
            feedback_text = st.text_area(
                "Enter corrections, improvements, or comments here...", height=100, disabled=not is_interaction_enabled, help="Provide detailed feedback to help improve the AI's responses."
            )
            submit_feedback_btn = st.form_submit_button("Submit Feedback", disabled=not is_interaction_enabled)
            if submit_feedback_btn and feedback_text:
                logger.info("Detailed feedback submitted.")
                # Call the global submit_feedback function, passing the current conversation history
                submit_feedback(feedback_text, st.session_state.messages, current_user_type) # Use current user type
                st.success("Thank you for your feedback!")

    with tab2:
        st.markdown("""
        ## Medical Chat Assistant

        This is an experimental medical chat assistant designed to provide information based on a knowledge graph and provided documents.

        **How it Works:**
        1.  **Medical Check:** It first assesses if your query is related to health or medicine.
        2.  **Symptom Analysis:** If the query is medical, it attempts to identify reported symptoms. If potential conditions are found in the Knowledge Graph but with low confidence, it may ask you to confirm or add symptoms via a checklist to improve accuracy.
        3.  **Information Retrieval:** It searches a Medical Knowledge Graph for related conditions, treatments, and remedies, and retrieves relevant passages from uploaded documents (RAG) if available.
        4.  **Context Selection:** It intelligently selects the most confident and relevant information from the Knowledge Graph and documents to use as context for the AI.
        5.  **Initial Answer:** It generates an initial answer using the selected context and its general medical knowledge (LLM).
        6.  **Self-Reflection:** It evaluates its initial answer for completeness based on your original query and the information it had access to.
        7.  **Information Gap Filling:** If the answer is incomplete, it attempts to find additional information using its general knowledge to fill the gaps.
        8.  **Final Answer:** It combines the initial and supplementary information into a coherent final response, formatted for the selected user type (Family/User or Physician).
        9.  **Triage Assessment:** For Family/User interactions, it performs an additional check to assess if the situation requires immediate medical attention and indicates a triage level.
        10. **Pathway Indication:** The response includes a small note indicating which primary sources (Knowledge Graph, Documents (RAG), LLM's general knowledge, Reflection Agent) were used to generate the answer.

        **Data Sources:**
        *   **Medical Knowledge Graph:** Contains structured information about diseases, symptoms, and treatments.
        *   **Documents (RAG):** Processes information from uploaded PDF files (e.g., `rawdata.pdf`).
        *   **LLM (Gemini 1.5 Flash):** Provides general medical knowledge and powers the multi-step reasoning (guardrail, symptom extraction, context selection, initial answer generation, reflection, gap filling, collation, triage).

        **Disclaimer:** This system is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read here.
        """)
    logger.debug("--- Streamlit App End of Rerun ---")


if __name__ == "__main__":
    # Set basic logging level before running the main app logic
    # Avoid double handlers if this is run by a Streamlit server internally
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Ensure logger is configured even if basicConfig was already called elsewhere
    logger = logging.getLogger(__name__)
    # Set level specifically for this logger if needed
    # logger.setLevel(logging.DEBUG)

    main()
