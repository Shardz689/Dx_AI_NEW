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
try:
    import torch
except ImportError:
     pass # Torch is not strictly required if you only use CPU embeddings and don't use torch directly elsewhere


# Configuration
from dotenv import load_dotenv
load_dotenv()

# !!! WARNING: Replace these with your actual credentials or ensure .env is configured correctly !!!
# It is strongly recommended to use environment variables loaded by dotenv instead of hardcoding.
# If using environment variables, remove the default "YOUR_..." values.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBv-I8Ld-k09Lxu9Yi7HPffZHKXIqGSdHU")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://1b47920f.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "eCqDfyhDcuGMLzbYfiqL6jsvjH3LIXr86xQGAEKmY8Y")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# Configure logging *before* potential errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Provide informative errors if required environment variables are missing
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    logger.critical("GEMINI_API_KEY environment variable is not set or is the placeholder value.")
    # Set to None so LLM initialization fails gracefully
    GEMINI_API_KEY = None
else:
    # Basic check if it looks like a key
    if len(GEMINI_API_KEY) < 10: # Keys are usually longer than this
         logger.critical("GEMINI_API_KEY appears invalid (too short). LLM initialization may fail.")

if not NEO4J_URI or NEO4J_URI == "YOUR_NEO4J_URI" or not NEO4J_USER or not NEO4J_PASSWORD or NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD":
     logger.critical("NEO4J environment variables (URI, USER, PASSWORD) are not fully set or are placeholder values. Cannot connect.")
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
    "high_kg_context_only": 0.8
}

# Load and convert the image to base64 for favicon
def get_image_as_base64(file_path):
    logger.debug(f"Attempting to load image: {file_path}")
    if not os.path.exists(file_path):
        logger.warning(f"Image file not found at {file_path}")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" # Tiny black pixel fallback
    try:
        with open(file_path, "rb") as image_file:
            b64_string = base64.b64encode(image_file.read()).decode()
            logger.debug("Image loaded and encoded to base64.")
            return b64_string
    except Exception as e:
        logger.error(f"Error encoding image {file_path} to base64: {e}")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" # Tiny black pixel fallback

image_path = "Zoom My Life.jpg" # Update with your actual logo path
icon = get_image_as_base64(image_path)

# Cache for expensive operations
CACHE = {}

def get_cached(key):
    key_str = json.dumps(key, sort_keys=True)
    if key_str in CACHE:
        # logger.debug(f"Cache hit for key: {key_str[:50]}...") # Suppress debug spam
        return CACHE[key_str]
    # logger.debug(f"Cache miss for key: {key_str[:50]}...") # Suppress debug spam
    return None

def set_cached(key, value):
    key_str = json.dumps(key, sort_keys=True)
    # logger.debug(f"Setting cache for key: {key_str[:50]}...") # Suppress debug spam
    CACHE[key_str] = value
    return value

# Hardcoded PDF files to use
HARDCODED_PDF_FILES = [
    "rawdata.pdf",
    # Add more PDF paths here if needed
]

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
            # Also remove the pathway info line at the very end
            sanitized_bot_msg = re.sub(r"\n\n<span style='font-size: 0.8em; color: grey;'>.*</span>$", "", sanitized_bot_msg).strip()


            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'user_type': user_type,
                'user_message': user_message.replace("\n", " "),
                'bot_message': sanitized_bot_msg.replace("\n", " "),
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
                history_string_parts.append(f"User: {u.replace('||', '')} | Bot: {sanitized_b.replace('||', '')}")

            history_string = " || ".join(history_string_parts)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_type,
                feedback_text.replace('\n', ' ').replace('||', ''),
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
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.vectordb: Optional[FAISS] = None
        # Removed chat_history from instance, using st.session_state
        self.followup_context = {"round": 0} # Unused in this refactor, can be removed

        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        try:
            # Check for CUDA availability and set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing SentenceTransformer embeddings on device: {device}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
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
        # Check if credentials are set before attempting connection
        if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD or NEO4J_URI == "YOUR_NEO4J_URI":
             logger.error("Neo4j credentials missing or are placeholder values. Cannot connect.")
             self.kg_driver = None
             self.kg_connection_ok = False
             return

        try:
            # The timeout parameter was removed from verify_connectivity, which was the error source.
            # The connection_timeout parameter in GraphDatabase.driver is correct.
            self.kg_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, connection_timeout=5.0)
            self.kg_driver.verify_connectivity() # This should now work without the 'timeout' parameter
            logger.info("Successfully connected to Neo4j.")
            self.kg_connection_ok = True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}. KG features will be unavailable.")
            self.kg_driver = None
            self.kg_connection_ok = False

    def create_vectordb(self):
        logger.info("Creating vector database...")
        # Filter for existing files
        pdf_files = [Path(pdf_file) for pdf_file in HARDCODED_PDF_FILES if Path(pdf_file).exists()]
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
             if any(Path(pdf_file).exists() for pdf_file in HARDCODED_PDF_FILES):
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
        logger.debug(f"Checking medical relevance for query: {query}")
        cache_key = {"type": "medical_relevance", "query": query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("Medical relevance check from cache.")
            return cached

        # Fallback if LLM is not available
        if self.llm is None:
             logger.warning("LLM not initialized. Falling back to keyword medical relevance check.")
             medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose", "medicine", "doctor", "hospital", "clinic", "condition", "illness", "sick", "diagnosed"]
             result = (any(keyword in query.lower() for keyword in medical_keywords), "Fallback heuristic match (LLM unavailable)")
             logger.debug(f"Medical relevance fallback result: {result}")
             return set_cached(cache_key, result)

        medical_relevance_prompt = f'''
        Analyze the user query. Is it directly related to health, medical conditions, symptoms, treatments, medication, diagnostics, or any other medical or health science topic?
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
                    is_medical = data.get("is_medical", False)
                    confidence_raw = data.get("confidence", 0.0)
                    confidence = max(0.0, min(1.0, float(confidence_raw))) # Clamp confidence to [0, 1]
                    reasoning = data.get("reasoning", "")
                    result = (is_medical and confidence >= THRESHOLDS.get("medical_relevance", 0.6), reasoning)
                    logger.info(f"Medical relevance check result: {result}")
                    return set_cached(cache_key, result)
                except (json.JSONDecodeError, ValueError) as e: # Catch ValueError from float conversion too
                    logger.warning(f"Could not parse or process medical relevance JSON from LLM response: {e}.")
            else:
                logger.warning("No JSON found in medical relevance response.")

        except ValueError as e: # Catch ValueError from local_generate if LLM fails
             logger.error(f"Error during medical relevance LLM call: {e}")

        # Fallback if LLM call/parsing fails
        medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose", "medicine", "doctor", "hospital", "clinic", "condition", "illness", "sick", "diagnosed"]
        result = (any(keyword in query.lower() for keyword in medical_keywords), "Fallback heuristic match (LLM failed)")
        logger.debug(f"Medical relevance final fallback result: {result}")
        return set_cached(cache_key, result)


    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        logger.debug(f"Attempting symptom extraction for query: {user_query}")
        cache_key = {"type": "symptom_extraction", "query": user_query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("Symptom extraction from cache.")
            # Cache stores lowercase symptoms, ensure we return list of strings
            return list(cached[0]), cached[1]

        # Expanded common symptom keywords for fallback
        common_symptom_keywords = ["fever", "cough", "headache", "sore throat", "nausea", "dizziness", "chest pain", "shortness of breath", "fatigue", "body aches", "runny nose", "congestion", "chills", "sweats", "joint pain", "muscle aches", "rash", "swelling", "pain", "ache", "burning", "itching", "numbness", "tingling", "diarrhea", "vomiting", "difficulty breathing", "difficulty swallowing", "weakness", "loss of taste", "loss of smell", "skin discoloration", "blurred vision", "ringing in ears", "loss of appetite", "weight loss", "tired", "sick"]
        query_lower = user_query.lower()
        fallback_symptoms_lower = list(set([s.strip().lower() for s in common_symptom_keywords if s in query_lower])) # Use set for uniqueness, ensure lowercase

        # Fallback if LLM is not available
        if self.llm is None:
             logger.warning("LLM not initialized. Falling back to keyword symptom extraction.")
             result = (fallback_symptoms_lower, 0.4) # Assign a low confidence for heuristic extraction
             logger.info(f"Symptom extraction fallback result: {result}")
             return set_cached(cache_key, result)

        SYMPTOM_PROMPT = f'''
        Extract all potential medical symptoms mentioned in the following user query.
        For each symptom, assign a confidence score between 0.0 and 1.0 indicating how certain you are that it is a symptom.
        Return ONLY a JSON object formatted as: {{"Extracted Symptoms": [{{"symptom": "symptom1", "confidence": 0.9}}, {{"symptom": "symptom2", "confidence": 0.8}}, ...]}}
        If no symptoms are found, return: {{"Extracted Symptoms": []}}

        User Query: "{user_query}"
        '''
        llm_symptoms_lower = []
        llm_avg_confidence = 0.0
        try:
            response = self.local_generate(SYMPTOM_PROMPT, max_tokens=500).strip()
            # logger.debug(f"Symptom extraction LLM raw response: {response}") # Suppress spam
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    symptom_data = data.get("Extracted Symptoms", []) # Get list safely

                    # Filter by threshold AND ensure data is structured correctly
                    llm_symptoms_confident_items = [
                         item for item in symptom_data
                         if isinstance(item, dict) and "symptom" in item and isinstance(item.get("symptom"), str) and item.get("confidence", 0) >= THRESHOLDS.get("symptom_extraction", 0.6)
                    ]
                    llm_symptoms_lower = sorted(list(set([item["symptom"].strip().lower() for item in llm_symptoms_confident_items]))) # Ensure lowercase and unique

                    # Calculate average confidence only for items with valid confidence scores
                    valid_confidences = [item.get("confidence", 0) for item in symptom_data if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float))]
                    llm_avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0

                    logger.debug(f"LLM extracted {len(llm_symptoms_lower)} confident symptoms above threshold {THRESHOLDS.get('symptom_extraction', 0.6)}. Avg LLM confidence: {llm_avg_confidence:.4f}")
                except json.JSONDecodeError:
                    logger.warning("Could not parse symptom JSON from LLM response")
                except Exception as parse_e:
                     logger.warning(f"Error processing symptom data structure: {parse_e}")
            else:
                 logger.warning("Could not find JSON object in LLM response for symptom extraction.")
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"Error during symptom extraction LLM call: {e}")
            # If LLM call fails, combine only fallback symptoms
            combined_symptoms_lower = list(set(fallback_symptoms_lower))
            final_confidence = 0.4 if combined_symptoms_lower else 0.0 # Low confidence
            logger.info(f"Final extracted symptoms (LLM failed): {combined_symptoms_lower} (Confidence: {final_confidence:.4f})")
            result = (combined_symptoms_lower, final_confidence)
            return set_cached(cache_key, result)


        # Combine unique symptoms from LLM and fallback (case-insensitive comparison for uniqueness)
        combined_symptom_set_lower = set(llm_symptoms_lower + fallback_symptoms_lower)
        final_symptoms_lower = sorted(list(combined_symptom_set_lower))


        # Calculate final confidence. If LLM found *any* symptoms above threshold, use its average (clamped),
        # otherwise use the fallback confidence (0.4 if fallback found symptoms, 0.0 otherwise).
        final_confidence = llm_avg_confidence if llm_symptoms_lower else (0.4 if fallback_symptoms_lower else 0.0)
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

        logger.debug(f"Checking if query is symptom-related heuristic: {query}")
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
        extracted_symptoms, symptom_confidence = self.extract_symptoms(query)

        # Check 1: Is the symptom extraction confident and did it find symptoms?
        if extracted_symptoms and symptom_confidence >= THRESHOLDS.get("symptom_extraction", 0.6):
            logger.debug("Query determined symptom-related based on confident extraction.")
            return set_cached(cache_key, True)

        # Check 2: Basic keyword matching (fallback if extraction isn't highly confident)
        health_keywords = ["symptom", "pain", "sick", "health", "disease", "condition", "diagnosis", "feel", "experiencing", "ache", "diagnosed"]
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
                 "symptoms_list": symptoms_for_kg, # Use the list passed into the function
                 "confidence": 0.0
            },
            "kg_content_other": "Medical Knowledge Graph information is unavailable.",
        }

        if not self.kg_connection_ok or self.kg_driver is None:
             logger.warning("📚 KG Agent: Connection not OK. Skipping KG queries.")
             kg_results["kg_content_other"] = "Medical Knowledge Graph is currently unavailable."
             logger.info("📚 Knowledge Graph Agent Finished (Connection Error).")
             return kg_results

        try:
            logger.debug("Attempting to acquire Neo4j session.")
            with self.kg_driver.session() as session:
                logger.debug("Neo4j session acquired.")
                if symptoms_for_kg:
                    logger.info("📚 KG Task: Identify Diseases from symptoms: %s", symptoms_for_kg)
                    # Call the helper to query diseases
                    # Pass the symptoms_for_kg directly
                    disease_data_from_kg: List[Dict[str, Any]] = self._query_disease_from_symptoms_with_session(session, symptoms_for_kg)
                    logger.debug("KG disease query returned %d results.", len(disease_data_from_kg))

                    if disease_data_from_kg:
                        kg_results["identified_diseases_data"] = disease_data_from_kg
                        # The first record is the top one due to ORDER BY confidence_score DESC
                        top_disease_record = disease_data_from_kg[0]
                        # Ensure keys match what _query_disease_from_symptoms_with_session returns ('Disease', 'Confidence', 'MatchedSymptoms')
                        top_disease_name = top_disease_record.get("Disease")
                        top_disease_conf = float(top_disease_record.get("Confidence", 0.0))
                        kg_results["top_disease_confidence"] = top_disease_conf
                        # Store the matched symptoms from the KG query result for the top disease
                        kg_results["kg_matched_symptoms"] = top_disease_record.get("MatchedSymptoms", [])

                        logger.info("✔️ Diseases Identified: %s (Top Confidence: %s)", [(d.get('Disease'), d.get('Confidence')) for d in disease_data_from_kg], top_disease_conf)

                        # Only query treatments if the top disease confidence meets the basic matching threshold
                        if top_disease_conf >= THRESHOLDS.get("disease_matching", 0.5) and top_disease_name:
                            logger.info("📚 KG Task: Find Treatments for %s", top_disease_name)
                            # Call the helper to query treatments
                            kg_results["kg_treatments"], kg_results["kg_treatment_confidence"] = self._query_treatments_with_session(session, top_disease_name)
                            logger.info("✔️ Treatments found: %s (Confidence: %s)", kg_results['kg_treatments'], kg_results['kg_treatment_confidence'])
                        else:
                            logger.info("📚 KG Task: Treatments skipped - Top disease confidence below basic matching threshold (%s < %s) or no top disease name.", top_disease_conf, THRESHOLDS.get("disease_matching", 0.5))
                else:
                     logger.info("📚 KG Task: Identify Diseases skipped - No symptoms provided.")


                # Prepare data for LLM phrasing, even if confidence is low or no diseases found
                # Use the top disease found by the KG query, or a placeholder
                top_disease_name_for_llm = kg_results["identified_diseases_data"][0].get("Disease", "an unidentifiable condition") if kg_results["identified_diseases_data"] else "an unidentifiable condition"

                kg_results["kg_content_diagnosis_data_for_llm"] = {
                      "disease_name": top_disease_name_for_llm,
                      "symptoms_list": symptoms_for_kg, # Use the symptoms list passed in, reflecting the user's input/confirmation
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
                 "symptoms_list": symptoms_for_kg, # Use the input symptom list even on error
                 "confidence": 0.0
            }
            kg_results["kg_content_other"] = f"An error occurred while querying the Medical Knowledge Graph: {str(e)}"
            kg_results["top_disease_confidence"] = 0.0 # Ensure confidence is 0 on failure
            kg_results["identified_diseases_data"] = [] # Clear potential partial results
            kg_results["kg_matched_symptoms"] = []
            kg_results["kg_treatments"] = []
            kg_results["kg_treatment_confidence"] = 0.0

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
            # Cache stores lowercase symptoms, ensure we return list of strings for symptoms
            cached_data = list(cached) # Ensure it's a mutable list copy
            for item in cached_data:
                if 'MatchedSymptoms' in item:
                    item['MatchedSymptoms'] = list(item['MatchedSymptoms'])
                if 'AllDiseaseSymptomsKG' in item:
                    item['AllDiseaseSymptomsKG'] = list(item['AllDiseaseSymptomsKG'])
            return cached_data
    
        # Ensure symptom names are lowercase for the Cypher query parameter
        symptom_names_lower = [s.lower() for s in valid_symptom_names]
    
        # Cypher query to find diseases based on input symptoms and calculate confidence
        # Confidence is the ratio of matching input symptoms to the total number of symptoms for the disease in the KG.
        cypher_query = """
        UNWIND $symptomNamesLower AS input_symptom_name_lower
        MATCH (s:symptom)
        WHERE toLower(s.Name) = input_symptom_name_lower
        MATCH (s)-[:INDICATES]->(d:disease)
        // Collect the original-cased names of matching symptoms from the KG node
        WITH d, COLLECT(DISTINCT s.Name) AS matched_symptoms_from_input_in_kg_case
        OPTIONAL MATCH (d)<-[:INDICATES]-(all_s:symptom)
        WITH d, matched_symptoms_from_input_in_kg_case,
             COLLECT(DISTINCT all_s.Name) AS all_disease_symptoms_in_kg, // All symptoms linked to the disease in KG
             size(COLLECT(DISTINCT all_s)) AS total_disease_symptoms_count, // Total symptoms for the disease in KG
             size(matched_symptoms_from_input_in_kg_case) AS matching_symptoms_count // Count of input symptoms found linked to the disease
        // The WHERE clause must come after the WITH clause that defines 'matching_symptoms_count'
        WHERE matching_symptoms_count > 0 // <-- Corrected placement
        // Calculate confidence: ratio of matched symptoms (from input) to total symptoms for the disease (in KG)
        WITH d.Name AS Disease,
             CASE WHEN total_disease_symptoms_count = 0 THEN 0.0 ELSE matching_symptoms_count * 1.0 / total_disease_symptoms_count END AS confidence_score,
             matched_symptoms_from_input_in_kg_case AS MatchedSymptoms,
             all_disease_symptoms_in_kg AS AllDiseaseSymptomsKG
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
                # Convert Neo4j Record to dictionary (records are NOT dict instances)
                disease_data.append({
                    "Disease": rec["Disease"],
                    "Confidence": float(rec["confidence_score"]),  # Field name matches Cypher query
                    "MatchedSymptoms": list(rec["MatchedSymptoms"]),  # Convert to list to ensure it's mutable
                    "AllDiseaseSymptomsKG": list(rec["AllDiseaseSymptomsKG"])  # Convert to list
                })
    
            # Filter out any records with None or empty Disease name
            disease_data = [d for d in disease_data if d.get("Disease")]
    
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
               // Simple confidence heuristic based on number of relationships (can be refined)
               CASE WHEN COUNT(r) >= 3 THEN 0.9 WHEN COUNT(r) >= 1 THEN 0.7 ELSE 0.0 END as Confidence
        ORDER BY Confidence DESC, Treatment // Order by confidence, then alphabetically by treatment name
        """
        
        try:
            logger.debug("Executing Cypher query for treatments for disease: %s", disease_name_lower)
            result = session.run(cypher_query, diseaseNameLower=disease_name_lower)
            records = list(result)
            
            # Process Neo4j Record objects directly without checking if they're dictionaries
            treatments_with_conf = []
            for rec in records:
                try:
                    # Access the Record object using dictionary-style notation
                    treatment_name = rec["Treatment"]
                    confidence = float(rec["Confidence"])
                    if treatment_name:  # Ensure treatment name is not empty
                        treatments_with_conf.append((treatment_name, confidence))
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Error processing treatment record: {rec}, error: {e}")
    
            treatments_list = [t[0] for t in treatments_with_conf]
            avg_confidence = sum(t[1] for t in treatments_with_conf) / len(treatments_with_conf) if treatments_with_conf else 0.0
            
            logger.debug("💊 Executed KG Treatment Query for %s, found %d treatments.", disease, len(treatments_list))
            return set_cached(cache_key, (treatments_list, avg_confidence))
        
        except Exception as e:
            logger.error("⚠️ Error executing KG query for treatments: %s", e, exc_info=True)
            return [], 0.0

    def retrieve_rag_context(self, query: str) -> Tuple[List[str], float]:
        logger.info(f"📄 RAG Retrieval Initiated for query: {query[:50]}...")
        # RAG_THRESHOLD_FOR_SELECTION = THRESHOLDS.get("rag_context_selection", 0.7) # Not used to filter here

        # Cache key includes the query. We retrieve top K always, filtering comes later.
        cache_key = {"type": "rag_retrieval_topk_chunks_and_scores", "query": query}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("RAG retrieval (topk chunks and scores) from cache.")
             # Return the cached data directly
             # Ensure cached chunks are treated as list of strings, score as float
             return list(cached['chunks']), float(cached['avg_score'])


        # Check if the vector database and embedding model are initialized
        if self.vectordb is None or self.embedding_model is None:
            logger.warning("📄 RAG Retrieval: Vector database or Embedding model not initialized. Skipping RAG retrieval.")
            return [], 0.0 # Return empty results and 0 confidence

        try:
            # Define the number of top results to retrieve
            k = 10
            logger.debug(f"Performing vector search for query: {query[:50]}... to retrieve top {k} documents with scores.")

            # Call the vector database's method to get the top k documents and their scores.
            # This method typically returns (Document, score) pairs.
            # The score is often distance (like cosine distance), where lower is better.
            retrieved_docs_with_scores = self.vectordb.similarity_search_with_score(query, k=k)
            logger.debug(f"📄 RAG: Retrieved {len(retrieved_docs_with_scores)} potential documents from vector DB.")

            top_k_chunks_content: List[str] = [] # List to store text content of the top K chunks
            top_k_similarity_scores: List[float] = [] # List to store calculated similarity scores of the top K chunks

            # Process the retrieved documents
            for doc, score in retrieved_docs_with_scores:
                # logger.debug(f"Processing retrieved chunk. Raw score type: {type(score)}, value: {score}") # Suppress spam

                # Ensure the score is a valid numeric type. Skip if not.
                if not isinstance(score, (int, float, np.floating)):
                     logger.warning(f"Received unexpected non-numeric raw score type ({type(score)}) for a retrieved chunk. Skipping this chunk for scoring/context.")
                     continue

                # Calculate similarity score from distance. Assume cosine distance where lower score is better.
                # Similarity = 1 - Distance. Clamp to [0, 1].
                similarity_score = max(0.0, min(1.0, 1 - float(score)))
                # logger.debug(f"Calculated similarity score: {similarity_score:.4f} (from raw score: {float(score):.4f})") # Suppress spam


                # Add the chunk content and its calculated similarity score to the lists
                # We add ALL top K chunks here based on the vector search result, regardless of their individual score relative to the *selection* threshold.
                # The selection threshold is applied *later* in select_context or implicit in the prompt if all chunks are passed.
                if doc and doc.page_content: # Ensure document and content exist
                    top_k_chunks_content.append(doc.page_content)
                    top_k_similarity_scores.append(similarity_score)
                    # logger.debug(f"📄 RAG: Processed top chunk (Sim: {similarity_score:.4f})") # Suppress spam


            # Calculate the overall RAG confidence score (S_RAG)
            # This is the average of the similarity scores of the *top K* chunks that were successfully processed.
            # If no chunks were processed (e.g., k=0 or all scores were invalid), S_RAG is 0.0
            srag = sum(top_k_similarity_scores) / len(top_k_similarity_scores) if top_k_similarity_scores else 0.0

            logger.info(f"📄 RAG Retrieval Finished. Processed {len(top_k_chunks_content)} chunks out of {len(retrieved_docs_with_scores)} initially retrieved. Overall S_RAG (Avg of Processed Top K): {srag:.4f}")

            # Store the result in the cache before returning
            # Cache stores the content of the top K chunks and their average score
            # We only cache the content and the average score, as the individual scores aren't used downstream in the *current* select_context logic.
            # If select_context were modified to filter chunks by individual score, we'd need to cache (chunk, score) pairs.
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
        if is_symptom_query and s_kg > high_kg_only_threshold:
            # Check if KG results are actually available before selecting
            if kg_results:
                 logger.info("📦 Applying High KG Confidence Rule (%.4f > %.4f). Selecting KG ONLY, ignoring RAG for this turn.", s_kg, high_kg_only_threshold)
                 # In this scenario, we ONLY select KG
                 selected_context["kg"] = kg_results
                 # RAG is explicitly excluded here
            else:
                 # This case is unlikely if s_kg > 0 but handle defensively
                 logger.warning("📦 High KG Confidence Rule triggered (%.4f), but kg_results is empty. Falling back to standard selection.", s_kg)
                 # Fall through to the standard logic


        # --- If the NEW Rule was NOT met or KG results were empty, fall back to standard selection logic ---
        # This 'else' block covers:
        # 1. Not a symptom query (is_symptom_query is False)
        # 2. Symptom query, but s_kg is <= high_kg_only_threshold
        # 3. Symptom query, s_kg > high_kg_only_threshold, BUT kg_results was empty (unlikely, but safe)

        if not selected_context: # Only execute standard logic if no context was selected by the high KG rule
            logger.debug("High KG Confidence Rule NOT applied or resulted in empty context. Applying standard selection logic.")
            # Decide based on query type and standard thresholds
            if is_symptom_query:
                logger.debug("Processing symptom query (standard logic).")
                # KG is selected if its standard threshold is met
                if s_kg >= kg_threshold:
                    selected_context["kg"] = kg_results
                    logger.debug("KG meets standard threshold (%.4f >= %.4f). KG selected.", s_kg, kg_threshold)
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


    def generate_initial_answer(self, query: str, selected_context: Optional[Dict[str, Any]]) -> str:
        logger.info("🧠 Initial Answer Generation Initiated")
        # Cache key includes query and a hash of the context structure/content
        cache_key = {"type": "initial_answer", "query": query, "context_hash": abs(hash(json.dumps(selected_context, sort_keys=True)))}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Initial answer from cache.")
             return cached

        base_prompt_instructions = "You are a medical assistant." # Keep simple
        context_info_for_prompt = ""
        context_type_description = ""
        prompt_for_initial_answer = "" # Initialize here

        if selected_context is None or not selected_context:
            logger.info("🧠 Initial Answer Generation: No external context available.")
            # --- MODIFICATION START: Corrected string formatting ---
            context_info_for_prompt = "" # No external context provided
            context_type_description = (
                "You have not been provided with any specific external medical knowledge or document snippets for this query.\n"
                "Therefore, generate only a minimal placeholder answer that indicates lack of specific information.\n"
                "Do NOT attempt to answer the user query using your general knowledge in this step.\n"
                "Do NOT mention external documents or knowledge graphs.\n"
            )
           prompt_for_initial_answer = (
    f"{base_prompt_instructions.strip()}.\n"
    f"{context_type_description.strip()}.\n\n"
    f"User Query: \"{query}\"\n\n"
    "Minimal Placeholder Answer:\n"
)
            # --- MODIFICATION END --- #
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
                     confidence = diag_data.get("confidence", 0)
                     if confidence > THRESHOLDS.get("kg_context_selection", 0.6):
                          kg_info_str += f"- Identified Potential Condition: {disease_name} (KG Confidence: {confidence:.2f})\n"
                     elif confidence > THRESHOLDS.get("disease_matching", 0.5):
                          kg_info_str += f"- Potential Condition: {disease_name} (KG Confidence: {confidence:.2f})\n"
                     else:
                          kg_info_str += f"- Possible Condition based on limited info: {disease_name} (KG Confidence: {confidence:.2f})\n"

                     if diag_data.get('symptoms_list'):
                         kg_info_str += f"- Reported Symptoms: {', '.join(diag_data.get('symptoms_list', []))}\n"
                         if kg_data.get('kg_matched_symptoms'):
                              kg_info_str += f"- Matching KG Symptoms for this condition: {', '.join(kg_data.get('kg_matched_symptoms', []))}\n"
                         else:
                              kg_info_str += "- No matching symptoms found in KG for this condition from your input.\n"

                other_kg_content = kg_data.get("kg_content_other")
                if other_kg_content and other_kg_content.strip() and "Medical Knowledge Graph did not find" not in other_kg_content:
                      kg_info_str += "\n" + other_kg_content

                if len(kg_info_str.splitlines()) > 1 or kg_info_str.strip() != "Knowledge Graph Information:":
                     context_parts_for_prompt.append(kg_info_str)
                     logger.debug("Added KG context to prompt.")
                else:
                     logger.debug("KG context was empty or only header, not added to prompt.")


            if "rag" in selected_context:
                rag_chunks = selected_context.get("rag", [])
                if rag_chunks:
                    rag_info_str = "Relevant Passages from Documents:\n---\n" + "\n---\n".join(rag_chunks) + "\n---"
                    context_parts_for_prompt.append(rag_info_str)
                    logger.debug(f"Added {len(rag_chunks)} RAG chunks to prompt.")
                else:
                     logger.debug("RAG context was empty, not added to prompt.")


            if context_parts_for_prompt:
                 context_info_for_prompt = "\n\n".join(context_parts_for_prompt)
                 if "kg" in selected_context and "rag" in selected_context:
                      context_type_description = "Based on the following structured information from a medical knowledge graph and relevant passages from medical documents, synthesize a comprehensive answer."
                 elif "kg" in selected_context:
                      context_type_description = "Based on the following information from a medical knowledge graph, answer the user query. Only use the information provided here where possible. Do not refer to external documents unless absolutely necessary for general medical facts."
                 elif "rag" in selected_context:
                      context_type_description = "Based on the following relevant passages from medical documents, answer the user query. Only use the information provided here where possible. Do not refer to a knowledge graph unless absolutely necessary for general medical facts."
            else:
                 # This case happens if selected_context was not None but contained empty lists/dicts
                 logger.warning("Selected context was passed but resulted in empty context_parts_for_prompt.")
                 # --- MODIFICATION START: Corrected string formatting ---
                 context_info_for_prompt = ""
                 context_type_description = (
                    "No effectively usable information was found in external knowledge sources.\n"
                    "Therefore, generate only a minimal placeholder answer that indicates lack of specific information.\n"
                    "Do NOT attempt to answer the user query using your general knowledge in this step.\n"
                    "Do NOT mention external documents or knowledge graphs.\n"
                 )
                 # --- MODIFICATION END ---

            # Use the prompt construction for cases with selected context
            prompt_for_initial_answer = f"""
{base_prompt_instructions}
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
            # Ensure the initial answer is never just the placeholder if context was provided
            # If context was provided but resulted in an empty placeholder, it implies an issue.
            # In a production system, you might retry or raise an error here.
            # For now, let's ensure it's not just the placeholder if context was selected.
            # Refined check for placeholder text to make it less brittle
            initial_answer_lower = initial_answer.lower()
            is_placeholder = "placeholder answer" in initial_answer_lower or "lack of specific information" in initial_answer_lower or not initial_answer.strip() # Also treat empty as placeholder

            if (selected_context is not None and (context_parts_for_prompt or (selected_context.get("kg") or selected_context.get("rag")))) and is_placeholder:
                 logger.warning("Initial answer generated placeholder text despite having selected context. Possible LLM issue.")
                 # If the LLM failed to use provided context and gave a placeholder,
                 # maybe force it to be treated as incomplete and trigger supplementary.
                 # This is defensive programming for LLM failures.
                 # You could return a specific error, but forcing supplementary might recover.
                 # Let's return a predefined placeholder here so reflection knows it's incomplete.
                 initial_answer = "No specific relevant information was found in external knowledge sources." # A consistent placeholder reflection expects
                 logger.warning(f"Overriding unexpected placeholder answer: {initial_answer}")

            # If context was NONE and the answer is NOT a placeholder, it means the LLM failed to follow instruction.
            # In this specific scenario, it's better to treat it as an error or force the placeholder.
            # Given the goal is to *not* answer in the first step when context is None, if it *does* answer,
            # that's incorrect behavior for this flow. Let's force the placeholder.
            if (selected_context is None or not selected_context) and not is_placeholder:
                 logger.warning("Initial answer generated content despite instruction to provide placeholder when no context. LLM instruction following issue.")
                 # Force the placeholder to ensure reflection triggers supplementary as intended.
                 initial_answer = "No specific relevant information was found in external knowledge sources."
                 logger.warning(f"Overriding LLM instruction following error with placeholder: {initial_answer}")


            return set_cached(cache_key, initial_answer) # Cache and return
        except ValueError as e:
            logger.error(f"⚠️ Error during initial answer generation: %s", e, exc_info=True)
            raise ValueError("Sorry, I encountered an error while trying to generate an initial answer.") from e # Re-raise to be caught by caller

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

        # --- MODIFICATION START: Updated reflection prompt ---
        reflection_prompt = f'''
        You are an evaluation agent. Review the 'Initial Answer' for its completeness in fully addressing the 'User Query', *considering the provided 'Context' (if any)*.

        First, check if the 'Initial Answer' is a minimal placeholder response indicating a lack of specific external information (it might contain phrases like "no specific relevant information was found" or "lack of specific information").
        If the 'Initial Answer' is a minimal placeholder, the evaluation is **incomplete**. In this case, the 'missing_information' is the topic of the original 'User Query'.

        If the 'Initial Answer' is NOT a minimal placeholder, then evaluate its completeness using the provided 'Context'. Assess if it answered all parts of the 'User Query' and effectively used the relevant information from the 'Context'. If incomplete, identify *exactly* what specific information is missing or incomplete *from the perspective of the User Query*.

        Return ONLY a JSON object: {{"evaluation": "complete" or "incomplete", "missing_information": "Description of what is missing or empty string if complete"}}
        User Query: "{query}"
        Context:
        {context_for_reflection_prompt}
        Initial Answer:
        "{initial_answer}"
        '''
        # --- MODIFICATION END ---

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
                         logger.warning("🔍 Reflection Result: Incomplete. Missing Info: %s...", missing_info_description[:100])
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
                  if diag_data and diag_data.get("disease_name"):
                       kg_info_str += f"  Potential Condition: {diag_data['disease_name']} (Confidence: {diag_data.get('confidence', 0):.2f})\n"
                  if kg_data.get("kg_matched_symptoms"):
                       kg_info_str += f"  Matched Symptoms: {', '.join(kg_data['kg_matched_symptoms'])}\n"
                  if kg_data.get("kg_treatments"):
                       kg_info_str += f"  Treatments: {', '.join(kg_data['kg_treatments'])}\n"
                  # Add other KG content if significant
                  other_kg_content = kg_data.get("kg_content_other")
                  if other_kg_content and other_kg_content.strip() and "Medical Knowledge Graph did not find" not in other_kg_content:
                      # Limit other content length for reflection prompt
                      kg_info_str += "\n" + other_kg_content[:300] + ("..." if len(other_kg_content) > 300 else "")
                  context_str_parts.append(kg_info_str)

             if "rag" in selected_context:
                  rag_chunks = selected_context.get("rag", [])
                  if rag_chunks:
                       # Limit RAG chunks for reflection prompt brevity
                       rag_info_str = "Relevant Passages:\n---\n" + "\n---\n".join(rag_chunks[:3]) + "\n---" # Use top 3 chunks
                       context_str_parts.append(rag_info_str)

        if context_str_parts:
             return "\n\n".join(context_str_parts)
        return "None"


    def get_supplementary_answer(self, query: str, missing_info_description: str) -> str:
        logger.info(f"🌐 External Agent (Gap Filling) Initiated. Missing Info: {missing_info_description[:100]}...")
        # Cache key based on the hash of the combination
        cache_key = {"type": "supplementary_answer", "missing_info_hash": abs(hash(missing_info_description)), "query_context_hash": abs(hash(query))}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Supplementary answer from cache.")
             return cached

        # Fallback if LLM is not available
        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform supplementary generation.")
             return "\n\n-- Additional Information --\nSupplementary information could not be generated because the AI model is unavailable."

        # Refined prompt for supplementary info
        supplementary_prompt = f'''
        You are a medical information agent designed to provide *only* specific missing details.
        Based on the user's original medical query and a description of information missing from a previous answer, provide *only* the missing information.
        Do NOT restate the original query or the information already provided in the previous answer.
        Focus precisely on the gap described.
        **You MUST include evidence or source attribution for any medical claims.** Provide links (URLs), names of reliable sources [Source Name], or indicate if it's general knowledge. Use markdown for formatting.
        Original User Query (for context to understand the scope): "{query}"
        Information Missing from Previous Answer: "{missing_info_description}"
        Provide ONLY the supplementary information addressing the missing part, including evidence/sources:
        '''
        try:
            # Generate the supplementary answer
            supplementary_answer = self.local_generate(supplementary_prompt, max_tokens=750)

            # Post-process to ensure it doesn't start with conversational filler
            if supplementary_answer and supplementary_answer.strip().lower().startswith(("based on", "here is", "the missing", "it seems")):
                 # Simple heuristic to remove potential filler
                 lines = supplementary_answer.strip().splitlines()
                 if lines:
                     # If the first line is short and looks like filler, skip it
                     if len(lines[0].split()) < 15 and any(word in lines[0].lower() for word in ["based on", "here is", "the missing", "it seems"]):
                          supplementary_answer = "\n".join(lines[1:]).strip()
                     else:
                          supplementary_answer = "\n".join(lines).strip() # Keep as is if not clear filler

            # Prefix to make it clear this is additional info
            formatted_supplementary_answer = "\n\n-- Additional Information --\n" + supplementary_answer.strip()
            if not supplementary_answer.strip():
                 formatted_supplementary_answer += "The AI could not find specific additional information."


            logger.info("🌐 Supplementary Answer Generated successfully.")
            return set_cached(cache_key, formatted_supplementary_answer) # Cache and return
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"⚠️ Error during supplementary answer generation: {e}", exc_info=True)
            return f"\n\n-- Additional Information --\nSorry, an error occurred while trying to find additional information about: '{missing_info_description[:50]}...'"

    def collate_answers(self, initial_answer: str, supplementary_answer: str) -> str:
        logger.info("✨ Final Answer Collation Initiated")
        # Cache key based on hash of inputs
        cache_key = {"type": "final_collation", "initial_answer_hash": abs(hash(initial_answer)), "supplementary_answer_hash": abs(hash(supplementary_answer))}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Final collation from cache.")
             return cached

        # Fallback if LLM is not available - just concatenate with a separator
        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform final answer collation.")
             return f"{initial_answer}\n\n{supplementary_answer}"

        # Check if supplementary answer is just the error/no-info placeholder
        if "-- Additional Information --\n" in supplementary_answer and ("The AI could not find specific additional information" in supplementary_answer or "an error occurred while trying to find additional information" in supplementary_answer):
             logger.debug("Supplementary answer appears to be an error or 'no info' placeholder. Returning initial answer with placeholder appended.")
             return initial_answer.strip() + "\n\n" + supplementary_answer.strip()


        collation_prompt = f'''
        You are a medical communicator tasked with combining information into a single, coherent response.
        Combine the following two parts:
        1. An 'Initial Answer' to a medical query.
        2. 'Supplementary Information' that addresses gaps in the initial answer.

        Create a single, fluent, and easy-to-understand final answer. Ensure a natural flow.
        Remove any redundancy between the two parts.
        Preserve all factual medical information and any source attributions or links provided in either part.
        Format the final response clearly using markdown (e.g., headings, lists) if appropriate.
        Do NOT include the medical disclaimer in the answer text itself; that will be added separately.

        Initial Answer Part:
        "{initial_answer}"

        Supplementary Information Part:
        "{supplementary_answer}"

        Provide ONLY the combined, final answer content:
        '''
        try:
            # Generate the combined answer content
            combined_answer_content = self.local_generate(collation_prompt, max_tokens=1500) # Allow more tokens for combined answer

            logger.info("✨ Final Answer Collated successfully.")
            return set_cached(cache_key, combined_answer_content) # Cache and return
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"⚠️ Error during final answer collation: {e}", exc_info=True)
            # If collation fails, provide a combined answer with an error notice
            error_message = f"\n\n-- Collation Error --\nAn error occurred while finalizing the answer ({e}). The information below is the initial answer followed by supplementary information, presented uncollated.\n\n"
            return initial_answer.strip() + error_message + supplementary_answer.strip()


    def reset_conversation(self):
        """Resets the internal state of the chatbot instance (currently minimal)."""
        logger.info("🔄 Resetting chatbot internal state.")
        # Clear any instance variables that hold conversation context if they were added
        self.followup_context = {"round": 0} # Example, currently unused


    # The Main Orchestrator Function
    # Returns (response_text, ui_action, ui_payload)
    # ui_action: "display_final_answer", "show_symptom_ui", "none"
    # ui_payload: Depends on ui_action, e.g., symptom options for "show_symptom_ui"
    def process_user_query(self,
                           user_query: str,
                           user_type: str = "User / Family",
                           confirmed_symptoms: Optional[List[str]] = None,
                           original_query_if_followup: Optional[str] = None
                           ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        logger.info("--- Processing User Query: '%s' ---", user_query[:100])
        logger.info("User Type: %s, Confirmed Symptoms Received: %s", user_type, len(confirmed_symptoms) if confirmed_symptoms is not None else 'None')

        processed_query = user_query # Query text used for retrieval and generation
        current_symptoms_for_retrieval: List[str] = [] # Symptoms used for KG and symptom UI check
        is_symptom_query = False # Flag for symptom path
        medical_check_ok = False # Assume false initially

        # --- Step 1: Handle Symptom Confirmation Rerun Logic ---
        if confirmed_symptoms is not None:
             logger.info("--- Step 1: Handling Symptom Confirmation Rerun ---")
             # In a rerun triggered by symptom confirmation, the 'user_query' param holds the original query text.
             processed_query = user_query
             if not processed_query or not isinstance(processed_query, str):
                  logger.error("Original query text (user_query param) is invalid during symptom rerun.")
                  return "An error occurred during symptom confirmation processing.", "display_final_answer", None

             is_symptom_query = True # Assume symptom path for UI follow-up
             # Use the confirmed symptoms as the list for KG in this rerun, ensure lowercase and valid
             current_symptoms_for_retrieval = [s.strip().lower() for s in confirmed_symptoms if isinstance(s, str) and s.strip()]
             logger.info("Reprocessing '%s...' with confirmed symptoms: %s", processed_query[:50], current_symptoms_for_retrieval)
             medical_check_ok = True # Assume valid as we were in a medical flow
             # Skip initial medical check and symptom extraction


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
             # extracted_symptoms will be lowercase list
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
             if is_symptom_query and current_symptoms_for_retrieval:
                  logger.info("Triggering KG Pipeline with %d symptoms.", len(current_symptoms_for_retrieval))
                  # Pass the *current* list of symptoms (initial extracted OR confirmed UI list)
                  kg_results = self.knowledge_graph_agent(processed_query, current_symptoms_for_retrieval)
                  s_kg = kg_results.get("top_disease_confidence", 0.0)
                  logger.info("KG Pipeline finished. S_KG: %.4f", s_kg)
             elif is_symptom_query and not current_symptoms_for_retrieval:
                  logger.warning("Query identified as symptom-related but no symptoms extracted/confirmed. Skipping KG.")
             else:
                  logger.info("Query not identified as symptom-related. Skipping KG.")


             # RAG Retrieval (for all medical queries if VDB and embeddings are available)
             if self.vectordb is not None and self.embedding_model is not None:
                 logger.info("Triggering RAG Pipeline.")
                 rag_chunks, s_rag = self.retrieve_rag_context(processed_query)
                 logger.info("RAG Pipeline finished. S_RAG: %.4f", s_rag)
             else:
                 logger.warning("Vector DB or Embedding Model not available. Skipping RAG.")


             # --- Step 5: Symptom Confirmation UI Decision Point (only for initial query path) ---
             # Check if this is the *initial* query processing (not a rerun after UI) AND it's a symptom query
             # AND KG successfully found *any* diseases (confidence > 0.0) AND KG confidence is below UI trigger threshold
             if confirmed_symptoms is None and is_symptom_query:
                  logger.info("--- Step 5: Symptom Confirmation UI Check ---")
                  ui_trigger_threshold = THRESHOLDS.get("disease_symptom_followup_threshold", 0.8)
                  kg_found_diseases = len(kg_results.get("identified_diseases_data", [])) > 0

                  # Condition to show UI:
                  # 1. It's a symptom query based on initial analysis.
                  # 2. KG successfully identified at least one potential disease (confidence > 0).
                  # 3. The top KG disease confidence is below the specific UI trigger threshold.
                  # 4. There are *new* symptoms suggested by the KG that weren't in the user's original input.
                  if kg_found_diseases and s_kg > 0.0 and s_kg < ui_trigger_threshold:
                       logger.info(f"KG confidence (%.4f) below UI trigger threshold (%.4f) and diseases found (%d). Checking for new symptoms for UI.", s_kg, ui_trigger_threshold, len(kg_results['identified_diseases_data']))

                       symptom_options_for_ui: Dict[str, List[str]] = {}
                       initial_symptoms_lower = set([s.strip().lower() for s in current_symptoms_for_retrieval if isinstance(s, str)])

                       if kg_results.get("identified_diseases_data"):
                            # Use a few top diseases to suggest symptoms
                            top_diseases_data = kg_results["identified_diseases_data"][:3]
                            for disease_data in top_diseases_data:
                                 disease_name = disease_data.get("Disease", "Unknown")
                                 # Get *all* symptoms associated with this disease in the KG
                                 all_disease_symptoms_kg = disease_data.get("AllDiseaseSymptomsKG", [])
                                 # Filter out symptoms the user already gave (case-insensitive) and ensure uniqueness
                                 filtered_symptoms = sorted(list(set([s for s in all_disease_symptoms_kg if isinstance(s, str) and s.strip() and s.strip().lower() not in initial_symptoms_lower])))

                                 if filtered_symptoms:
                                      # Store original casing for display in UI
                                      symptom_options_for_ui[disease_name] = filtered_symptoms


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
                       logger.info("KG confidence below threshold, but no NEW symptoms were found to suggest for UI. Proceeding to answer generation based on available info.")


             # --- Step 6: Context Selection Logic ---
             # This step runs if no symptom UI was shown (either not symptom query, high confidence, or rerun from UI)
             logger.info("--- Step 6: Context Selection Logic ---")
             # Pass the results from Step 4
             selected_context = self.select_context(kg_results, s_kg, rag_chunks, s_rag, is_symptom_query)

             # --- Determine Pathway Information ---
             # Determine which external sources were SELECTED
             context_sources_used: List[str] = []
             if selected_context is not None:
                 if "kg" in selected_context:
                     context_sources_used.append("Knowledge Graph")
                 if "rag" in selected_context:
                     context_sources_used.append("Documents (RAG)")

             initial_pathway_parts = context_sources_used if context_sources_used else ["LLM (General Knowledge for Initial Phrasing)"] # Indicate LLM is used even if no external context


             # --- Step 7: Initial Answer Generation ---
             logger.info("--- Step 7: Initial Answer Generation ---")
             try:
                 initial_answer = self.generate_initial_answer(processed_query, selected_context)
             except ValueError as e: # Catch specific LLM generation errors
                 logger.error(f"Initial answer generation failed: {e}")
                 # Return error message directly, bypass reflection etc.
                 # Indicate the sources attempted for initial generation
                 pathway_info = ", ".join(initial_pathway_parts) + " (Initial Generation Failed)"
                 error_response = f"Sorry, I could not generate an initial answer due to an error: {e}\n\n<span style='font-size: 0.8em; color: grey;'>*Sources attempted for this response: {pathway_info}*</span>"
                 return error_response, "display_final_answer", None


             # --- Step 8: Reflection and Evaluation ---
             logger.info("--- Step 8: Reflection and Evaluation ---")
             reflection_failed = False
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


             # --- Step 9: Conditional External Agent & Collation ---
             # This combines steps 9, 10, 11 from the previous numbering
             final_answer_content = initial_answer # Start with the initial answer
             supplementary_step_triggered = False

             if evaluation_result == 'incomplete':
                  logger.warning("--- Step 9: Reflection Incomplete. Triggering supplementary pipeline. ---")
                  logger.debug(f"Missing Info Description: {missing_info_description[:100]}...")
                  supplementary_step_triggered = True # Flag that supplementary step is happening

                  # --- Step 10: External Agent (Gap Filling) ---
                  logger.info("--- Step 10: External Agent (Gap Filling) ---")
                  supplementary_answer = "" # Initialize empty
                  try:
                       supplementary_answer = self.get_supplementary_answer(processed_query, missing_info_description)
                  except ValueError as e:
                       logger.error(f"Supplementary answer generation failed: {e}")
                       # Ensure supplementary_answer is a string indicating failure
                       supplementary_answer = f"\n\n-- Additional Information --\nSorry, I could not find supplementary information due to an error: {e}"
                       logger.warning("Proceeding to collation with supplementary error message.")


                  # --- Step 11: Final Answer Collation ---
                  logger.info("--- Step 11: Final Answer Collation ---")
                  # Check if supplementary answer is meaningful (not just the initial header or error)
                  # The get_supplementary_answer function is designed to return a string starting with "-- Additional Information --"
                  # even if empty or failed.
                  if supplementary_answer.strip().startswith("-- Additional Information --") and len(supplementary_answer.strip()) > len("-- Additional Information --"):
                       # Supplement was generated (even if it's an error message indicating lookup failed)
                       try:
                            final_answer_content = self.collate_answers(initial_answer, supplementary_answer)
                       except ValueError as e:
                            logger.error(f"Final answer collation failed: {e}")
                            # If collation fails, just combine with a separator and add error message
                            final_answer_content = f"{initial_answer.strip()}\n\n-- Collation Failed --\nAn error occurred while finalizing the answer ({e}).\n\n{supplementary_answer.strip()}"
                            logger.warning("Proceeding with simple answer concatenation due to collation failure.")
                  else:
                      # Supplementary answer was empty or just the header, implying no additional info was found/generated successfully.
                      logger.debug("Supplementary answer was empty or just header. Skipping collation.")
                      final_answer_content = initial_answer.strip() # Use initial answer as final content


             # --- Final Assembly: Determine Final Pathway Info, Add Disclaimer and Pathway Note ---

             final_pathway_parts = context_sources_used # Start with the external sources used for initial context

             # Add 'LLM' to the pathway if the supplementary step was triggered OR if no external context was used at all
             # (indicating the initial phrasing relied solely on the LLM's general knowledge)
             if supplementary_step_triggered or not context_sources_used:
                  final_pathway_parts.append("LLM (General Knowledge)") # Indicate LLM was used beyond just context phrasing if it filled gaps or was the sole source

             # Ensure uniqueness and order
             final_pathway_parts = sorted(list(set(final_pathway_parts)))

             if not final_pathway_parts: # Should not happen if logic is correct, but defensive
                 pathway_info = "Unknown Pathway"
             else:
                 pathway_info = ", ".join(final_pathway_parts)


             # Add the medical disclaimer consistently at the end of the main content
             disclaimer = "\n\nIMPORTANT MEDICAL DISCLAIMER: This information is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read here."

             # Add the pathway information using HTML for smaller, grey text below the disclaimer
             pathway_note = f"<span style='font-size: 0.8em; color: grey;'>*Sources used for this response: {pathway_info}*</span>"

             # Combine content, disclaimer, and pathway note
             final_response_text = f"{final_answer_content.strip()}{disclaimer}\n\n{pathway_note}"

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
    form_key_base = f"symptom_confirmation_form_{abs(hash(original_query))}"
    form_key = f"{form_key_base}_{st.session_state.get('form_timestamp', datetime.now().timestamp())}"

    # Define a session state key specific to this form instance to store selected symptoms
    # We will store lowercase symptoms in this set
    local_confirmed_symptoms_state_key = f'{form_key}_selected_symptoms_local_set'

    # Initialize the local set if it's not in session state for this form key.
    # It should be initialized as an empty set when this form is first rendered in a session/rerun cycle.
    if local_confirmed_symptoms_state_key not in st.session_state:
        logger.debug(f"Initializing local symptom set state variable: {local_confirmed_symptoms_state_key}")
        st.session_state[local_confirmed_symptoms_state_key] = set()
    else:
        logger.debug(f"Using existing local symptom set for form {form_key} with {len(st.session_state.get(local_confirmed_symptoms_state_key, set()))} items.")


    all_unique_symptoms_original_case = sorted(list(set(
        s.strip() for symptoms_list in symptom_options.values()
        if isinstance(symptoms_list, list) for s in symptoms_list
        if isinstance(s, str) and s.strip()
    )))
    logger.debug(f"Total unique suggested symptoms for UI: {len(all_unique_symptoms_original_case)}")


    with st.form(form_key):
        st.markdown("Please check all symptoms that apply to you from the list below:")
        if not all_unique_symptoms_original_case:
            st.info("No specific additional symptoms were found for potential conditions to suggest. Use the box below to add symptoms you are experiencing.")
        else:
            # Display symptoms in 4 columns
            cols = st.columns(4)
            for i, symptom in enumerate(all_unique_symptoms_original_case):
                col = cols[i % 4]
                # Checkbox key unique to the form and symptom
                checkbox_key = f"{form_key}_checkbox_{symptom}"
                symptom_lower = symptom.strip().lower()

                # Check the initial state from the local set (case-insensitive check)
                # Use .get with default set for safety
                initial_state = symptom_lower in st.session_state.get(local_confirmed_symptoms_state_key, set())

                # Display the checkbox with original capitalization, but store/check lowercase
                is_checked = col.checkbox(symptom, key=checkbox_key, value=initial_state)

                # Update the local symptom set in session state based on the checkbox interaction
                # We store the lowercase version in the set for consistent lookup
                if is_checked:
                     st.session_state[local_confirmed_symptoms_state_key].add(symptom_lower)
                else:
                     # Only discard if it was previously in the set and is now unchecked
                     st.session_state[local_confirmed_symptoms_state_key].discard(symptom_lower)


        st.markdown("**Other Symptoms (if any):**")
        # Use a unique key for the text input as well, tied to the form
        other_symptoms_text_key = f"{form_key}_other_symptoms_input"
        # Get current value from state if it exists, otherwise empty string
        other_symptoms_initial_value = st.session_state.get(other_symptoms_text_key, "")
        other_symptoms_text = st.text_input(
            "Enter additional symptoms here (comma-separated)",
            key=other_symptoms_text_key,
            value=other_symptoms_initial_value # Use state value
        )

        # Update the local symptom set with any symptoms typed into the text input
        # This handles cases where the user types, submits, then unchecks a box and resubmits -
        # the typed symptoms should persist in the local set.
        if other_symptoms_text:
             # Add typed symptoms to the local set as lowercase
             typed_symptoms_lower = [s.strip().lower() for s in other_symptoms_text.split(',') if s.strip()]
             if typed_symptoms_lower:
                 logger.debug(f"Adding other symptoms from input to local set: {typed_symptoms_lower}")
                 st.session_state[local_confirmed_symptoms_state_key].update(typed_symptoms_lower)
             # Store the raw text input value in state to preserve it across reruns if needed
             st.session_state[other_symptoms_text_key] = other_symptoms_text
        elif other_symptoms_text_key in st.session_state:
             # If the text input is cleared, remove it from state and remove any symptoms it might have added from the local set
             old_value = st.session_state[other_symptoms_text_key]
             old_typed_symptoms_lower = [s.strip().lower() for s in old_value.split(',') if s.strip()]
             if old_typed_symptoms_lower:
                  logger.debug(f"Removing previously typed symptoms from local set: {old_typed_symptoms_lower}")
                  st.session_state[local_confirmed_symptoms_state_key].difference_update(old_typed_symptoms_lower)
             del st.session_state[other_symptoms_text_key]
             logger.debug(f"Cleared {other_symptoms_text_key} from state.")


        # --- Form Submission ---
        submit_button = st.form_submit_button("Confirm and Continue")
        if submit_button:
            logger.info(f"Symptom confirmation form submitted for query: '{original_query[:50]}...'.")

            # IMPORTANT: The local state variable (st.session_state[local_confirmed_symptoms_state_key])
            # now contains the *final* set of confirmed symptoms (checkboxes + text input), in lowercase.
            # We need to pass this set back to the main processing logic.
            # Store the final confirmed symptoms (as a list, lowercase) in the session state variable
            # that the main loop watches for form submissions.
            final_confirmed_symptoms_list = sorted(list(st.session_state.get(local_confirmed_symptoms_state_key, set())))
            st.session_state.confirmed_symptoms_from_ui = final_confirmed_symptoms_list
            logger.info(f"Set st.session_state.confirmed_symptoms_from_ui with {len(st.session_state.confirmed_symptoms_from_ui)} symptoms.")

            # Reset the UI state back to 'input' so the chat input appears on the next rerun, hiding the form.
            st.session_state.ui_state = {"step": "input", "payload": None}
            logger.debug("Set ui_state back to 'input' after form submission.")

            # Set a new timestamp for the next potential form display to ensure a fresh form key.
            st.session_state.form_timestamp = datetime.now().timestamp()
            logger.debug(f"Reset form_timestamp.")

            # Clear the local symptom set explicitely for cleanliness and to prevent persistence across forms.
            if local_confirmed_symptoms_state_key in st.session_state:
                 del st.session_state[local_confirmed_symptoms_state_key]
                 logger.debug(f"Cleared {local_confirmed_symptoms_state_key} from state.")

            # Explicitly clear the text input value in session state so it's empty next time
            # This prevents typed symptoms from reappearing if the form is shown again for a new query.
            other_symptoms_text_key = f"{form_key}_other_symptoms_input" # Re-create key based on current form_key
            if other_symptoms_text_key in st.session_state:
                 del st.session_state[other_symptoms_text_key]
                 logger.debug(f"Cleared {other_symptoms_text_key} from state.")

            # The rerun is triggered automatically by st.form_submit_button


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
    if 'chatbot' not in st.session_state:
        logger.info("Initializing chatbot instance and performing full backend setup.")
        try:
             with st.spinner("Initializing chat assistant..."):
                  st.session_state.chatbot = DocumentChatBot() # Create the chatbot instance
                  # Perform the initialization that sets up LLM, VDB, KG connection
                  success, init_message = st.session_state.chatbot.initialize_qa_chain()
                  # Store the final status
                  st.session_state.init_status = (success, init_message)
             logger.info(f"Chatbot initialization complete. Status: {st.session_state.init_status}")
        except Exception as e:
             logger.critical(f"CRITICAL UNCAUGHT ERROR DURING INITIALIZATION: {e}", exc_info=True)
             st.session_state.init_status = (False, f"Critical initialization error: {e}") # Store failure status
             st.session_state.chatbot = None # Ensure chatbot is None if init fails


    # --- Retrieve Initialization Status for Subsequent Reruns ---
    # This happens on every rerun to know if the bot is usable.
    init_success, init_msg = st.session_state.get('init_status', (False, "Initialization status not set."))
    logger.debug(f"Current init status (retrieved from state): Success={init_success}, Msg='{init_msg}'")

    # Determine if interaction is enabled based on init success and chatbot instance existing
    # This guards against trying to use the bot if init failed or the instance wasn't created
    is_interaction_enabled = init_success and st.session_state.get('chatbot') is not None


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
         st.session_state.processing_input_payload = None # Dict like {"query": ..., "confirmed_symptoms": ..., "original_query_context": ...}

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


    # --- Sidebar ---
    user_type = st.sidebar.selectbox(
        "Who is asking?",
        ["User / Family", "Physician"],
        index=0,
        key="user_type_select" # Add key for uniqueness
    )
    logger.debug(f"User type selected: {user_type}")

    st.sidebar.info("DxAI-Agent helps answer medical questions using our medical knowledge base.")

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
        # --- Examples Section (Removed per previous request) ---
        # st.subheader("Try these examples")
        # ... (rest of example button code) ...


        # --- Chat Messages Display ---
        # Display messages from state. Add feedback buttons to the last assistant message
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
                            b1, b2 = st.columns([0.05, 0.95]) # Small column for button, large for spacing

                            # Find the preceding user message for logging
                            user_msg_content = ""
                            for j in range(i - 1, -1, -1):
                                if st.session_state.messages[j][1] is True: # Check if it's a user message
                                    user_msg_content = st.session_state.messages[j][0]
                                    break # Found the last user message before this assistant message

                            with b1:
                                if st.button("👍", key=feedback_key_up):
                                     logger.info(f"Thumbs Up feedback for user query: '{user_msg_content[:50]}...'")
                                     # Call the global vote_message function
                                     vote_message(user_msg_content, msg_content, "thumbs_up", user_type)
                                     st.toast("Feedback recorded: Thumbs Up!")
                            with b2:
                                if st.button("👎", key=feedback_key_down):
                                    logger.info(f"Thumbs Down feedback for user query: '{user_msg_content[:50]}...'")
                                    # Call the global vote_message function
                                    vote_message(user_msg_content, msg_content, "thumbs_down", user_type)
                                    st.toast("Feedback recorded: Thumbs Down!")


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

                    # Set the input to be processed in the *next* rerun.
                    # This payload tells the processing block below *what* to process.
                    st.session_state.processing_input_payload = {
                        "query": user_query, # The new query text
                        "confirmed_symptoms": None, # No confirmed symptoms for an initial query
                        "original_query_context": None # No original query context for an initial query
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
                  "original_query_context": original_query_to_pass # Redundant but useful context flag for process_user_query internal logic
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

            # Ensure chatbot instance exists before calling its method
            if st.session_state.get('chatbot') is None:
                 logger.critical("Attempted to call process_user_query, but chatbot instance is None.")
                 st.session_state.messages.append(("Sorry, the chat assistant is not initialized properly.", False))
                 st.session_state.ui_state = {"step": "input", "payload": None} # Reset UI state
                 # Clean up symptom rerun state if it was active before this error
                 if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                 st.rerun()
                 return # Stop processing


            # Extract the input data from the payload
            # query_to_process will be the *new* user query for initial inputs,
            # and the *original* query text for symptom confirmation reruns.
            query_to_process = input_data.get("query", "")
            confirmed_symps = input_data.get("confirmed_symptoms") # Will be None for initial inputs
            original_query_context = input_data.get("original_query_context") # Will be None for initial inputs


            # Prevent processing if query_to_process is empty (e.g., from a bug)
            if not query_to_process:
                 logger.error("processing_input_payload contained empty query. Skipping processing.")
                 st.session_state.messages.append(("Sorry, I received an empty query for processing.", False))
                 st.session_state.ui_state = {"step": "input", "payload": None} # Reset UI state
                 # Clean up symptom rerun state if applicable
                 if original_query_context is not None and 'original_query_for_symptom_rerun' in st.session_state:
                     del st.session_state.original_query_for_symptom_rerun
                     logger.debug("Cleared original_query_for_symptom_rerun state due to empty query.")
                 st.rerun()
                 return # Stop processing


            with st.spinner("Thinking..."):
                 try:
                     # Call the chatbot's main processing function
                     response_text, ui_action, ui_payload = st.session_state.chatbot.process_user_query(
                          query_to_process, user_type,
                          confirmed_symptoms=confirmed_symps, # Pass confirmed symptoms if available (from form submit)
                          original_query_if_followup=original_query_context # Pass original query context if available (from form submit)
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


                     elif ui_action == "show_symptom_ui":
                          logger.info("UI Action: show_symptom_ui. Adding prompt message.")
                          # Add the prompt message returned by process_user_query
                          st.session_state.messages.append((response_text, False))
                          # Update UI state to show the symptom checklist on the next rerun
                          st.session_state.ui_state = {"step": "confirm_symptoms", "payload": ui_payload}
                          # Set a new timestamp for the form key to ensure a fresh form rendering
                          st.session_state.form_timestamp = datetime.now().timestamp()
                          logger.debug("UI state set to 'confirm_symptoms'.")

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

                     else:
                          logger.error("Unknown ui_action returned: %s. Defaulting to input state.", ui_action)
                          st.session_state.messages.append((f"An internal error occurred (Unknown UI action: {ui_action}).", False))
                          st.session_state.ui_state = {"step": "input", "payload": None}
                          # Clean up symptom rerun state on error
                          if confirmed_symps is not None and 'original_query_for_symptom_rerun' in st.session_state:
                              del st.session_state.original_query_for_symptom_rerun
                              logger.debug("Cleared original_query_for_symptom_rerun after unknown UI action error.")


                 except ValueError as e: # Catch ValueErrors raised by local_generate and its callers
                     logger.error(f"LLM/Processing Error during chatbot execution: {e}", exc_info=True)
                     # Add error message including pathway info attempt
                     error_pathway_info = "Error during AI processing"
                     # Attempt to derive pathway info based on available context if not already set
                     if 'pathway_info' in locals():
                         error_pathway_info = locals()['pathway_info'] # Use determined pathway if available

                     error_response = f"Sorry, an AI processing error occurred: {e}\n\n<span style='font-size: 0.8em; color: grey;'>*Sources attempted for this response: {error_pathway_info}*</span>"
                     st.session_state.messages.append((error_response, False))

                     # Reset UI state to input on error
                     st.session_state.ui_state = {"step": "input", "payload": None}
                     # Clean up symptom rerun state on error
                     if confirmed_symps is not None and 'original_query_for_symptom_rerun' in st.session_state:
                          del st.session_state.original_query_for_symptom_rerun
                          logger.debug("Cleared original_query_for_symptom_rerun after ValueError.")

                 except Exception as e: # Catch any other unexpected errors
                     logger.error(f"Unexpected Error during chatbot process_user_query execution: {e}", exc_info=True)
                     # Add error message including pathway info attempt
                     error_pathway_info = "Unexpected error during processing"
                     if 'pathway_info' in locals():
                         error_pathway_info = locals()['pathway_info'] # Use determined pathway if available

                     error_response = f"Sorry, an unexpected error occurred: {e}\n\n<span style='font-size: 0.8em; color: grey;'>*Sources attempted for this response: {error_pathway_info}*</span>"
                     st.session_state.messages.append((error_response, False))
                     # Reset UI state to input on error
                     st.session_state.ui_state = {"step": "input", "payload": None}
                     # Clean up symptom rerun state on error
                     if confirmed_symps is not None and 'original_query_for_symptom_rerun' in st.session_state:
                          del st.session_state.original_query_for_symptom_rerun
                          logger.debug("Cleared original_query_for_symptom_rerun after unexpected error.")


            # Always trigger a rerun after processing a payload.
            # This ensures the UI updates based on the new st.session_state.ui_state.
            logger.debug("Triggering rerun after processing_input_payload.")
            st.rerun()

        st.divider()
        # Ensure reset button is disabled if init failed
        if st.button("Reset Conversation", key="reset_conversation_button_main", disabled=not is_interaction_enabled):
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


            logger.debug("Triggering rerun after reset.")
            st.rerun()

        st.divider()
        st.subheader("🩺 Detailed Feedback")
        # Disable feedback form if init failed
        # Added unique key for feedback form
        with st.form(key="detailed_feedback_form", clear_on_submit=True):
            feedback_text = st.text_area(
                "Enter corrections, improvements, or comments here...", height=100, disabled=not is_interaction_enabled
            )
            submit_feedback_btn = st.form_submit_button("Submit Feedback", disabled=not is_interaction_enabled)
            if submit_feedback_btn and feedback_text:
                logger.info("Detailed feedback submitted.")
                # Call the global submit_feedback function
                submit_feedback(feedback_text, st.session_state.messages, user_type)
                st.success("Thank you for your feedback!")

    with tab2:
        st.markdown("""
        ## Medical Chat Assistant

        This is an experimental medical chat assistant designed to provide information based on a knowledge graph and provided documents.

        **How it Works:**
        1.  **Medical Check:** It first assesses if your query is related to health or medicine.
        2.  **Symptom Analysis:** If the query is medical, it attempts to identify reported symptoms. If potential conditions are found in the Knowledge Graph but with low confidence, it may ask you to confirm or add symptoms via a checklist.
        3.  **Information Retrieval:** It searches a Medical Knowledge Graph for related conditions, treatments, and remedies, and retrieves relevant passages from uploaded documents (RAG).
        4.  **Context Selection:** It selects the most confident and relevant information from the Knowledge Graph and documents to use as context.
        5.  **Initial Answer:** It generates an initial answer using the selected context and its general knowledge (LLM).
        6.  **Self-Reflection:** It evaluates its initial answer for completeness based on your original query.
        7.  **Information Gap Filling:** If the answer is incomplete, it attempts to find additional information to fill the gaps, seeking reliable sources or general medical knowledge.
        8.  **Final Answer:** It combines the initial and supplementary information into a coherent final response.
        9.  **Pathway Indication:** The response includes a small note indicating which primary sources (Knowledge Graph, Documents (RAG), and the LLM's general knowledge) were used to generate the answer.

        **Disclaimer:** This system is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read here.
        """)
    logger.debug("--- Streamlit App End of Rerun ---")


if __name__ == "__main__":
    # Set basic logging level before running the main app logic
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Pass arguments if needed, or call main directly
    main()
