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

GEMINI_API_KEY = "AIzaSyAifk9Gntw6eYfaZkLOsd9d1-TkfOR1el0" # Your actual Gemini Key
NEO4J_URI = "neo4j+s://1b47920f.databases.neo4j.io" # Your actual Neo4j URI
NEO4J_USER = "neo4j" # Your actual Neo4j User
NEO4J_PASSWORD = "eCqDfyhDcuGMLzbYfiqL6jsvjH3LIXr86xQGAEKmY8Y" # Your actual Neo4j Password# Ensure this is set in .env
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# Provide informative errors if required environment variables are missing
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    logger.error("GEMINI_API_KEY environment variable is not set or is the placeholder value.")
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD or NEO4J_URI == "YOUR_NEO4J_URI":
     logger.error("NEO4J environment variables (URI, USER, PASSWORD) are not fully set or are placeholder values.")


# Threshold settings
THRESHOLDS = {
    "symptom_extraction": 0.6,
    "disease_matching": 0.5, # Base threshold for KG to identify *a* disease
    "disease_symptom_followup_threshold": 0.8, # Below this confidence for a disease query, trigger symptom confirmation UI
    "kg_context_selection": 0.8, # Threshold for KG confidence to be included in context sent to LLM (for symptom queries)
    "rag_context_selection": 0.7, # Threshold for RAG confidence to be included in context sent to LLM (for both symptom and non-symptom queries)
    "medical_relevance": 0.6 # Threshold for medical relevance check
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.debug(f"Cache hit for key: {key_str[:50]}...")
        return CACHE[key_str]
    logger.debug(f"Cache miss for key: {key_str[:50]}...")
    return None

def set_cached(key, value):
    key_str = json.dumps(key, sort_keys=True)
    logger.debug(f"Setting cache for key: {key_str[:50]}...")
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

            sanitized_bot_msg = bot_message.split("IMPORTANT MEDICAL DISCLAIMER:", 1)[0].strip()
            sanitized_bot_msg = sanitized_bot_msg.split("## References:", 1)[0].strip()

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

            history_string = " || ".join([f"User: {u.replace('||', '')} | Bot: {b.replace('||', '').split('IMPORTANT MEDICAL DISCLAIMER:', 1)[0].strip()}" for u, b in conversation_history])

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
        self.chat_history: List[Tuple[str, str]] = []
        self.followup_context = {"round": 0}

        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        try:
            # Check for CUDA availability and set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing SentenceTransformer embeddings on device: {device}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-mpnet-base-v2',
                cache_folder='./cache',
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            # Test the embedding model
            try:
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
        self._init_kg_connection()
        logger.info("DocumentChatBot initialization finished.")

    def _init_kg_connection(self):
        logger.info("Attempting to connect to Neo4j...")
        # Check if credentials are set before attempting connection
        if not NEO4J_URI or NEO4J_URI == "YOUR_NEO4J_URI" or not NEO4J_USER or not NEO4J_PASSWORD:
             logger.error("Neo4j credentials missing or are placeholder values. Cannot connect.")
             self.kg_driver = None
             self.kg_connection_ok = False
             return

        try:
            # This line seems okay based on the error message you got
            self.kg_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, connection_timeout=5.0)
            # Corrected: Removed the 'timeout' parameter from verify_connectivity
            # The error message "Unexpected config keys: timeout" points to this specific key being the issue
            self.kg_driver.verify_connectivity() # <-- Removed timeout=2.0 here
            logger.info("Successfully connected to Neo4j.")
            self.kg_connection_ok = True
        except Exception as e:
            # This is where the error you saw was caught and logged
            logger.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}. KG features will be unavailable.")
            self.kg_driver = None
            self.kg_connection_ok = False
            
    def create_vectordb(self):
        logger.info("Creating vector database...")
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
            vectordb = FAISS.from_documents(splits, self.embedding_model)
            logger.info("FAISS vectorstore created.")
            return vectordb, "Vector database created successfully."
        except Exception as e:
            logger.error(f"Error creating FAISS vector database: {e}")
            return None, f"Failed to create vector database: {str(e)}"

    def initialize_qa_chain(self):
        logger.info("Initializing QA chain components (LLM, Vector DB)...")
        llm_init_message = "LLM initialization skipped."
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
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
                    test_response = self.llm.invoke("Hello, are you ready?")
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
             self.vectordb, vdb_message = self.create_vectordb()
             if self.vectordb is not None:
                  logger.info("Vector DB successfully created/loaded.")
             else:
                  logger.warning(f"Vector DB creation failed: {vdb_message}")


        status_parts = []
        if self.llm is not None: status_parts.append("LLM OK")
        else: status_parts.append("LLM Failed")
        if self.embedding_model is not None: status_parts.append("Embeddings OK")
        else: status_parts.append("Embeddings Failed")
        if self.vectordb is not None: status_parts.append("Vector DB OK")
        else: status_parts.append("Vector DB Failed")
        if self.kg_connection_ok: status_parts.append("KG OK")
        else: status_parts.append("KG Failed")

        overall_message = f"{', '.join(status_parts)}."
        overall_success = self.llm is not None # LLM is minimal requirement for any useful response

        logger.info(f"Initialization Result: Success={overall_success}, Message='{overall_message}'")
        return overall_success, overall_message

    def local_generate(self, prompt, max_tokens=500):
        logger.debug(f"Attempting LLM generation with prompt (first 100 chars): {prompt[:100]}...")
        # Ensure LLM is accessible, raise informative error if not
        if self.llm is None:
             logger.critical("LLM is not initialized. Generation failed.")
             raise ValueError("LLM is not initialized. Cannot generate response.")

        try:
            response = self.llm.invoke(prompt)
            logger.debug("LLM generation successful.")
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
             medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose"]
             result = (any(keyword in query.lower() for keyword in medical_keywords), "Fallback heuristic match (LLM unavailable)")
             logger.debug(f"Medical relevance fallback result: {result}")
             return set_cached(cache_key, result)

        medical_relevance_prompt = f'''
        Analyze the user query. Is it related to health, medical conditions, symptoms, treatments, medication, diagnostics, or any other medical or health science topic?
        Return ONLY a JSON object: {{"is_medical": true, "confidence": 0.0, "reasoning": "brief explanation"}}
        Query: "{query}"
        '''
        try:
            response = self.local_generate(medical_relevance_prompt, max_tokens=150)
            logger.debug(f"Medical relevance LLM raw response: {response}")
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    is_medical = data.get("is_medical", False)
                    confidence = data.get("confidence", 0.0)
                    reasoning = data.get("reasoning", "")
                    result = (is_medical and confidence >= THRESHOLDS.get("medical_relevance", 0.6), reasoning)
                    logger.info(f"Medical relevance check result: {result}")
                    return set_cached(cache_key, result)
                except json.JSONDecodeError:
                    logger.warning("Could not parse medical relevance JSON from LLM response.")
            else:
                logger.warning("No JSON found in medical relevance response.")

        except ValueError as e: # Catch ValueError from local_generate if LLM fails
             logger.error(f"Error during medical relevance LLM call: {e}")

        # Fallback if LLM call/parsing fails
        medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose"]
        result = (any(keyword in query.lower() for keyword in medical_keywords), "Fallback heuristic match (LLM failed)")
        logger.debug(f"Medical relevance final fallback result: {result}")
        return set_cached(cache_key, result)


    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        logger.debug(f"Attempting symptom extraction for query: {user_query}")
        cache_key = {"type": "symptom_extraction", "query": user_query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("Symptom extraction from cache.")
            return cached

        common_symptom_keywords = ["fever", "cough", "headache", "sore throat", "nausea", "dizziness", "chest pain", "shortness of breath", "fatigue", "body aches", "runny nose", "congestion", "chills", "sweats", "joint pain", "muscle aches", "rash", "swelling", "pain", "ache", "burning", "itching", "numbness", "tingling", "diarrhea", "vomiting", "difficulty breathing", "difficulty swallowing"]
        query_lower = user_query.lower()
        fallback_symptoms = [s.capitalize() for s in common_symptom_keywords if s in query_lower]

        # Fallback if LLM is not available
        if self.llm is None:
             logger.warning("LLM not initialized. Falling back to keyword symptom extraction.")
             result = (fallback_symptoms, 0.4)
             logger.info(f"Symptom extraction fallback result: {result}")
             return set_cached(cache_key, result)

        SYMPTOM_PROMPT = f'''
        Extract all medical symptoms mentioned in the following user query.
        For each symptom, assign a confidence score between 0.0 and 1.0.
        Return ONLY in this format: Extracted Symptoms: [{{"symptom": "symptom1", "confidence": 0.9}}, {{"symptom": "symptom2", "confidence": 0.8}}, ...]

        User Query: "{user_query}"
        '''
        llm_symptoms = []
        llm_avg_confidence = 0.0
        try:
            response = self.local_generate(SYMPTOM_PROMPT, max_tokens=500).strip()
            logger.debug(f"Symptom extraction LLM raw response: {response}")
            match = re.search(r"Extracted Symptoms:\s*(\[.*?\])", response, re.DOTALL)
            if match:
                try:
                    symptom_data = json.loads(match.group(1))
                    llm_symptoms_confident = [item["symptom"].strip() for item in symptom_data if item.get("confidence", 0) >= THRESHOLDS.get("symptom_extraction", 0.6)]
                    if symptom_data:
                        llm_avg_confidence = sum(item.get("confidence", 0) for item in symptom_data) / len(symptom_data)
                    llm_symptoms = llm_symptoms_confident
                    logger.debug(f"LLM extracted {len(llm_symptoms)} confident symptoms.")
                except json.JSONDecodeError:
                    logger.warning("Could not parse symptom JSON from LLM response")
            else:
                 logger.warning("Could not find 'Extracted Symptoms: [...]: in LLM response.")
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"Error during symptom extraction LLM call: {e}")
            # If LLM call fails, combine only fallback symptoms
            combined_symptoms = list(set(fallback_symptoms))
            final_confidence = 0.4 if combined_symptoms else 0.0 # Low confidence
            logger.info(f"Final extracted symptoms (LLM failed): {combined_symptoms} (Confidence: {final_confidence:.4f})")
            result = (combined_symptoms, final_confidence)
            return set_cached(cache_key, result)


        combined_symptoms = list(set(llm_symptoms + fallback_symptoms))
        final_confidence = llm_avg_confidence if llm_symptoms else (0.4 if combined_symptoms else 0.0)

        result = (combined_symptoms, final_confidence)
        logger.info(f"Final extracted symptoms: {combined_symptoms} (Confidence: {final_confidence:.4f})")
        return set_cached(cache_key, result)


    def is_symptom_related_query(self, query: str) -> bool:
        logger.debug(f"Checking if query is symptom-related: {query}")
        if not query or not query.strip():
            logger.debug("Query is empty or whitespace, not symptom related.")
            return False
        cache_key = {"type": "symptom_query_detection", "query": query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("Symptom query detection from cache.")
            return cached

        # is_symptom_related_query uses extract_symptoms internally, call it first for initial symptoms
        extracted_symptoms, symptom_confidence = self.extract_symptoms(query)

        # Primary check based on confidence of symptom extraction
        if extracted_symptoms and symptom_confidence >= THRESHOLDS.get("symptom_extraction", 0.6):
            logger.debug("Query determined symptom-related based on confident extraction.")
            return set_cached(cache_key, True) # Cache and return True

        # Fallback to LLM intent analysis if LLM is available and extraction confidence is low
        if self.llm is None:
            logger.warning("LLM not initialized. Falling back to keyword symptom query detection.")
            health_keywords = ["symptom", "pain", "sick", "health", "disease", "condition", "diagnosis"]
            result = any(keyword in query.lower() for keyword in health_keywords)
            logger.debug(f"Symptom query detection fallback result: {result}")
            return set_cached(cache_key, result)

        QUERY_INTENT_PROMPT = f'''
        Analyze the user query. Is it primarily about health symptoms, medical conditions, or seeking health-related information?
        User Query: "{query}"
        Return ONLY "YES" or "NO".
        '''
        try:
            response = self.local_generate(QUERY_INTENT_PROMPT, max_tokens=50).strip().upper()
            logger.debug(f"Symptom query detection LLM raw response: {response}")
            is_symptom_query = "YES" in response
            logger.info(f"Symptom query detection result: {is_symptom_query}")
            return set_cached(cache_key, is_symptom_query)
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"Error during symptom query detection LLM call: {e}")

        # Ultimate fallback to keyword matching if LLM call/parsing fails
        basic_health_terms = ["symptom", "pain", "sick", "fever", "headache"]
        result = any(term in query.lower() for term in basic_health_terms)
        logger.debug(f"Symptom query detection final fallback result: {result}")
        return set_cached(cache_key, result)


    def knowledge_graph_agent(self, user_query: str, all_symptoms: List[str]) -> Dict[str, Any]:
        logger.info("📚 Knowledge Graph Agent Initiated for query: %s...", user_query[:50])
        logger.debug("Input symptoms for KG: %s", all_symptoms)

        kg_results: Dict[str, Any] = {
            "extracted_symptoms": all_symptoms,
            "identified_diseases_data": [],
            "top_disease_confidence": 0.0,
            "kg_matched_symptoms": [],
            "kg_treatments": [],
            "kg_treatment_confidence": 0.0,
            # Removed kg_home_remedies and kg_remedy_confidence
            "kg_content_diagnosis_data_for_llm": {
                 "disease_name": "an unidentifiable condition",
                 "symptoms_list": all_symptoms,
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
                if all_symptoms:
                    logger.info("📚 KG Task: Identify Diseases from symptoms: %s", all_symptoms)
                    # Call the helper to query diseases
                    disease_data_from_kg: List[Dict[str, Any]] = self._query_disease_from_symptoms_with_session(session, all_symptoms)
                    logger.debug("KG disease query returned %d results.", len(disease_data_from_kg))

                    if disease_data_from_kg:
                        kg_results["identified_diseases_data"] = disease_data_from_kg
                        top_disease_record = disease_data_from_kg[0]
                        # Ensure keys match what _query_disease_from_symptoms_with_session returns ('Disease', 'Confidence')
                        top_disease_name = top_disease_record.get("Disease")
                        top_disease_conf = float(top_disease_record.get("Confidence", 0.0))
                        kg_results["top_disease_confidence"] = top_disease_conf
                        kg_results["kg_matched_symptoms"] = top_disease_record.get("MatchedSymptoms", [])
                        logger.info("✔️ Diseases Identified: %s (Top Confidence: %s)", [(d.get('Disease'), d.get('Confidence')) for d in disease_data_from_kg], top_disease_conf)

                        if top_disease_conf >= THRESHOLDS.get("disease_matching", 0.5):
                            logger.info("📚 KG Task: Find Treatments for %s", top_disease_name)
                            # Call the helper to query treatments
                            kg_results["kg_treatments"], kg_results["kg_treatment_confidence"] = self._query_treatments_with_session(session, top_disease_name)
                            logger.info("✔️ Treatments found: %s (Confidence: %s)", kg_results['kg_treatments'], kg_results['kg_treatment_confidence'])
                            # Removed call to _query_home_remedies_with_session
                        else:
                            logger.info("📚 KG Task: Treatments skipped - Top disease confidence below basic matching threshold (%s < %s).", top_disease_conf, THRESHOLDS.get("disease_matching", 0.5))
                else:
                     logger.info("📚 KG Task: Identify Diseases skipped - No symptoms provided.")

                # Prepare data for LLM phrasing, even if confidence is low or no diseases found
                logger.debug("Preparing KG diagnosis data for LLM.")
                kg_results["kg_content_diagnosis_data_for_llm"] = {
                      "disease_name": kg_results["identified_diseases_data"][0].get("Disease", "an unidentifiable condition") if kg_results["identified_diseases_data"] else "an unidentifiable condition",
                      "symptoms_list": all_symptoms,
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
                # Removed formatting for home remedies

                kg_results["kg_content_other"] = "\n".join(other_parts).strip()
                if not kg_results["kg_content_other"]:
                    kg_results["kg_content_other"] = "Medical Knowledge Graph did not find specific relevant information on treatments or remedies."
                logger.debug("Formatted kg_content_other: %s", kg_results["kg_content_other"][:100])

                logger.info("📚 Knowledge Graph Agent Finished successfully.")
                return kg_results

        except Exception as e:
            logger.error("⚠️ Error within KG Agent: %s", e, exc_info=True)
            # Ensure fallback data is set on failure
            kg_results["kg_content_diagnosis_data_for_llm"] = {
                 "disease_name": "an unidentifiable condition",
                 "symptoms_list": all_symptoms, # Corrected key name for consistency
                 "confidence": 0.0
            }
            kg_results["kg_content_other"] = f"An error occurred while querying the Medical Knowledge Graph: {str(e)}"
            kg_results["top_disease_confidence"] = 0.0 # Ensure confidence is 0 on failure
            logger.info("📚 Knowledge Graph Agent Finished (Error).")
            return kg_results

    def _query_disease_from_symptoms_with_session(self, session, symptoms: List[str]) -> List[Dict[str, Any]]:
         logger.debug("Querying KG for diseases based on symptoms: %s", symptoms)
         if not symptoms:
              logger.debug("No symptoms provided for KG disease query.")
              return []
         cache_key = {"type": "disease_matching_v2", "symptoms": tuple(sorted([s.lower() for s in symptoms]))}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("Disease match query from cache.")
             return cached

         cypher_query = """
         UNWIND $symptomNames AS input_symptom_name
         MATCH (s:symptom) WHERE toLower(s.Name) = toLower(input_symptom_name)
         MATCH (s)-[:INDICATES]->(d:disease)
         WITH d, COLLECT(DISTINCT s.Name) AS matched_symptoms_from_input
         OPTIONAL MATCH (d)<-[:INDICATES]-(all_s:symptom)
         WITH d, matched_symptoms_from_input, COLLECT(DISTINCT all_s.Name) AS all_disease_symptoms_in_kg, size(COLLECT(DISTINCT all_s)) AS total_disease_symptoms_count, size(matched_symptoms_from_input) AS matching_symptoms_count
         WITH d.Name AS Disease, matched_symptoms_from_input, all_disease_symptoms_in_kg,
              CASE WHEN total_disease_symptoms_count = 0 THEN 0 ELSE matching_symptoms_count * 1.0 / total_disease_symptoms_count END AS confidence_score
              WHERE matching_symptoms_count > 0
         RETURN Disease, confidence_score AS Confidence, matched_symptoms_from_input AS MatchedSymptoms, all_disease_symptoms_in_kg AS AllDiseaseSymptomsKG
         ORDER BY confidence_score DESC
         LIMIT 5
         """
         try:
              logger.debug("Executing Cypher query for diseases with symptoms: %s", symptoms)
              result = session.run(cypher_query, symptomNames=[s.lower() for s in symptoms if s])
              records = list(result)
              disease_data = [{"Disease": rec["Disease"], "Confidence": float(rec["Confidence"]), "MatchedSymptoms": rec["MatchedSymptoms"], "AllDiseaseSymptomsKG": rec["AllDiseaseSymptomsKG"]} for rec in records]
              logger.debug("🦠 Executed KG Disease Query, found %d results.", len(disease_data))
              return set_cached(cache_key, disease_data)
         except Exception as e:
              logger.error("⚠️ Error executing KG query for diseases: %s", e)
              return []


    def _query_treatments_with_session(self, session, disease: str) -> Tuple[List[str], float]:
         logger.debug("Querying KG for treatments for disease: %s", disease)
         if not disease:
             logger.debug("No disease provided for KG treatments query.")
             return [], 0.0
         cache_key = {"type": "treatment_query_kg", "disease": disease.lower()}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("KG treatments query from cache.")
             return cached
         cypher_query = """
         MATCH (d:disease)-[r:TREATED_BY]->(t:treatment)
         WHERE toLower(d.Name) = toLower($diseaseName)
         RETURN t.Name as Treatment,
                CASE WHEN COUNT(r) > 3 THEN 0.9 WHEN COUNT(r) > 1 THEN 0.8 ELSE 0.7 END as Confidence
         ORDER BY Confidence DESC
         """
         try:
              logger.debug("Executing Cypher query for treatments for disease: %s", disease)
              result = session.run(cypher_query, diseaseName=disease)
              records = list(result)
              treatments = [(rec["Treatment"], float(rec["Confidence"])) for rec in records]
              treatments_list = [t[0] for t in treatments]
              avg_confidence = sum(t[1] for t in treatments) / len(treatments) if treatments else 0.0
              logger.debug("💊 Executed KG Treatment Query for %s, found %d treatments.", disease, len(treatments_list))
              return set_cached(cache_key, (treatments_list, avg_confidence))
         except Exception as e:
              logger.error("⚠️ Error executing KG query for treatments: %s", e)
              return [], 0.0

    

    def retrieve_rag_context(self, query: str) -> Tuple[List[str], float]:
        logger.info(f"📄 RAG Retrieval Initiated for query: {query[:50]}")
        RAG_RELEVANCE_THRESHOLD = THRESHOLDS.get("rag_context_selection", 0.7)
        logger.debug(f"RAG retrieval threshold: {RAG_RELEVANCE_THRESHOLD}")
        cache_key = {"type": "rag_retrieval", "query": query, "threshold": RAG_RELEVANCE_THRESHOLD}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("RAG retrieval from cache.")
             return cached

        if self.vectordb is None:
            logger.warning("📄 RAG Retrieval: Vector database not initialized.")
            return [], 0.0

        try:
            k = 10
            logger.debug(f"Performing vector search for query: {query[:50]}... (k={k})")
            if not hasattr(self.vectordb, 'similarity_search_with_score'):
                 logger.error("Vector database object does not have 'similarity_search_with_score' method.")
                 return [], 0.0

            retrieved_docs_with_scores = self.vectordb.similarity_search_with_score(query, k=k)
            logger.debug(f"📄 RAG: Retrieved {len(retrieved_docs_with_scores)} initial documents from vector DB.")

            relevant_chunks: List[str] = []
            relevant_scores: List[float] = []

            for doc, score in retrieved_docs_with_scores:
                # Log the type and value before the check
                logger.debug(f"Processing retrieved chunk. Received score type: {type(score)}, value: {score}")

                # Corrected check: Use numbers.Real to include standard floats, ints, and numpy floats.
                if not isinstance(score, (int, float, np.floating)): # Added np.floating for clarity, numbers.Real is also good
                     logger.warning(f"Received unexpected non-numeric score type ({type(score)}) from vector DB. Skipping chunk.")
                     continue

                logger.debug("Score is a valid numeric type.")

                # Cosine distance typically ranges from 0 (identical) to 2 (opposite)
                similarity_score = max(0.0, 1 - float(score))

                logger.debug(f"📄 RAG: Chunk (Sim: {similarity_score:.4f}, Dist: {score:.4f}) from {doc.metadata.get('source', 'N/A')} Page {doc.metadata.get('page', 'N/A')}")
                if similarity_score >= RAG_RELEVANCE_THRESHOLD:
                    relevant_chunks.append(doc.page_content)
                    relevant_scores.append(similarity_score)
                    logger.debug(f"📄 RAG: Added relevant chunk (Sim: {similarity_score:.4f})")
                else:
                    logger.debug(f"📄 RAG: Skipped chunk (Sim: {similarity_score:.4f}) - below threshold {RAG_RELEVANCE_THRESHOLD:.4f}")


            srag = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
            logger.info(f"📄 RAG Retrieval Finished. Found {len(relevant_chunks)} relevant chunks. Overall S_RAG: {srag:.4f}")
            return set_cached(cache_key, (relevant_chunks, srag))

        except Exception as e:
            logger.error(f"⚠️ Error during RAG retrieval: {e}", exc_info=True)
            return [], 0.0

    def select_context(self,
                       kg_results: Dict[str, Any],
                       s_kg: float,
                       rag_chunks: List[str],
                       s_rag: float,
                       is_symptom_query: bool
                      ) -> Optional[Dict[str, Any]]:
        logger.info("📦 Context Selection Initiated. Symptom Query: %s, S_KG: %.4f, S_RAG: %.4f", is_symptom_query, s_kg, s_rag)
        kg_threshold = THRESHOLDS.get("kg_context_selection", 0.8)
        rag_threshold = THRESHOLDS.get("rag_context_selection", 0.7)
        logger.debug("Context selection thresholds: KG > %s, RAG > %s", kg_threshold, rag_threshold)
        selected_context: Optional[Dict[str, Any]] = None
        context_parts: List[str] = []

        if is_symptom_query:
            logger.debug("Processing symptom query context selection logic.")
            if s_kg > kg_threshold and s_rag > rag_threshold:
                logger.info("📦 Symptom query, both KG and RAG thresholds met. Selecting KG + RAG.")
                selected_context = {"kg": kg_results, "rag": rag_chunks}
                context_parts.extend(["KG", "RAG"])
            elif s_kg > kg_threshold:
                logger.info("📦 Symptom query, only KG threshold met. Selecting KG Only.")
                selected_context = {"kg": kg_results}
                context_parts.append("KG")
            elif s_rag > rag_threshold:
                logger.info("📦 Symptom query, only RAG threshold met. Selecting RAG Only.")
                selected_context = {"rag": rag_chunks}
                context_parts.append("RAG")
            else:
                logger.info("📦 Symptom query, neither KG nor RAG met individual thresholds. Selected Context: None.")
                selected_context = None
        else:
            logger.debug("Processing non-symptom query context selection logic (RAG only).")
            if s_rag > rag_threshold:
                logger.info("📦 Non-symptom query, RAG threshold met. Selecting RAG Only.")
                selected_context = {"rag": rag_chunks}
                context_parts.append("RAG")
            else:
                 logger.info("📦 Non-symptom query, RAG threshold not met. Selected Context: None.")
                 selected_context = None

        if selected_context is not None:
            logger.info("📦 Context Selection Final: Includes: %s.", ', '.join(context_parts))
        else:
             logger.info("📦 Context Selection Final: No context selected.")
        return selected_context
                          
    def generate_initial_answer(self, query: str, selected_context: Optional[Dict[str, Any]]) -> str:
        logger.info("🧠 Initial Answer Generation Initiated")
        base_prompt_instructions = "You are a helpful and knowledgeable medical assistant. Answer the user query to the best of your ability, using the provided information. Be concise, medically accurate, and easy for a general user to understand."
        context_info_for_prompt = ""
        context_type_description = ""

        if selected_context is None:
            logger.info("🧠 Generating initial answer WITHOUT context.")
            context_info_for_prompt = "No specific relevant information was found in external knowledge sources."
            context_type_description = "Relying only on your vast general knowledge, answer the user query. Do not mention external documents or knowledge graphs."
        else:
            context_types = []
            if 'kg' in selected_context: context_types.append('KG')
            if 'rag' in selected_context: context_types.append('RAG')
            logger.debug("Generating initial answer with context type: %s", " + ".join(context_types))

            context_type_description = "Based on the following information,"
            if "kg" in selected_context:
                kg_data = selected_context.get("kg", {})
                kg_info_str = "Knowledge Graph Information:\n"
                diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                # Use disease_matching threshold for basic phrasing inclusion, kg_context_selection for 'Identified'
                if diag_data and diag_data.get("confidence", 0) > THRESHOLDS.get("disease_matching", 0.5):
                     disease_name = diag_data.get("disease_name", "an unidentifiable condition")
                     confidence = diag_data.get("confidence", 0)
                     if confidence > THRESHOLDS.get("kg_context_selection", 0.8):
                          kg_info_str += f"- Identified Condition: {disease_name} (KG Confidence: {confidence:.2f})\n"
                     else:
                          kg_info_str += f"- Potential Condition: {disease_name} (KG Confidence: {confidence:.2f})\n"
                other_kg_content = kg_data.get("kg_content_other")
                if other_kg_content and other_kg_content.strip() and "Medical Knowledge Graph did not find" not in other_kg_content and "Home Remedies (from KG)" not in other_kg_content: # Added check to exclude remedies if present
                      kg_info_str += "\n" + other_kg_content
                context_info_for_prompt += kg_info_str

            if "rag" in selected_context:
                rag_chunks = selected_context.get("rag", [])
                rag_info_str = "Relevant Passages from Documents:\n" + "\n---\n".join(rag_chunks) + "\n---"
                if "kg" in selected_context:
                    context_info_for_prompt += "\n\n" + rag_info_str
                    context_type_description = "Based on the following structured information from a medical knowledge graph and relevant passages from medical documents, synthesize a comprehensive answer."
                else:
                    context_info_for_prompt += rag_info_str
                    context_type_description = "Based on the following relevant passages from medical documents, answer the user query. Only use the information provided here. Do not refer to a knowledge graph."
            elif "kg" in selected_context:
                 context_type_description = "Based on the following information from a medical knowledge graph, answer the user query. Only use the information provided here. Do not refer to external documents."

        if not context_info_for_prompt.strip() and selected_context is not None:
             logger.warning("Selected context was passed but formatted into an empty string for the prompt.")
             context_info_for_prompt = "No specific relevant information was effectively utilized from external knowledge sources."
             context_type_description = "Relying only on your vast general knowledge, answer the user query."


        prompt = f"""
{base_prompt_instructions}
{context_type_description}

{context_info_for_prompt}

User Query: {query}

Answer:
"""
        logger.debug("Initial Answer Generation Prompt: %s...", prompt[:500])
        try:
            initial_answer = self.local_generate(prompt, max_tokens=1000)
            logger.info("🧠 Initial Answer Generated successfully.")
            logger.debug("Initial Answer: %s...", initial_answer[:100])
            return initial_answer
        except ValueError as e:
            logger.error("⚠️ Error during initial answer generation: %s", e, exc_info=True)
            return "Sorry, I encountered an error while trying to generate an initial answer."


    def reflect_on_answer(self,
                          query: str,
                          initial_answer: str,
                          selected_context: Optional[Dict[str, Any]]
                         ) -> Tuple[str, Optional[str]]:
        logger.info("🔍 Reflection and Evaluation Initiated")
        context_for_prompt = "None"
        if selected_context is not None:
            context_types = []
            if 'kg' in selected_context: context_types.append('KG')
            if 'rag' in selected_context: context_types.append('RAG')
            logger.debug("Reflection using context type: %s", " + ".join(context_types))

            context_for_prompt = "Provided Context:\n---\n"
            if "kg" in selected_context:
                kg_data = selected_context.get("kg", {})
                kg_info_str = "Knowledge Graph Info:\n"
                diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                if diag_data and diag_data.get("disease_name"):
                     kg_info_str += f"  Potential Condition: {diag_data['disease_name']} (Confidence: {diag_data.get('confidence', 0):.2f})\n"
                if kg_data.get("kg_treatments"):
                     kg_info_str += f"  Treatments: {', '.join(kg_data['kg_treatments'])}\n"
                # Removed kg_home_remedies formatting
                context_for_prompt += kg_info_str + "\n"
            if "rag" in selected_context:
                rag_chunks = selected_context.get("rag", [])
                # Limit RAG chunks for reflection prompt brevity
                rag_info_str = "Relevant Passages:\n" + "\n---\n".join(rag_chunks[:3]) + "\n---"
                context_for_prompt += rag_info_str

        cache_key = {"type": "reflection", "query": query, "initial_answer": initial_answer, "context_hash": abs(hash(str(selected_context)))}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Reflection from cache.")
             return cached

        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform reflection.")
             return ('incomplete', 'Reflection LLM is unavailable.')

        reflection_prompt = f'''
        You are an evaluation agent. Review the 'Initial Answer' for its completeness and correctness in fully addressing the 'User Query', *considering the provided 'Context' (if any)*.
        If incomplete, identify *exactly* what specific information is missing or incomplete *from the perspective of the User Query*.
        Return ONLY a JSON object: {{"evaluation": "complete" or "incomplete", "missing_information": "Description of what is missing"}}
        User Query: "{query}"
        Context:
        {context_for_prompt}
        Initial Answer:
        "{initial_answer}"
        '''
        logger.debug("Reflection Prompt: %s...", reflection_prompt[:500])
        try:
            response = self.local_generate(reflection_prompt, max_tokens=500)
            logger.debug("Reflection LLM raw response: %s", response)
            json_match = re.search(r'\{[\s\S]*\}', response)
            evaluation_result = 'incomplete'
            missing_info_description = "Could not parse reflection response or missing information was not provided by evaluator."
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    evaluation_result = data.get("evaluation", "incomplete").lower()
                    missing_info_description = data.get("missing_information", "").strip()
                    if evaluation_result == 'complete':
                        missing_info_description = None
                        logger.info("🔍 Reflection Result: Complete.")
                    else:
                         if not missing_info_description:
                              missing_info_description = f"Answer incomplete (evaluator provided no details for '{query[:50]}...')"
                         logger.warning("🔍 Reflection Result: Incomplete. Missing Info: %s...", missing_info_description[:100])
                except json.JSONDecodeError:
                    logger.error("⚠️ Reflection: Could not parse JSON from LLM response.")
            else:
                logger.warning("⚠️ Reflection: No JSON object found in LLM response.")
            result = (evaluation_result, missing_info_description)
            return set_cached(cache_key, result)
        except ValueError as e:
            logger.error("⚠️ Error during reflection process: %s", e, exc_info=True)
            return ('incomplete', f"An error occurred during reflection: {str(e)}")


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
             return "Supplementary information could not be generated because the AI model is unavailable."

        supplementary_prompt = f'''
        You are a medical information agent. Provide *only* the specific information missing from a previous answer, based on the description below.
        Do NOT re-answer the original user query entirely. Focus strictly on the missing part.
        **You MUST include evidence or source attribution.** Provide links (URLs) or names of reliable sources [Source Name or URL].
        Original User Query (for context): "{query}"
        Information Missing from Previous Answer: "{missing_info_description}"
        Provide ONLY the supplementary information addressing the missing part, including evidence/sources:
        '''
        try:
            supplementary_answer = self.local_generate(supplementary_prompt, max_tokens=750)
            logger.info("🌐 Supplementary Answer Generated successfully.")
            return set_cached(cache_key, supplementary_answer)
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"⚠️ Error during supplementary answer generation: {e}", exc_info=True)
            return f"Sorry, an error occurred while trying to find additional information about: '{missing_info_description[:50]}...'"

    def collate_answers(self, initial_answer: str, supplementary_answer: str) -> str:
        logger.info("✨ Final Answer Collation Initiated")
        # Cache key based on hash of inputs
        cache_key = {"type": "final_collation", "initial_answer_hash": abs(hash(initial_answer)), "supplementary_answer_hash": abs(hash(supplementary_answer))}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Final collation from cache.")
             return cached

        # Fallback if LLM is not available
        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform final answer collation.")
             return f"{initial_answer}\n\n-- Additional Information --\n{supplementary_answer}"

        collation_prompt = f'''
        You are a medical communicator. Combine two parts of an answer related to a medical query.
        The first part is an initial answer. The second part contains supplementary information that was missing from the first.
        Combine these into a single, smooth, coherent, and easy-to-understand answer.
        Ensure a natural flow, remove redundancy, and preserve any source attributions or links.
        Present the final response as a unified answer.
        Initial Answer Part: "{initial_answer}"
        Supplementary Information Part: "{supplementary_answer}"
        Provide ONLY the combined, final answer:
        '''
        try:
            final_answer = self.local_generate(collation_prompt, max_tokens=1200)
            logger.info("✨ Final Answer Collated successfully.")
            return set_cached(cache_key, final_answer)
        except ValueError as e: # Catch ValueError from local_generate if LLM fails
            logger.error(f"⚠️ Error during final answer collation: {e}", exc_info=True)
            return f"Sorry, an error occurred while finalizing the answer. Parts:\n\nInitial: {initial_answer}\n\nSupplementary: {supplementary_answer}"

    def reset_conversation(self):
        logger.info("🔄 Resetting chatbot internal state.")
        self.chat_history = []
        self.followup_context = {"round": 0}


    # The Main Orchestrator Function
    def process_user_query(self,
                           user_query: str,
                           user_type: str = "User / Family",
                           confirmed_symptoms: Optional[List[str]] = None,
                           original_query_if_followup: Optional[str] = None
                           ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        logger.info("--- Processing User Query: '%s' ---", user_query[:100])
        logger.info("User Type: %s, Confirmed Symptoms Received: %s", user_type, len(confirmed_symptoms) if confirmed_symptoms is not None else 'None')

        processed_query = user_query # Query text used for retrieval and generation
        current_symptoms_for_retrieval: List[str] = [] # Symptoms used for KG
        is_symptom_query = False # Flag for symptom path

        # --- Handle Symptom Confirmation Rerun Logic ---
        if confirmed_symptoms is not None:
             logger.info("--- Step 6 (Rerun): Handling Symptom Confirmation Rerun ---")
             # Use the original query text that triggered the UI
             processed_query = original_query_if_followup
             if not processed_query: # Sanity check
                  logger.error("original_query_if_followup is None during symptom rerun.")
                  return "An error occurred during symptom confirmation processing.", "display_final_answer", None

             is_symptom_query = True # Assume symptom path for UI follow-up
             # Use the confirmed symptoms as the list for KG in this rerun
             current_symptoms_for_retrieval = confirmed_symptoms
             logger.info("Reprocessing '%s...' with confirmed symptoms: %s", processed_query[:50], current_symptoms_for_retrieval)
             medical_check_ok = True # Assume valid as we were in a medical flow

        else: # Standard initial query processing
             logger.info("--- Initial Query Processing ---")
             # --- Step 2: Guardrail Check ---
             logger.info("--- Step 2: Guardrail Check ---")
             is_medical, medical_reason = self.is_medical_query(processed_query)
             if not is_medical:
                 logger.warning("Query flagged as non-medical: %s. Workflow terminates.", medical_reason)
                 response = f"I can only answer medical-related questions. Please rephrase your query. ({medical_reason})"
                 return response, "display_final_answer", None
             logger.info("Query passed medical guardrail.")
             medical_check_ok = True

        # --- Proceed only if medical check is OK ---
        if medical_check_ok:
             # --- Step 3: Symptom Processing (only for initial query path) ---
             if confirmed_symptoms is None: # Only run symptom processing on the first pass
                  logger.info("--- Step 3: Symptom Processing ---")
                  extracted_symptoms, symptom_extraction_confidence = self.extract_symptoms(processed_query)
                  current_symptoms_for_retrieval = extracted_symptoms
                  is_symptom_query = self.is_symptom_related_query(processed_query)
                  logger.info("Initial Symptom Extraction: %s (Confidence: %.4f)", current_symptoms_for_retrieval, symptom_extraction_confidence)
                  logger.info("Is Symptom Query: %s", is_symptom_query)
                  # Note: If combining original + confirmed later, store extracted_symptoms here in state

             # --- Step 4: Context Retrieval (KG & RAG) ---
             logger.info("--- Step 4: Context Retrieval ---")
             kg_results = {}
             s_kg = 0.0
             rag_chunks = []
             s_rag = 0.0

             # KG Retrieval (if query is symptom-related)
             if is_symptom_query:
                  logger.info("Triggering KG Pipeline.")
                  # Pass the *current* list of symptoms (initial extracted OR confirmed UI list)
                  kg_results = self.knowledge_graph_agent(processed_query, current_symptoms_for_retrieval)
                  s_kg = kg_results.get("top_disease_confidence", 0.0)
                  logger.info("KG Pipeline finished. S_KG: %.4f", s_kg)

             # RAG Retrieval (for all medical queries)
             logger.info("Triggering RAG Pipeline.")
             rag_chunks, s_rag = self.retrieve_rag_context(processed_query)
             logger.info("RAG Pipeline finished. S_RAG: %.4f", s_rag)

             # --- Step 5: Symptom Confirmation UI Decision Point (only after initial symptom retrieval) ---
             # Check if this is the *initial* query processing (not a rerun after UI) AND it's a symptom query
             if confirmed_symptoms is None and is_symptom_query:
                  logger.info("--- Step 5: Symptom Confirmation UI Check ---")
                  ui_trigger_threshold = THRESHOLDS.get("disease_symptom_followup_threshold", 0.8)
                  kg_found_diseases = len(kg_results.get("identified_diseases_data", [])) > 0

                  # Condition: Query is symptom-related AND KG found diseases AND KG confidence < threshold
                  if kg_found_diseases and s_kg < ui_trigger_threshold:
                       logger.info(f"KG confidence (%.4f) below UI trigger threshold (%.4f) and diseases found (%d). Preparing symptom UI.", s_kg, ui_trigger_threshold, len(kg_results['identified_diseases_data']))
                       symptom_options_for_ui = {}
                       if kg_results.get("identified_diseases_data"):
                            top_diseases_data = kg_results["identified_diseases_data"][:3]
                            # Filter out symptoms already provided in the *initial* query extraction
                            # This requires the initial extracted symptoms list from Step 3
                            # Assuming current_symptoms_for_retrieval currently holds initial extracted symptoms on first pass
                            initial_symptoms_lower = set([s.strip().lower() for s in current_symptoms_for_retrieval if isinstance(s, str)])

                            filtered_symptom_options_for_ui = {}
                            for disease_data in top_diseases_data:
                                 disease_name = disease_data.get("Disease", "Unknown")
                                 symptoms_list = disease_data.get("AllDiseaseSymptomsKG", []) # Symptoms from KG for this disease
                                 # Filter out symptoms the user already gave
                                 filtered_symptoms = [s for s in symptoms_list if isinstance(s, str) and s.strip().lower() not in initial_symptoms_lower]
                                 if filtered_symptoms:
                                      filtered_symptom_options_for_ui[disease_name] = filtered_symptoms

                            if filtered_symptom_options_for_ui:
                                 total_suggested_symps = sum(len(v) for v in filtered_symptom_options_for_ui.values())
                                 logger.info(f"Prepared {total_suggested_symps} potential symptoms for UI checklist.")
                                 prompt_text = "Based on your symptoms, I've identified a few potential conditions. To help me provide the most accurate information, could you please confirm any additional symptoms you are experiencing from the list below?"
                                 ui_payload = {
                                      "symptom_options": filtered_symptom_options_for_ui,
                                      "original_query": processed_query # Store the query that triggered this UI
                                  }
                                 logger.info("Returning UI action: show_symptom_ui")
                                 return prompt_text, "show_symptom_ui", ui_payload

                       # If KG confidence below threshold, but no diseases found or no NEW symptoms to suggest,
                       # proceed to context selection and generation.
                       logger.info("KG confidence below threshold, but no diseases found or no NEW symptoms to suggest. Proceeding to answer generation.")


             # --- Step 7: Context Selection Logic ---
             logger.info("--- Step 7: Context Selection Logic ---")
             selected_context = self.select_context(kg_results, s_kg, rag_chunks, s_rag, is_symptom_query)

             # --- Step 8: Initial Answer Generation ---
             logger.info("--- Step 8: Initial Answer Generation ---")
             initial_answer = self.generate_initial_answer(processed_query, selected_context)

             # --- Step 9: Reflection and Evaluation ---
             logger.info("--- Step 9: Reflection and Evaluation ---")
             evaluation_result, missing_info_description = self.reflect_on_answer(
                  processed_query, initial_answer, selected_context
             )
             logger.info(f"Reflection Result: {evaluation_result}")

             # --- Step 10: Conditional External Agent Invocation Logic ---
             if evaluation_result == 'complete':
                  logger.info("--- Step 10: Reflection Complete. Skipping supplementary pipeline. ---")
                  final_answer = initial_answer
                  self.followup_context = {"round": 0} # Reset state for next query thread
                  logger.info("--- Workflow Finished (Complete) ---")
                  return final_answer, "display_final_answer", None

             else: # evaluation_result is 'incomplete'
                  logger.warning("--- Step 10: Reflection Incomplete. Triggering supplementary pipeline. ---")
                  logger.debug(f"Missing Info Description: {missing_info_description[:100]}...")

                  # --- Step 11: External Agent - Gap Filling & Evidence ---
                  logger.info("--- Step 11: External Agent (Gap Filling) ---")
                  supplementary_answer = self.get_supplementary_answer(processed_query, missing_info_description)

                  # --- Step 12: Final Answer Collation ---
                  logger.info("--- Step 12: Final Answer Collation ---")
                  final_answer = self.collate_answers(initial_answer, supplementary_answer)

                  self.followup_context = {"round": 0} # Reset state for next query thread
                  logger.info("--- Workflow Finished (Incomplete path, collated) ---")
                  return final_answer, "display_final_answer", None
        else:
             # This case should ideally not be reachable due to the initial medical_check_ok guard
             logger.error("Reached unexpected state: medical_check_ok was False but processing continued.")
             return "An internal error occurred during processing.", "display_final_answer", None


# Streamlit UI Component - Symptom Checklist Form
def display_symptom_checklist(symptom_options: Dict[str, List[str]], original_query: str):
    logger.debug("Rendering symptom checklist UI.")
    st.subheader("Confirm Your Symptoms")
    st.info(f"Based on your query: '{original_query}' and initial analysis, please confirm any additional symptoms you are experiencing from the list below:")

    form_key = f"symptom_confirmation_form_{abs(hash(original_query))}_{st.session_state.get('form_timestamp', datetime.now().timestamp())}"
    local_confirmed_symptoms_key = f'{form_key}_confirmed_symptoms_local'

    # Initialize the local set if it's not in session state for this form key
    # This ensures symptoms selected on one form don't carry over to another if keys change unexpectedly
    if local_confirmed_symptoms_key not in st.session_state:
        logger.debug(f"Initializing local symptom set for form {form_key}")
        st.session_state[local_confirmed_symptoms_key] = set()
    else:
        logger.debug(f"Using existing local symptom set for form {form_key} with {len(st.session_state[local_confirmed_symptoms_key])} items.")


    all_unique_symptoms = set()
    # Iterate through diseases/symptom lists provided for the UI
    for symptoms_list in symptom_options.values():
        if isinstance(symptoms_list, list):
            for symptom in symptoms_list:
                 if isinstance(symptom, str):
                      all_unique_symptoms.add(symptom.strip()) # Add stripped symptom, keep original case for display initially

    sorted_all_symptoms = sorted(list(all_unique_symptoms))
    logger.debug(f"Total unique symptoms suggested in UI: {len(sorted_all_symptoms)}")

    with st.form(form_key):
        st.markdown("Please check all symptoms that apply to you from the list below:")
        if not sorted_all_symptoms:
            st.info("No specific additional symptoms were found for potential conditions to suggest. Use the box below to add symptoms you are experiencing.")
        else:
            cols = st.columns(4)
            for i, symptom in enumerate(sorted_all_symptoms):
                col = cols[i % 4]
                checkbox_key = f"{form_key}_checkbox_{symptom}"
                # Check the initial state from the local set
                initial_state = symptom.strip().lower() in st.session_state.get(local_confirmed_symptoms_key, set())
                # Checkbox returns True if checked, False if unchecked.
                is_checked = col.checkbox(symptom, key=checkbox_key, value=initial_state)

                # Update the local set based on the current state of the checkbox
                symptom_lower = symptom.strip().lower()
                if is_checked:
                     st.session_state[local_confirmed_symptoms_key].add(symptom_lower)
                else:
                     # Only discard if it was previously in the set and is now unchecked
                     if symptom_lower in st.session_state.get(local_confirmed_symptoms_key, set()):
                         st.session_state[local_confirmed_symptoms_key].discard(symptom_lower)


        st.markdown("**Other Symptoms (if any):**")
        # Use a unique key for the text input as well
        other_symptoms_text_key = f"{form_key}_other_symptoms_input"
        other_symptoms_text = st.text_input("Enter additional symptoms here (comma-separated)", key=other_symptoms_text_key)
        if other_symptoms_text:
             other_symptoms_list = [s.strip().lower() for s in other_symptoms_text.split(',') if s.strip()]
             if other_symptoms_list:
                 logger.debug(f"Adding other symptoms from input: {other_symptoms_list}")
                 # Add these to the local set as well
                 st.session_state[local_confirmed_symptoms_key].update(other_symptoms_list)

        submit_button = st.form_submit_button("Confirm and Continue")
        if submit_button:
            logger.info(f"Symptom confirmation form submitted for query: '{original_query[:50]}...'. Final confirmed symptoms count: {len(st.session_state[local_confirmed_symptoms_key])}")
            # Store the final confirmed symptoms in a session state variable the main loop looks for
            # Ensure we are getting the list from the *local* set state variable
            st.session_state.confirmed_symptoms_from_ui = sorted(list(st.session_state[local_confirmed_symptoms_key]))
            logger.debug(f"Set st.session_state.confirmed_symptoms_from_ui.")

            # Reset the UI state to 'input' so the chat input appears on the next rerun
            st.session_state.ui_state = {"step": "input", "payload": None}
            logger.debug("Set ui_state back to 'input' after form submission.")

            # Set a new timestamp for the next potential form display
            st.session_state.form_timestamp = datetime.now().timestamp()
            logger.debug(f"Reset form_timestamp.")

            # The main loop will detect confirmed_symptoms_from_ui and trigger processing
            # No need to set processing_input_payload here; main() does that based on detecting confirmed_symptoms_from_ui


# --- Main Streamlit App Function ---
def main():
    logger.info("--- Streamlit App Start ---")
    try:
        st.set_page_config(
            page_title="DxAI-Agent",
            page_icon=f"data:image/png;base64,{icon}",
            layout="wide"
        )
        logger.info("Page config set.")
    except Exception as e:
        logger.error(f"Error setting page config: {e}")
        st.set_page_config(page_title="DxAI-Agent", layout="wide")
        logger.warning("Using fallback page config.")

    try:
        # Assumes image_path is defined globally or accessible
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


    # Initialize session state variables if they don't exist
    if 'chatbot' not in st.session_state:
        logger.info("Initializing chatbot instance in session state.")
        st.session_state.chatbot = DocumentChatBot()
        # Initialization happens implicitly the first time we check init_status

    # Store init status separately
    # Perform initialization check/run only once per session unless reset
    if 'init_status' not in st.session_state:
         logger.info("Checking/Performing chatbot initialization.")
         with st.spinner("Initializing chat assistant..."):
              success, init_message = st.session_state.chatbot.initialize_qa_chain()
              st.session_state.init_status = (success, init_message)
         logger.info(f"Chatbot initialization complete. Status: {st.session_state.init_status}")


    # --- UI State Variables ---
    # Controls what the main input/action area shows: "input" or "confirm_symptoms"
    if 'ui_state' not in st.session_state:
        logger.info("Initializing ui_state.")
        st.session_state.ui_state = {"step": "input", "payload": None}

    # Messages for display
    if 'messages' not in st.session_state:
        logger.info("Initializing messages state.")
        st.session_state.messages = [] # List of (content, is_user) tuples for UI display

    # Variable to hold the input data that needs processing by process_user_query
    # Set by user input or form submission. Cleared before calling process_user_query.
    if 'processing_input_payload' not in st.session_state:
         logger.info("Initializing processing_input_payload state.")
         st.session_state.processing_input_payload = None # Dict like {"query": ..., "confirmed_symptoms": ..., "original_query_context": ...}

    # Variable to store the symptoms confirmed by the UI form, detected by the main loop
    # Set by display_symptom_checklist on form submit. Cleared by the main loop after detection.
    if 'confirmed_symptoms_from_ui' not in st.session_state:
         logger.info("Initializing confirmed_symptoms_from_ui state.")
         st.session_state.confirmed_symptoms_from_ui = None

    # Variable to store the original query text that triggered the symptom UI, needed for rerun
    # Set by main() when process_user_query returns "show_symptom_ui". Cleared after the rerun processing completes.
    if 'original_query_for_symptom_rerun' not in st.session_state:
         logger.info("Initializing original_query_for_symptom_rerun state.")
         st.session_state.original_query_for_symptom_rerun = None

    # Add a timestamp for the symptom confirmation form key to ensure uniqueness across reruns
    if 'form_timestamp' not in st.session_state:
         logger.info("Initializing form_timestamp state.")
         st.session_state.form_timestamp = datetime.now().timestamp()

    user_type = st.sidebar.selectbox(
        "Who is asking?",
        ["User / Family", "Physician"],
        index=0
    )
    logger.debug(f"User type selected: {user_type}")

    st.sidebar.info("DxAI-Agent helps answer medical questions using our medical knowledge base.")

    # Display initialization status in sidebar
    init_success, init_msg = st.session_state.init_status
    if not init_success:
         st.sidebar.error(f"Initialization Failed: {init_msg}")
         logger.error(f"Initialization failed. Message: {init_msg}")
         # If initialization failed, disable interaction
         is_interaction_enabled = False
    else:
         st.sidebar.success(f"Initialization Status: {init_msg}")
         logger.info("Chatbot initialized successfully.")
         is_interaction_enabled = True


    tab1, tab2 = st.tabs(["Chat", "About"])

    with tab1:
        st.subheader("Try these examples")
        examples = [
            "What are treatments for cough and cold?",
            "I have a headache and sore throat. What could it be?",
            "What home remedies help with flu symptoms?",
            "I have chest pain and shortness of breath. What could i do?"
        ]

        # Examples disabled if init failed OR if UI is not in the main 'input' state (e.g., showing symptom form)
        examples_disabled = not is_interaction_enabled or st.session_state.ui_state["step"] != "input"
        cols = st.columns(len(examples))
        for i, col in enumerate(cols):
            if col.button(examples[i], key=f"example_{i}", disabled=examples_disabled):
                logger.info(f"Example '{examples[i][:50]}...' clicked. Triggering processing.")
                # Reset conversation history for a new example thread
                st.session_state.messages = []
                # Ensure UI state is back to input
                st.session_state.ui_state = {"step": "input", "payload": None}
                # Reset backend state
                st.session_state.chatbot.reset_conversation()
                # Reset UI form/symptom state variables
                st.session_state.form_timestamp = datetime.now().timestamp() # New timestamp for any potential forms
                if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun

                # Set the input to be processed in the next rerun
                st.session_state.processing_input_payload = {
                    "query": examples[i], "confirmed_symptoms": None, "original_query_context": None
                }
                logger.debug("Set processing_input_payload for example click.")
                st.rerun()

        # --- Chat Messages Display ---
        for i, (msg_content, is_user) in enumerate(st.session_state.messages):
            if is_user:
                with st.chat_message("user"): st.write(msg_content)
            else:
                with st.chat_message("assistant"):
                    st.write(msg_content)
                    # Add feedback buttons only to final answers (when input is enabled again)
                    # A message is considered a final answer display if it's the last message AND the UI is back to the 'input' state.
                    is_final_answer_display = (i == len(st.session_state.messages) - 1) and (st.session_state.ui_state["step"] == "input")

                    if is_final_answer_display:
                        col = st.container()
                        with col:
                            # Ensure unique keys for feedback buttons
                            feedback_key_up = f"thumbs_up_{i}_{abs(hash(msg_content))}"
                            feedback_key_down = f"thumbs_down_{i}_{abs(hash(msg_content))}"
                            b1, b2 = st.columns([0.05, 0.95])
                            with b1:
                                if st.button("👍", key=feedback_key_up):
                                     # Find the preceding user message for context
                                     user_msg_content = next((st.session_state.messages[j][0] for j in range(i - 1, -1, -1) if st.session_state.messages[j][1] is True), "")
                                     logger.info(f"Thumbs Up feedback for user query: '{user_msg_content[:50]}...'")
                                     vote_message(user_msg_content, msg_content, "thumbs_up", user_type)
                                     st.toast("Feedback recorded: Thumbs Up!")
                            with b2:
                                if st.button("👎", key=feedback_key_down):
                                    user_msg_content = next((st.session_state.messages[j][0] for j in range(i - 1, -1, -1) if st.session_state.messages[j][1] is True), "")
                                    logger.info(f"Thumbs Down feedback for user query: '{user_msg_content[:50]}...'")
                                    vote_message(user_msg_content, msg_content, "thumbs_down", user_type)
                                    st.toast("Feedback recorded: Thumbs Down!")

        input_area_container = st.container()
        st.write("  \n" * 5) # Add space at the end of the tab

        with input_area_container:
            # Disable input area completely if initialization failed
            if not is_interaction_enabled:
                 st.error("Chat assistant failed to initialize. Please check the logs and configuration.")
            elif st.session_state.ui_state["step"] == "confirm_symptoms":
                logger.debug("UI state is 'confirm_symptoms', displaying checklist.")
                # Ensure payload is a dictionary before accessing its contents
                ui_payload = st.session_state.ui_state.get("payload")
                if ui_payload is None:
                     logger.error("ui_state is 'confirm_symptoms' but payload is None! Resetting UI state.")
                     st.session_state.ui_state = {"step": "input", "payload": None}
                     st.error("An error occurred displaying the symptom checklist.")
                     st.rerun() # Trigger rerun to fix state
                     return # Stop processing this rerun

                display_symptom_checklist(
                     ui_payload.get("symptom_options", {}),
                     ui_payload.get("original_query", "") # Provide a default empty string just in case
                )
                # Chat input is disabled while symptom form is active
                st.chat_input("Confirm symptoms above...", disabled=True, key="disabled_chat_input")

            elif st.session_state.ui_state["step"] == "input":
                logger.debug("UI state is 'input', displaying chat input.")
                # Chat input is enabled only if initialization was successful
                user_query = st.chat_input("Ask your medical question...", disabled=not is_interaction_enabled, key="main_chat_input")
                if user_query:
                    logger.info(f"Detected new chat input: '{user_query[:50]}...'. Triggering processing.")
                    # Add user message to state immediately for display
                    st.session_state.messages.append((user_query, True))
                    # Reset backend state for a brand new conversation thread starting with this query
                    st.session_state.chatbot.reset_conversation()
                    # Reset UI form/symptom state variables for a new thread
                    st.session_state.form_timestamp = datetime.now().timestamp()
                    if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                    if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun

                    # Set the input to be processed in the next rerun
                    st.session_state.processing_input_payload = {
                        "query": user_query, "confirmed_symptoms": None, "original_query_context": None
                    }
                    logger.debug("Set processing_input_payload for new chat input.")
                    st.rerun() # Trigger rerun to process the input

        # --- Check for Symptom Form Submission and Trigger Processing ---
        # This block runs *after* the symptom form (if active) might have been submitted in this rerun.
        # The display_symptom_checklist function sets st.session_state.confirmed_symptoms_from_ui
        # and updates ui_state on submit.
        if 'confirmed_symptoms_from_ui' in st.session_state and st.session_state.confirmed_symptoms_from_ui is not None:
             logger.info("Detected symptom confirmation form submission via state. Preparing processing payload.")
             confirmed_symps_to_pass = st.session_state.confirmed_symptoms_from_ui

             # --- NEW: Retrieve original query from the dedicated state variable ---
             # This variable was set when process_user_query returned "show_symptom_ui"
             original_query_to_pass = st.session_state.get('original_query_for_symptom_rerun')
             if original_query_to_pass is None:
                  logger.error("confirmed_symptoms_from_ui set, but original_query_for_symptom_rerun is None! Cannot re-process.")
                  # Clean up state and return an error message
                  del st.session_state.confirmed_symptoms_from_ui
                  if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                  st.session_state.ui_state = {"step": "input", "payload": None}
                  st.session_state.messages.append(("Sorry, an internal error occurred during symptom confirmation.", False))
                  st.rerun() # Trigger rerun to display error and clear state
                  return # Stop processing this rerun


             # Clear the temporary state variable used by the form submission
             del st.session_state.confirmed_symptoms_from_ui
             logger.debug("Cleared confirmed_symptoms_from_ui state.")

             # Note: We DO NOT clear original_query_for_symptom_rerun here yet.
             # It will be cleared after the rerun process is finished (in the processing_input_payload block).

             # Set the input into processing_input_payload to trigger the backend call
             st.session_state.processing_input_payload = {
                  "query": original_query_to_pass, # Process using original query text
                  "confirmed_symptoms": confirmed_symps_to_pass, # Pass the confirmed symptoms
                  "original_query_context": original_query_to_pass # Original query context
             }
             logger.info("Set processing_input_payload for symptom confirmation rerun.")
             st.rerun() # Trigger rerun to process the confirmed symptoms

        # --- Call the Backend Orchestrator if Processing is Needed ---
        # This is the SINGLE point where process_user_query is called based on the state.
        if st.session_state.processing_input_payload is not None:
            logger.info("Detected processing_input_payload. Calling chatbot.process_user_query.")
            input_data = st.session_state.processing_input_payload
            # Clear the processing flag immediately
            st.session_state.processing_input_payload = None
            logger.debug("Cleared processing_input_payload state.")

            query_to_process = input_data["query"]
            confirmed_symps = input_data["confirmed_symptoms"]
            original_query_context = input_data["original_query_context"]

            with st.spinner("Thinking..."):
                 try:
                     # Call the chatbot's main processing function
                     response_text, ui_action, ui_payload = st.session_state.chatbot.process_user_query(
                          query_to_process, user_type,
                          confirmed_symptoms=confirmed_symps,
                          original_query_if_followup=original_query_context
                     )

                     logger.info("process_user_query returned ui_action: %s", ui_action)

                     if ui_action == "display_final_answer":
                          logger.info("UI Action: display_final_answer. Adding message.")
                          st.session_state.messages.append((response_text, False))
                          # Reset UI state back to input, clearing symptom UI if it was active
                          st.session_state.ui_state = {"step": "input", "payload": None}
                          logger.debug("UI state set to 'input'.")
                          # --- NEW: Clear original_query_for_symptom_rerun if this rerun came from symptom UI ---
                          # Check if the input payload that triggered *this* process_user_query call had confirmed_symptoms
                          if input_data.get("confirmed_symptoms") is not None:
                              st.session_state.original_query_for_symptom_rerun = None
                              logger.debug("Cleared original_query_for_symptom_rerun after symptom rerun finished.")


                     elif ui_action == "show_symptom_ui":
                          logger.info("UI Action: show_symptom_ui. Adding prompt message.")
                          st.session_state.messages.append((response_text, False)) # Add the prompt message
                          st.session_state.ui_state = {"step": "confirm_symptoms", "payload": ui_payload}
                          st.session_state.form_timestamp = datetime.now().timestamp()
                          logger.debug("UI state set to 'confirm_symptoms'.")
                          # --- NEW: Store original query for the symptom rerun ---
                          st.session_state.original_query_for_symptom_rerun = ui_payload.get("original_query")
                          logger.debug(f"Stored original_query_for_symptom_rerun: {st.session_state.original_query_for_symptom_rerun}")


                     elif ui_action == "none":
                          logger.info("UI Action: none. No message added.")
                          pass

                     else:
                          logger.error("Unknown ui_action returned: %s. Defaulting to input state.", ui_action)
                          st.session_state.messages.append((f"An internal error occurred (Unknown UI action).", False))
                          st.session_state.ui_state = {"step": "input", "payload": None}

                 except ValueError as e: # Catch ValueErrors raised by local_generate and its callers
                     logger.error(f"LLM/Processing Error during chatbot execution: {e}", exc_info=True)
                     st.session_state.messages.append((f"Sorry, an AI processing error occurred: {e}", False))
                     # Reset UI state to input on error
                     st.session_state.ui_state = {"step": "input", "payload": None}
                     # Also clear symptom rerun state if it was active
                     if input_data.get("confirmed_symptoms") is not None:
                          st.session_state.original_query_for_symptom_rerun = None
                          logger.debug("Cleared original_query_for_symptom_rerun after error.")


                 except Exception as e: # Catch any other unexpected errors
                     logger.error(f"Unexpected Error during chatbot process_user_query execution: {e}", exc_info=True)
                     st.session_state.messages.append((f"Sorry, an unexpected error occurred: {e}", False))
                     # Reset UI state to input on error
                     st.session_state.ui_state = {"step": "input", "payload": None}
                     # Also clear symptom rerun state if it was active
                     if input_data.get("confirmed_symptoms") is not None:
                          st.session_state.original_query_for_symptom_rerun = None
                          logger.debug("Cleared original_query_for_symptom_rerun after error.")


            logger.debug("Triggering rerun after processing_input_payload.")
            st.rerun()

        st.divider()
        if st.button("Reset Conversation", key="reset_conversation_button_main"):
            logger.info("Conversation reset triggered by user.")
            st.session_state.chatbot.reset_conversation()
            st.session_state.messages = []
            st.session_state.ui_state = {"step": "input", "payload": None}
            st.session_state.processing_input_payload = None
            st.session_state.form_timestamp = datetime.now().timestamp()
            # Clear symptom specific state variables on reset
            if 'confirmed_symptoms_from_ui' in st.session_state:
                del st.session_state.confirmed_symptoms_from_ui
                logger.debug("Cleared confirmed_symptoms_from_ui state.")
            if 'original_query_for_symptom_rerun' in st.session_state:
                del st.session_state.original_query_for_symptom_rerun
                logger.debug("Cleared original_query_for_symptom_rerun state.")


            logger.debug("Triggering rerun after reset.")
            st.rerun()

        st.divider()
        st.subheader("🩺 Detailed Feedback")
        with st.form("feedback_form"):
            feedback_text = st.text_area(
                "Enter corrections, improvements, or comments here...", height=100
            )
            submit_feedback_btn = st.form_submit_button("Submit Feedback")
            if submit_feedback_btn and feedback_text:
                logger.info("Detailed feedback submitted.")
                submit_feedback(feedback_text, st.session_state.messages, user_type) # Pass UI messages
                st.success("Thank you for your feedback!")

    with tab2:
        st.markdown("""
        ## Medical Chat Assistant

        **How it Works:**
        1.  **Medical Check:** Ensures your query is related to health or medicine.
        2.  **Symptom Analysis:** Identifies symptoms. If uncertain, may ask to confirm symptoms (UI).
        3.  **Information Retrieval:** Searches Knowledge Graph (KG) and documents (RAG).
        4.  **Context Selection:** Selects relevant info based on confidence (KG >0.8, RAG >0.7).
        5.  **Initial Answer:** Generates answer using selected context.
        6.  **Self-Reflection:** Evaluates answer completeness.
        7.  **Information Gap Filling:** If incomplete, gets missing info with sources.
        8.  **Final Answer:** Combines parts into a coherent response.

        **Disclaimer:** This system is informational only and not a substitute for professional medical advice. Always consult a healthcare provider.
        """)
    logger.debug("--- Streamlit App End of Rerun ---")


if __name__ == "__main__":
    main()
