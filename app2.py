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

# Import chain and memory components (Note: ConversationalRetrievalChain itself isn't directly used in this specific workflow logic, but its components are)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Import Neo4j components
from neo4j import GraphDatabase

# Ensure torch compatibility hint is still handled if needed, but often unnecessary now
try:
    import torch
except ImportError:
     pass

# Configuration
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY") # Ensure this is set in .env
NEO4J_URI = os.getenv("NEO4J_URI", "YOUR_NEO4J_URI") # Ensure this is set in .env
NEO4J_USER = os.getenv("NEO4J_USER", "YOUR_NEO4J_USER") # Ensure this is set in .env
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "YOUR_NEO4J_PASSWORD") # Ensure this is set in .env
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

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
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing SentenceTransformer embeddings on device: {device}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                cache_folder='./cache',
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            test_embedding = self.embedding_model.embed_query("test query")
            if test_embedding is not None and len(test_embedding) > 0:
                 logger.info("Embedding model initialized and tested successfully.")
            else:
                raise ValueError("Test embedding was empty.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Could not initialize embedding model: {e}")
            self.embedding_model = None

        self.llm: Optional[ChatGoogleGenerativeAI] = None

        self.kg_driver = None
        self.kg_connection_ok = False
        self._init_kg_connection()
        logger.info("DocumentChatBot initialization finished.")

    def _init_kg_connection(self):
        logger.info("Attempting to connect to Neo4j...")
        try:
            self.kg_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, connection_timeout=5.0)
            self.kg_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
            self.kg_connection_ok = True
        except Exception as e:
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
                logger.info(f"Loaded {len(loaded_pages)} pages from {loader._file_path}.") # Use _file_path for PyPDFLoader
            except Exception as e:
                logger.error(f"Error loading pages from PDF {loader._file_path}: {e}") # Use _file_path

        if not pages:
             logger.warning("No pages were loaded from the PDFs.")
             return None, "No pages were loaded from the PDFs."

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
                test_response = self.llm.invoke("Hello, are you ready?")
                if test_response.content:
                    logger.info("Successfully connected to Gemini Flash 1.5")
                    llm_init_message = "Gemini Flash 1.5 initialized."
                else:
                   raise ValueError("LLM test response was empty.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Flash 1.5: {e}")
                self.llm = None
                llm_init_message = f"Failed to initialize Gemini Flash 1.5: {str(e)}"

        vdb_message = "Vector database initialization skipped."
        if self.embedding_model is None:
             vdb_message = "Embedding model not initialized."
             self.vectordb = None
        else:
             self.vectordb, vdb_message = self.create_vectordb()

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
        if self.llm is None:
            logger.error("LLM is not initialized. Cannot generate.")
            raise ValueError("LLM is not initialized")
        try:
            response = self.llm.invoke(prompt)
            logger.debug("LLM generation successful.")
            return response.content
        except Exception as e:
            logger.error(f"Error generating with Gemini (configured LLM): {e}")
            # Fallback direct generation using genai library if configured LLM fails
            try:
                logger.warning("Attempting fallback LLM generation with genai.")
                model = genai.GenerativeModel('gemini-1.5-flash')
                result = model.generate_content(prompt)
                logger.debug("Fallback LLM generation successful.")
                return result.text
            except Exception as inner_e:
                logger.critical(f"CRITICAL ERROR: Error in fallback generation: {inner_e}")
                return "Error generating response. Please try again."

    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        logger.debug(f"Checking medical relevance for query: {query}")
        cache_key = {"type": "medical_relevance", "query": query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("Medical relevance check from cache.")
            return cached

        if self.llm is None:
             logger.warning("LLM not initialized. Falling back to keyword medical relevance check.")
             medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose"]
             result = (any(keyword in query.lower() for keyword in medical_keywords), "Fallback heuristic match")
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

        except Exception as e:
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
                    logger.warning("Could not parse symptom JSON from LLM response.")
            else:
                 logger.warning("Could not find 'Extracted Symptoms: [...]: in LLM response.")
        except Exception as e:
            logger.error(f"Error during symptom extraction LLM call: {e}")

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

        extracted_symptoms, symptom_confidence = self.extract_symptoms(query)
        if extracted_symptoms and symptom_confidence >= THRESHOLDS.get("symptom_extraction", 0.6):
            logger.debug("Query determined symptom-related based on confident extraction.")
            return set_cached(cache_key, True)

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
        except Exception as e:
            logger.error(f"Error during symptom query detection LLM call: {e}")

        # Fallback to symptom extraction result if LLM analysis fails
        if extracted_symptoms:
             logger.debug("Falling back to symptom extraction result for symptom query detection.")
             return set_cached(cache_key, True)

        # Ultimate fallback to keyword matching
        basic_health_terms = ["symptom", "pain", "sick", "fever", "headache"]
        result = any(term in query.lower() for term in basic_health_terms)
        logger.debug(f"Symptom query detection final fallback result: {result}")
        return set_cached(cache_key, result)


    def knowledge_graph_agent(self, user_query: str, all_symptoms: List[str]) -> Dict[str, Any]:
        logger.info("📚 Knowledge Graph Agent Initiated")
        kg_results: Dict[str, Any] = {
            "extracted_symptoms": all_symptoms,
            "identified_diseases_data": [],
            "top_disease_confidence": 0.0,
            "kg_matched_symptoms": [],
            "kg_treatments": [],
            "kg_treatment_confidence": 0.0,
            "kg_home_remedies": [],
            "kg_remedy_confidence": 0.0,
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
             return kg_results

        try:
            with self.kg_driver.session() as session:
                if all_symptoms:
                    logger.info(f"📚 KG Task: Identify Diseases from symptoms: {all_symptoms}")
                    disease_data_from_kg: List[Dict[str, Any]] = self._query_disease_from_symptoms_with_session(session, all_symptoms)

                    if disease_data_from_kg:
                        kg_results["identified_diseases_data"] = disease_data_from_kg
                        top_disease_record = disease_data_from_kg[0]
                        top_disease_name = top_disease_record.get("Disease")
                        top_disease_conf = float(top_disease_record.get("Confidence", 0.0))
                        kg_results["top_disease_confidence"] = top_disease_conf
                        kg_results["kg_matched_symptoms"] = top_disease_record.get("MatchedSymptoms", [])
                        logger.info(f"✔️ Diseases Identified: {[(d.get('Disease'), d.get('Confidence')) for d in disease_data_from_kg]} (Top Confidence: {top_disease_conf:.4f})")

                        if top_disease_conf >= THRESHOLDS.get("disease_matching", 0.5):
                            logger.info(f"📚 KG Tasks: Find Treatments & Remedies for {top_disease_name}")
                            kg_results["kg_treatments"], kg_results["kg_treatment_confidence"] = self._query_treatments_with_session(session, top_disease_name)
                            kg_results["kg_home_remedies"], kg_results["kg_remedy_confidence"] = self._query_home_remedies_with_session(session, top_disease_name)
                            logger.info(f"✔️ Treatments found: {kg_results['kg_treatments']} (Confidence: {kg_results['kg_treatment_confidence']:.4f})")
                            logger.info(f"✔️ Home Remedies found: {kg_results['kg_remedies']} (Confidence: {kg_results['kg_remedy_confidence']:.4f})")
                        else:
                            logger.info("📚 KG Tasks: Treatments/Remedies skipped - Top disease confidence below threshold.")
                else:
                     logger.info("📚 KG Task: Identify Diseases skipped - No symptoms provided.")

                kg_results["kg_content_diagnosis_data_for_llm"] = {
                      "disease_name": kg_results["identified_diseases_data"][0]["Disease"] if kg_results["identified_diseases_data"] else "an unidentifiable condition",
                      "symptoms_list": all_symptoms,
                      "confidence": kg_results["top_disease_confidence"]
                }

                other_parts: List[str] = []
                if kg_results["kg_treatments"]:
                     other_parts.append("## Recommended Treatments (from KG)")
                     other_parts.extend([f"- {t}" for t in kg_results["kg_treatments"]])
                     other_parts.append("")
                if kg_results["kg_home_remedies"]:
                     other_parts.append("## Home Remedies (from KG)")
                     other_parts.extend([f"- {h}" for h in kg_results["kg_home_remedies"]])
                     other_parts.append("")

                kg_results["kg_content_other"] = "\n".join(other_parts).strip()
                if not kg_results["kg_content_other"]:
                    kg_results["kg_content_other"] = "Medical Knowledge Graph did not find specific relevant information on treatments or remedies."

                logger.info("📚 Knowledge Graph Agent Finished successfully.")
                return kg_results

        except Exception as e:
            logger.error(f"⚠️ Error within KG Agent: {e}", exc_info=True)
            kg_results["kg_content_diagnosis_data_for_llm"] = {
                 "disease_name": "an unidentifiable condition",
                 "symptoms_list": all_symptoms,
                 "confidence": 0.0
            }
            kg_results["kg_content_other"] = f"An error occurred while querying the Medical Knowledge Graph: {str(e)}"
            kg_results["top_disease_confidence"] = 0.0
            return kg_results

    def _query_disease_from_symptoms_with_session(self, session, symptoms: List[str]) -> List[Dict[str, Any]]:
         logger.debug(f"Querying KG for diseases based on symptoms: {symptoms}")
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
              result = session.run(cypher_query, symptomNames=[s.lower() for s in symptoms if s])
              records = list(result)
              disease_data = [{"Disease": rec["Disease"], "Confidence": float(rec["Confidence"]), "MatchedSymptoms": rec["MatchedSymptoms"], "AllDiseaseSymptomsKG": rec["AllDiseaseSymptomsKG"]} for rec in records]
              logger.debug(f"🦠 Executed KG Disease Query, found {len(disease_data)} results.")
              return set_cached(cache_key, disease_data)
         except Exception as e:
              logger.error(f"⚠️ Error executing KG query for diseases: {e}")
              return []

    def _query_treatments_with_session(self, session, disease: str) -> Tuple[List[str], float]:
         logger.debug(f"Querying KG for treatments for disease: {disease}")
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
              result = session.run(cypher_query, diseaseName=disease)
              records = list(result)
              treatments = [(rec["Treatment"], float(rec["Confidence"])) for rec in records]
              treatments_list = [t[0] for t in treatments]
              avg_confidence = sum(t[1] for t in treatments) / len(treatments) if treatments else 0.0
              logger.debug(f"💊 Executed KG Treatment Query for {disease}, found {len(treatments_list)} treatments.")
              return set_cached(cache_key, (treatments_list, avg_confidence))
         except Exception as e:
              logger.error(f"⚠️ Error executing KG query for treatments: {e}")
              return [], 0.0

    def _query_home_remedies_with_session(self, session, disease: str) -> Tuple[List[str], float]:
         logger.debug(f"Querying KG for home remedies for disease: {disease}")
         if not disease:
             logger.debug("No disease provided for KG remedies query.")
             return [], 0.0
         cache_key = {"type": "remedy_query_kg", "disease": disease.lower()}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("KG home remedies query from cache.")
             return cached
         cypher_query = """
         MATCH (d:disease)-[r:HAS_HOMEREMEDY]->(h:homeremedy)
         WHERE toLower(d.Name) = toLower($diseaseName)
         RETURN h.Name as HomeRemedy,
                CASE WHEN COUNT(r) > 2 THEN 0.85 WHEN COUNT(r) > 1 THEN 0.75 ELSE 0.65 END as Confidence
         ORDER BY Confidence DESC
         """
         try:
             result = session.run(cypher_query, diseaseName=disease)
             records = list(result)
             remedies = [(rec["HomeRemedy"], float(rec["Confidence"])) for rec in records]
             remedies_list = [r[0] for r in remedies]
             avg_confidence = sum(r[1] for r in remedies) / len(remedies) if remedies else 0.0
             logger.debug(f"🏡 Executed KG Remedy Query for {disease}, found {len(remedies_list)} remedies.")
             return set_cached(cache_key, (remedies_list, avg_confidence))
         except Exception as e:
             logger.error(f"⚠️ Error executing KG query for home remedies: {e}")
             return [], 0.0

    def retrieve_rag_context(self, query: str) -> Tuple[List[str], float]:
        logger.info("📄 RAG Retrieval Initiated")
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
            retrieved_docs_with_scores = self.vectordb.similarity_search_with_score(query, k=k)
            logger.debug(f"📄 RAG: Retrieved {len(retrieved_docs_with_scores)} initial documents from vector DB.")

            relevant_chunks: List[str] = []
            relevant_scores: List[float] = []

            for doc, score in retrieved_docs_with_scores:
                similarity_score = max(0.0, 1 - score)
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
        logger.info(f"📦 Context Selection Initiated. Symptom Query: {is_symptom_query}, S_KG: {s_kg:.4f}, S_RAG: {s_rag:.4f}")
        kg_threshold = THRESHOLDS.get("kg_context_selection", 0.8)
        rag_threshold = THRESHOLDS.get("rag_context_selection", 0.7)
        logger.debug(f"Context selection thresholds: KG > {kg_threshold}, RAG > {rag_threshold}")
        selected_context: Optional[Dict[str, Any]] = None
        context_parts: List[str] = []

        if is_symptom_query:
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
            if s_rag > rag_threshold:
                logger.info("📦 Non-symptom query, RAG threshold met. Selecting RAG Only.")
                selected_context = {"rag": rag_chunks}
                context_parts.append("RAG")
            else:
                 logger.info("📦 Non-symptom query, RAG threshold not met. Selected Context: None.")
                 selected_context = None

        if selected_context is not None:
            logger.info(f"📦 Context Selection Final: Includes: {', '.join(context_parts)}.")
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
            logger.debug(f"Generating initial answer with context type: {'KG' if 'kg' in selected_context else ''} {'RAG' if 'rag' in selected_context else ''}".strip())
            context_type_description = "Based on the following information,"
            if "kg" in selected_context:
                kg_data = selected_context.get("kg", {})
                kg_info_str = "Knowledge Graph Information:\n"
                diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                if diag_data and diag_data.get("disease_name"):
                     confidence = diag_data.get("confidence", 0)
                     if confidence > THRESHOLDS.get("kg_context_selection", 0.8):
                          kg_info_str += f"- Identified Condition: {diag_data['disease_name']} (KG Confidence: {confidence:.2f})\n"
                     else:
                          kg_info_str += f"- Potential Condition: {diag_data['disease_name']} (KG Confidence: {confidence:.2f})\n"
                other_kg_content = kg_data.get("kg_content_other")
                if other_kg_content and other_kg_content.strip() and "Medical Knowledge Graph did not find" not in other_kg_content:
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
            elif "kg" in selected_context: # Only KG was selected
                 context_type_description = "Based on the following information from a medical knowledge graph, answer the user query. Only use the information provided here. Do not refer to external documents."

        if not context_info_for_prompt.strip() and selected_context is not None: # Check if context was passed but formatted into empty string
             logger.warning("Selected context was passed but formatted into an empty string for the prompt.")
             # Fallback to general knowledge prompt even if context was technically selected
             context_info_for_prompt = "No specific relevant information was effectively utilized from external knowledge sources."
             context_type_description = "Relying only on your vast general knowledge, answer the user query."


        prompt = f"""
{base_prompt_instructions}
{context_type_description}

{context_info_for_prompt}

User Query: {query}

Answer:
"""
        try:
            initial_answer = self.local_generate(prompt, max_tokens=1000)
            logger.info("🧠 Initial Answer Generated successfully.")
            return initial_answer
        except Exception as e:
            logger.error(f"⚠️ Error during initial answer generation: {e}", exc_info=True)
            return "Sorry, I encountered an error while trying to generate an initial answer."

    def reflect_on_answer(self,
                          query: str,
                          initial_answer: str,
                          selected_context: Optional[Dict[str, Any]]
                         ) -> Tuple[str, Optional[str]]:
        logger.info("🔍 Reflection and Evaluation Initiated")
        context_for_prompt = "None"
        if selected_context is not None:
            logger.debug(f"Reflection using context type: {'KG' if 'kg' in selected_context else ''} {'RAG' if 'rag' in selected_context else ''}".strip())
            context_for_prompt = "Provided Context:\n---\n"
            if "kg" in selected_context:
                kg_data = selected_context.get("kg", {})
                kg_info_str = "Knowledge Graph Info:\n"
                diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                if diag_data and diag_data.get("disease_name"):
                     kg_info_str += f"  Potential Condition: {diag_data['disease_name']} (Confidence: {diag_data.get('confidence', 0):.2f})\n"
                if kg_data.get("kg_treatments"):
                     kg_info_str += f"  Treatments: {', '.join(kg_data['kg_treatments'])}\n"
                if kg_data.get("kg_home_remedies"):
                     kg_info_str += f"  Home Remedies: {', '.join(kg_data['kg_home_remedies'])}\n"
                context_for_prompt += kg_info_str + "\n"
            if "rag" in selected_context:
                rag_chunks = selected_context.get("rag", [])
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
        try:
            response = self.local_generate(reflection_prompt, max_tokens=500)
            logger.debug(f"Reflection LLM raw response: {response}")
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
                         logger.warning(f"🔍 Reflection Result: Incomplete. Missing Info: {missing_info_description[:100]}...")
                except json.JSONDecodeError:
                    logger.error("⚠️ Reflection: Could not parse JSON from LLM response.")
            else:
                logger.warning("⚠️ Reflection: No JSON object found in LLM response.")
            result = (evaluation_result, missing_info_description)
            return set_cached(cache_key, result)
        except Exception as e:
            logger.error(f"⚠️ Error during reflection process: {e}", exc_info=True)
            return ('incomplete', f"An error occurred during reflection: {str(e)}")

    def get_supplementary_answer(self, query: str, missing_info_description: str) -> str:
        logger.info(f"🌐 External Agent (Gap Filling) Initiated. Missing Info: {missing_info_description[:100]}...")
        cache_key = {"type": "supplementary_answer", "missing_info_hash": abs(hash(missing_info_description)), "query_context_hash": abs(hash(query))}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Supplementary answer from cache.")
             return cached

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
        except Exception as e:
            logger.error(f"⚠️ Error during supplementary answer generation: {e}", exc_info=True)
            return f"Sorry, an error occurred while trying to find additional information about: '{missing_info_description[:50]}...'"

    def collate_answers(self, initial_answer: str, supplementary_answer: str) -> str:
        logger.info("✨ Final Answer Collation Initiated")
        cache_key = {"type": "final_collation", "initial_answer_hash": abs(hash(initial_answer)), "supplementary_answer_hash": abs(hash(supplementary_answer))}
        cached = get_cached(cache_key)
        if cached:
             logger.debug("Final collation from cache.")
             return cached

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
        except Exception as e:
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
        logger.info(f"--- Processing User Query: '{user_query}' ---")
        logger.info(f"User Type: {user_type}, Confirmed Symptoms Received: {len(confirmed_symptoms) if confirmed_symptoms is not None else 'None'}")

        processed_query = user_query # Query text used for retrieval and generation
        current_symptoms_for_retrieval: List[str] = [] # Symptoms used for KG
        is_symptom_query = False # Flag for symptom path

        # --- Handle Symptom Confirmation Rerun Logic ---
        if confirmed_symptoms is not None:
             logger.info("--- Step 6 (Rerun): Handling Symptom Confirmation Rerun ---")
             processed_query = original_query_if_followup # Use original query for processing
             is_symptom_query = True # Assume symptom path for UI follow-up
             current_symptoms_for_retrieval = confirmed_symptoms # Use the confirmed symptoms as the list for KG
             logger.info(f"Reprocessing '{processed_query[:50]}...' with confirmed symptoms: {current_symptoms_for_retrieval}")
             medical_check_ok = True # Assume valid as we were in a medical flow
             # Note: If you needed to combine original extracted symptoms with confirmed_symptoms,
             # you'd store original_extracted_symptoms in session state during the initial run
             # and combine them here. Using confirmed_symptoms directly for simplicity now.

        else: # Standard initial query processing
             logger.info("--- Initial Query Processing ---")
             # --- Step 2: Guardrail Check ---
             logger.info("--- Step 2: Guardrail Check ---")
             is_medical, medical_reason = self.is_medical_query(processed_query)
             if not is_medical:
                 logger.warning(f"Query flagged as non-medical: {medical_reason}. Workflow terminates.")
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
                  logger.info(f"Initial Symptom Extraction: {current_symptoms_for_retrieval} (Confidence: {symptom_extraction_confidence:.4f})")
                  logger.info(f"Is Symptom Query: {is_symptom_query}")
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
                  logger.info(f"KG Pipeline finished. S_KG: {s_kg:.4f}")

             # RAG Retrieval (for all medical queries)
             logger.info("Triggering RAG Pipeline.")
             rag_chunks, s_rag = self.retrieve_rag_context(processed_query)
             logger.info(f"RAG Pipeline finished. S_RAG: {s_rag:.4f}")

             # --- Step 5: Symptom Confirmation UI Decision Point (only after initial symptom retrieval) ---
             # Check if this is the *initial* query processing (not a rerun after UI) AND it's a symptom query
             if confirmed_symptoms is None and is_symptom_query:
                  logger.info("--- Step 5: Symptom Confirmation UI Check ---")
                  ui_trigger_threshold = THRESHOLDS.get("disease_symptom_followup_threshold", 0.8)
                  kg_found_diseases = len(kg_results.get("identified_diseases_data", [])) > 0

                  # Condition: Query is symptom-related AND KG found diseases AND KG confidence < threshold
                  if kg_found_diseases and s_kg < ui_trigger_threshold:
                       logger.info(f"KG confidence ({s_kg:.4f}) below UI trigger threshold ({ui_trigger_threshold:.4f}) and diseases found ({len(kg_results['identified_diseases_data'])}). Preparing symptom UI.")
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
             logger.error("Reached unexpected state: medical_check_ok was False but processing continued.")
             return "An internal error occurred during processing.", "display_final_answer", None


# Streamlit UI Component - Symptom Checklist Form
def display_symptom_checklist(symptom_options: Dict[str, List[str]], original_query: str):
    logger.debug("Rendering symptom checklist UI.")
    st.subheader("Confirm Your Symptoms")
    st.info(f"Based on your query: '{original_query}' and initial analysis, please confirm the symptoms you are experiencing from the list below to help narrow down possibilities.")

    form_key = f"symptom_confirmation_form_{abs(hash(original_query))}_{st.session_state.get('form_timestamp', datetime.now().timestamp())}"
    local_confirmed_symptoms_key = f'{form_key}_confirmed_symptoms_local'

    if local_confirmed_symptoms_key not in st.session_state:
        logger.debug(f"Initializing local symptom set for form {form_key}")
        st.session_state[local_confirmed_symptoms_key] = set()
    else:
        logger.debug(f"Using existing local symptom set for form {form_key} with {len(st.session_state[local_confirmed_symptoms_key])} items.")


    all_unique_symptoms = set()
    for disease_label, symptoms_list in symptom_options.items():
        if isinstance(symptoms_list, list):
            for symptom in symptoms_list:
                 if isinstance(symptom, str):
                      all_unique_symptoms.add(symptom.strip())

    sorted_all_symptoms = sorted(list(all_unique_symptoms))
    logger.debug(f"Total unique symptoms to suggest: {len(sorted_all_symptoms)}")

    with st.form(form_key):
        st.markdown("Please check all symptoms that apply to you from the list below:")
        if not sorted_all_symptoms:
            st.info("No specific symptoms were found for potential conditions. Use the box below to add symptoms.")
        else:
            cols = st.columns(4)
            for i, symptom in enumerate(sorted_all_symptoms):
                col = cols[i % 4]
                checkbox_key = f"{form_key}_checkbox_{symptom}"
                initial_state = symptom.strip().lower() in st.session_state[local_confirmed_symptoms_key]
                # Note: Streamlit's checkbox `value` is the default, it reads the current state.
                # We update the session state in the if/else block below based on the *return* of checkbox.
                # Checkbox returns True if checked, False if unchecked.
                is_checked = col.checkbox(symptom, key=checkbox_key, value=initial_state)

                if is_checked:
                     st.session_state[local_confirmed_symptoms_key].add(symptom.strip().lower())
                else:
                     # Only discard if it was previously in the set and is now unchecked
                     if symptom.strip().lower() in st.session_state[local_confirmed_symptoms_key]:
                         st.session_state[local_confirmed_symptoms_key].discard(symptom.strip().lower())


        st.markdown("**Other Symptoms (if any):**")
        other_symptoms_text = st.text_input("Enter additional symptoms here (comma-separated)", key=f"{form_key}_other_symptoms_input")
        if other_symptoms_text:
             other_symptoms_list = [s.strip().lower() for s in other_symptoms_text.split(',') if s.strip()]
             if other_symptoms_list:
                 logger.debug(f"Adding other symptoms from input: {other_symptoms_list}")
                 st.session_state[local_confirmed_symptoms_key].update(other_symptoms_list)

        submit_button = st.form_submit_button("Confirm and Continue")
        if submit_button:
            logger.info(f"Symptom confirmation form submitted. Final confirmed symptoms: {st.session_state[local_confirmed_symptoms_key]}")
            # Store the final confirmed symptoms in a session state variable the main loop looks for
            st.session_state.confirmed_symptoms_from_ui = sorted(list(st.session_state[local_confirmed_symptoms_key]))
            # Reset the UI state to 'input' so the chat input appears on the next rerun
            st.session_state.ui_state = {"step": "input", "payload": None}
            # Set a new timestamp for the next potential form display
            st.session_state.form_timestamp = datetime.now().timestamp()
            # The main loop will detect confirmed_symptoms_from_ui and trigger processing


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
    if 'init_status' not in st.session_state:
         logger.info("Checking chatbot initialization status.")
         with st.spinner("Initializing chat assistant..."):
              success, init_message = st.session_state.chatbot.initialize_qa_chain()
              st.session_state.init_status = (success, init_message)
         logger.info(f"Chatbot initialization complete. Status: {st.session_state.init_status}")


    # --- UI State Variables ---
    # Use a single state variable to control what the main input/action area shows
    if 'ui_state' not in st.session_state:
        logger.info("Initializing ui_state.")
        st.session_state.ui_state = {"step": "input", "payload": None} # step: "input", "confirm_symptoms"

    # Messages for display
    if 'messages' not in st.session_state:
        logger.info("Initializing messages state.")
        st.session_state.messages = [] # List of (content, is_user) tuples for UI display

    # Variable to hold the user query or confirmed symptoms that needs processing
    if 'processing_input_payload' not in st.session_state:
         logger.info("Initializing processing_input_payload state.")
         st.session_state.processing_input_payload = None # Dict like {"query": ..., "confirmed_symptoms": ..., "original_query_context": ...}

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
    else:
         st.sidebar.success(f"Initialization Status: {init_msg}")
         logger.info("Chatbot initialized successfully.")


    tab1, tab2 = st.tabs(["Chat", "About"])

    with tab1:
        st.subheader("Try these examples")
        examples = [
            "What are treatments for cough and cold?",
            "I have a headache and sore throat. What could it be?",
            "What home remedies help with flu symptoms?",
            "I have chest pain and shortness of breath. What could i do?"
        ]

        examples_disabled = not init_success or st.session_state.ui_state["step"] != "input"
        cols = st.columns(len(examples))
        for i, col in enumerate(cols):
            if col.button(examples[i], key=f"example_{i}", disabled=examples_disabled):
                logger.info(f"Example '{examples[i]}' clicked. Triggering processing.")
                # On example click, reset UI state and set processing_input_payload
                st.session_state.messages = [] # Clear previous conversation for example
                st.session_state.ui_state = {"step": "input", "payload": None} # Reset UI state
                st.session_state.chatbot.reset_conversation() # Reset backend state
                st.session_state.form_timestamp = datetime.now().timestamp() # New timestamp for any potential forms
                if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui # Ensure cleared

                st.session_state.processing_input_payload = {
                    "query": examples[i], "confirmed_symptoms": None, "original_query_context": None
                }
                st.rerun()

        # --- Chat Messages Display ---
        for i, (msg_content, is_user) in enumerate(st.session_state.messages):
            if is_user:
                with st.chat_message("user"): st.write(msg_content)
            else:
                with st.chat_message("assistant"):
                    st.write(msg_content)
                    # Add feedback buttons only to final answers (when input is enabled again)
                    is_final_answer_display = (i == len(st.session_state.messages) - 1) and (st.session_state.ui_state["step"] == "input")
                    if is_final_answer_display:
                        col = st.container()
                        with col:
                            feedback_key_up = f"thumbs_up_{i}_{abs(hash(msg_content))}"
                            feedback_key_down = f"thumbs_down_{i}_{abs(hash(msg_content))}"
                            b1, b2 = st.columns([0.05, 0.95])
                            with b1:
                                if st.button("👍", key=feedback_key_up):
                                     # Find the preceding user message for context
                                     user_msg_content = next((st.session_state.messages[j][0] for j in range(i - 1, -1, -1) if st.session_state.messages[j][1] is True), "")
                                     vote_message(user_msg_content, msg_content, "thumbs_up", user_type)
                                     st.toast("Feedback recorded: Thumbs Up!")
                            with b2:
                                if st.button("👎", key=feedback_key_down):
                                    user_msg_content = next((st.session_state.messages[j][0] for j in range(i - 1, -1, -1) if st.session_state.messages[j][1] is True), "")
                                    vote_message(user_msg_content, msg_content, "thumbs_down", user_type)
                                    st.toast("Feedback recorded: Thumbs Down!")

        input_area_container = st.container()
        st.write("  \n" * 5) # Add space at the end of the tab

        with input_area_container:
            if not init_success:
                 st.error("Chat assistant failed to initialize. Please check the logs and configuration.")
            elif st.session_state.ui_state["step"] == "confirm_symptoms":
                logger.debug("UI state is 'confirm_symptoms', displaying checklist.")
                ui_payload = st.session_state.ui_state.get("payload", {})
                display_symptom_checklist(
                     ui_payload.get("symptom_options", {}),
                     ui_payload.get("original_query", "")
                )
                st.chat_input("Confirm symptoms above...", disabled=True, key="disabled_chat_input")

            elif st.session_state.ui_state["step"] == "input":
                logger.debug("UI state is 'input', displaying chat input.")
                user_query = st.chat_input("Ask your medical question...", disabled=not init_success, key="main_chat_input")
                if user_query:
                    logger.info(f"Detected new chat input: '{user_query[:50]}...'. Triggering processing.")
                    st.session_state.messages.append((user_query, True))
                    # Reset backend state for a brand new conversation thread starting with this query
                    st.session_state.chatbot.reset_conversation()
                    st.session_state.form_timestamp = datetime.now().timestamp() # New timestamp for any potential forms
                    if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui # Ensure cleared

                    st.session_state.processing_input_payload = {
                        "query": user_query, "confirmed_symptoms": None, "original_query_context": None
                    }
                    st.rerun()

        # --- Check for Symptom Form Submission and Trigger Processing ---
        # This runs after the symptom form (if active) has processed its input in the current rerun.
        # The display_symptom_checklist function sets st.session_state.confirmed_symptoms_from_ui and updates ui_state on submit.
        if 'confirmed_symptoms_from_ui' in st.session_state and st.session_state.confirmed_symptoms_from_ui is not None:
             logger.info("Detected symptom confirmation form submission via state. Preparing processing payload.")
             confirmed_symps_to_pass = st.session_state.confirmed_symptoms_from_ui
             # original_query is stored in the ui_state payload *before* the form was displayed
             original_query_to_pass = st.session_state.ui_state.get("payload", {}).get("original_query", "")

             # Clear the temporary state variable used by the form submission
             del st.session_state.confirmed_symptoms_from_ui
             logger.debug("Cleared confirmed_symptoms_from_ui state.")

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
            # Clear the processing flag immediately to avoid infinite loops on subsequent reruns
            # if process_user_query itself causes a rerun (which it shouldn't explicitly unless needed for UI change).
            # The UI state update and subsequent rerun is how the flow is managed.
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

                     # --- Update UI state based on the action flag returned ---
                     logger.info(f"process_user_query returned ui_action: {ui_action}")

                     if ui_action == "display_final_answer":
                          logger.info("UI Action: display_final_answer.")
                          st.session_state.messages.append((response_text, False))
                          # Reset UI state back to input, clearing symptom UI if it was active
                          st.session_state.ui_state = {"step": "input", "payload": None}
                          logger.debug("UI state set to 'input'.")

                     elif ui_action == "show_symptom_ui":
                          logger.info("UI Action: show_symptom_ui.")
                          st.session_state.messages.append((response_text, False)) # Add the prompt message
                          # Update UI state to show the symptom checklist next rerun
                          st.session_state.ui_state = {"step": "confirm_symptoms", "payload": ui_payload}
                          st.session_state.form_timestamp = datetime.now().timestamp() # New timestamp for form key
                          logger.debug("UI state set to 'confirm_symptoms'.")


                     elif ui_action == "none":
                          logger.info("UI Action: none. No message added.")
                          pass # No response text to add, state change might have happened internally

                     else:
                          logger.error(f"Unknown ui_action returned: {ui_action}. Defaulting to input state.")
                          st.session_state.messages.append((f"An internal error occurred (Unknown UI action: {ui_action}).", False))
                          st.session_state.ui_state = {"step": "input", "payload": None}

                 except Exception as e:
                     logger.error(f"Error during chatbot process_user_query execution: {e}", exc_info=True)
                     st.session_state.messages.append((f"Sorry, an error occurred while processing your request: {e}", False))
                     # Reset UI state to input on error
                     st.session_state.ui_state = {"step": "input", "payload": None}

            # Force a rerun to update the UI based on state/messages
            logger.debug("Triggering rerun after processing_input_payload.")
            st.rerun()


        # --- Reset Conversation Button ---
        st.divider()
        if st.button("Reset Conversation", key="reset_conversation_button_main"):
            logger.info("Conversation reset triggered by user.")
            st.session_state.chatbot.reset_conversation()
            st.session_state.messages = []
            st.session_state.ui_state = {"step": "input", "payload": None}
            st.session_state.processing_input_payload = None
            st.session_state.form_timestamp = datetime.now().timestamp()
            # Clear symptom form specific state variable if it exists
            if 'confirmed_symptoms_from_ui' in st.session_state:
                del st.session_state.confirmed_symptoms_from_ui
                logger.debug("Cleared confirmed_symptoms_from_ui state.")

            logger.debug("Triggering rerun after reset.")
            st.rerun()


        # Physician feedback section
        st.divider()
        st.subheader("🩺 Detailed Feedback")
        with st.form("feedback_form"):
            feedback_text = st.text_area(
                "Enter corrections, improvements, or comments here...", height=100
            )
            submit_feedback_btn = st.form_submit_button("Submit Feedback")
            if submit_feedback_btn and feedback_text:
                logger.info("Detailed feedback submitted.")
                submit_feedback(feedback_text, st.session_state.messages, user_type)
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
