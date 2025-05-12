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

# Import Neo4j components
from neo4j import GraphDatabase

# Attempt basic torch import handling.
try:
    import torch
except ImportError:
     pass

try:
    import numpy as np
except ImportError:
    class DummyNumpy:
        def floating(self, *args, **kwargs): return float
    np = DummyNumpy()
    logging.warning("NumPy not found, some float checks might be less robust.")


# Configuration
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_PLACEHOLDER")
NEO4J_URI = os.getenv("NEO4J_URI", "YOUR_NEO4J_URI_PLACEHOLDER")
NEO4J_USER = os.getenv("NEO4J_USER", "YOUR_NEO4J_USER_PLACEHOLDER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "YOUR_NEO4J_PASSWORD_PLACEHOLDER")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_PLACEHOLDER":
    logger.critical("GEMINI_API_KEY environment variable is not set or is the placeholder value.")
    GEMINI_API_KEY = None
elif len(GEMINI_API_KEY) < 20: # Basic check
    logger.warning("GEMINI_API_KEY appears short, possibly invalid.")

if not NEO4J_URI or NEO4J_URI == "YOUR_NEO4J_URI_PLACEHOLDER" or \
   not NEO4J_USER or NEO4J_USER == "YOUR_NEO4J_USER_PLACEHOLDER" or \
   not NEO4J_PASSWORD or NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD_PLACEHOLDER":
    logger.critical("NEO4J environment variables are not fully set or are placeholder values.")
    NEO4J_URI = None # Prevent connection attempts
    NEO4J_USER = None
    NEO4J_PASSWORD = None

THRESHOLDS = {
    "symptom_extraction": 0.6, "disease_matching": 0.5,
    "disease_symptom_followup_threshold": 0.8, "kg_context_selection": 0.6,
    "rag_context_selection": 0.7, "medical_relevance": 0.6,
    "high_kg_context_only": 0.8
}

def get_image_as_base64(file_path):
    if not Path(file_path).is_file():
        logger.warning(f"Image file not found: {file_path}")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        logger.error(f"Error encoding image {file_path}: {e}")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

image_path = "Zoom My Life.jpg"
try:
    icon = get_image_as_base64(image_path)
except Exception: # Catch all for safety during init
    icon = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

CACHE = {}
def get_cached(key):
    try: key_str = json.dumps(key, sort_keys=True)
    except Exception: key_str = str(key) # Fallback for non-serializable keys
    return CACHE.get(key_str)

def set_cached(key, value):
    try: key_str = json.dumps(key, sort_keys=True)
    except Exception: key_str = str(key)
    CACHE[key_str] = value
    return value

HARDCODED_PDF_FILES = ["rawdata.pdf"]

def get_system_prompt(user_type):
    internal_user_type = "family" if user_type == "User / Family" else user_type
    if internal_user_type not in ["physician", "family"]:
        logger.warning(f"Unknown user type '{user_type}', defaulting to 'family'.")
        internal_user_type = "family"
    base_prompt = "You are MediAssist, an AI assistant specialized in medical information. "
    if internal_user_type == "physician":
        return base_prompt + (
            "Respond using professional medical terminology and consider offering differential diagnoses "
            "when appropriate. Provide detailed clinical insights and evidence-based recommendations. "
            "Use medical jargon freely, assuming high medical literacy. Cite specific guidelines or studies "
            "when possible. Structure your responses with clear clinical reasoning."
        )
    else: # family user
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

def vote_message(user_message: str, bot_message: str, vote: str, user_type: str):
    logger.info(f"Logging vote: {vote} for user_type: {user_type}")
    try:
        feedback_file = "feedback_log.csv"
        file_exists = os.path.isfile(feedback_file)
        with open(feedback_file, mode='a', newline='', encoding='utf-8') as file:
            fieldnames = ['timestamp', 'user_type', 'user_message', 'bot_message', 'vote']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            sanitized_bot_msg = bot_message.split("IMPORTANT MEDICAL DISCLAIMER:", 1)[0].strip()
            sanitized_bot_msg = re.sub(r"\n\n<span style='font-size: 0.8em; color: grey;'>.*</span>$", "", sanitized_bot_msg).strip().replace('||', '')
            sanitized_user_msg = user_message.replace('||', '')
            writer.writerow({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'user_type': user_type,
                'user_message': sanitized_user_msg.replace("\n", " "),
                'bot_message': sanitized_bot_msg.replace("\n", " "), 'vote': vote
            })
        logger.info(f"Feedback '{vote}' logged successfully.")
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")

def submit_feedback(feedback_text: str, conversation_history: List[Tuple[str, str]], user_type: str):
    logger.info(f"Logging detailed feedback for user_type: {user_type}")
    try:
        feedback_file = "detailed_feedback_log.csv"
        file_exists = os.path.isfile(feedback_file)
        with open(feedback_file, mode='a', newline='\n', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists: writer.writerow(['Timestamp', 'User Type', 'Feedback', 'Conversation History'])
            history_string_parts = []
            for u, b in conversation_history:
                sanitized_b = b.split("IMPORTANT MEDICAL DISCLAIMER:", 1)[0].strip()
                sanitized_b = re.sub(r"\n\n<span style='font-size: 0.8em; color: grey;'>.*</span>$", "", sanitized_b).strip()
                history_string_parts.append(f"User: {u.replace('||', '')} | Bot: {sanitized_b.replace('||', '')}")
            history_string = " ~~~ ".join(history_string_parts)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_type,
                feedback_text.replace('\n', ' ').replace('||', ''), history_string
            ])
        logger.info("Detailed feedback submitted successfully.")
    except Exception as e:
        logger.error(f"Error submitting detailed feedback: {e}")

class DocumentChatBot:
    def __init__(self):
        logger.info("DocumentChatBot initializing...")
        self.vectordb: Optional[FAISS] = None
        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        try:
            device = 'cuda' if 'torch' in globals() and torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing SentenceTransformer embeddings on device: {device}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name='pritamdeka/S-PubMedBert-MS-MARCO', cache_folder='./cache',
                model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True}
            )
            if self.embedding_model.embed_query("test query"): # Simple test
                 logger.info("Embedding model initialized and tested successfully.")
            else: logger.warning("Test embedding was empty, but embedding model object exists.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Could not initialize embedding model: {e}")
            self.embedding_model = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.kg_driver = None
        self.kg_connection_ok = False
        self._init_kg_connection()

    def _init_kg_connection(self):
        logger.info("Attempting to connect to Neo4j...")
        if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD: # Check if any are None or empty
             logger.error("Neo4j credentials missing or are placeholder values. Cannot connect.")
             self.kg_connection_ok = False; return
        try:
            self.kg_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, connection_timeout=10.0)
            self.kg_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
            self.kg_connection_ok = True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}. KG features will be unavailable.")
            self.kg_connection_ok = False
    
    def enhance_with_triage_detection(self, query: str, response_content: str, user_type: str) -> str:
        # Only enhance if user is a family user
        if user_type not in ["User / Family", "family"]:
            # logger.debug(f"Triage detection skipped for user type: {user_type}")
            return response_content

        # Avoid re-triaging if triage info is already explicitly in the response
        if "TRIAGE ASSESSMENT:" in response_content.upper() or \
           any(cat in response_content for cat in ["1. Emergency", "2. Urgent Care", "3. Primary Care", "4. Self-care"]):
             # logger.debug("Triage detection skipped, response already contains triage info.")
             return response_content

        logger.debug(f"Attempting triage detection for family user. Query: {query[:50]}..., Response: {response_content[:100]}...")
        triage_prompt = (
            f"You are an AI assistant analyzing a medical conversation. "
            f"Determine if the user's query, given the assistant's response, represents a triage situation.\n\n"
            f"QUERY: {query}\n\n"
            f"ASSISTANT RESPONSE CONTENT: {response_content}\n\n"
            "If this interaction suggests a need for urgency or specific care steps, classify it according to these categories based on the *most critical* implied level:\n"
            "1. Emergency (Call 911/Emergency Services immediately)\n2. Urgent Care (See a doctor within 24 hours)\n"
            "3. Primary Care (Schedule a regular appointment)\n4. Self-care (Can be managed at home)\n\n"
            "Provide ONLY the triage category number and title (e.g., '1. Emergency') followed by a brief, one-sentence explanation of *why* this category is suggested. "
            "If it's clearly NOT a triage situation, respond with ONLY 'NO_TRIAGE_NEEDED'. Ensure your response is very concise (max 50 words total)."
        )
        cache_key = {"type": "triage_detection", "query": query, "response": response_content}
        cached = get_cached(cache_key)
        if cached is not None:
             # logger.debug("Triage detection from cache.")
             triage_text = cached
             if triage_text != "NO_TRIAGE_NEEDED": return f"{response_content}\n\n**TRIAGE ASSESSMENT:**\n{triage_text}"
             return response_content
        try:
            triage_analysis = self.local_generate(triage_prompt, max_tokens=100)
            triage_text = triage_analysis.strip()
            # logger.debug(f"Triage detection LLM raw response: {triage_text}")
            set_cached(cache_key, triage_text)
            if "NO_TRIAGE_NEEDED" not in triage_text:
                # logger.info("Triage detection: Enhanced response with assessment.")
                return f"{response_content}\n\n**TRIAGE ASSESSMENT:**\n{triage_text}"
            # logger.debug("Triage detection: NO_TRIAGE_NEEDED detected.")
            return response_content
        except Exception as e: # Catch ValueError from local_generate and other errors
            logger.error(f"Error in triage detection: {e}")
            return response_content # Return original content on error

    def create_vectordb(self):
        logger.info("Creating vector database...")
        pdf_files = [Path(pdf_file) for pdf_file in HARDCODED_PDF_FILES if Path(pdf_file).is_file()]
        if not pdf_files: return None, "No PDF files found."
        
        loaders = []
        for pdf_file_path in pdf_files:
            try: loaders.append(PyPDFLoader(str(pdf_file_path)))
            except Exception as e: logger.error(f"Error creating loader for {pdf_file_path}: {e}")
        if not loaders: return None, "No valid PDF loaders."

        pages = []
        for loader in loaders:
            try: pages.extend(loader.load())
            except Exception as e: logger.error(f"Error loading from {getattr(loader, 'file_path', 'unknown PDF')}: {e}")
        if not pages: return None, "No pages loaded from PDFs."

        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(pages)
        if not splits: return None, "No text chunks created from PDF pages."
        if not self.embedding_model: return None, "Embedding model not initialized for VDB."
        try:
            vectordb = FAISS.from_documents(splits, self.embedding_model)
            logger.info("FAISS vectorstore created successfully.")
            return vectordb, "Vector database created successfully."
        except Exception as e:
            logger.error(f"Error creating FAISS vector database: {e}")
            return None, f"Failed to create vector database: {str(e)}"

    def initialize_qa_chain(self):
        logger.info("Initializing QA chain components (LLM, Vector DB)...")
        llm_init_message = "LLM initialization skipped."
        if GEMINI_API_KEY: # Check if key exists and is not placeholder
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY,
                    temperature=0.3, top_p=0.95, top_k=40, convert_system_message_to_human=True
                )
                # Test LLM
                test_response = self.llm.invoke("Hello, are you ready?", config={"max_output_tokens": 10})
                if test_response and test_response.content:
                    llm_init_message = "Gemini Flash 1.5 initialized."
                else:
                   llm_init_message = "Gemini Flash 1.5 initialized, but test response was empty."
            except Exception as e:
                 self.llm = None # Ensure LLM is None on failure
                 llm_init_message = f"Gemini LLM initialization/test failed: {e}"
        else:
            llm_init_message = "Gemini API key not found or invalid."

        vdb_message = "Vector database initialization skipped."
        if self.embedding_model:
             if any(Path(pdf_file).is_file() for pdf_file in HARDCODED_PDF_FILES):
                self.vectordb, vdb_message = self.create_vectordb()
             else:
                  vdb_message = "No PDF files found to create vector database."
        else:
             vdb_message = "Embedding model not initialized, skipping VDB creation."
        
        status_parts = [
            "LLM OK" if self.llm else "LLM Failed",
            "Embeddings OK" if self.embedding_model else "Embeddings Failed",
            "Vector DB OK" if self.vectordb else "Vector DB Failed",
            "KG OK" if self.kg_connection_ok else "KG Failed"
        ]
        overall_message = f"{llm_init_message}. {vdb_message}. KG Status: {'Connected' if self.kg_connection_ok else 'Failed'}. Overall: {', '.join(status_parts)}."
        overall_success = bool(self.llm) # LLM is minimal requirement for useful response
        logger.info(f"Initialization Result: Success={overall_success}, Message='{overall_message}'")
        return overall_success, overall_message

    def local_generate(self, prompt, max_tokens=500):
        if not self.llm: raise ValueError("LLM is not initialized. Cannot generate response.")
        try:
            response = self.llm.invoke(prompt, config={"max_output_tokens": max_tokens})
            if response and response.content: return response.content
            raise ValueError("LLM returned empty response.") # Treat empty as failure
        except Exception as e: raise ValueError(f"LLM generation failed: {e}") from e # Re-raise with context

    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        cache_key = {"type": "medical_relevance", "query": query}
        if cached := get_cached(cache_key): return cached
        
        medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose", "medicine", "doctor", "hospital", "clinic", "condition", "illness", "sick", "diagnosed"]
        if not self.llm: # LLM unavailable fallback
            return set_cached(cache_key, (any(k in query.lower() for k in medical_keywords), "Fallback heuristic (LLM unavailable)"))

        prompt = f'''Analyze if the query is health/medical. JSON: {{"is_medical": boolean, "confidence": float, "reasoning": "explanation"}}. Query: "{query}"'''
        try:
            response = self.local_generate(prompt, max_tokens=150)
            if match := re.search(r'\{[\s\S]*\}', response):
                data = json.loads(match.group(0))
                is_medical_llm = data.get("is_medical", False)
                confidence_raw = data.get("confidence", "0.0") # Default to string "0.0"
                try:
                    confidence = float(confidence_raw if confidence_raw is not None else "0.0")
                except (ValueError, TypeError):
                    logger.warning(f"Medical relevance: could not convert confidence '{confidence_raw}' to float. Defaulting to 0.0.")
                    confidence = 0.0
                
                final_is_medical = is_medical_llm and confidence >= THRESHOLDS.get("medical_relevance", 0.6)
                return set_cached(cache_key, (final_is_medical, data.get("reasoning", "")))
        except Exception as e: # Catch LLM call errors or JSON parsing errors
            logger.warning(f"Medical relevance LLM call or JSON processing failed: {e}.")
        
        # Fallback if LLM/JSON processing fails
        return set_cached(cache_key, (any(k in query.lower() for k in medical_keywords), "Fallback heuristic (LLM/JSON processing failed)"))

    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        cache_key = {"type": "symptom_extraction", "query": user_query}
        if cached := get_cached(cache_key): return list(cached[0]), cached[1]
        
        common_keywords = ["fever", "cough", "headache", "sore throat", "nausea", "dizziness", "chest pain", "shortness of breath", "fatigue", "body aches", "runny nose", "congestion", "chills", "sweats", "joint pain", "muscle aches", "rash", "swelling", "pain", "ache", "burning", "itching", "numbness", "tingling", "diarrhea", "vomiting"] # Truncated for example
        query_lower = user_query.lower()
        fallback_symptoms = sorted(list(set(s.strip().lower() for s in common_keywords if s.strip().lower() in query_lower)))
        
        if not self.llm:
            return set_cached(cache_key, (fallback_symptoms, 0.4 if fallback_symptoms else 0.0))

        SYMPTOM_PROMPT = f'''Extract medical symptoms from query. JSON: {{"Extracted Symptoms": [{{"symptom": "symptom1", "confidence": 0.9}}, ...]}}. Query: "{user_query}"'''
        llm_symptoms_lower, llm_avg_confidence, llm_found_any = [], 0.0, False
        try:
            response = self.local_generate(SYMPTOM_PROMPT, max_tokens=500).strip()
            if json_match := re.search(r'\{[\s\S]*\}', response):
                data = json.loads(json_match.group(0))
                symptom_data = data.get("Extracted Symptoms", [])
                if symptom_data: llm_found_any = True

                llm_symptoms_confident_items = []
                all_numeric_llm_confidences = []

                for item in symptom_data:
                    if isinstance(item, dict) and "symptom" in item and isinstance(item.get("symptom"), str) and item.get("symptom").strip():
                        try:
                            raw_confidence = item.get("confidence", "0.0") # Default to string "0.0"
                            item_confidence_float = float(raw_confidence if raw_confidence is not None else "0.0")
                            all_numeric_llm_confidences.append(item_confidence_float)
                            if item_confidence_float >= THRESHOLDS.get("symptom_extraction", 0.6):
                                llm_symptoms_confident_items.append(item)
                        except (ValueError, TypeError):
                            logger.warning(f"Symptom extraction: Could not convert confidence '{item.get('confidence')}' to float. Skipping.")
                
                llm_symptoms_lower = sorted(list(set(item["symptom"].strip().lower() for item in llm_symptoms_confident_items)))
                if all_numeric_llm_confidences:
                    llm_avg_confidence = sum(all_numeric_llm_confidences) / len(all_numeric_llm_confidences)
            else: logger.warning("No JSON in symptom extraction LLM response.")
        except Exception as e: logger.error(f"Symptom extraction LLM call/parse error: {e}")

        combined_symptom_set_lower = set(llm_symptoms_lower)
        combined_symptom_set_lower.update(fallback_symptoms)
        final_symptoms_lower = sorted(list(combined_symptom_set_lower))

        final_confidence = llm_avg_confidence if llm_found_any and all_numeric_llm_confidences else (0.4 if fallback_symptoms_lower else 0.0)
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        logger.info(f"Final extracted symptoms: {final_symptoms_lower} (Confidence: {final_confidence:.4f})")
        return set_cached(cache_key, (final_symptoms_lower, final_confidence))

    def is_symptom_related_query(self, query: str) -> bool:
        if not query or not query.strip(): return False
        cache_key = {"type": "symptom_query_detection_heuristic", "query": query}
        if cached := get_cached(cache_key): return cached
        
        extracted_symptoms, symptom_confidence = self.extract_symptoms(query)
        if extracted_symptoms and symptom_confidence >= THRESHOLDS.get("symptom_extraction", 0.6):
            return set_cached(cache_key, True)

        health_keywords = ["symptom", "pain", "sick", "health", "disease", "condition", "diagnosis", "feel", "experiencing", "ache", "diagnosed"]
        result = any(keyword in query.lower() for keyword in health_keywords)
        return set_cached(cache_key, result)

    def knowledge_graph_agent(self, user_query: str, symptoms_for_kg: List[str]) -> Dict[str, Any]:
        logger.info("📚 KG Agent Initiated for query: %s...", user_query[:50])
        # logger.debug("Input symptoms for KG: %s", symptoms_for_kg)

        kg_results_template = lambda s_list: {
            "extracted_symptoms": s_list, "identified_diseases_data": [], "top_disease_confidence": 0.0,
            "kg_matched_symptoms": [], "all_disease_symptoms_kg_for_top_disease": [],
            "kg_treatments": [], "kg_treatment_confidence": 0.0,
            "kg_content_diagnosis_data_for_llm": {"disease_name": "an unidentifiable condition", "symptoms_list": s_list, "confidence": 0.0},
            "kg_content_other": "Medical Knowledge Graph information is unavailable."
        }
        valid_symptom_names = [s.strip() for s in symptoms_for_kg if isinstance(s, str) and s.strip()]
        if not valid_symptom_names:
             logger.info("📚 KG Agent: No valid symptoms provided.")
             return kg_results_template(symptoms_for_kg) # Return template with original list

        kg_results = kg_results_template(valid_symptom_names) # Use valid names for the template's symptom list

        if not self.kg_connection_ok or not self.kg_driver:
             logger.warning("📚 KG Agent: Connection not OK.")
             return kg_results # Already has appropriate default message

        try:
            with self.kg_driver.session(connection_acquisition_timeout=10.0) as session:
                disease_data_from_kg = self._query_disease_from_symptoms_with_session(session, valid_symptom_names)
                
                if disease_data_from_kg:
                    # Ensure Confidence is float and sort
                    for item in disease_data_from_kg: item['Confidence'] = float(item.get('Confidence', 0.0))
                    sorted_disease_data = sorted(disease_data_from_kg, key=lambda x: x.get("Confidence", 0.0), reverse=True)
                    
                    if sorted_disease_data: # Check if list is not empty after potential filtering/sorting
                        kg_results["identified_diseases_data"] = sorted_disease_data
                        top_disease_record = sorted_disease_data[0]
                        top_disease_name = top_disease_record.get("Disease")
                        top_disease_conf = top_disease_record.get("Confidence", 0.0) # Already float
                        kg_results["top_disease_confidence"] = top_disease_conf
                        kg_results["kg_matched_symptoms"] = list(top_disease_record.get("MatchedSymptoms", []))
                        kg_results["all_disease_symptoms_kg_for_top_disease"] = list(top_disease_record.get("AllDiseaseSymptomsKG", []))

                        if top_disease_conf >= THRESHOLDS.get("disease_matching", 0.5) and top_disease_name:
                            treatments, treat_conf = self._query_treatments_with_session(session, top_disease_name)
                            kg_results["kg_treatments"], kg_results["kg_treatment_confidence"] = treatments, treat_conf
                
                top_name_for_llm = kg_results["identified_diseases_data"][0].get("Disease") if kg_results["identified_diseases_data"] else "an unidentifiable condition"
                kg_results["kg_content_diagnosis_data_for_llm"].update({
                    "disease_name": top_name_for_llm, 
                    "symptoms_list": valid_symptom_names, # Use the validated input symptoms
                    "confidence": kg_results["top_disease_confidence"] # This is already a float
                })
                
                other_parts = []
                if kg_results["kg_treatments"]:
                     other_parts.append("## Recommended Treatments (from KG)")
                     other_parts.extend([f"- {t}" for t in kg_results["kg_treatments"]])
                kg_results["kg_content_other"] = "\n".join(other_parts).strip() or "Medical KG did not find specific relevant treatment information."
            logger.info("📚 Knowledge Graph Agent Finished successfully.")
        except Exception as e:
            logger.error("⚠️ Error within KG Agent: %s", e, exc_info=True)
            # Return the template, but ensure its symptom list reflects the validated input
            return kg_results_template(valid_symptom_names) 
        return kg_results

    def _query_disease_from_symptoms_with_session(self, session, symptoms: List[str]) -> List[Dict[str, Any]]:
        # logger.debug("Querying KG for diseases based on symptoms: %s", symptoms)
        symptom_names_lower = [s.lower() for s in symptoms]
        cache_key = {"type": "disease_matching_v2", "symptoms": tuple(sorted(symptom_names_lower))}
        if cached := get_cached(cache_key):
            # logger.debug("Disease match query from cache.")
            return [dict(item, MatchedSymptoms=list(item['MatchedSymptoms']), AllDiseaseSymptomsKG=list(item['AllDiseaseSymptomsKG']), Confidence=float(item['Confidence'])) for item in cached]

        cypher_query = """
        UNWIND $symptomNamesLower AS input_symptom_name_lower
        MATCH (s:symptom) WHERE toLower(s.Name) = input_symptom_name_lower
        MATCH (s)-[:INDICATES]->(d:disease)
        WITH d, COLLECT(DISTINCT s.Name) AS matched_symptoms_from_input_in_kg_case
        OPTIONAL MATCH (d)<-[:INDICATES]-(all_s:symptom)
        WITH d, matched_symptoms_from_input_in_kg_case,
             COLLECT(DISTINCT all_s.Name) AS all_disease_symptoms_in_kg,
             size(COLLECT(DISTINCT all_s)) AS total_disease_symptoms_count,
             size(matched_symptoms_from_input_in_kg_case) AS matching_symptoms_count
        WHERE matching_symptoms_count > 0
        RETURN d.Name AS Disease,
               CASE WHEN total_disease_symptoms_count = 0 THEN 0.0 ELSE matching_symptoms_count * 1.0 / total_disease_symptoms_count END AS confidence_score,
               matched_symptoms_from_input_in_kg_case AS MatchedSymptoms,
               all_disease_symptoms_in_kg AS AllDiseaseSymptomsKG
        ORDER BY confidence_score DESC LIMIT 5
        """
        try:
            # logger.debug("Executing Cypher query for diseases with symptoms: %s", symptom_names_lower)
            result = session.run(cypher_query, symptomNamesLower=symptom_names_lower)
            disease_data = []
            for rec in result:
                if disease_name := rec.get("Disease"):
                    disease_data.append({
                        "Disease": disease_name, 
                        "Confidence": float(rec.get("confidence_score", 0.0)), # Ensure float here
                        "MatchedSymptoms": list(rec.get("MatchedSymptoms", [])),
                        "AllDiseaseSymptomsKG": list(rec.get("AllDiseaseSymptomsKG", []))
                    })
            # logger.debug("🦠 Executed KG Disease Query, found %d results.", len(disease_data))
            # Cache with tuples for list-like fields
            set_cached(cache_key, [dict(d, MatchedSymptoms=tuple(d['MatchedSymptoms']), AllDiseaseSymptomsKG=tuple(d['AllDiseaseSymptomsKG'])) for d in disease_data])
            return disease_data
        except Exception as e:
            logger.error(f"Error querying diseases from symptoms: {e}")
            return []

    def _query_treatments_with_session(self, session, disease: str) -> Tuple[List[str], float]:
        # logger.debug("Querying KG for treatments for disease: %s", disease)
        disease_name_lower = disease.strip().lower()
        cache_key = {"type": "treatment_query_kg", "disease": disease_name_lower}
        if cached := get_cached(cache_key):
            # logger.debug("KG treatments query from cache.")
            # Ensure float when retrieving from cache
            if isinstance(cached, tuple) and len(cached) == 2: return list(cached[0]), float(cached[1])
            return cached # Should not happen if set correctly
        
        cypher_query = """
        MATCH (d:disease)-[:TREATED_BY]->(t:treatment) WHERE toLower(d.Name) = $diseaseNameLower
        RETURN t.Name as Treatment ORDER BY Treatment
        """
        try:
            result = session.run(cypher_query, diseaseNameLower=disease_name_lower)
            treatments_list = sorted(list(set(
                rec["Treatment"].strip() for rec in result if rec.get("Treatment") and isinstance(rec.get("Treatment"), str) and rec.get("Treatment").strip()
            )))
            avg_confidence = 1.0 if treatments_list else 0.0 # Confidence in the link
            # logger.debug("💊 Executed KG Treatment Query for %s, found %d treatments.", disease, len(treatments_list))
            return set_cached(cache_key, (treatments_list, avg_confidence))
        except Exception as e:
            logger.error("⚠️ Error executing KG query for treatments: %s", e, exc_info=True)
            return [], 0.0

    def retrieve_rag_context(self, query: str) -> Tuple[List[str], float]:
        logger.info(f"📄 RAG Retrieval Initiated for query: {query[:50]}...")
        k = 5 
        cache_key = {"type": "rag_retrieval_topk_chunks_and_scores", "query": query, "k": k}
        if cached := get_cached(cache_key):
             # logger.debug(f"RAG retrieval (top {k} chunks and scores) from cache.")
             return list(cached['chunks']), float(cached['avg_score']) # Ensure float
        if not self.vectordb or not self.embedding_model:
            logger.warning("📄 RAG: VDB or Embedding model not initialized.")
            return [], 0.0
        try:
            retrieved_docs_with_scores = self.vectordb.similarity_search_with_score(query, k=k)
            top_k_chunks_content, top_k_similarity_scores = [], []
            for doc, score_val in retrieved_docs_with_scores:
                if isinstance(score_val, (int, float)): # Check if score_val is numeric
                    raw_score_float = float(score_val)
                    similarity_score = max(0.0, min(1.0, 1 - (raw_score_float if raw_score_float >=0 else 0.0) ))
                    if doc and doc.page_content:
                        top_k_chunks_content.append(doc.page_content)
                        top_k_similarity_scores.append(similarity_score)
            srag = sum(top_k_similarity_scores) / len(top_k_similarity_scores) if top_k_similarity_scores else 0.0
            logger.info(f"📄 RAG Finished. {len(top_k_chunks_content)} chunks. S_RAG: {srag:.4f}")
            return set_cached(cache_key, {'chunks': top_k_chunks_content, 'avg_score': srag})
        except Exception as e:
            logger.error(f"⚠️ Error during RAG retrieval: {e}", exc_info=True)
            return [], 0.0
            
    def select_context(self, kg_results, s_kg, rag_chunks, s_rag, is_symptom_query) -> Optional[Dict[str, Any]]:
        logger.info("📦 Context Selection. SymptomQ: %s, S_KG: %.4f, S_RAG: %.4f", is_symptom_query, s_kg, s_rag)
        kg_thresh = THRESHOLDS.get("kg_context_selection", 0.6)
        rag_thresh = THRESHOLDS.get("rag_context_selection", 0.7)
        high_kg_thresh = THRESHOLDS.get("high_kg_context_only", 0.8)
        selected = {}
        
        # Ensure s_kg and s_rag are floats before comparison
        s_kg_float = float(s_kg) if isinstance(s_kg, (int, float, str)) else 0.0
        s_rag_float = float(s_rag) if isinstance(s_rag, (int, float, str)) else 0.0

        kg_has_data = kg_results and kg_results.get("identified_diseases_data")

        if is_symptom_query and s_kg_float > high_kg_thresh and kg_has_data:
            # logger.info("📦 Applying High KG Confidence Rule. Selecting KG ONLY.")
            selected["kg"] = kg_results
        else:
            if is_symptom_query:
                if s_kg_float >= kg_thresh and kg_has_data:
                    selected["kg"] = kg_results
                    # logger.debug("KG meets standard threshold and has data. KG selected.")
            if s_rag_float >= rag_thresh and rag_chunks:
                selected["rag"] = rag_chunks
                # logger.debug("RAG meets threshold and chunks exist. RAG selected.")
        
        if not selected: logger.info("📦 Context Selection: No context source met thresholds."); return None
        # logger.info("📦 Context Selection Final: Includes: %s.", ', '.join(selected.keys()))
        return selected

    def generate_initial_answer(self, query: str, selected_context: Optional[Dict[str, Any]], user_type: str) -> str:
        # logger.info("🧠 Initial Answer Generation Initiated")
        cache_key = {"type": "initial_answer", "query": query, "user_type": user_type, "context_hash": abs(hash(json.dumps(selected_context, sort_keys=True)))}
        if cached := get_cached(cache_key): 
            # logger.debug("Initial answer from cache.")
            return cached

        base_prompt_instructions = get_system_prompt(user_type)
        context_info_for_prompt, context_type_description = "", ""
        
        context_parts_for_prompt = []
        if selected_context:
            if "kg" in selected_context:
                kg_data = selected_context["kg"]
                kg_info_str = "Knowledge Graph Information:\n"
                diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                # Ensure confidence is float for comparison
                diag_confidence = float(diag_data.get("confidence", 0.0)) if diag_data else 0.0

                if diag_data and diag_confidence > 0.0:
                    disease_name = diag_data.get("disease_name", "an unidentifiable condition")
                    # Phrasing based on confidence thresholds
                    if diag_confidence > THRESHOLDS["high_kg_context_only"]: kg_info_str += f"- **Highly Probable Condition:** {disease_name} (KG Conf: {diag_confidence:.2f})\n"
                    elif diag_confidence > THRESHOLDS["kg_context_selection"]: kg_info_str += f"- **Potential Condition:** {disease_name} (KG Conf: {diag_confidence:.2f})\n"
                    elif diag_confidence > THRESHOLDS["disease_matching"]: kg_info_str += f"- **Possible Condition:** {disease_name} (KG Conf: {diag_confidence:.2f})\n"
                    else: kg_info_str += f"- Possible Condition (low match): {disease_name} (KG Conf: {diag_confidence:.2f})\n"
                    if kg_data.get('kg_matched_symptoms'): kg_info_str += f"- Relevant Symptoms (matched in KG): {', '.join(kg_data['kg_matched_symptoms'])}\n"
                
                other_kg_content = kg_data.get("kg_content_other", "")
                if other_kg_content and "did not find" not in other_kg_content: kg_info_str += "\n" + other_kg_content
                if len(kg_info_str.splitlines()) > 1 or (diag_data and diag_confidence > 0.0):
                    context_parts_for_prompt.append(kg_info_str.strip())

            if "rag" in selected_context and selected_context["rag"]:
                context_parts_for_prompt.append("Relevant Passages from Documents:\n---\n" + "\n---\n".join(selected_context["rag"][:3]) + "\n---")

        if not selected_context or not context_parts_for_prompt:
            context_type_description = "You have not been provided with specific external medical knowledge. Generate only a minimal placeholder answer (e.g., 'No specific relevant information was found'). Do NOT answer using general knowledge."
            prompt_for_initial_answer = f"{base_prompt_instructions.strip()}\n\n{context_type_description}\n\nUser Query: \"{query}\"\n\nMinimal Placeholder Answer:"
        else:
            context_info_for_prompt = "\n\n".join(context_parts_for_prompt)
            ctx_desc_key = "_".join(sorted(k for k in selected_context if context_parts_for_prompt)) # Only consider keys that contributed
            desc_map = {"kg_rag": "Based on KG and RAG docs...", "kg": "Based on KG...", "rag": "Based on RAG docs..."}
            context_type_description = desc_map.get(ctx_desc_key, "Based on available information...")
            prompt_for_initial_answer = f"{base_prompt_instructions.strip()}\n{context_type_description}\n\n{context_info_for_prompt}\n\nUser Query: {query}\n\nAnswer:"
        
        try:
            initial_answer = self.local_generate(prompt_for_initial_answer, max_tokens=1000)
            placeholder_frags = ["no specific relevant information was found", "lack of specific information"]
            is_placeholder = not initial_answer.strip() or any(f in initial_answer.lower() for f in placeholder_frags)

            if (selected_context and context_parts_for_prompt) and is_placeholder:
                initial_answer = "No specific relevant information was found in external knowledge sources."
                logger.warning(f"Overriding unexpected placeholder with consistent one: {initial_answer}")
            elif (not selected_context or not context_parts_for_prompt) and not is_placeholder:
                initial_answer = "No specific relevant information was found in external knowledge sources."
                logger.warning(f"Overriding LLM non-placeholder due to no context: {initial_answer}")
            return set_cached(cache_key, initial_answer)
        except ValueError as e:
            logger.error(f"⚠️ Error during initial answer generation: {e}", exc_info=True)
            raise ValueError("Sorry, I encountered an error while trying to generate an initial answer.") from e

    def reflect_on_answer(self, query, initial_answer, selected_context) -> Tuple[str, Optional[str]]:
        # logger.info("🔍 Reflection and Evaluation Initiated")
        context_for_reflection_prompt = self._format_context_for_reflection(selected_context)
        cache_key = {"type": "reflection", "query": query, "initial_answer": initial_answer, "context_hash": abs(hash(context_for_reflection_prompt))}
        if cached := get_cached(cache_key): 
            # logger.debug("Reflection from cache.")
            return cached
        if not self.llm: return ('incomplete', 'Reflection LLM is unavailable.')

        placeholder_check_fragment = "no specific relevant information was found"
        reflection_prompt = f'''You are an evaluation agent. Review 'Initial Answer' for completeness regarding 'User Query' and 'Context'.
        If 'Initial Answer' is placeholder (like "{placeholder_check_fragment}"), eval is "incomplete", missing_info is original 'User Query' topic.
        Else, if incomplete, identify missing info.
        Return ONLY JSON: {{"evaluation": "complete" or "incomplete", "missing_information": "Description or empty string"}}
        User Query: "{query}"
        Context:\n{context_for_reflection_prompt}
        Initial Answer:\n"{initial_answer}"'''
        try:
            response = self.local_generate(reflection_prompt, max_tokens=300) # Reduced max_tokens
            json_match = re.search(r'\{[\s\S]*\}', response)
            evaluation_result, missing_info_description = 'incomplete', "Reflection parse error: No JSON."
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    evaluation_result = data.get("evaluation", "incomplete").lower()
                    missing_info_description = data.get("missing_information", "").strip()
                    if evaluation_result == 'complete': missing_info_description = None
                    elif not missing_info_description: missing_info_description = f"Answer incomplete, specific missing info not detailed."
                except json.JSONDecodeError as e: missing_info_description = f"Reflection JSON parse error: {e}"
            # logger.info(f"🔍 Reflection Result: {evaluation_result}. Missing: {missing_info_description[:100] if missing_info_description else 'None'}")
            return set_cached(cache_key, (evaluation_result, missing_info_description))
        except ValueError as e: # Catch LLM generation error
            logger.error(f"⚠️ Error during reflection LLM call: {e}")
            raise ValueError(f"An error occurred during reflection process: {e}") from e # Re-raise
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error during reflection: {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred during reflection: {e}") from e


    def _format_context_for_reflection(self, selected_context: Optional[Dict[str, Any]]) -> str:
        context_str_parts: List[str] = []
        if selected_context:
             if "kg" in selected_context:
                  kg_data = selected_context["kg"]
                  kg_info_str = "Knowledge Graph Info:\n"
                  diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                  # Ensure confidence is float before formatting
                  diag_confidence = float(diag_data.get("confidence", 0.0)) if diag_data else 0.0
                  if diag_data and diag_data.get("disease_name") and diag_confidence > 0.0:
                       kg_info_str += f"  Potential Condition: {diag_data['disease_name']} (Confidence: {diag_confidence:.2f})\n"
                  if kg_data.get("kg_matched_symptoms"): kg_info_str += f"  Matched Symptoms: {', '.join(kg_data['kg_matched_symptoms'])}\n"
                  if kg_data.get("kg_treatments"): kg_info_str += f"  Treatments: {', '.join(kg_data['kg_treatments'])}\n"
                  other_kg_content = kg_data.get("kg_content_other","")
                  if other_kg_content and "did not find" not in other_kg_content:
                      kg_info_str += "\n" + other_kg_content[:200] + ("..." if len(other_kg_content) > 200 else "") # Shorter for reflection
                  if len(kg_info_str.splitlines()) > 1 : context_str_parts.append(kg_info_str.strip())

             if "rag" in selected_context and selected_context["rag"]:
                  valid_chunks = [c for c in selected_context["rag"] if isinstance(c,str)]
                  if valid_chunks: context_str_parts.append("Relevant Passages:\n---\n" + "\n---\n".join(valid_chunks[:2]) + "\n---") # Top 2 for brevity
        return "\n\n".join(context_str_parts) if context_str_parts else "None"

    def get_supplementary_answer(self, query: str, missing_info_description: str, user_type: str) -> str:
        # logger.info(f"🌐 Gap Filling. Missing: {missing_info_description[:100]}...")
        cache_key = {"type": "supplementary_answer", "missing_info_hash": abs(hash(missing_info_description)), "query_hash": abs(hash(query)), "user_type": user_type}
        if cached := get_cached(cache_key): 
            # logger.debug("Supplementary answer from cache.")
            return cached
        if not self.llm: return "\n\n-- Additional Information --\nSupplementary information could not be generated (LLM unavailable)."

        base_prompt_instructions = get_system_prompt(user_type)
        supplementary_prompt = f'''{base_prompt_instructions.strip()}
        You are acting as an external agent to provide *only* specific missing details.
        Original User Query (for context): "{query}"
        Information Missing from Previous Answer: "{missing_info_description}"
        Provide ONLY the supplementary information addressing the missing part. Include evidence/sources where possible (URLs, [Source Name], or [General Medical Knowledge]). If you cannot find specific information, state this concisely. Start directly.
        '''
        try:
            supplementary_answer = self.local_generate(supplementary_prompt, max_tokens=750).strip()
            if not supplementary_answer: supplementary_answer = "The AI could not find specific additional information for the identified gap."
            # logger.info("🌐 Supplementary Answer Generated successfully.")
            final_supplementary_text = "\n\n-- Additional Information --\n" + supplementary_answer
            return set_cached(cache_key, final_supplementary_text)
        except ValueError as e:
            logger.error(f"⚠️ Error during supplementary answer generation: {e}", exc_info=True)
            error_msg = f"Sorry, an error occurred while trying to find additional information about: '{missing_info_description[:50]}...'"
            final_supplementary_text = f"\n\n-- Additional Information --\n{error_msg}"
            return set_cached(cache_key, final_supplementary_text)

    def collate_answers(self, initial_answer: str, supplementary_answer: str, user_type: str) -> str:
        # logger.info("✨ Final Answer Collation Initiated")
        cache_key = {"type": "final_collation", "initial_answer_hash": abs(hash(initial_answer)), "supplementary_answer_hash": abs(hash(supplementary_answer)), "user_type": user_type}
        if cached := get_cached(cache_key): 
            # logger.debug("Final collation from cache.")
            return cached
        if not self.llm: return f"{initial_answer.strip()}\n\n{supplementary_answer.strip()}" # Fallback

        supp_content_after_header = supplementary_answer.split("-- Additional Information --\n", 1)[-1].strip()
        if not supp_content_after_header or "could not find specific additional information" in supp_content_after_header.lower() or "error occurred while trying to find additional information" in supp_content_after_header.lower():
             # logger.debug("Supplementary answer empty/placeholder/error. Appending directly.")
             return initial_answer.strip() + supplementary_answer.strip()

        base_prompt_instructions = get_system_prompt(user_type)
        collation_prompt = f'''{base_prompt_instructions.strip()}
        Combine 'Initial Answer' and 'Supplementary Information' into a single, coherent response.
        Remove redundancy. Preserve all factual medical information and source attributions.
        Format clearly using markdown. Do NOT include the medical disclaimer or pathway note.
        Initial Answer Part:\n"{initial_answer}"
        Supplementary Information Part:\n"{supplementary_answer}"
        Provide ONLY the combined, final answer content:'''
        try:
            combined_answer_content = self.local_generate(collation_prompt, max_tokens=1500)
            # logger.info("✨ Final Answer Collated successfully.")
            return set_cached(cache_key, combined_answer_content)
        except ValueError as e:
            logger.error(f"⚠️ Error during final answer collation: {e}", exc_info=True)
            error_message = f"\n\n-- Collation Failed --\nAn error occurred: {e}\n\n"
            final_collated_text = initial_answer.strip() + error_message + supplementary_answer.strip()
            return set_cached(cache_key, final_collated_text)

    def reset_conversation(self): logger.info("🔄 Resetting chatbot internal state.")

    def process_user_query(self, user_query, user_type, confirmed_symptoms=None, original_query_if_followup=None) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        logger.info("--- Processing User Query: '%s' ---", user_query[:50])
        processed_query, current_symptoms_for_retrieval, is_symptom_query, medical_check_ok = user_query, [], False, False

        if confirmed_symptoms is not None:
            # logger.info("--- Step 1: Handling Symptom Confirmation Rerun ---")
            if not user_query: return "An error occurred: Original query missing for symptom rerun.", "display_final_answer", None
            is_symptom_query = True
            current_symptoms_for_retrieval = sorted(list(set(s.strip().lower() for s in confirmed_symptoms if isinstance(s,str) and s.strip())))
            medical_check_ok = True
        else:
            # logger.info("--- Step 1: Initial Query Processing ---")
            medical_check_ok, medical_reason = self.is_medical_query(processed_query)
            if not medical_check_ok: return f"I can only answer medical-related questions. Please rephrase. ({medical_reason})", "display_final_answer", None
            current_symptoms_for_retrieval, _ = self.extract_symptoms(processed_query)
            is_symptom_query = self.is_symptom_related_query(processed_query)
        
        if not medical_check_ok: return "Internal error: Medical check did not pass unexpectedly.", "display_final_answer", None

        # logger.info("--- Step 4: Context Retrieval ---")
        kg_results, s_kg, rag_chunks, s_rag = {}, 0.0, [], 0.0
        if is_symptom_query and current_symptoms_for_retrieval:
            kg_results = self.knowledge_graph_agent(processed_query, current_symptoms_for_retrieval)
            s_kg = float(kg_results.get("top_disease_confidence", 0.0)) # Ensure float
        if self.vectordb and self.embedding_model:
            rag_chunks, s_rag = self.retrieve_rag_context(processed_query) # s_rag is float
            s_rag = float(s_rag) # Ensure float

        if confirmed_symptoms is None and is_symptom_query and \
           len(kg_results.get("identified_diseases_data",[])) > 0 and \
           0.0 < s_kg < THRESHOLDS["disease_symptom_followup_threshold"]: # s_kg is float here
            
            top_disease_data = kg_results["identified_diseases_data"][0] # Assumes list is not empty due to len check
            all_kg_symps_lower = set(s.lower() for s in kg_results.get("all_disease_symptoms_kg_for_top_disease",[]) if isinstance(s,str))
            initial_symps_lower = set(s.lower() for s in current_symptoms_for_retrieval if isinstance(s,str)) # current_symptoms_for_retrieval are already lower
            suggested_symps_lower = sorted(list(all_kg_symps_lower - initial_symps_lower))
            
            original_case_map = {s.lower(): s for s in kg_results.get("all_disease_symptoms_kg_for_top_disease",[]) if isinstance(s,str)}
            suggested_original_case = [original_case_map[s_low] for s_low in suggested_symps_lower if s_low in original_case_map]

            if suggested_original_case:
                # logger.info(f"Suggesting {len(suggested_original_case)} new symptoms for UI.")
                ui_payload = {
                    "symptom_options": {top_disease_data.get("Disease", "Potential Condition"): suggested_original_case}, 
                    "original_query": processed_query
                }
                return "To help provide more accurate information, please confirm if you are experiencing any of these additional symptoms:", "show_symptom_ui", ui_payload
        
        # logger.info("--- Step 6: Context Selection Logic ---")
        selected_context = self.select_context(kg_results, s_kg, rag_chunks, s_rag, is_symptom_query) # s_kg, s_rag are floats
        initial_context_sources_used = [k.upper() for k in selected_context.keys()] if selected_context else []
        
        # logger.info("--- Step 7: Initial Answer Generation ---")
        try:
            initial_answer = self.generate_initial_answer(processed_query, selected_context, user_type)
        except ValueError as e:
            pathway_info = ", ".join(initial_context_sources_used) if initial_context_sources_used else "LLM (Initial Phrasing)"
            error_pathway_info = pathway_info + " (Initial Generation Failed)"
            return f"Sorry, I could not generate an initial answer due to an error: {e}\n\n<span style='font-size: 0.8em; color: grey;'>*Sources attempted: {error_pathway_info}*</span>", "display_final_answer", None

        # logger.info("--- Step 8: Reflection and Evaluation ---")
        reflection_failed, evaluation_result, missing_info_description = False, 'complete', None # Assume complete initially
        try:
            evaluation_result, missing_info_description = self.reflect_on_answer(processed_query, initial_answer, selected_context)
        except Exception as e: # Catch ValueError from LLM or RuntimeError
            logger.error(f"Reflection step failed: {e}")
            reflection_failed = True; evaluation_result = 'incomplete' 
            missing_info_description = f"Reflection failed ({e}). Attempting general supplementary info."
        
        final_answer_content = initial_answer
        supplementary_step_triggered = False
        if evaluation_result == 'incomplete':
            # logger.warning("--- Step 9: Reflection Incomplete/failed. Triggering supplementary. ---")
            supplementary_step_triggered = True
            description_for_supplementary = missing_info_description or f"Gap filling needed for query: {processed_query[:50]}..."
            supplementary_answer = self.get_supplementary_answer(processed_query, description_for_supplementary, user_type)
            final_answer_content = self.collate_answers(initial_answer, supplementary_answer, user_type)
            
        # logger.info("--- Step 12: Applying Triage Enhancement ---")
        final_answer_content_with_triage = self.enhance_with_triage_detection(processed_query, final_answer_content, user_type)

        final_pathway_parts = list(set(initial_context_sources_used)) # Start with initial
        if supplementary_step_triggered or not initial_context_sources_used or reflection_failed: 
            final_pathway_parts.append("LLM (General Knowledge)")
        if not reflection_failed : final_pathway_parts.append("Reflection Agent")
        elif reflection_failed and evaluation_result == 'incomplete': final_pathway_parts.append("Reflection Agent (Failed)")
        
        pathway_info = ", ".join(sorted(list(set(final_pathway_parts)))) or "Unknown Pathway"
        disclaimer = "\n\nIMPORTANT MEDICAL DISCLAIMER: This information is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
        pathway_note = f"<span style='font-size: 0.8em; color: grey;'>*Sources used for this response: {pathway_info.strip()}*</span>"
        final_response_text = f"{final_answer_content_with_triage.strip()}{disclaimer}\n\n{pathway_note}"
        
        # logger.info("--- Workflow Finished ---")
        return final_response_text, "display_final_answer", None

# Streamlit UI (largely unchanged from your last version, focus was backend fix)
# ... (Streamlit UI code - display_symptom_checklist, create_user_type_selector, main) ...
# The Streamlit UI part of your previous code was mostly fine regarding state management.
# The primary issue was the TypeError in the backend.
# The following is a truncated version of main, assuming the Streamlit part is correct.

def display_symptom_checklist(symptom_options: Dict[str, List[str]], original_query: str):
    st.subheader("Confirm Your Symptoms")
    st.info(f"Based on your query: '{original_query}' and initial analysis, please confirm additional symptoms:")

    user_type_key = st.session_state.get("user_type_select", "family") # Use consistent key
    form_key = f"symptom_form_{abs(hash(original_query))}_{user_type_key}_{st.session_state.get('form_timestamp',0)}"
    
    local_set_key = f'{form_key}_local_symptoms_set' # Changed key to avoid conflicts
    text_input_key = f"{form_key}_other_symptoms_text_input" # Changed key

    if local_set_key not in st.session_state:
        st.session_state[local_set_key] = set()
        # Clear text input if form is new instance
        if text_input_key in st.session_state: del st.session_state[text_input_key]

    # Pre-populate local_set_key with any existing text input from this form instance
    current_other_text_val = st.session_state.get(text_input_key, "")
    if current_other_text_val: # If there's text, update the set
        st.session_state[local_set_key].update(
            s.strip().lower() for s in current_other_text_val.split(',') if s.strip()
        )

    all_unique_suggested = sorted(list(set(
        s.strip() for sl in symptom_options.values() if isinstance(sl,list) for s in sl if isinstance(s,str) and s.strip()
    )))

    with st.form(form_key):
        st.markdown("Please check all symptoms that apply to you from the list below:")
        if not all_unique_suggested: st.info("No specific additional symptoms were found to suggest.")
        else:
            num_cols = min(4, max(1, len(all_unique_suggested))) # Ensure at least 1 col
            cols = st.columns(num_cols)
            for i, symptom_orig_case in enumerate(all_unique_suggested):
                col_idx = i % num_cols
                cb_key = f"{form_key}_checkbox_{abs(hash(symptom_orig_case))}"
                s_lower = symptom_orig_case.strip().lower()
                
                # Default checkbox state from the set
                is_currently_checked = s_lower in st.session_state.get(local_set_key, set())
                
                # Let Streamlit manage the checkbox state via its key for this form rendering
                new_checked_state = cols[col_idx].checkbox(symptom_orig_case, key=cb_key, value=is_currently_checked)
                
                # Update our set based on the checkbox's current state in this rerun
                if new_checked_state:
                    st.session_state[local_set_key].add(s_lower)
                else:
                    st.session_state[local_set_key].discard(s_lower)
        
        st.markdown("**Other Symptoms (if any, comma-separated):**")
        # Text input value is managed by its key
        other_symptoms_text_val_input = st.text_input("", key=text_input_key, value=st.session_state.get(text_input_key,""))

        if st.form_submit_button("Confirm and Continue"):
            logger.info(f"Symptom confirmation form submitted for: '{original_query[:50]}...'.")
            
            # Explicitly add symptoms from text input to the set *on submission*
            if other_symptoms_text_val_input: # Use the value from input at submission time
                st.session_state[local_set_key].update(
                    s.strip().lower() for s in other_symptoms_text_val_input.split(',') if s.strip()
                )

            final_symptoms_list = sorted(list(st.session_state.get(local_set_key, set())))
            st.session_state.confirmed_symptoms_from_ui = final_symptoms_list
            st.session_state.ui_state = {"step": "input", "payload": None}
            
            # Clean up this specific form's state variables
            if local_set_key in st.session_state: del st.session_state[local_set_key]
            if text_input_key in st.session_state: del st.session_state[text_input_key]
            
            st.session_state.form_timestamp = datetime.now().timestamp() # New timestamp for next potential form

def create_user_type_selector():
    if 'last_user_type' not in st.session_state:
        st.session_state.last_user_type = "User / Family"  # Default to initial selectbox value

    selected_type = st.sidebar.selectbox(
        "Who is asking?",
        ["User / Family", "Physician"],
        key="user_type_select",
        index=["User / Family", "Physician"].index(st.session_state.get("user_type_select", "User / Family")),
        help="Select user type. This may reset chat."
    )

    if selected_type != st.session_state.last_user_type:
        logger.info(f"User type changed from '{st.session_state.last_user_type}' to '{selected_type}'. Flagging for reset.")
        st.session_state.reset_requested_by_type_change = True
        st.session_state.last_user_type = selected_type # Update last_user_type immediately

def main():
    logger.info("--- Streamlit App Start ---")
    st.set_page_config(page_title="DxAI-Agent", page_icon=f"data:image/png;base64,{icon}", layout="wide")
    
    create_user_type_selector()
    current_user_type = st.session_state.get("user_type_select", "User / Family")

    if st.session_state.get('reset_requested_by_type_change', False):
        logger.info("Executing conversation reset due to user type change.")
        if st.session_state.get('chatbot'): st.session_state.chatbot.reset_conversation()
        # Selective reset for user type change
        st.session_state.messages = []
        st.session_state.ui_state = {"step": "input", "payload": None}
        st.session_state.processing_input_payload = None
        st.session_state.form_timestamp = datetime.now().timestamp()
        
        # Clear symptom-specific states
        for k_suffix in ['_from_ui', '_for_symptom_rerun']:
            state_key = f'confirmed_symptoms{k_suffix}'
            if state_key in st.session_state: del st.session_state[state_key]
        
        # Clear any active local form states (based on current original_query and user_type if available)
        # This is a bit trickier as the form_key might be dynamic
        # A simpler approach for type change is to clear all symptom_form_ prefixed keys
        keys_to_delete = [k for k in st.session_state if k.startswith("symptom_form_")]
        for k_del in keys_to_delete: del st.session_state[k_del]
        
        del st.session_state.reset_requested_by_type_change
        logger.info("Reset due to type change complete. Rerunning.")
        st.rerun()

    try:
        logo = Image.open(image_path)
        c1, c2 = st.columns([1,10]); c1.image(logo,width=100); c2.markdown("# DxAI-Agent")
    except Exception: st.markdown("# DxAI-Agent") # Fallback

    if 'chatbot_initialized_flag' not in st.session_state: # Use a more distinct flag
        st.session_state.chatbot_initialized_flag = False
        st.session_state.chatbot = None
        st.session_state.init_status = (False, "Initialization not started.")
        logger.info("Starting chatbot instance and backend setup (first run or after full clear)...")
        with st.spinner("Initializing chat assistant (LLM, KG, Documents)... This may take a moment."):
            try:
                st.session_state.chatbot = DocumentChatBot()
                success, msg = st.session_state.chatbot.initialize_qa_chain()
                st.session_state.init_status = (success, msg)
                st.session_state.chatbot_initialized_flag = success
                logger.info(f"Chatbot initialization attempt complete. Status: {success}, Msg: {msg}")
            except Exception as e:
                logger.critical(f"CRITICAL UNCAUGHT ERROR DURING INITIALIZATION: {e}", exc_info=True)
                st.session_state.init_status = (False, f"Critical initialization error: {e}")
                st.session_state.chatbot_initialized_flag = False
    
    init_success, init_msg = st.session_state.get('init_status', (False, "Status unknown."))
    is_interaction_enabled = st.session_state.get('chatbot_initialized_flag', False) and st.session_state.get('chatbot') is not None

    # Initialize other session state variables if they don't exist
    default_states = {
        'ui_state': {"step": "input", "payload": None},
        'messages': [],
        'processing_input_payload': None,
        'confirmed_symptoms_from_ui': None,
        'original_query_for_symptom_rerun': None,
        'form_timestamp': datetime.now().timestamp()
    }
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            logger.info(f"Initializing session state for '{key}'.")

    st.sidebar.info("DxAI-Agent helps answer medical questions using our medical knowledge base.")
    if is_interaction_enabled: st.sidebar.success(f"Initialization Status: {init_msg}")
    else: st.sidebar.error(f"Initialization Failed: {init_msg}")

    tab1, tab2 = st.tabs(["Chat", "About"])
    with tab1:
        for i, (msg_content, is_user_msg) in enumerate(st.session_state.messages):
            role = "user" if is_user_msg else "assistant"
            with st.chat_message(role):
                st.markdown(msg_content, unsafe_allow_html=(not is_user_msg)) # Allow HTML for assistant pathway notes
                if not is_user_msg and i == len(st.session_state.messages) - 1 and \
                   st.session_state.ui_state["step"] == "input" and is_interaction_enabled:
                    # Feedback buttons logic (simplified for brevity, assume it's mostly correct from previous version)
                    cols_fb = st.columns([0.05, 0.05, 0.9])
                    # ... (Your existing feedback button logic can go here) ...

        st.write(" \n" * 2) # Spacer
        input_area_container = st.container()
        with input_area_container:
            if not is_interaction_enabled:
                st.error("Chat assistant failed to initialize. Please check logs and configuration.")
                st.chat_input("Initializing...", disabled=True, key="init_fail_chat_input_disabled")
            elif st.session_state.ui_state["step"] == "confirm_symptoms":
                payload = st.session_state.ui_state.get("payload")
                if not payload or "symptom_options" not in payload or "original_query" not in payload:
                    logger.error("Symptom UI state error: Payload invalid. Resetting.")
                    st.session_state.messages.append(("Error showing symptom checklist. Please retry.", False))
                    st.session_state.ui_state = {"step": "input", "payload": None}
                    # Clean up symptom state
                    if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                    if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                    st.rerun()
                else:
                    display_symptom_checklist(payload["symptom_options"], payload["original_query"])
                    st.chat_input("Confirm symptoms above...", disabled=True, key="symptom_confirm_chat_input_disabled")
            elif st.session_state.ui_state["step"] == "input":
                user_query_input = st.chat_input("Ask your medical question...", disabled=not is_interaction_enabled, key="main_chat_input_field")
                if user_query_input and is_interaction_enabled:
                    st.session_state.messages.append((user_query_input, True))
                    if st.session_state.get('chatbot'): st.session_state.chatbot.reset_conversation()
                    st.session_state.form_timestamp = datetime.now().timestamp() # New form instance if UI triggered
                    # Clear states for a new query
                    if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                    if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                    # Clear previous feedback button states
                    for k_fb in [k for k in st.session_state if k.startswith("fb_")]: del st.session_state[k_fb]

                    st.session_state.processing_input_payload = {"query": user_query_input, "confirmed_symptoms": None}
                    st.rerun()

        if st.session_state.get('confirmed_symptoms_from_ui') is not None:
            confirmed_symps_payload = st.session_state.confirmed_symptoms_from_ui
            original_q_payload = st.session_state.get('original_query_for_symptom_rerun')
            
            del st.session_state.confirmed_symptoms_from_ui # Clear after reading

            if not original_q_payload:
                logger.error("Symptom form submitted but original query missing. Resetting UI.")
                st.session_state.messages.append(("Error processing symptom confirmation. Please retry query.", False))
                if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                st.session_state.ui_state = {"step":"input", "payload":None}
                st.rerun()
            else:
                logger.info("Symptom form submitted. Preparing to re-process with confirmed symptoms.")
                st.session_state.processing_input_payload = {"query": original_q_payload, "confirmed_symptoms": confirmed_symps_payload}
                st.rerun()

        if st.session_state.get('processing_input_payload') is not None:
            payload_to_process = st.session_state.processing_input_payload
            st.session_state.processing_input_payload = None # Clear immediately
            
            chatbot_instance = st.session_state.get('chatbot')
            if not chatbot_instance or not st.session_state.get('chatbot_initialized_flag'):
                logger.critical("Processing triggered but chatbot not ready.")
                st.session_state.messages.append(("Chatbot not ready. Please wait or re-initialize.", False))
                st.rerun()
            
            query_to_run = payload_to_process.get("query","")
            confirmed_symptoms_to_run = payload_to_process.get("confirmed_symptoms")

            if not query_to_run:
                logger.error("Empty query in processing payload. Skipping.")
                st.session_state.messages.append(("Received empty query for processing.", False))
                st.rerun() # Avoid getting stuck
            
            with st.spinner("Thinking..."):
                try:
                    response_text, ui_action, ui_payload_from_bot = chatbot_instance.process_user_query(
                        query_to_run, current_user_type, confirmed_symptoms_to_run
                    )
                    if ui_action == "display_final_answer":
                        st.session_state.messages.append((response_text, False))
                        st.session_state.ui_state = {"step": "input", "payload": None}
                        if confirmed_symptoms_to_run and 'original_query_for_symptom_rerun' in st.session_state:
                            del st.session_state.original_query_for_symptom_rerun
                    elif ui_action == "show_symptom_ui":
                        st.session_state.messages.append((response_text, False)) # This is the prompt for the UI
                        st.session_state.ui_state = {"step": "confirm_symptoms", "payload": ui_payload_from_bot}
                        st.session_state.form_timestamp = datetime.now().timestamp()
                        if ui_payload_from_bot and ui_payload_from_bot.get("original_query"):
                            st.session_state.original_query_for_symptom_rerun = ui_payload_from_bot["original_query"]
                        else: 
                            logger.error("Symptom UI requested but original_query missing in payload. Resetting.")
                            st.session_state.messages.append(("Error setting up symptom checklist.", False))
                            st.session_state.ui_state={"step":"input", "payload":None}
                    # else: ui_action == "none" or unknown - typically means UI should revert to input
                    # if st.session_state.ui_state["step"] != "confirm_symptoms": # Don't override if already set to show form
                    # st.session_state.ui_state = {"step": "input", "payload": None}
                except Exception as e_process: # Catch any error from process_user_query
                    logger.error(f"Error during process_user_query: {e_process}", exc_info=True)
                    st.session_state.messages.append((f"Sorry, an error occurred during processing: {e_process}", False))
                    st.session_state.ui_state = {"step": "input", "payload": None} # Reset UI on error
                    # Clean up symptom state if error occurred during symptom rerun
                    if confirmed_symptoms_to_run and 'original_query_for_symptom_rerun' in st.session_state:
                        del st.session_state.original_query_for_symptom_rerun
                st.rerun()

        st.divider()
        if st.button("Reset Conversation", key="reset_conversation_button", disabled=not is_interaction_enabled, help="Clear chat history and start over."):
            logger.info("Full conversation reset triggered by user button.")
            # More thorough reset: clear all session state and rerun.
            # This will re-trigger initialization checks and default state settings.
            # Store user_type_select before clearing, then restore if needed, or let it default.
            # current_selected_user_type = st.session_state.get("user_type_select", "User / Family")
            st.session_state.clear()
            # st.session_state.user_type_select = current_selected_user_type # Optionally restore user type
            # st.session_state.last_user_type = current_selected_user_type # Also reset this
            logger.info("Session state cleared. Rerunning for full reset.")
            st.rerun()
        
        st.divider()
        st.subheader("🩺 Detailed Feedback")
        with st.form(key="detailed_feedback_main_form", clear_on_submit=True):
            feedback_input_text = st.text_area("Enter corrections, improvements, or comments here...", height=100, disabled=not is_interaction_enabled)
            if st.form_submit_button("Submit Feedback", disabled=not is_interaction_enabled) and feedback_input_text:
                submit_feedback(feedback_input_text, st.session_state.get('messages',[]), current_user_type)
                st.success("Thank you for your feedback!")

    with tab2:
        st.markdown(""" ## Medical Chat Assistant ... (your about text) ... """)
    logger.debug("--- Streamlit App End of Rerun ---")

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers(): # Check if handlers are already configured
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Ensure logger is getLogger(__name__)
    main()
