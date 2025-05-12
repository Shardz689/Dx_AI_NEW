
import streamlit as st
from pathlib import Path
import csv
import os
import re
# import torch # Keep for sentence-transformers, but handle import error if not strictly needed
import json
# import numpy as np # Keep for sentence-transformers, but handle import error
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
import base64
import logging
from PIL import Image
import io

# Attempt basic torch import handling.
try:
    import torch
except ImportError:
    logger = logging.getLogger(__name__) # Ensure logger is defined
    logger.warning("PyTorch not found. Some functionalities might be affected if they rely on it directly.")
    torch = None # Define as None if not found

try:
    import numpy as np
except ImportError:
    logger = logging.getLogger(__name__) # Ensure logger is defined
    logger.warning("NumPy not found. Some functionalities might be affected if they rely on it directly.")
    class DummyNumpy:
        def floating(self, *args, **kwargs): return float # type: ignore
    np = DummyNumpy() # type: ignore


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


# Configuration
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBv-I8Ld-k09Lxu9Yi7HPffZHKXIqGSdHU")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://1b47920f.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "eCqDfyhDcuGMLzbYfiqL6jsvjH3LIXr86xQGAEKmY8Y")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger after basicConfig

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_PLACEHOLDER":
    logger.critical("GEMINI_API_KEY environment variable is not set or is the placeholder value.")
    GEMINI_API_KEY = None
elif len(GEMINI_API_KEY) < 20: # Basic check
    logger.warning("GEMINI_API_KEY appears short, possibly invalid.")

if not NEO4J_URI or NEO4J_URI == "YOUR_NEO4J_URI_PLACEHOLDER" or \
   not NEO4J_USER or NEO4J_USER == "YOUR_NEO4J_USER_PLACEHOLDER" or \
   not NEO4J_PASSWORD or NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD_PLACEHOLDER":
    logger.critical("NEO4J environment variables are not fully set or are placeholder values. KG connection will fail.")
    NEO4J_URI = None # Prevent connection attempts
    NEO4J_USER = None
    NEO4J_PASSWORD = None

THRESHOLDS = {
    "symptom_extraction": 0.6, "disease_matching": 0.5,
    "disease_symptom_followup_threshold": 0.8, "kg_context_selection": 0.6,
    "rag_context_selection": 0.7, "medical_relevance": 0.6,
    "high_kg_context_only": 0.8
}

def get_image_as_base64(file_path_str: str) -> str:
    file_path = Path(file_path_str)
    if not file_path.is_file():
        logger.warning(f"Image file not found: {file_path}")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        logger.error(f"Error encoding image {file_path}: {e}")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

image_path_str = "Zoom My Life.jpg" # Ensure this is a string
try:
    icon = get_image_as_base64(image_path_str)
except Exception: 
    icon = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

CACHE: Dict[str, Any] = {}
def get_cached(key: Any) -> Optional[Any]:
    try: key_str = json.dumps(key, sort_keys=True)
    except Exception: key_str = str(key) 
    return CACHE.get(key_str)

def set_cached(key: Any, value: Any) -> Any:
    try: key_str = json.dumps(key, sort_keys=True)
    except Exception: key_str = str(key)
    CACHE[key_str] = value
    return value

HARDCODED_PDF_FILES = ["rawdata.pdf"]


def vote_message(user_message: str, bot_message: str, vote: str, user_type: str) -> None:
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

def submit_feedback(feedback_text: str, conversation_history: List[Tuple[str, str]], user_type: str) -> None:
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
    def __init__(self) -> None:
        logger.info("DocumentChatBot initializing...")
        self.vectordb: Optional[FAISS] = None
        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        try:
            device_name = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing SentenceTransformer embeddings on device: {device_name}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name='pritamdeka/S-PubMedBert-MS-MARCO', 
                cache_folder=os.path.abspath('./cache'), # Use absolute path string
                model_kwargs={'device': device_name}, 
                encode_kwargs={'normalize_embeddings': True}
            )
            if self.embedding_model.embed_query("test query"): 
                 logger.info("Embedding model initialized and tested successfully.")
            else: logger.warning("Test embedding was empty, but embedding model object exists.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Could not initialize embedding model: {e}")
            self.embedding_model = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.kg_driver: Optional[GraphDatabase.driver] = None
        self.kg_connection_ok: bool = False
        self._init_kg_connection()

    def _init_kg_connection(self) -> None:
        logger.info("Attempting to connect to Neo4j...")
        if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD: 
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
            if self.kg_driver: # Close driver if partially initialized then failed
                try: self.kg_driver.close()
                except Exception as e_close: logger.error(f"Error closing Neo4j driver during init failure: {e_close}")
                self.kg_driver = None
    
    def enhance_with_triage_detection(self, query: str, response_content: str, user_type: str) -> str:
        logger.debug(f"enhance_with_triage_detection called with user_type: '{user_type}'")
        
        normalized_user_type = ""
        if user_type == "User / Family" or user_type == "family":
            normalized_user_type = "family"
        elif user_type == "Physician" or user_type == "physician":
            normalized_user_type = "physician"
        else:
            # Default to family if unsure, but log it
            logger.warning(f"Unknown user_type '{user_type}' in enhance_with_triage_detection, defaulting behavior to 'family' for safety (though triage should be skipped).")
            normalized_user_type = "family" 
    
        if normalized_user_type != "family":
            logger.info(f"Triage detection skipped for user type: '{user_type}' (normalized to '{normalized_user_type}')")
            return response_content
        
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
            "1. Emergency (Call 911/Emergency Services immediately)\n2. Urgent Care (See a doctor within 24 hours)\n"
            "3. Primary Care (Schedule a regular appointment)\n4. Self-care (Can be managed at home)\n\n"
            "Provide ONLY the triage category number and title (e.g., '1. Emergency') followed by a brief, one-sentence explanation of *why* this category is suggested. "
            "If it's clearly NOT a triage situation, respond with ONLY 'NO_TRIAGE_NEEDED'. Ensure your response is very concise (max 50 words total)."
        )
        cache_key = {"type": "triage_detection", "query": query, "response_hash": hash(response_content)}
        if (cached := get_cached(cache_key)) is not None:
             if cached != "NO_TRIAGE_NEEDED": return f"{response_content}\n\n**TRIAGE ASSESSMENT:**\n{cached}"
             return response_content
        try:
            triage_text = self.local_generate(triage_prompt, max_tokens=100).strip()
            set_cached(cache_key, triage_text)
            if "NO_TRIAGE_NEEDED" not in triage_text: return f"{response_content}\n\n**TRIAGE ASSESSMENT:**\n{triage_text}"
            return response_content
        except Exception as e: 
            logger.error(f"Error in triage detection: {e}", exc_info=True)
            return response_content # Return original content on any error during triage

    def create_vectordb(self) -> Tuple[Optional[FAISS], str]:
        logger.info("Creating vector database...")
        pdf_files = [Path(pdf_file) for pdf_file in HARDCODED_PDF_FILES if Path(pdf_file).is_file()]
        if not pdf_files: return None, "No PDF files found."
        loaders = [PyPDFLoader(str(pdf)) for pdf in pdf_files if Path(pdf).is_file()] # Re-check is_file
        if not loaders: return None, "No valid PDF loaders created."
        pages = [p for loader in loaders for p_list in [loader.load()] if p_list for p in p_list] # Ensure load() result is handled
        if not pages: return None, "No pages loaded from PDFs."
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(pages)
        if not splits: return None, "No text chunks created."
        if not self.embedding_model: return None, "Embedding model not initialized."
        try:
            vectordb = FAISS.from_documents(splits, self.embedding_model)
            logger.info("FAISS vectorstore created successfully.")
            return vectordb, "Vector database created successfully."
        except Exception as e:
            logger.error(f"Error creating FAISS DB: {e}", exc_info=True)
            return None, f"Failed to create vector database: {str(e)}"

    def initialize_qa_chain(self) -> Tuple[bool, str]:
        logger.info("Initializing QA chain components...")
        llm_msg = "LLM init skipped."
        if GEMINI_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY,
                    temperature=0.3, top_p=0.95, top_k=40, convert_system_message_to_human=True
                )
                if self.llm.invoke("Hello?", config={"max_output_tokens": 10}).content: llm_msg = "Gemini Flash 1.5 initialized."
                else: llm_msg = "Gemini Flash 1.5 init, but test response empty."
            except Exception as e: self.llm = None; llm_msg = f"Gemini LLM init/test failed: {e}"
        else: llm_msg = "Gemini API key not found or invalid."

        vdb_msg = "VDB init skipped."
        if self.embedding_model:
            if any(Path(pdf).is_file() for pdf in HARDCODED_PDF_FILES):
                self.vectordb, vdb_msg = self.create_vectordb()
            else: vdb_msg = "No PDF files found for VDB creation."
        else: vdb_msg = "Embedding model not init, VDB creation skipped."
        
        status = [
            "LLM OK" if self.llm else "LLM Failed",
            "Embeddings OK" if self.embedding_model else "Embeddings Failed",
            "Vector DB OK" if self.vectordb else "Vector DB Failed",
            "KG OK" if self.kg_connection_ok else "KG Failed"
        ]
        overall_msg = f"{llm_msg}. {vdb_msg}. KG: {'Connected' if self.kg_connection_ok else 'Failed'}. Overall: {', '.join(status)}."
        success = bool(self.llm)
        logger.info(f"Initialization Result: Success={success}, Message='{overall_msg}'")
        return success, overall_msg

    def local_generate(self, prompt: str, max_tokens: int = 500) -> str:
        if not self.llm: raise ValueError("LLM is not initialized. Cannot generate response.")
        try:
            response = self.llm.invoke(prompt, config={"max_output_tokens": max_tokens})
            if response and response.content: return response.content
            logger.warning("LLM invoke returned empty or None response.")
            raise ValueError("LLM returned empty response.") 
        except Exception as e: 
            logger.error(f"Error during Gemini LLM generation: {e}. Prompt (start): {prompt[:100]}")
            raise ValueError(f"LLM generation failed: {e}") from e
            
    def get_system_prompt(self, user_type: str) -> str: # Added self as it's a method
        logger.debug(f"get_system_prompt called with user_type: '{user_type}' (type: {type(user_type)})")
        
        normalized_user_type = ""
        # Explicitly check for the exact strings returned by the selectbox or common variations
        if user_type == "User / Family" or user_type == "family":
            normalized_user_type = "family"
        elif user_type == "Physician" or user_type == "physician": # Handles "Physician" from selectbox
            normalized_user_type = "physician"
        else:
            logger.warning(f"Unknown user_type '{user_type}' in get_system_prompt, defaulting to 'family'.")
            normalized_user_type = "family"
    
        base_prompt = "You are MediAssist, an AI assistant specialized in medical information. "
        if normalized_user_type == "physician":
            logger.info("Using PHYSICIAN system prompt.")
            return base_prompt + (
                "Respond using professional medical terminology and consider offering differential diagnoses "
                "when appropriate. Provide detailed clinical insights and evidence-based recommendations. "
                "Use medical jargon freely, assuming high medical literacy. Cite specific guidelines or studies "
                "when possible. Structure your responses with clear clinical reasoning. "
                "Avoid addressing the user as if they are the patient; assume you are consulting with a fellow medical professional about a case. Do not provide basic patient education unless specifically asked."
            )
        else:  # family user
            logger.info("Using FAMILY system prompt.")
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
        
    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        cache_key = {"type": "medical_relevance", "query": query}
        if (cached := get_cached(cache_key)) is not None: return cached # type: ignore
        
        medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose", "medicine", "doctor"]
        if not self.llm: 
            return set_cached(cache_key, (any(k in query.lower() for k in medical_keywords), "Fallback heuristic (LLM unavailable)"))

        prompt = f'''Analyze if query is health/medical. JSON: {{"is_medical": boolean, "confidence": float, "reasoning": "explanation"}}. Query: "{query}"'''
        try:
            response = self.local_generate(prompt, max_tokens=150)
            if (match := re.search(r'\{[\s\S]*\}', response)):
                data = json.loads(match.group(0))
                is_medical_llm = data.get("is_medical", False)
                confidence_raw = data.get("confidence", "0.0") 
                confidence = 0.0
                try: confidence = float(confidence_raw if confidence_raw is not None else "0.0")
                except (ValueError, TypeError): logger.warning(f"Medical relevance: bad confidence '{confidence_raw}'. Defaulting to 0.0.")
                
                final_is_medical = is_medical_llm and confidence >= THRESHOLDS.get("medical_relevance", 0.6)
                return set_cached(cache_key, (final_is_medical, data.get("reasoning", "")))
        except Exception as e: logger.warning(f"Medical relevance LLM/JSON processing failed: {e}.")
        return set_cached(cache_key, (any(k in query.lower() for k in medical_keywords), "Fallback heuristic (LLM/JSON fail)"))

    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        cache_key = {"type": "symptom_extraction", "query": user_query}
        if (cached := get_cached(cache_key)) is not None: return list(cached[0]), float(cached[1])
        
        common_keywords = ["fever", "cough", "headache", "pain", "nausea", "fatigue", "rash", "dizziness"] 
        query_lower = user_query.lower()
        fallback_symptoms = sorted(list(set(s for s in common_keywords if s in query_lower)))
        
        if not self.llm:
            return set_cached(cache_key, (fallback_symptoms, 0.4 if fallback_symptoms else 0.0))

        SYMPTOM_PROMPT = f'''Extract medical symptoms from query. JSON: {{"Extracted Symptoms": [{{"symptom": "symptom1", "confidence": 0.9}}, ...]}}. Query: "{user_query}"'''
        llm_symptoms_lower: List[str] = []
        llm_avg_confidence: float = 0.0
        llm_found_any: bool = False
        all_numeric_llm_confidences: List[float] = []

        try:
            response = self.local_generate(SYMPTOM_PROMPT, max_tokens=500).strip()
            if (json_match := re.search(r'\{[\s\S]*\}', response)):
                data = json.loads(json_match.group(0))
                symptom_data = data.get("Extracted Symptoms", [])
                if symptom_data: llm_found_any = True

                for item in symptom_data:
                    if isinstance(item, dict) and "symptom" in item and isinstance(item.get("symptom"), str) and item.get("symptom","").strip():
                        try:
                            raw_confidence = item.get("confidence", "0.0") 
                            item_confidence_float = float(raw_confidence if raw_confidence is not None else "0.0")
                            all_numeric_llm_confidences.append(item_confidence_float)
                            if item_confidence_float >= THRESHOLDS.get("symptom_extraction", 0.6):
                                llm_symptoms_lower.append(item["symptom"].strip().lower())
                        except (ValueError, TypeError):
                            logger.warning(f"Symptom extraction: Bad confidence '{item.get('confidence')}' for '{item.get('symptom')}'. Skipping.")
                
                llm_symptoms_lower = sorted(list(set(llm_symptoms_lower)))
                if all_numeric_llm_confidences:
                    llm_avg_confidence = sum(all_numeric_llm_confidences) / len(all_numeric_llm_confidences)
            else: logger.warning("No JSON in symptom extraction LLM response.")
        except Exception as e: logger.error(f"Symptom extraction LLM call/parse error: {e}", exc_info=True)

        combined_symptom_set_lower = set(llm_symptoms_lower)
        combined_symptom_set_lower.update(fallback_symptoms)
        final_symptoms_lower = sorted(list(combined_symptom_set_lower))

        final_confidence = llm_avg_confidence if llm_found_any and all_numeric_llm_confidences else \
                           (0.4 if fallback_symptoms else 0.0)
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        logger.info(f"Final extracted symptoms: {final_symptoms_lower} (Confidence: {final_confidence:.4f})")
        return set_cached(cache_key, (final_symptoms_lower, final_confidence)) # type: ignore

    def is_symptom_related_query(self, query: str) -> bool:
        if not query or not query.strip(): return False
        cache_key = {"type": "symptom_query_detection_heuristic", "query": query}
        if (cached := get_cached(cache_key)) is not None: return cached # type: ignore
        
        extracted_symptoms, symptom_confidence = self.extract_symptoms(query)
        if extracted_symptoms and symptom_confidence >= THRESHOLDS.get("symptom_extraction", 0.6):
            return set_cached(cache_key, True) # type: ignore

        health_keywords = ["symptom", "pain", "sick", "health", "disease", "condition", "diagnosis", "feel"]
        result = any(keyword in query.lower() for keyword in health_keywords)
        return set_cached(cache_key, result) # type: ignore

    def knowledge_graph_agent(self, user_query: str, symptoms_for_kg: List[str]) -> Dict[str, Any]:
        logger.info("📚 KG Agent Initiated for query: %s...", user_query[:50])
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
             return kg_results_template(symptoms_for_kg)

        kg_results = kg_results_template(valid_symptom_names)

        if not self.kg_connection_ok or not self.kg_driver:
             logger.warning("📚 KG Agent: Connection not OK.")
             return kg_results

        try:
            with self.kg_driver.session(connection_acquisition_timeout=10.0) as session:
                disease_data_from_kg = self._query_disease_from_symptoms_with_session(session, valid_symptom_names)
                
                if disease_data_from_kg: # Already sorted and confidence is float
                    kg_results["identified_diseases_data"] = disease_data_from_kg
                    top_disease_record = disease_data_from_kg[0]
                    top_disease_name = top_disease_record.get("Disease")
                    top_disease_conf = top_disease_record.get("Confidence", 0.0) 
                    kg_results["top_disease_confidence"] = top_disease_conf
                    kg_results["kg_matched_symptoms"] = list(top_disease_record.get("MatchedSymptoms", []))
                    kg_results["all_disease_symptoms_kg_for_top_disease"] = list(top_disease_record.get("AllDiseaseSymptomsKG", []))

                    if top_disease_conf >= THRESHOLDS.get("disease_matching", 0.5) and top_disease_name:
                        treatments, treat_conf = self._query_treatments_with_session(session, top_disease_name)
                        kg_results["kg_treatments"], kg_results["kg_treatment_confidence"] = treatments, treat_conf
                
                top_name_for_llm = kg_results["identified_diseases_data"][0].get("Disease") if kg_results["identified_diseases_data"] else "an unidentifiable condition"
                kg_results["kg_content_diagnosis_data_for_llm"].update({
                    "disease_name": top_name_for_llm, 
                    "symptoms_list": valid_symptom_names, 
                    "confidence": kg_results["top_disease_confidence"] 
                })
                
                other_parts = []
                if kg_results["kg_treatments"]:
                     other_parts.append("## Recommended Treatments (from KG)")
                     other_parts.extend([f"- {t}" for t in kg_results["kg_treatments"]])
                kg_results["kg_content_other"] = "\n".join(other_parts).strip() or "Medical KG did not find specific relevant treatment information."
            logger.info("📚 Knowledge Graph Agent Finished successfully.")
        except Exception as e:
            logger.error("⚠️ Error within KG Agent: %s", e, exc_info=True)
            return kg_results_template(valid_symptom_names) 
        return kg_results

    def _query_disease_from_symptoms_with_session(self, session: Any, symptoms: List[str]) -> List[Dict[str, Any]]:
        symptom_names_lower = [s.lower() for s in symptoms]
        cache_key = {"type": "disease_matching_v2", "symptoms": tuple(sorted(symptom_names_lower))}
        if (cached := get_cached(cache_key)) is not None:
            return [dict(item, MatchedSymptoms=list(item['MatchedSymptoms']), AllDiseaseSymptomsKG=list(item['AllDiseaseSymptomsKG']), Confidence=float(item['Confidence'])) for item in cached] # type: ignore

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
            result_cursor = session.run(cypher_query, symptomNamesLower=symptom_names_lower)
            disease_data = []
            for rec in result_cursor:
                if (disease_name := rec.get("Disease")):
                    disease_data.append({
                        "Disease": disease_name, 
                        "Confidence": float(rec.get("confidence_score", 0.0)), 
                        "MatchedSymptoms": list(rec.get("MatchedSymptoms", [])),
                        "AllDiseaseSymptomsKG": list(rec.get("AllDiseaseSymptomsKG", []))
                    })
            set_cached(cache_key, [dict(d, MatchedSymptoms=tuple(d['MatchedSymptoms']), AllDiseaseSymptomsKG=tuple(d['AllDiseaseSymptomsKG'])) for d in disease_data])
            return disease_data
        except Exception as e:
            logger.error(f"Error querying diseases from symptoms: {e}", exc_info=True)
            return []

    def _query_treatments_with_session(self, session: Any, disease: str) -> Tuple[List[str], float]:
        disease_name_lower = disease.strip().lower()
        cache_key = {"type": "treatment_query_kg", "disease": disease_name_lower}
        if (cached := get_cached(cache_key)) is not None:
            if isinstance(cached, tuple) and len(cached) == 2: return list(cached[0]), float(cached[1])
            logger.warning(f"Cache for treatments of '{disease}' had unexpected format. Recalculating.")
        
        cypher_query = """
        MATCH (d:disease)-[:TREATED_BY]->(t:treatment) WHERE toLower(d.Name) = $diseaseNameLower
        RETURN t.Name as Treatment ORDER BY Treatment
        """
        try:
            result_cursor = session.run(cypher_query, diseaseNameLower=disease_name_lower)
            treatments_list = sorted(list(set(
                rec["Treatment"].strip() for rec in result_cursor if rec.get("Treatment") and isinstance(rec.get("Treatment"), str) and rec.get("Treatment").strip()
            )))
            avg_confidence = 1.0 if treatments_list else 0.0 
            return set_cached(cache_key, (treatments_list, avg_confidence)) # type: ignore
        except Exception as e:
            logger.error(f"⚠️ Error executing KG query for treatments: {e}", exc_info=True)
            return [], 0.0

    def retrieve_rag_context(self, query: str) -> Tuple[List[str], float]:
        # logger.info(f"📄 RAG Retrieval Initiated for query: {query[:50]}...")
        k = 5 
        cache_key = {"type": "rag_retrieval_topk_chunks_and_scores", "query": query, "k": k}
        if (cached_value := get_cached(cache_key)) is not None:
             if isinstance(cached_value, dict) and 'chunks' in cached_value and 'avg_score' in cached_value:
                 try:
                     # logger.debug(f"RAG from cache. Avg score: {cached_value.get('avg_score')}")
                     return list(cached_value['chunks']), float(cached_value['avg_score'])
                 except (ValueError, TypeError) as e:
                     logger.error(f"Error converting cached RAG 'avg_score' ('{cached_value.get('avg_score')}') to float: {e}. Recalculating.")
             else:
                 logger.warning(f"RAG cache for key {cache_key} had unexpected format. Recalculating.")
        
        if not self.vectordb or not self.embedding_model:
            logger.warning("📄 RAG: VDB or Embedding model not initialized. Skipping.")
            return [], 0.0
        
        # logger.info(f"📄 RAG Cache miss or error for query: {query[:50]}. Recalculating...")
        try:
            retrieved_docs_with_scores = self.vectordb.similarity_search_with_score(query, k=k)
            top_k_chunks_content: List[str] = []
            top_k_similarity_scores: List[float] = []

            # logger.debug(f"--- Processing {len(retrieved_docs_with_scores)} Retrieved RAG Chunks ---")
            for i, (doc, score_val) in enumerate(retrieved_docs_with_scores):
                try:
                    # Attempt to convert score_val to a standard Python float
                    current_score_float = float(score_val) 

                    # FAISS scores are distances (lower is better, assume non-negative)
                    # Similarity = 1 - Distance. Clamp to [0, 1].
                    similarity_score = max(0.0, min(1.0, 1.0 - current_score_float))
                    
                    # logger.debug(f"RAG Chunk {i+1}: Raw Score='{score_val}', Converted Score={current_score_float:.4f}, Similarity={similarity_score:.4f}")

                    if doc and doc.page_content:
                        top_k_chunks_content.append(doc.page_content)
                        top_k_similarity_scores.append(similarity_score)
                    else:
                        logger.warning(f"RAG Chunk {i+1}: Document or content is None. Skipping.")
                except (ValueError, TypeError) as e_conv:
                    logger.warning(f"RAG Chunk {i+1}: Could not convert score value '{score_val}' (type: {type(score_val)}) to float. Error: {e_conv}. Skipping this document-score pair.")
                    continue 
            # logger.debug(f"--- Finished Processing RAG Chunks ---")

            srag_calculated = sum(top_k_similarity_scores) / len(top_k_similarity_scores) if top_k_similarity_scores else 0.0
            srag_float_to_return = float(srag_calculated) # Ensure final srag is float
            
            data_to_cache = {'chunks': top_k_chunks_content, 'avg_score': srag_float_to_return}
            set_cached(cache_key, data_to_cache) 
            
            logger.info(f"📄 RAG Finished. {len(top_k_chunks_content)} chunks processed. S_RAG: {srag_float_to_return:.4f}")
            return top_k_chunks_content, srag_float_to_return
        
        except Exception as e_main_logic:
            logger.error(f"⚠️ Error during RAG retrieval main logic: {e_main_logic}", exc_info=True)
            return [], 0.0
            
    def select_context(self, kg_results: Dict[str, Any], s_kg: float, rag_chunks: List[str], s_rag: float, is_symptom_query: bool) -> Optional[Dict[str, Any]]:
        logger.info("📦 Context Selection. SymptomQ: %s, S_KG: %.4f, S_RAG: %.4f", is_symptom_query, s_kg, s_rag)
        kg_thresh = THRESHOLDS.get("kg_context_selection", 0.6)
        rag_thresh = THRESHOLDS.get("rag_context_selection", 0.7)
        high_kg_thresh = THRESHOLDS.get("high_kg_context_only", 0.8)
        selected: Dict[str, Any] = {}
        
        # Ensure s_kg and s_rag are floats before comparison
        s_kg_float = s_kg # Already float from KG agent
        s_rag_float = s_rag # Already float from RAG

        kg_has_data = kg_results and kg_results.get("identified_diseases_data")

        if is_symptom_query and s_kg_float > high_kg_thresh and kg_has_data:
            selected["kg"] = kg_results
        else:
            if is_symptom_query:
                if s_kg_float >= kg_thresh and kg_has_data: selected["kg"] = kg_results
            if s_rag_float >= rag_thresh and rag_chunks: selected["rag"] = rag_chunks
        
        if not selected: logger.info("📦 Context Selection: No context source met thresholds."); return None
        return selected

    def generate_initial_answer(self, query: str, selected_context: Optional[Dict[str, Any]], user_type: str) -> str:
        logger.info(f"🧠 Initial Answer Gen. User Type: '{user_type}'. Query: '{query[:30]}...'")
        cache_key = {"type": "initial_answer", "query": query, "user_type": user_type, 
                     "context_hash": abs(hash(json.dumps(selected_context, sort_keys=True, default=str)))} # default=str for non-serializable
        if (cached := get_cached(cache_key)) is not None: 
            logger.debug("Initial answer from cache.")
            return cached # type: ignore
    
        base_prompt_instructions = self.get_system_prompt(user_type) # Use self.get_system_prompt
        logger.debug(f"Initial Answer Gen - System Prompt used (start): {base_prompt_instructions[:150]}...")
        
        context_info_for_prompt = ""
        context_type_description = ""
        context_parts_for_prompt: List[str] = []
        
        if selected_context:
            if "kg" in selected_context:
                kg_data = selected_context["kg"]
                kg_info_str_parts = ["Knowledge Graph Information:"] # Start with a header
                diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                diag_confidence = float(diag_data.get("confidence", 0.0)) if diag_data else 0.0
    
                if diag_data and diag_confidence > 0.0:
                    disease_name = diag_data.get("disease_name", "an unidentifiable condition")
                    # Phrasing based on confidence thresholds
                    if diag_confidence > THRESHOLDS["high_kg_context_only"]: 
                        kg_info_str_parts.append(f"- **Highly Probable Condition:** {disease_name} (KG Confidence: {diag_confidence:.2f})")
                    elif diag_confidence > THRESHOLDS["kg_context_selection"]: 
                        kg_info_str_parts.append(f"- **Potential Condition:** {disease_name} (KG Confidence: {diag_confidence:.2f})")
                    elif diag_confidence > THRESHOLDS["disease_matching"]: 
                        kg_info_str_parts.append(f"- **Possible Condition:** {disease_name} (KG Confidence: {diag_confidence:.2f})")
                    else: 
                        kg_info_str_parts.append(f"- Possible Condition based on limited match: {disease_name} (KG Confidence: {diag_confidence:.2f})")
                    
                    if kg_data.get('kg_matched_symptoms'): 
                        kg_info_str_parts.append(f"- Relevant Symptoms (matched in KG): {', '.join(kg_data['kg_matched_symptoms'])}")
                
                other_kg_content = kg_data.get("kg_content_other", "")
                if other_kg_content and "did not find" not in other_kg_content and other_kg_content.strip(): # Ensure it's meaningful
                    kg_info_str_parts.append(other_kg_content)
                
                if len(kg_info_str_parts) > 1: # Only add if more than just the header
                    context_parts_for_prompt.append("\n".join(kg_info_str_parts))
    
            if "rag" in selected_context and selected_context["rag"]:
                context_parts_for_prompt.append("Relevant Passages from Documents:\n---\n" + "\n---\n".join(selected_context["rag"][:3]) + "\n---")
    
        if not selected_context or not context_parts_for_prompt: # No context or context parts ended up empty
            context_type_description = ("You have not been provided with any specific external medical knowledge or document snippets for this query, "
                                        "or the available information did not meet confidence thresholds. "
                                        "Therefore, generate only a minimal placeholder answer that indicates lack of specific information "
                                        "(e.g., 'No specific relevant information was found in available knowledge sources.'). "
                                        "Do NOT attempt to answer the user query using your general knowledge in this step. "
                                        "Do NOT mention external documents or knowledge graphs unless explicitly told they were empty.")
            prompt_for_initial_answer = (
                f"{base_prompt_instructions.strip()}\n\n"
                f"{context_type_description.strip()}\n\n"
                f"User Query: \"{query}\"\n\n"
                "Minimal Placeholder Answer (be very concise):\n"
            )
        else:
            context_info_for_prompt = "\n\n".join(context_parts_for_prompt)
            # Determine context description based on what parts were actually added
            active_ctx_keys = []
            if any("Knowledge Graph Information:" in part for part in context_parts_for_prompt): active_ctx_keys.append("kg")
            if any("Relevant Passages from Documents:" in part for part in context_parts_for_prompt): active_ctx_keys.append("rag")
            
            ctx_desc_key = "_".join(sorted(active_ctx_keys))
            desc_map = {
                "kg_rag": "Based on the following structured medical knowledge from a knowledge graph AND relevant passages from medical documents, synthesize a comprehensive answer.",
                "kg": "Based on the following information from a medical knowledge graph, answer the user query. Focus on the provided KG information.",
                "rag": "Based on the following relevant passages from medical documents, answer the user query. Focus on the provided document excerpts."
            }
            context_type_description = desc_map.get(ctx_desc_key, "Based on the available information...")
            prompt_for_initial_answer = (
                f"{base_prompt_instructions.strip()}\n"
                f"{context_type_description.strip()}\n\n"
                f"Context Provided:\n{context_info_for_prompt}\n\n"
                f"User Query: {query}\n\n"
                "Answer (synthesize from context if provided, otherwise follow instructions for placeholder if no context was usable):\n"
            )
        
        logger.debug(f"Initial Answer Prompt (start):\n{prompt_for_initial_answer[:500]}...")
        try:
            initial_answer = self.local_generate(prompt_for_initial_answer, max_tokens=1000)
            logger.debug(f"Initial Answer Raw LLM Output (start): {initial_answer[:100]}")
    
            placeholder_frags = ["no specific relevant information was found", "lack of specific information", "unable to provide specific information"]
            initial_answer_lower = initial_answer.lower()
            is_placeholder = not initial_answer.strip() or any(f in initial_answer_lower for f in placeholder_frags)
    
            has_provided_context = selected_context and context_parts_for_prompt
            if has_provided_context and is_placeholder:
                logger.warning("LLM generated placeholder despite context. Overriding.")
                initial_answer = "No specific relevant information was found in external knowledge sources that could address the query." 
            elif not has_provided_context and not is_placeholder:
                logger.warning("LLM generated content despite no usable context and instruction for placeholder. Overriding.")
                initial_answer = "No specific relevant information was found in available knowledge sources to address the query."
            
            return set_cached(cache_key, initial_answer)
        except ValueError as e: # From local_generate
            logger.error(f"⚠️ Error during initial answer LLM call: {e}", exc_info=True)
            # Return a specific error message that can be displayed to the user
            raise ValueError(f"Sorry, I encountered an AI processing error while generating the initial part of the answer: {e}") from e

    def reflect_on_answer(self, query: str, initial_answer: str, selected_context: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
        context_for_reflection_prompt = self._format_context_for_reflection(selected_context)
        cache_key = {"type": "reflection", "query": query, "initial_answer_hash": hash(initial_answer), "context_hash": hash(context_for_reflection_prompt)}
        if (cached := get_cached(cache_key)) is not None: return cached # type: ignore
        if not self.llm: return ('incomplete', 'Reflection LLM unavailable.')

        placeholder_check = "no specific relevant information was found"
        prompt = f'''Evaluate 'Initial Answer' for completeness for 'User Query' using 'Context'.
        If Initial Answer is placeholder (like "{placeholder_check}"), eval="incomplete", missing_info=query topic.
        Else, if incomplete, identify missing.
        JSON ONLY: {{"evaluation": "complete"|"incomplete", "missing_information": "Description|empty"}}
        Query: "{query}"
        Context:\n{context_for_reflection_prompt}
        Initial Answer:\n"{initial_answer}"'''
        try:
            response = self.local_generate(prompt, max_tokens=300)
            eval_res, missing_desc = 'incomplete', "Reflection parse error: No JSON."
            if (match := re.search(r'\{[\s\S]*\}', response)):
                try:
                    data = json.loads(match.group(0))
                    eval_res = data.get("evaluation", "incomplete").lower()
                    missing_desc = data.get("missing_information", "").strip()
                    if eval_res == 'complete': missing_desc = None
                    elif not missing_desc: missing_desc = f"Answer incomplete, details not provided by evaluator."
                except json.JSONDecodeError as e_json: missing_desc = f"Reflection JSON parse error: {e_json}"
            return set_cached(cache_key, (eval_res, missing_desc)) # type: ignore
        except Exception as e_reflect: 
            logger.error(f"Error during reflection: {e_reflect}", exc_info=True)
            raise ValueError(f"Error during reflection: {e_reflect}") from e_reflect

    def _format_context_for_reflection(self, selected_context: Optional[Dict[str, Any]]) -> str:
        parts = []
        if selected_context:
            if "kg" in selected_context:
                kg_data, kg_str_parts = selected_context["kg"], ["KG Info:"]
                diag = kg_data.get("kg_content_diagnosis_data_for_llm")
                diag_conf = float(diag.get("confidence", 0.0)) if diag else 0.0
                if diag and diag.get("disease_name") and diag_conf > 0: 
                    kg_str_parts.append(f"  Potential Condition: {diag['disease_name']} (Conf: {diag_conf:.2f})")
                if kg_data.get("kg_matched_symptoms"): kg_str_parts.append(f"  Matched Symptoms: {', '.join(kg_data['kg_matched_symptoms'])}")
                if kg_data.get("kg_treatments"): kg_str_parts.append(f"  Treatments: {', '.join(kg_data['kg_treatments'])}")
                other_kg = kg_data.get("kg_content_other","")
                if other_kg and "did not find" not in other_kg: kg_str_parts.append(other_kg[:150] + "...")
                if len(kg_str_parts) > 1 : parts.append("\n".join(kg_str_parts))
            if "rag" in selected_context and selected_context["rag"]:
                valid_chunks = [c for c in selected_context["rag"] if isinstance(c,str)]
                if valid_chunks: parts.append("Relevant Passages:\n---\n" + "\n---\n".join(valid_chunks[:2]) + "\n---")
        return "\n\n".join(parts) if parts else "None"

    def get_supplementary_answer(self, query: str, missing_info_description: str, user_type: str) -> str:
        logger.info(f"🌐 Gap Filling. User Type: '{user_type}'. Missing: {missing_info_description[:50]}...")
        cache_key = {"type": "supplementary_answer", "missing_info_hash": abs(hash(missing_info_description)), 
                     "query_hash": abs(hash(query)), "user_type": user_type}
        if (cached := get_cached(cache_key)) is not None: 
            logger.debug("Supplementary answer from cache.")
            return cached # type: ignore
        if not self.llm: 
            return "\n\n-- Additional Information --\nSupplementary information could not be generated because the AI model is unavailable."
    
        base_prompt_instructions = self.get_system_prompt(user_type) # Use self.get_system_prompt
        logger.debug(f"Supplementary Answer Gen - System Prompt used (start): {base_prompt_instructions[:150]}...")
        
        supplementary_prompt = f'''{base_prompt_instructions.strip()}
    You are an AI assistant acting to provide *only* specific missing details to supplement a previous incomplete answer.
    Your response should be suitable for the specified user type.
    Original User Query (for context to understand the scope and what was asked): "{query}"
    Description of Information Missing from Previous Answer (this is what you need to address): "{missing_info_description}"
    
    Provide ONLY the supplementary information that directly addresses the 'Missing Information' described above.
    Do NOT restate the original query or any information already provided in the previous answer.
    Focus precisely on filling the identified gap.
    **If you make medical claims, you MUST include evidence or source attribution where possible.** Use formats like [Source Name], [General Medical Knowledge], or provide URLs if appropriate.
    If you cannot find specific information for the described gap, state this concisely (e.g., "No further specific details could be found regarding X.").
    Start your response directly with the supplementary information.
    '''
        try:
            supplementary_answer = self.local_generate(supplementary_prompt, max_tokens=750).strip()
            if not supplementary_answer: 
                supplementary_answer = "The AI could not find specific additional information for the identified gap."
            logger.info("🌐 Supplementary Answer Generated successfully.")
            final_supplementary_text = "\n\n-- Additional Information --\n" + supplementary_answer
            return set_cached(cache_key, final_supplementary_text) # type: ignore
        except ValueError as e: # From local_generate
            logger.error(f"⚠️ Error during supplementary answer LLM call: {e}", exc_info=True)
            error_msg = f"Sorry, an AI processing error occurred while trying to find additional information about: '{missing_info_description[:50]}...'"
            final_supplementary_text = f"\n\n-- Additional Information --\n{error_msg}"
            return set_cached(cache_key, final_supplementary_text) # type: ignore

    def collate_answers(self, initial_answer: str, supplementary_answer: str, user_type: str) -> str:
            logger.info(f"✨ Final Answer Collation. User Type: '{user_type}'.")
            cache_key = {"type": "final_collation", "initial_answer_hash": abs(hash(initial_answer)), 
                         "supplementary_answer_hash": abs(hash(supplementary_answer)), "user_type": user_type}
            if (cached := get_cached(cache_key)) is not None: 
                logger.debug("Final collation from cache.")
                return cached # type: ignore
            if not self.llm: 
                logger.warning("LLM not available for collation. Concatenating answers.")
                return f"{initial_answer.strip()}\n\n{supplementary_answer.strip()}"
        
            # Check if supplementary answer is just the error/no-info placeholder string structure
            supp_content_after_header = supplementary_answer.split("-- Additional Information --\n", 1)[-1].strip()
            if not supp_content_after_header or \
               "could not find specific additional information" in supp_content_after_header.lower() or \
               "error occurred while trying to find additional information" in supp_content_after_header.lower() or \
               "ai model is unavailable" in supp_content_after_header.lower():
                 logger.debug("Supplementary answer appears to be empty, placeholder, or error. Appending directly without LLM collation.")
                 return initial_answer.strip() + "\n" + supplementary_answer.strip() # Ensure newline if supp_answer is not just header
        
            base_prompt_instructions = self.get_system_prompt(user_type) # Use self.get_system_prompt
            logger.debug(f"Collate Answers - System Prompt used (start): {base_prompt_instructions[:150]}...")
            
            collation_prompt = f'''{base_prompt_instructions.strip()}
        You are a medical communicator tasked with combining information from two sources into a single, coherent final response, appropriate for the specified user type.
        Combine the following two parts:
        1. An 'Initial Answer' to a medical query.
        2. 'Supplementary Information' that addresses gaps identified in the initial answer.
        
        Create a single, fluent, and easy-to-understand final answer. Ensure a natural flow.
        Remove any redundancy between the two parts. If the supplementary part largely repeats or doesn't add significant new value to the initial answer, prioritize the initial answer or integrate minimally.
        Preserve all factual medical information and any source attributions or links provided in either part. Do NOT add new sources or make claims not supported by the provided parts.
        Format the final response clearly using markdown (e.g., headings, lists) if appropriate.
        Do NOT include the headers "Initial Answer Part:" or "Supplementary Information Part:" in your final output.
        Do NOT include the medical disclaimer or source pathway note in the answer text itself; these will be added separately.
        
        Initial Answer Part:
        "{initial_answer}"
        
        Supplementary Information Part (this part starts with "-- Additional Information --\\n" which you should ignore as a header):
        "{supplementary_answer}"
        
        Provide ONLY the combined, final answer content. Start directly with the answer.
        '''
            try:
                combined_answer_content = self.local_generate(collation_prompt, max_tokens=1500)
                logger.info("✨ Final Answer Collated successfully.")
                return set_cached(cache_key, combined_answer_content) # type: ignore
            except ValueError as e: # From local_generate
                logger.error(f"⚠️ Error during final answer collation LLM call: {e}", exc_info=True)
                error_message = f"\n\n-- Collation Failed --\nAn error occurred while finalizing the answer ({e}). The information below is the initial answer followed by supplementary information, presented uncollated.\n\n"
                final_collated_text = initial_answer.strip() + error_message + supplementary_answer.strip()
                return set_cached(cache_key, final_collated_text) # type: ignore

    def reset_conversation(self) -> None: logger.info("🔄 Resetting chatbot internal state.")

    def process_user_query(self, user_query: str, user_type: str, 
                           confirmed_symptoms: Optional[List[str]] = None, 
                           original_query_if_followup: Optional[str] = None # Keep for API consistency, though not directly used
                           ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        logger.info("--- Processing User Query: '%s' ---", user_query[:50])
        processed_query: str = user_query
        current_symptoms_for_retrieval: List[str] = []
        is_symptom_query_flag: bool = False # Renamed to avoid conflict
        medical_check_ok: bool = False

        if confirmed_symptoms is not None:
            if not user_query: return "Error: Original query missing for symptom rerun.", "display_final_answer", None
            is_symptom_query_flag = True
            current_symptoms_for_retrieval = sorted(list(set(s.strip().lower() for s in confirmed_symptoms if isinstance(s,str) and s.strip())))
            medical_check_ok = True
        else:
            medical_check_ok, medical_reason = self.is_medical_query(processed_query)
            if not medical_check_ok: return f"I only answer medical questions. ({medical_reason})", "display_final_answer", None
            current_symptoms_for_retrieval, _ = self.extract_symptoms(processed_query)
            is_symptom_query_flag = self.is_symptom_related_query(processed_query)
        
        if not medical_check_ok: return "Internal error: Medical check failed.", "display_final_answer", None

        kg_results: Dict[str, Any] = {}
        s_kg: float = 0.0
        rag_chunks: List[str] = []
        s_rag_val: float = 0.0 # Renamed to avoid conflict with s_rag as a string

        if is_symptom_query_flag and current_symptoms_for_retrieval:
            kg_results = self.knowledge_graph_agent(processed_query, current_symptoms_for_retrieval)
            s_kg = float(kg_results.get("top_disease_confidence", 0.0))

        if self.vectordb and self.embedding_model:
            retrieved_rag_data = self.retrieve_rag_context(processed_query)
            logger.info(f"retrieve_rag_context returned: Type={type(retrieved_rag_data)}, Value='{str(retrieved_rag_data)[:100]}...'")
            if isinstance(retrieved_rag_data, tuple) and len(retrieved_rag_data) == 2:
                rag_chunks = retrieved_rag_data[0] if isinstance(retrieved_rag_data[0], list) else []
                s_rag_candidate = retrieved_rag_data[1]
                try: s_rag_val = float(s_rag_candidate)
                except (ValueError, TypeError) as e:
                    logger.error(f"Could not convert s_rag_candidate ('{s_rag_candidate}') to float: {e}. Defaulting s_rag to 0.0.")
                    s_rag_val = 0.0
            else: logger.error(f"retrieve_rag_context bad return. Defaulting RAG."); rag_chunks, s_rag_val = [], 0.0
        logger.info(f"After RAG processing, s_rag_val = {s_rag_val} (Type: {type(s_rag_val)})")


        if confirmed_symptoms is None and is_symptom_query_flag and \
           len(kg_results.get("identified_diseases_data",[])) > 0 and \
           0.0 < s_kg < THRESHOLDS["disease_symptom_followup_threshold"]:
            top_disease_data = kg_results["identified_diseases_data"][0]
            all_kg_symps_lower = set(s.lower() for s in kg_results.get("all_disease_symptoms_kg_for_top_disease",[]) if isinstance(s,str))
            initial_symps_lower = set(s.lower() for s in current_symptoms_for_retrieval if isinstance(s,str))
            suggested_symps_lower = sorted(list(all_kg_symps_lower - initial_symps_lower))
            original_case_map = {s.lower(): s for s in kg_results.get("all_disease_symptoms_kg_for_top_disease",[]) if isinstance(s,str)}
            suggested_original_case = [original_case_map[s_low] for s_low in suggested_symps_lower if s_low in original_case_map]
            if suggested_original_case:
                ui_payload = {"symptom_options": {top_disease_data.get("Disease", "Condition"): suggested_original_case}, "original_query": processed_query}
                return "To help, please confirm additional symptoms:", "show_symptom_ui", ui_payload
        
        selected_context = self.select_context(kg_results, s_kg, rag_chunks, s_rag_val, is_symptom_query_flag)
        initial_sources = [k.upper() for k in selected_context.keys()] if selected_context else []
        
        try: initial_answer = self.generate_initial_answer(processed_query, selected_context, user_type)
        except ValueError as e:
            path_info = ", ".join(initial_sources) or "LLM (Initial)"
            return f"Error generating initial answer: {e}\n\n<span>*Attempted: {path_info} (Failed)*</span>", "display_final_answer", None

        reflection_failed, eval_res, missing_desc = False, 'complete', None
        try: eval_res, missing_desc = self.reflect_on_answer(processed_query, initial_answer, selected_context)
        except Exception as e: reflection_failed = True; eval_res = 'incomplete'; missing_desc = f"Reflection fail ({e})."
        
        final_answer = initial_answer; supp_triggered = False
        if eval_res == 'incomplete':
            supp_triggered = True
            desc_for_supp = missing_desc or f"Gap for: {processed_query[:30]}..."
            supp_answer = self.get_supplementary_answer(processed_query, desc_for_supp, user_type)
            final_answer = self.collate_answers(initial_answer, supp_answer, user_type)
            
        final_answer_triaged = self.enhance_with_triage_detection(processed_query, final_answer, user_type)

        final_path = list(set(initial_sources))
        if supp_triggered or not initial_sources or reflection_failed: final_path.append("LLM (General Knowledge)")
        if not reflection_failed : final_path.append("Reflection Agent")
        elif reflection_failed and eval_res == 'incomplete': final_path.append("Reflection Agent (Failed)")
        
        path_str = ", ".join(sorted(list(set(final_path)))) or "Unknown Pathway"
        disclaimer = "\n\nIMPORTANT MEDICAL DISCLAIMER: Not professional medical advice. Consult a doctor." # Shorter
        path_note = f"<span style='font-size:0.8em;color:grey;'>*Sources: {path_str.strip()}*</span>"
        return f"{final_answer_triaged.strip()}{disclaimer}\n\n{path_note}", "display_final_answer", None


# --- Streamlit UI ---
# (Assuming the Streamlit UI part from your provided code is largely correct and placed here)
# For brevity, I'll only include the main function and necessary UI helper.
# Ensure the UI calls `chatbot.process_user_query` correctly as in your previous version.

def display_symptom_checklist(symptom_options: Dict[str, List[str]], original_query: str) -> None:
    st.subheader("Confirm Your Symptoms")
    st.info(f"Based on your query: '{original_query}' and initial analysis, please confirm additional symptoms:")

    user_type_key = st.session_state.get("user_type_select", "family") 
    form_key = f"symptom_form_{abs(hash(original_query))}_{user_type_key}_{st.session_state.get('form_timestamp',0)}"
    
    local_set_key = f'{form_key}_local_symptoms_set_v2' 
    text_input_key = f"{form_key}_other_symptoms_text_input_v2" 

    if local_set_key not in st.session_state:
        st.session_state[local_set_key] = set()
        if text_input_key in st.session_state: del st.session_state[text_input_key]

    current_other_text_val = st.session_state.get(text_input_key, "")
    if current_other_text_val: 
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
            num_cols = min(4, max(1, len(all_unique_suggested))) 
            cols = st.columns(num_cols)
            for i, symptom_orig_case in enumerate(all_unique_suggested):
                col_idx = i % num_cols
                cb_key = f"{form_key}_checkbox_{abs(hash(symptom_orig_case))}" # Unique key for checkbox
                s_lower = symptom_orig_case.strip().lower()
                
                is_currently_checked = s_lower in st.session_state.get(local_set_key, set())
                new_checked_state = cols[col_idx].checkbox(symptom_orig_case, key=cb_key, value=is_currently_checked)
                
                if new_checked_state: st.session_state[local_set_key].add(s_lower)
                else: st.session_state[local_set_key].discard(s_lower)
        
        st.markdown("**Other Symptoms (if any, comma-separated):**")
        other_symptoms_text_val_input = st.text_input("", key=text_input_key, value=st.session_state.get(text_input_key,""))

        if st.form_submit_button("Confirm and Continue"):
            logger.info(f"Symptom confirmation form submitted for: '{original_query[:50]}...'.")
            if other_symptoms_text_val_input: 
                st.session_state[local_set_key].update(
                    s.strip().lower() for s in other_symptoms_text_val_input.split(',') if s.strip()
                )

            final_symptoms_list = sorted(list(st.session_state.get(local_set_key, set())))
            st.session_state.confirmed_symptoms_from_ui = final_symptoms_list
            st.session_state.ui_state = {"step": "input", "payload": None}
            
            if local_set_key in st.session_state: del st.session_state[local_set_key]
            if text_input_key in st.session_state: del st.session_state[text_input_key]
            st.session_state.form_timestamp = datetime.now().timestamp()

def create_user_type_selector() -> None:
    if 'last_user_type' not in st.session_state:
        st.session_state.last_user_type = st.session_state.get("user_type_select", "User / Family")

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
        st.session_state.last_user_type = selected_type 

def main() -> None:
    logger.info("--- Streamlit App Start ---")
    st.set_page_config(page_title="DxAI-Agent", page_icon=f"data:image/png;base64,{icon}", layout="wide")
    
    create_user_type_selector()
    current_user_type = st.session_state.get("user_type_select", "User / Family")

    if st.session_state.get('reset_requested_by_type_change', False):
        logger.info("Executing conversation reset due to user type change.")
        if st.session_state.get('chatbot'): st.session_state.chatbot.reset_conversation()
        st.session_state.messages = []
        st.session_state.ui_state = {"step": "input", "payload": None}
        st.session_state.processing_input_payload = None
        st.session_state.form_timestamp = datetime.now().timestamp()
        for k_suffix in ['_from_ui', '_for_symptom_rerun']:
            if f'confirmed_symptoms{k_suffix}' in st.session_state: del st.session_state[f'confirmed_symptoms{k_suffix}']
        keys_to_delete = [k for k in st.session_state if k.startswith("symptom_form_")]
        for k_del in keys_to_delete: del st.session_state[k_del]
        del st.session_state.reset_requested_by_type_change
        logger.info("Reset due to type change complete. Rerunning.")
        st.rerun()

    try:
        logo = Image.open(image_path_str)
        c1, c2 = st.columns([1,10]); c1.image(logo,width=100); c2.markdown("# DxAI-Agent")
    except Exception: st.markdown("# DxAI-Agent")

    if 'chatbot_initialized_flag' not in st.session_state: 
        st.session_state.chatbot_initialized_flag = False
        st.session_state.chatbot = None
        st.session_state.init_status = (False, "Initialization not started.")
        with st.spinner("Initializing chat assistant... This may take a moment."):
            try:
                st.session_state.chatbot = DocumentChatBot()
                success, msg = st.session_state.chatbot.initialize_qa_chain()
                st.session_state.init_status = (success, msg)
                st.session_state.chatbot_initialized_flag = success
            except Exception as e:
                st.session_state.init_status = (False, f"Critical init error: {e}")
                st.session_state.chatbot_initialized_flag = False
    
    init_success, init_msg = st.session_state.get('init_status', (False, "Status unknown."))
    is_interaction_enabled = st.session_state.get('chatbot_initialized_flag', False) and st.session_state.get('chatbot') is not None

    default_states = {
        'ui_state': {"step": "input", "payload": None}, 'messages': [],
        'processing_input_payload': None, 'confirmed_symptoms_from_ui': None,
        'original_query_for_symptom_rerun': None,
        'form_timestamp': datetime.now().timestamp()
    }
    for key, default_value in default_states.items():
        if key not in st.session_state: st.session_state[key] = default_value

    st.sidebar.info("DxAI-Agent for medical queries.")
    if is_interaction_enabled: st.sidebar.success(f"Status: {init_msg}")
    else: st.sidebar.error(f"Init Failed: {init_msg}")

    tab1, tab2 = st.tabs(["Chat", "About"])
    with tab1:
        for i, (msg_content, is_user_msg) in enumerate(st.session_state.messages):
            role = "user" if is_user_msg else "assistant"
            with st.chat_message(role):
                st.markdown(msg_content, unsafe_allow_html=(not is_user_msg))
                if not is_user_msg and i == len(st.session_state.messages) - 1 and \
                   st.session_state.ui_state["step"] == "input" and is_interaction_enabled:
                    cols_fb = st.columns([0.05, 0.05, 0.9])
                    user_q_for_fb = next((m[0] for j,m in enumerate(st.session_state.messages) if j < i and m[1]), "") 
                    fb_key_base = f"fb_{i}_{abs(hash(msg_content.split('IMPORTANT MEDICAL DISCLAIMER:',1)[0]))}"
                    up_key, down_key = f'{fb_key_base}_up', f'{fb_key_base}_down'
                    up_btn_key, down_btn_key = f'{up_key}_btn', f'{down_key}_btn'

                    if up_key not in st.session_state and down_key not in st.session_state:
                        if cols_fb[0].button("👍", key=up_btn_key, help="Good response"):
                            vote_message(user_q_for_fb, msg_content, "thumbs_up", current_user_type)
                            st.session_state[up_key] = True; st.toast("Thanks!"); st.rerun()
                        if cols_fb[1].button("👎", key=down_btn_key, help="Bad response"):
                            vote_message(user_q_for_fb, msg_content, "thumbs_down", current_user_type)
                            st.session_state[down_key] = True; st.toast("Thanks!"); st.rerun()
                    elif up_key in st.session_state: cols_fb[0].button("👍", key=up_btn_key, disabled=True)
                    elif down_key in st.session_state: cols_fb[1].button("👎", key=down_btn_key, disabled=True)
        
        st.write(" \n" * 2) 
        input_area_container = st.container()
        with input_area_container:
            if not is_interaction_enabled:
                st.error("Chat assistant failed to initialize.")
                st.chat_input("Initializing...", disabled=True, key="init_fail_chat_input_disabled_v2")
            elif st.session_state.ui_state["step"] == "confirm_symptoms":
                payload = st.session_state.ui_state.get("payload")
                if not payload or "symptom_options" not in payload or "original_query" not in payload:
                    st.session_state.messages.append(("Error with symptom checklist. Please retry.", False))
                    st.session_state.ui_state = {"step": "input", "payload": None}; st.rerun()
                else:
                    display_symptom_checklist(payload["symptom_options"], payload["original_query"])
                    st.chat_input("Confirm symptoms above...", disabled=True, key="symptom_confirm_chat_input_disabled_v2")
            elif st.session_state.ui_state["step"] == "input":
                if (user_query_input := st.chat_input("Ask your medical question...", disabled=not is_interaction_enabled, key="main_chat_input_field_v2")):
                    st.session_state.messages.append((user_query_input, True))
                    if st.session_state.get('chatbot'): st.session_state.chatbot.reset_conversation()
                    st.session_state.form_timestamp = datetime.now().timestamp()
                    for k_s in ['confirmed_symptoms_from_ui', 'original_query_for_symptom_rerun']:
                        if k_s in st.session_state: del st.session_state[k_s]
                    for k_fb in [k for k in st.session_state if k.startswith("fb_")]: del st.session_state[k_fb]
                    st.session_state.processing_input_payload = {"query": user_query_input, "confirmed_symptoms": None}
                    st.rerun()

        if st.session_state.get('confirmed_symptoms_from_ui') is not None:
            confirmed_symps_payload = st.session_state.confirmed_symptoms_from_ui
            original_q_payload = st.session_state.get('original_query_for_symptom_rerun')
            del st.session_state.confirmed_symptoms_from_ui 
            if not original_q_payload:
                st.session_state.messages.append(("Error processing symptoms. Retry query.", False))
                if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                st.session_state.ui_state = {"step":"input"}; st.rerun()
            else:
                st.session_state.processing_input_payload = {"query": original_q_payload, "confirmed_symptoms": confirmed_symps_payload}
                st.rerun()

        if st.session_state.get('processing_input_payload') is not None:
            payload_to_process = st.session_state.processing_input_payload
            st.session_state.processing_input_payload = None 
            
            chatbot_instance = st.session_state.get('chatbot')
            if not chatbot_instance or not st.session_state.get('chatbot_initialized_flag'):
                st.session_state.messages.append(("Chatbot not ready.", False)); st.rerun()
            
            query_to_run = payload_to_process.get("query","")
            confirmed_symptoms_to_run = payload_to_process.get("confirmed_symptoms")

            if not query_to_run: st.session_state.messages.append(("Empty query received.", False)); st.rerun()
            
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
                        st.session_state.messages.append((response_text, False))
                        st.session_state.ui_state = {"step": "confirm_symptoms", "payload": ui_payload_from_bot}
                        st.session_state.form_timestamp = datetime.now().timestamp()
                        if ui_payload_from_bot and ui_payload_from_bot.get("original_query"):
                            st.session_state.original_query_for_symptom_rerun = ui_payload_from_bot["original_query"]
                        else: 
                            st.session_state.messages.append(("Error preparing symptom checklist.", False))
                            st.session_state.ui_state={"step":"input"}
                except Exception as e_process: 
                    logger.error(f"Error during process_user_query: {e_process}", exc_info=True)
                    st.session_state.messages.append((f"Sorry, an error occurred: {str(e_process)[:200]}", False)) # Show limited error
                    st.session_state.ui_state = {"step": "input", "payload": None} 
                    if confirmed_symptoms_to_run and 'original_query_for_symptom_rerun' in st.session_state:
                        del st.session_state.original_query_for_symptom_rerun
                st.rerun()

        st.divider()
        if st.button("Reset Conversation", key="reset_conversation_button_v2", disabled=not is_interaction_enabled):
            st.session_state.clear()
            logger.info("Session state cleared for full reset. Rerunning.")
            st.rerun()
        
        st.divider()
        st.subheader("🩺 Detailed Feedback")
        with st.form(key="detailed_feedback_main_form_v2", clear_on_submit=True):
            feedback_input_text = st.text_area("Comments...", height=100, disabled=not is_interaction_enabled)
            if st.form_submit_button("Submit Feedback", disabled=not is_interaction_enabled) and feedback_input_text:
                submit_feedback(feedback_input_text, st.session_state.get('messages',[]), current_user_type)
                st.success("Thank you!")

    with tab2:
        st.markdown(""" ## Medical Chat Assistant ... (your about text) ... """)
    # logger.debug("--- Streamlit App End of Rerun ---")

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers(): 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) 
    main()
