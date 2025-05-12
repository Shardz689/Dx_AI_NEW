
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBv-I8Ld-k09Lxu9Yi7HPffZHKXIqGSdHU")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://1b47920f.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "eCqDfyhDcuGMLzbYfiqL6jsvjH3LIXr86xQGAEKmY8Y")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_PLACEHOLDER":
    logger.critical("GEMINI_API_KEY environment variable is not set or is the placeholder value.")
    GEMINI_API_KEY = None
elif len(GEMINI_API_KEY) < 20:
    logger.warning("GEMINI_API_KEY appears short, possibly invalid.")

if not NEO4J_URI or NEO4J_URI == "YOUR_NEO4J_URI_PLACEHOLDER" or \
   not NEO4J_USER or NEO4J_USER == "YOUR_NEO4J_USER_PLACEHOLDER" or \
   not NEO4J_PASSWORD or NEO4J_PASSWORD == "YOUR_NEO4J_PASSWORD_PLACEHOLDER":
    logger.critical("NEO4J environment variables are not fully set or are placeholder values.")
    NEO4J_URI = None
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
except Exception:
    icon = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

CACHE = {}
def get_cached(key):
    try: key_str = json.dumps(key, sort_keys=True)
    except Exception: key_str = str(key)
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
    else:
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
            if self.embedding_model.embed_query("test query"):
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
    
    def enhance_with_triage_detection(self, query: str, response_content: str, user_type: str) -> str:
        if user_type not in ["User / Family", "family"]:
            return response_content
        if "TRIAGE ASSESSMENT:" in response_content.upper() or any(cat in response_content for cat in ["1. Emergency", "2. Urgent Care", "3. Primary Care", "4. Self-care"]):
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
            "Provide ONLY the triage category number and title (e.g., '1. Emergency') followed by a brief, one-sentence explanation. If not a triage situation, respond with ONLY 'NO_TRIAGE_NEEDED'. Max 50 words."
        )
        cache_key = {"type": "triage_detection", "query": query, "response": response_content}
        cached = get_cached(cache_key)
        if cached is not None:
             triage_text = cached
             if triage_text != "NO_TRIAGE_NEEDED": return f"{response_content}\n\n**TRIAGE ASSESSMENT:**\n{triage_text}"
             return response_content
        try:
            triage_analysis = self.local_generate(triage_prompt, max_tokens=100)
            triage_text = triage_analysis.strip()
            set_cached(cache_key, triage_text)
            if "NO_TRIAGE_NEEDED" not in triage_text:
                return f"{response_content}\n\n**TRIAGE ASSESSMENT:**\n{triage_text}"
            return response_content
        except Exception as e:
            logger.error(f"Error in triage detection: {e}")
            return response_content

    def create_vectordb(self):
        logger.info("Creating vector database...")
        pdf_files = [Path(pdf_file) for pdf_file in HARDCODED_PDF_FILES if Path(pdf_file).is_file()]
        if not pdf_files: return None, "No PDF files found."
        loaders = [PyPDFLoader(str(pdf)) for pdf in pdf_files]
        pages = [p for loader in loaders for p in loader.load()]
        if not pages: return None, "No pages loaded."
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(pages)
        if not splits: return None, "No text chunks created."
        if not self.embedding_model: return None, "Embedding model not initialized."
        try:
            vectordb = FAISS.from_documents(splits, self.embedding_model)
            logger.info("FAISS vectorstore created.")
            return vectordb, "Vector database created successfully."
        except Exception as e:
            logger.error(f"Error creating FAISS DB: {e}")
            return None, f"Failed to create vector database: {str(e)}"

    def initialize_qa_chain(self):
        logger.info("Initializing QA chain components...")
        llm_msg = "LLM init skipped."
        if GEMINI_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY,
                    temperature=0.3, top_p=0.95, top_k=40, convert_system_message_to_human=True
                )
                if self.llm.invoke("Hello?", config={"max_output_tokens": 10}).content:
                    llm_msg = "Gemini Flash 1.5 initialized."
                else: llm_msg = "Gemini Flash 1.5 init, but test response empty."
            except Exception as e: self.llm = None; llm_msg = f"Gemini LLM init/test failed: {e}"
        else: llm_msg = "Gemini API key not found."

        vdb_msg = "VDB init skipped."
        if self.embedding_model:
            if any(Path(pdf).is_file() for pdf in HARDCODED_PDF_FILES):
                self.vectordb, vdb_msg = self.create_vectordb()
            else: vdb_msg = "No PDF files for VDB."
        else: vdb_msg = "Embedding model not init for VDB."
        
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

    def local_generate(self, prompt, max_tokens=500):
        if not self.llm: raise ValueError("LLM not initialized.")
        try:
            response = self.llm.invoke(prompt, config={"max_output_tokens": max_tokens})
            if response and response.content: return response.content
            raise ValueError("LLM returned empty response.")
        except Exception as e: raise ValueError(f"LLM generation failed: {e}") from e

    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        cache_key = {"type": "medical_relevance", "query": query}
        if cached := get_cached(cache_key): return cached
        if not self.llm:
            keywords = ["symptom", "disease", "health", "medical", "pain", "treatment", "doctor"]
            return set_cached(cache_key, (any(k in query.lower() for k in keywords), "Fallback heuristic (LLM unavailable)"))
        prompt = f'Is the query "{query}" related to health/medicine? JSON: {{"is_medical": boolean, "confidence": float, "reasoning": "explanation"}}'
        try:
            response = self.local_generate(prompt, max_tokens=150)
            if match := re.search(r'\{[\s\S]*\}', response):
                data = json.loads(match.group(0))
                is_med = data.get("is_medical", False) and float(data.get("confidence", 0.0)) >= THRESHOLDS["medical_relevance"]
                return set_cached(cache_key, (is_med, data.get("reasoning", "")))
        except Exception as e: logger.warning(f"Medical relevance LLM call/parse failed: {e}")
        keywords = ["symptom", "disease", "health", "medical", "pain", "treatment", "doctor"]
        return set_cached(cache_key, (any(k in query.lower() for k in keywords), "Fallback heuristic (LLM failed)"))

    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        cache_key = {"type": "symptom_extraction", "query": user_query}
        if cached := get_cached(cache_key): return list(cached[0]), cached[1]
        
        common_keywords = ["fever", "cough", "headache", "pain", "nausea", "fatigue", "rash"] # Simplified
        query_lower = user_query.lower()
        fallback_symptoms = sorted(list(set(s for s in common_keywords if s in query_lower)))
        
        if not self.llm:
            return set_cached(cache_key, (fallback_symptoms, 0.4 if fallback_symptoms else 0.0))

        prompt = f'Extract symptoms from "{user_query}". JSON: {{"Extracted Symptoms": [{{"symptom": "symptom1", "confidence": 0.9}}, ...]}}'
        llm_symptoms, llm_avg_conf, llm_found_any = [], 0.0, False
        try:
            response = self.local_generate(prompt, max_tokens=300).strip()
            if match := re.search(r'\{[\s\S]*\}', response):
                data = json.loads(match.group(0))
                symptom_data = data.get("Extracted Symptoms", [])
                if symptom_data: llm_found_any = True
                llm_symptoms = sorted(list(set(
                    item["symptom"].strip().lower() for item in symptom_data
                    if isinstance(item, dict) and item.get("symptom") and isinstance(item.get("symptom"), str) and
                       item.get("confidence", 0.0) >= THRESHOLDS["symptom_extraction"]
                )))
                valid_confs = [item.get("confidence",0) for item in symptom_data if isinstance(item,dict) and isinstance(item.get("confidence"),(int,float))]
                if valid_confs: llm_avg_conf = sum(valid_confs) / len(valid_confs)
        except Exception as e: logger.error(f"Symptom extraction LLM/parse error: {e}")

        combined_symptoms = sorted(list(set(llm_symptoms + fallback_symptoms)))
        final_conf = llm_avg_conf if llm_found_any else (0.4 if fallback_symptoms else 0.0)
        final_conf = max(0.0, min(1.0, final_conf))
        logger.info(f"Final extracted symptoms: {combined_symptoms} (Confidence: {final_conf:.4f})")
        return set_cached(cache_key, (combined_symptoms, final_conf))

    def is_symptom_related_query(self, query: str) -> bool:
        if not query or not query.strip(): return False
        cache_key = {"type": "symptom_query_detection_heuristic", "query": query}
        if cached := get_cached(cache_key): return cached
        
        symptoms, confidence = self.extract_symptoms(query)
        if symptoms and confidence >= THRESHOLDS["symptom_extraction"]:
            return set_cached(cache_key, True)
        
        health_keywords = ["symptom", "pain", "sick", "health", "disease", "condition", "feel"]
        result = any(keyword in query.lower() for keyword in health_keywords)
        return set_cached(cache_key, result)

    def knowledge_graph_agent(self, user_query: str, symptoms_for_kg: List[str]) -> Dict[str, Any]:
        logger.info("📚 KG Agent Initiated. Symptoms: %s", symptoms_for_kg)
        kg_results_template = lambda s: {
            "extracted_symptoms": s, "identified_diseases_data": [], "top_disease_confidence": 0.0,
            "kg_matched_symptoms": [], "kg_treatments": [], "kg_treatment_confidence": 0.0,
            "all_disease_symptoms_kg_for_top_disease": [],
            "kg_content_diagnosis_data_for_llm": {"disease_name": "an unidentifiable condition", "symptoms_list": s, "confidence": 0.0},
            "kg_content_other": "Medical KG info unavailable."
        }
        valid_symptoms = [s.strip() for s in symptoms_for_kg if isinstance(s, str) and s.strip()]
        if not valid_symptoms: return kg_results_template(symptoms_for_kg)
        if not self.kg_connection_ok or not self.kg_driver: return kg_results_template(valid_symptoms)

        kg_results = kg_results_template(valid_symptoms) # Initialize with valid symptoms
        try:
            with self.kg_driver.session(connection_acquisition_timeout=10.0) as session:
                disease_data = self._query_disease_from_symptoms_with_session(session, valid_symptoms)
                if disease_data:
                    disease_data = sorted([d for d in disease_data if d.get("Disease")], key=lambda x: float(x.get("Confidence", 0.0)), reverse=True)
                    if disease_data: # Check again after filtering
                        kg_results["identified_diseases_data"] = disease_data
                        top_disease = disease_data[0]
                        kg_results["top_disease_confidence"] = float(top_disease.get("Confidence", 0.0))
                        kg_results["kg_matched_symptoms"] = list(top_disease.get("MatchedSymptoms", []))
                        kg_results["all_disease_symptoms_kg_for_top_disease"] = list(top_disease.get("AllDiseaseSymptomsKG", []))
                        
                        if kg_results["top_disease_confidence"] >= THRESHOLDS["disease_matching"] and top_disease.get("Disease"):
                            treatments, treat_conf = self._query_treatments_with_session(session, top_disease["Disease"])
                            kg_results["kg_treatments"], kg_results["kg_treatment_confidence"] = treatments, treat_conf
                
                top_name = kg_results["identified_diseases_data"][0]["Disease"] if kg_results["identified_diseases_data"] else "an unidentifiable condition"
                kg_results["kg_content_diagnosis_data_for_llm"].update({
                    "disease_name": top_name, "confidence": kg_results["top_disease_confidence"]
                })
                other_parts = [f"## Recommended Treatments (from KG)\n" + "\n".join([f"- {t}" for t in kg_results["kg_treatments"]])] if kg_results["kg_treatments"] else []
                kg_results["kg_content_other"] = "\n".join(other_parts).strip() or "KG found no specific treatment info."
            logger.info("📚 KG Agent Finished successfully.")
        except Exception as e:
            logger.error("⚠️ Error in KG Agent: %s", e, exc_info=True)
            # Return template with valid_symptoms, as an error occurred after this point
            return kg_results_template(valid_symptoms) 
        return kg_results

    def _query_disease_from_symptoms_with_session(self, session, symptoms: List[str]) -> List[Dict[str, Any]]:
        s_lower = [s.lower() for s in symptoms]
        cache_key = {"type": "disease_matching_v2", "symptoms": tuple(sorted(s_lower))}
        if cached := get_cached(cache_key): return [dict(item, MatchedSymptoms=list(item['MatchedSymptoms']), AllDiseaseSymptomsKG=list(item['AllDiseaseSymptomsKG'])) for item in cached]

        cypher = """
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
            records = list(session.run(cypher, symptomNamesLower=s_lower))
            disease_data = [
                {
                    "Disease": rec["Disease"], "Confidence": float(rec["confidence_score"]),
                    "MatchedSymptoms": list(rec["MatchedSymptoms"]), "AllDiseaseSymptomsKG": list(rec["AllDiseaseSymptomsKG"])
                } for rec in records if rec.get("Disease")
            ]
            set_cached(cache_key, [dict(d, MatchedSymptoms=tuple(d['MatchedSymptoms']), AllDiseaseSymptomsKG=tuple(d['AllDiseaseSymptomsKG'])) for d in disease_data])
            return disease_data
        except Exception as e: logger.error(f"Error querying diseases: {e}"); return []

    def _query_treatments_with_session(self, session, disease: str) -> Tuple[List[str], float]:
        d_lower = disease.strip().lower()
        cache_key = {"type": "treatment_query_kg", "disease": d_lower}
        if cached := get_cached(cache_key): return cached
        
        cypher = """
        MATCH (d:disease)-[:TREATED_BY]->(t:treatment) WHERE toLower(d.Name) = $diseaseNameLower
        RETURN t.Name as Treatment ORDER BY Treatment
        """
        try:
            records = list(session.run(cypher, diseaseNameLower=d_lower))
            treatments = sorted(list(set(rec["Treatment"].strip() for rec in records if rec.get("Treatment"))))
            conf = 1.0 if treatments else 0.0
            return set_cached(cache_key, (treatments, conf))
        except Exception as e: logger.error(f"Error querying treatments: {e}"); return [], 0.0

    def retrieve_rag_context(self, query: str) -> Tuple[List[str], float]:
        k = 5
        cache_key = {"type": "rag_retrieval_topk_chunks_and_scores", "query": query, "k": k}
        if cached := get_cached(cache_key): return list(cached['chunks']), float(cached['avg_score'])
        if not self.vectordb or not self.embedding_model: return [], 0.0
        try:
            docs_scores = self.vectordb.similarity_search_with_score(query, k=k)
            chunks, scores = [], []
            for doc, score_val in docs_scores:
                if isinstance(score_val, (int, float)):
                    sim_score = max(0.0, min(1.0, 1 - float(score_val if score_val >=0 else 0.0)))
                    if doc and doc.page_content:
                        chunks.append(doc.page_content)
                        scores.append(sim_score)
            srag = sum(scores) / len(scores) if scores else 0.0
            logger.info(f"📄 RAG: Processed {len(chunks)} chunks. S_RAG: {srag:.4f}")
            return set_cached(cache_key, {'chunks': chunks, 'avg_score': srag})
        except Exception as e: logger.error(f"⚠️ Error RAG: {e}", exc_info=True); return [], 0.0
            
    def select_context(self, kg_results, s_kg, rag_chunks, s_rag, is_symptom_query) -> Optional[Dict[str, Any]]:
        logger.info("📦 Context Selection. SymptomQ: %s, S_KG: %.4f, S_RAG: %.4f", is_symptom_query, s_kg, s_rag)
        kg_thresh, rag_thresh, high_kg_thresh = THRESHOLDS["kg_context_selection"], THRESHOLDS["rag_context_selection"], THRESHOLDS["high_kg_context_only"]
        selected = {}
        kg_has_data = kg_results and kg_results.get("identified_diseases_data")

        if is_symptom_query and s_kg > high_kg_thresh and kg_has_data:
            logger.info("📦 High KG Rule: Selecting KG ONLY.")
            selected["kg"] = kg_results
        else:
            if is_symptom_query:
                if s_kg >= kg_thresh and kg_has_data: selected["kg"] = kg_results
            if s_rag >= rag_thresh and rag_chunks: selected["rag"] = rag_chunks
        
        if not selected: logger.info("📦 No context source met thresholds."); return None
        logger.info("📦 Context Selected: %s", ', '.join(selected.keys()))
        return selected

    def generate_initial_answer(self, query: str, selected_context: Optional[Dict[str, Any]], user_type: str) -> str:
        cache_key = {"type": "initial_answer", "query": query, "user_type": user_type, "context_hash": abs(hash(json.dumps(selected_context, sort_keys=True)))}
        if cached := get_cached(cache_key): return cached

        sys_prompt = get_system_prompt(user_type)
        ctx_info_prompt, ctx_type_desc = "", ""
        
        if not selected_context:
            ctx_type_desc = "No external medical knowledge provided. Generate a minimal placeholder answer indicating lack of specific information. Do NOT answer using general knowledge. Placeholder like 'No specific relevant information was found.'"
            final_prompt = f"{sys_prompt}\n\n{ctx_type_desc}\n\nUser Query: \"{query}\"\n\nMinimal Placeholder Answer:"
        else:
            ctx_parts = []
            if "kg" in selected_context:
                kg_data = selected_context["kg"]
                diag = kg_data.get("kg_content_diagnosis_data_for_llm")
                if diag and diag.get("confidence",0) > 0:
                    s = f"KG Info: Potential Condition: {diag['disease_name']} (Conf: {diag['confidence']:.2f}). "
                    if kg_data.get('kg_matched_symptoms'): s += f"Matched Symptoms: {', '.join(kg_data['kg_matched_symptoms'])}. "
                    other_kg = kg_data.get("kg_content_other", "")
                    if other_kg and "did not find" not in other_kg: s += other_kg
                    ctx_parts.append(s.strip())
            if "rag" in selected_context and selected_context["rag"]:
                ctx_parts.append("Relevant Passages:\n---\n" + "\n---\n".join(selected_context["rag"][:3]) + "\n---")

            if ctx_parts:
                ctx_info_prompt = "\n\n".join(ctx_parts)
                desc_map = {"kg_rag": "Based on KG and RAG docs...", "kg": "Based on KG...", "rag": "Based on RAG docs..."}
                key = "_".join(sorted(selected_context.keys()))
                ctx_type_desc = desc_map.get(key, "Based on available information...")
            else: # Selected context was passed but yielded no usable parts
                ctx_type_desc = "No effectively usable external info found. Generate a minimal placeholder. Do NOT use general knowledge. Placeholder: 'No specific relevant information was found.'"

            final_prompt = f"{sys_prompt}\n{ctx_type_desc}\n\n{ctx_info_prompt}\n\nUser Query: {query}\n\nAnswer:"
        
        try:
            initial_answer = self.local_generate(final_prompt, max_tokens=1000)
            placeholder_frags = ["no specific relevant information was found", "lack of specific information"]
            is_placeholder = not initial_answer.strip() or any(f in initial_answer.lower() for f in placeholder_frags)

            if (selected_context and ctx_parts) and is_placeholder: # Context provided, but got placeholder
                initial_answer = "No specific relevant information was found in external knowledge sources." # Force consistent placeholder
                logger.warning("Overriding unexpected placeholder: %s", initial_answer)
            elif (not selected_context or not ctx_parts) and not is_placeholder: # No context, but didn't get placeholder
                initial_answer = "No specific relevant information was found in external knowledge sources."
                logger.warning("Overriding non-placeholder due to no context: %s", initial_answer)
            return set_cached(cache_key, initial_answer)
        except ValueError as e: raise ValueError("Error generating initial answer.") from e

    def reflect_on_answer(self, query, initial_answer, selected_context) -> Tuple[str, Optional[str]]:
        ctx_for_reflect = self._format_context_for_reflection(selected_context)
        cache_key = {"type": "reflection", "query": query, "initial_answer": initial_answer, "context_hash": abs(hash(ctx_for_reflect))}
        if cached := get_cached(cache_key): return cached
        if not self.llm: return ('incomplete', 'Reflection LLM unavailable.')

        placeholder_frag = "no specific relevant information was found"
        prompt = f'''Evaluate 'Initial Answer' for completeness regarding 'User Query' and 'Context'.
        If 'Initial Answer' is placeholder (like "{placeholder_frag}"), evaluation is "incomplete", missing_information is original 'User Query' topic.
        Else, if incomplete, identify missing info from 'User Query' perspective.
        JSON: {{"evaluation": "complete" or "incomplete", "missing_information": "Description or empty string"}}
        User Query: "{query}"
        Context:\n{ctx_for_reflect}
        Initial Answer:\n"{initial_answer}"'''
        try:
            response = self.local_generate(prompt, max_tokens=300)
            if match := re.search(r'\{[\s\S]*\}', response):
                data = json.loads(match.group(0))
                eval_res = data.get("evaluation", "incomplete").lower()
                missing_desc = data.get("missing_information", "").strip()
                if eval_res == 'complete': missing_desc = None
                elif not missing_desc: missing_desc = f"Answer incomplete, specific missing info not detailed."
                return set_cached(cache_key, (eval_res, missing_desc))
            return set_cached(cache_key, ('incomplete', "Reflection parse error: no JSON"))
        except Exception as e: raise ValueError(f"Error during reflection: {e}") from e

    def _format_context_for_reflection(self, selected_context: Optional[Dict[str, Any]]) -> str:
        parts = []
        if selected_context:
            if "kg" in selected_context:
                kg_data, kg_str = selected_context["kg"], "KG Info:\n"
                diag = kg_data.get("kg_content_diagnosis_data_for_llm")
                if diag and diag.get("disease_name") and diag.get("confidence",0)>0: 
                    kg_str += f"  Potential Condition: {diag['disease_name']} (Conf: {diag.get('confidence',0):.2f})\n"
                if kg_data.get("kg_matched_symptoms"): kg_str += f"  Matched Symptoms: {', '.join(kg_data['kg_matched_symptoms'])}\n"
                if kg_data.get("kg_treatments"): kg_str += f"  Treatments: {', '.join(kg_data['kg_treatments'])}\n"
                other_kg = kg_data.get("kg_content_other","")
                if other_kg and "did not find" not in other_kg: kg_str += "\n" + other_kg[:150] + "..."
                if len(kg_str.splitlines()) > 1 : parts.append(kg_str.strip())
            if "rag" in selected_context and selected_context["rag"]:
                valid_chunks = [c for c in selected_context["rag"] if isinstance(c,str)]
                if valid_chunks: parts.append("Relevant Passages:\n---\n" + "\n---\n".join(valid_chunks[:2]) + "\n---") # Top 2 for brevity
        return "\n\n".join(parts) if parts else "None"

    def get_supplementary_answer(self, query: str, missing_info_description: str, user_type: str) -> str:
        cache_key = {"type": "supplementary_answer", "missing_info_hash": abs(hash(missing_info_description)), "query_hash": abs(hash(query)), "user_type": user_type}
        if cached := get_cached(cache_key): return cached
        if not self.llm: return "\n\n-- Additional Information --\nSupplementary info unavailable (LLM down)."

        sys_prompt = get_system_prompt(user_type)
        supp_prompt = f'''{sys_prompt}
        You provide *only* specific missing details for an incomplete medical answer.
        Original Query: "{query}"
        Missing Info: "{missing_info_description}"
        Provide ONLY the supplementary information. Include evidence/sources (URLs, [Source Name], [General Medical Knowledge]). If you cannot find info, state concisely.
        '''
        try:
            supp_answer = self.local_generate(supp_prompt, max_tokens=750).strip()
            if not supp_answer: supp_answer = "The AI could not find specific additional information."
            final_text = "\n\n-- Additional Information --\n" + supp_answer
            return set_cached(cache_key, final_text)
        except ValueError as e:
            final_text = f"\n\n-- Additional Information --\nError finding supplementary info: {e}"
            return set_cached(cache_key, final_text)

    def collate_answers(self, initial_answer: str, supplementary_answer: str, user_type: str) -> str:
        cache_key = {"type": "final_collation", "initial_answer_hash": abs(hash(initial_answer)), "supplementary_answer_hash": abs(hash(supplementary_answer)), "user_type": user_type}
        if cached := get_cached(cache_key): return cached
        if not self.llm: return f"{initial_answer}\n\n{supplementary_answer}"

        supp_content = supplementary_answer.split("-- Additional Information --\n", 1)[-1].strip()
        if not supp_content or "could not find specific additional information" in supp_content.lower() or "error occurred while trying to find additional information" in supp_content.lower():
            return initial_answer.strip() + supplementary_answer.strip()

        sys_prompt = get_system_prompt(user_type)
        collate_prompt = f'''{sys_prompt}
        Combine 'Initial Answer' and 'Supplementary Information' into one coherent response.
        Remove redundancy. Preserve facts & sources. Format with markdown. NO disclaimer/pathway.
        Initial Answer:\n"{initial_answer}"
        Supplementary Info:\n"{supplementary_answer}"
        Combined Answer:'''
        try:
            combined = self.local_generate(collate_prompt, max_tokens=1500)
            return set_cached(cache_key, combined)
        except ValueError as e:
            err_msg = f"\n\n-- Collation Failed --\nError: {e}\n\n"
            return set_cached(cache_key, initial_answer.strip() + err_msg + supplementary_answer.strip())

    def reset_conversation(self): logger.info("🔄 Resetting chatbot internal state.")

    def process_user_query(self, user_query, user_type, confirmed_symptoms=None, original_query_if_followup=None) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        logger.info("--- Processing User Query: '%s' ---", user_query[:50])
        processed_query, current_symptoms, is_symptom_q, medical_ok = user_query, [], False, False

        if confirmed_symptoms is not None:
            logger.info("--- Step 1: Symptom Confirmation Rerun ---")
            if not user_query: return "Error during symptom confirmation.", "display_final_answer", None
            is_symptom_q = True
            current_symptoms = sorted(list(set(s.strip().lower() for s in confirmed_symptoms if isinstance(s,str) and s.strip())))
            medical_ok = True
        else:
            logger.info("--- Step 1: Initial Query Processing ---")
            medical_ok, medical_reason = self.is_medical_query(user_query)
            if not medical_ok: return f"I only answer medical questions. ({medical_reason})", "display_final_answer", None
            current_symptoms, _ = self.extract_symptoms(user_query)
            is_symptom_q = self.is_symptom_related_query(user_query)
        
        if not medical_ok: return "Internal error: medical check failed unexpectedly.", "display_final_answer", None

        kg_results, s_kg, rag_chunks, s_rag = {}, 0.0, [], 0.0
        if is_symptom_q and current_symptoms:
            kg_results = self.knowledge_graph_agent(processed_query, current_symptoms)
            s_kg = kg_results.get("top_disease_confidence", 0.0)
        if self.vectordb and self.embedding_model:
            rag_chunks, s_rag = self.retrieve_rag_context(processed_query)

        if confirmed_symptoms is None and is_symptom_q and \
           len(kg_results.get("identified_diseases_data",[])) > 0 and \
           0.0 < s_kg < THRESHOLDS["disease_symptom_followup_threshold"]:
            
            top_disease_data = kg_results["identified_diseases_data"][0]
            all_kg_symps_lower = set(s.lower() for s in kg_results.get("all_disease_symptoms_kg_for_top_disease",[]) if isinstance(s,str))
            initial_symps_lower = set(s.lower() for s in current_symptoms if isinstance(s,str))
            suggested_symps_lower = sorted(list(all_kg_symps_lower - initial_symps_lower))
            
            # Get original casing for suggested symptoms
            original_case_map = {s.lower(): s for s in kg_results.get("all_disease_symptoms_kg_for_top_disease",[]) if isinstance(s,str)}
            suggested_original_case = [original_case_map[s_low] for s_low in suggested_symps_lower if s_low in original_case_map]

            if suggested_original_case:
                ui_payload = {"symptom_options": {top_disease_data.get("Disease", "Potential Condition"): suggested_original_case}, "original_query": processed_query}
                return "To help, please confirm additional symptoms:", "show_symptom_ui", ui_payload
        
        selected_ctx = self.select_context(kg_results, s_kg, rag_chunks, s_rag, is_symptom_q)
        initial_sources = [k.upper() for k in selected_ctx.keys()] if selected_ctx else []
        
        try: initial_answer = self.generate_initial_answer(processed_query, selected_ctx, user_type)
        except ValueError as e:
            path_info = ", ".join(initial_sources) if initial_sources else "LLM (Initial Phrasing)"
            return f"Error generating initial answer: {e}\n\n<span style='font-size:0.8em;color:grey;'>*Attempted: {path_info} (Failed)*</span>", "display_final_answer", None

        reflection_failed, eval_res, missing_desc = False, 'complete', None
        try: eval_res, missing_desc = self.reflect_on_answer(processed_query, initial_answer, selected_ctx)
        except Exception as e: reflection_failed = True; eval_res = 'incomplete'; missing_desc = f"Reflection failed ({e})."
        
        final_answer = initial_answer
        supp_triggered = False
        if eval_res == 'incomplete':
            supp_triggered = True
            desc_for_supp = missing_desc or f"Gap for query: {processed_query[:50]}..."
            supp_answer = self.get_supplementary_answer(processed_query, desc_for_supp, user_type)
            final_answer = self.collate_answers(initial_answer, supp_answer, user_type)
            
        final_answer_triaged = self.enhance_with_triage_detection(processed_query, final_answer, user_type)

        final_path_parts = list(set(initial_sources))
        if supp_triggered or not initial_sources or reflection_failed: final_path_parts.append("LLM (General Knowledge)")
        if not reflection_failed: final_path_parts.append("Reflection Agent")
        elif reflection_failed and eval_res == 'incomplete': final_path_parts.append("Reflection Agent (Failed)") # only if it ran and failed
        
        path_info_str = ", ".join(sorted(list(set(final_path_parts)))) or "Unknown Pathway"
        disclaimer = "\n\nIMPORTANT MEDICAL DISCLAIMER: This is not professional medical advice..." # Shortened for brevity
        path_note = f"<span style='font-size:0.8em;color:grey;'>*Sources: {path_info_str.strip()}*</span>"
        return f"{final_answer_triaged.strip()}{disclaimer}\n\n{path_note}", "display_final_answer", None

def display_symptom_checklist(symptom_options: Dict[str, List[str]], original_query: str):
    st.subheader("Confirm Your Symptoms")
    user_type_key = st.session_state.get("user_type_select", "family")
    form_key = f"symptom_form_{abs(hash(original_query))}_{user_type_key}_{st.session_state.get('form_timestamp',0)}"
    
    local_set_key = f'{form_key}_local_symptoms'
    text_input_key = f"{form_key}_other_symptoms_text"
    prev_text_key = f"{text_input_key}_previous_value"

    if local_set_key not in st.session_state:
        st.session_state[local_set_key] = set()
        if text_input_key in st.session_state: del st.session_state[text_input_key]
        if prev_text_key in st.session_state: del st.session_state[prev_text_key]

    current_other_text = st.session_state.get(text_input_key, "")
    if current_other_text:
        st.session_state[local_set_key].update(s.strip().lower() for s in current_other_text.split(',') if s.strip())

    all_unique_suggested = sorted(list(set(
        s.strip() for sl in symptom_options.values() if isinstance(sl,list) for s in sl if isinstance(s,str) and s.strip()
    )))

    with st.form(form_key):
        st.markdown("Please check all symptoms that apply:")
        if not all_unique_suggested: st.info("No specific additional symptoms to suggest.")
        else:
            num_cols = min(4, len(all_unique_suggested))
            cols = st.columns(num_cols) if num_cols > 0 else [st]
            for i, symptom_orig_case in enumerate(all_unique_suggested):
                col = cols[i % num_cols] if num_cols > 0 else cols[0]
                cb_key = f"{form_key}_cb_{abs(hash(symptom_orig_case))}"
                s_lower = symptom_orig_case.strip().lower()
                initial_state = s_lower in st.session_state.get(local_set_key, set())
                
                is_checked = col.checkbox(symptom_orig_case, key=cb_key, value=initial_state)
                # Update local set based on current checkbox state (Streamlit handles this via key)
                if is_checked: st.session_state[local_set_key].add(s_lower)
                else: st.session_state[local_set_key].discard(s_lower)
        
        st.markdown("**Other Symptoms (comma-separated):**")
        other_symptoms_text_val = st.text_input("", key=text_input_key, value=st.session_state.get(text_input_key,""))

        if submit_button := st.form_submit_button("Confirm and Continue"):
            logger.info(f"Symptom form submitted for: '{original_query[:50]}...'.")
            # Add typed symptoms from text input to the set upon submission
            if other_symptoms_text_val:
                st.session_state[local_set_key].update(s.strip().lower() for s in other_symptoms_text_val.split(',') if s.strip())

            final_symptoms = sorted(list(st.session_state.get(local_set_key, set())))
            st.session_state.confirmed_symptoms_from_ui = final_symptoms
            st.session_state.ui_state = {"step": "input", "payload": None}
            
            # Clean up form-specific state
            for k in [local_set_key, text_input_key, prev_text_key]:
                if k in st.session_state: del st.session_state[k]
            st.session_state.form_timestamp = datetime.now().timestamp() # For next form instance

def create_user_type_selector():
    # Initialize 'last_user_type' to avoid error on first check, default to initial selectbox value
    if 'last_user_type' not in st.session_state:
        st.session_state.last_user_type = "User / Family" 

    # The selectbox value is stored in st.session_state.user_type_select by its key
    selected_type = st.sidebar.selectbox(
        "Who is asking?",
        ["User / Family", "Physician"],
        key="user_type_select", # This key controls the widget's state
        index=["User / Family", "Physician"].index(st.session_state.get("user_type_select", "User / Family")),
        help="Select user type. This affects response style and may reset chat."
    )

    # Compare current widget value with the value from the *previous complete run of this function*
    if selected_type != st.session_state.last_user_type:
        logger.info(f"User type changed from '{st.session_state.last_user_type}' to '{selected_type}'. Flagging for reset.")
        st.session_state.reset_requested_by_type_change = True
        # Update last_user_type *immediately* after detecting the change.
        # This ensures that if a reset occurs, the *next* run of this function
        # will see last_user_type as the *new* type, preventing another immediate reset.
        st.session_state.last_user_type = selected_type

def main():
    logger.info("--- Streamlit App Start ---")
    st.set_page_config(page_title="DxAI-Agent", page_icon=f"data:image/png;base64,{icon}", layout="wide")
    
    create_user_type_selector() # Handles its own state for 'last_user_type'
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
        
        # Clear local form state more carefully
        keys_to_delete = [k for k in st.session_state if k.startswith("symptom_form_")]
        for k_del in keys_to_delete: del st.session_state[k_del]
        
        del st.session_state.reset_requested_by_type_change # Clear the flag
        logger.info("Reset complete. Rerunning.")
        st.rerun()

    try:
        logo = Image.open(image_path)
        c1, c2 = st.columns([1,10]); c1.image(logo,width=100); c2.markdown("# DxAI-Agent")
    except Exception: st.markdown("# DxAI-Agent")

    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False
        st.session_state.chatbot = None
        st.session_state.init_status = (False, "Init not started.")
        with st.spinner("Initializing chat assistant..."):
            try:
                st.session_state.chatbot = DocumentChatBot()
                success, msg = st.session_state.chatbot.initialize_qa_chain()
                st.session_state.init_status = (success, msg)
                st.session_state.chatbot_initialized = success
            except Exception as e:
                st.session_state.init_status = (False, f"Critical init error: {e}")
                st.session_state.chatbot_initialized = False
    
    init_success, init_msg = st.session_state.get('init_status', (False, "Status unknown."))
    is_interaction_enabled = st.session_state.get('chatbot_initialized', False) and st.session_state.get('chatbot')

    if 'ui_state' not in st.session_state: st.session_state.ui_state = {"step": "input", "payload": None}
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'processing_input_payload' not in st.session_state: st.session_state.processing_input_payload = None
    if 'confirmed_symptoms_from_ui' not in st.session_state: st.session_state.confirmed_symptoms_from_ui = None
    if 'original_query_for_symptom_rerun' not in st.session_state: st.session_state.original_query_for_symptom_rerun = None
    if 'form_timestamp' not in st.session_state: st.session_state.form_timestamp = datetime.now().timestamp()

    st.sidebar.info("DxAI-Agent for medical queries.")
    st.sidebar.success(f"Status: {init_msg}") if is_interaction_enabled else st.sidebar.error(f"Init Failed: {init_msg}")

    tab1, tab2 = st.tabs(["Chat", "About"])
    with tab1:
        for i, (msg_content, is_user_msg) in enumerate(st.session_state.messages):
            with st.chat_message("user" if is_user_msg else "assistant"):
                st.markdown(msg_content, unsafe_allow_html=not is_user_msg)
                if not is_user_msg and i == len(st.session_state.messages) - 1 and \
                   st.session_state.ui_state["step"] == "input" and is_interaction_enabled:
                    cols_fb = st.columns([0.05, 0.05, 0.9])
                    user_q_for_fb = next((m[0] for j,m in enumerate(st.session_state.messages) if j < i and m[1]), "") # Get preceding user query
                    fb_key_base = f"fb_{i}_{abs(hash(msg_content.split('IMPORTANT MEDICAL DISCLAIMER:',1)[0]))}"
                    if f'{fb_key_base}_up' not in st.session_state and f'{fb_key_base}_down' not in st.session_state:
                        if cols_fb[0].button("👍", key=f'{fb_key_base}_up_btn', help="Good response"):
                            vote_message(user_q_for_fb, msg_content, "thumbs_up", current_user_type)
                            st.session_state[f'{fb_key_base}_up'] = True; st.toast("Thanks!"); st.rerun()
                        if cols_fb[1].button("👎", key=f'{fb_key_base}_down_btn', help="Bad response"):
                            vote_message(user_q_for_fb, msg_content, "thumbs_down", current_user_type)
                            st.session_state[f'{fb_key_base}_down'] = True; st.toast("Thanks!"); st.rerun()
                    elif f'{fb_key_base}_up' in st.session_state: cols_fb[0].button("👍", disabled=True)
                    elif f'{fb_key_base}_down' in st.session_state: cols_fb[1].button("👎", disabled=True)
        
        st.write(" \n" * 2)
        if not is_interaction_enabled:
            st.error("Chat assistant failed to initialize.")
            st.chat_input("Initializing...", disabled=True, key="disabled_init_input")
        elif st.session_state.ui_state["step"] == "confirm_symptoms":
            payload = st.session_state.ui_state.get("payload")
            if not payload or "symptom_options" not in payload or "original_query" not in payload:
                st.error("Error showing symptom checklist. Resetting...")
                st.session_state.ui_state = {"step": "input", "payload": None}
                if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                st.rerun()
            else:
                display_symptom_checklist(payload["symptom_options"], payload["original_query"])
                st.chat_input("Confirm symptoms above...", disabled=True, key="disabled_symptom_input")
        elif st.session_state.ui_state["step"] == "input":
            if user_query := st.chat_input("Ask your medical question...", disabled=not is_interaction_enabled, key="main_chat_input"):
                st.session_state.messages.append((user_query, True))
                if st.session_state.get('chatbot'): st.session_state.chatbot.reset_conversation()
                st.session_state.form_timestamp = datetime.now().timestamp()
                for k_suffix in ['_from_ui', '_for_symptom_rerun']:
                     if f'confirmed_symptoms{k_suffix}' in st.session_state: del st.session_state[f'confirmed_symptoms{k_suffix}']
                # Clear previous feedback states
                for k_fb in [k for k in st.session_state if k.startswith("fb_")]: del st.session_state[k_fb]

                st.session_state.processing_input_payload = {"query": user_query, "confirmed_symptoms": None}
                st.rerun()

        if st.session_state.get('confirmed_symptoms_from_ui') is not None:
            confirmed_symps = st.session_state.confirmed_symptoms_from_ui
            original_q = st.session_state.get('original_query_for_symptom_rerun')
            del st.session_state.confirmed_symptoms_from_ui # Clear immediately
            if not original_q:
                st.error("Internal error with symptom confirmation. Please retry.")
                if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                st.session_state.ui_state = {"step":"input", "payload":None}; st.rerun()
            else:
                st.session_state.processing_input_payload = {"query": original_q, "confirmed_symptoms": confirmed_symps}
                st.rerun()

        if st.session_state.get('processing_input_payload') is not None:
            payload_data = st.session_state.processing_input_payload
            st.session_state.processing_input_payload = None # Clear immediately
            
            if not st.session_state.get('chatbot') or not st.session_state.get('chatbot_initialized'):
                st.error("Chatbot not ready."); st.rerun()
            
            query_proc = payload_data.get("query","")
            confirmed_s = payload_data.get("confirmed_symptoms")

            if not query_proc:
                st.error("Empty query received for processing."); st.rerun()
            
            with st.spinner("Thinking..."):
                try:
                    resp_text, ui_act, ui_pay = st.session_state.chatbot.process_user_query(
                        query_proc, current_user_type, confirmed_s
                    )
                    if ui_act == "display_final_answer":
                        st.session_state.messages.append((resp_text, False))
                        st.session_state.ui_state = {"step": "input", "payload": None}
                        if confirmed_s and 'original_query_for_symptom_rerun' in st.session_state:
                            del st.session_state.original_query_for_symptom_rerun
                    elif ui_act == "show_symptom_ui":
                        st.session_state.messages.append((resp_text, False)) # Prompt message
                        st.session_state.ui_state = {"step": "confirm_symptoms", "payload": ui_pay}
                        st.session_state.form_timestamp = datetime.now().timestamp()
                        if ui_pay and ui_pay.get("original_query"):
                            st.session_state.original_query_for_symptom_rerun = ui_pay["original_query"]
                        else: # Error case
                            st.error("Error preparing symptom checklist. Resetting."); st.session_state.ui_state={"step":"input"};
                    # else: "none" or unknown, default to input state (already handled if not changed)
                except Exception as e:
                    logger.error(f"Error during query processing: {e}", exc_info=True)
                    st.session_state.messages.append((f"Sorry, an error occurred: {e}", False))
                    st.session_state.ui_state = {"step": "input", "payload": None}
                st.rerun()

        st.divider()
        if st.button("Reset Conversation", key="reset_main", disabled=not is_interaction_enabled):
            if st.session_state.get('chatbot'): st.session_state.chatbot.reset_conversation()
            st.session_state.clear() # Clears ALL session state
            st.rerun() # Will re-init everything including first-run user_type
        
        st.divider()
        st.subheader("🩺 Detailed Feedback")
        with st.form(key="detailed_fb_form", clear_on_submit=True):
            fb_text = st.text_area("Comments...", height=100, disabled=not is_interaction_enabled)
            if st.form_submit_button("Submit Feedback", disabled=not is_interaction_enabled) and fb_text:
                submit_feedback(fb_text, st.session_state.get('messages',[]), current_user_type)
                st.success("Thank you!")

    with tab2:
        st.markdown("## Medical Chat Assistant\n\nDetails about how it works...") # Shortened

if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
