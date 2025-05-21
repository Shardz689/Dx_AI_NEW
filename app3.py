
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
image_path_str = "Zoom My Life.jpg" # Ensure this is a string
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
try:
    ICON_B64 = get_image_as_base64(image_path_str) 
except Exception as e_icon: # Catch any unexpected error during this global setup
    logger.error(f"Failed to load/encode favicon from '{image_path_str}': {e_icon}")
    ICON_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" 

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

    def local_generate(self, messages_for_llm: List[Dict[str, str]], 
                       max_tokens: int = 500) -> str:
        if not self.llm: 
            raise ValueError("LLM is not initialized. Cannot generate response.")
        if not messages_for_llm:
            raise ValueError("No messages to send to LLM.")
        
        # logger.debug(f"Final messages to LLM ({len(messages_for_llm)} items): {json.dumps(messages_for_llm, indent=2)[:500]}...")
        try:
            response = self.llm.invoke(messages_for_llm, config={"max_output_tokens": max_tokens}) # type: ignore
            if response and response.content: 
                return response.content
            logger.warning("LLM invoke returned empty or None response content.")
            raise ValueError("LLM returned empty response content.") 
        except Exception as e: 
            logger.error(f"Error during Gemini LLM generation: {e}", exc_info=True)
            raise ValueError(f"LLM generation failed: {e}") from e
            
    def get_system_prompt(self, user_type: str) -> str:
        logger.debug(f"get_system_prompt called with user_type: '{user_type}'")
        
        normalized_user_type = ""
        if user_type == "User / Family" or user_type == "family":
            normalized_user_type = "family"
        elif user_type == "Physician" or user_type == "physician":
            normalized_user_type = "physician"
        else:
            logger.warning(f"Unknown user_type '{user_type}' in get_system_prompt, defaulting to 'family'.")
            normalized_user_type = "family"
    
        base_prompt = "You are MediAssist, an AI assistant. Your primary goal is to provide helpful medical information. "
        
        common_instructions_for_all_users = ( # Renamed for clarity
            "When answering, prioritize information from the 'Context Provided' (which may include 'Knowledge Graph Information' and 'Relevant Passages from Internal Data').\n"
            "1. **If context is sufficient:** Base your answer primarily on this context. If the context contains mentions of specific source names (e.g., specific studies, document titles within the internal data), include these attributions. If the context explicitly contains URLs, you may reproduce these URLs as clickable Markdown links (e.g., [Descriptive Link Text from Context](URL_FROM_CONTEXT)) if they are directly relevant and appear reliable.\n"
            "2. **If context is insufficient or you need to use general knowledge:** You may supplement with your general medical knowledge. When doing so:\n"
            "   a. If you reference information generally attributable to well-known, reputable medical organizations (like the CDC, WHO, NIH, Mayo Clinic, American Heart Association, American Diabetes Association, etc.), **mention the organization by name.**\n"
            "   b. For these highly reputable organizations mentioned in point 2a, if you are confident in recalling their main public-facing homepage URL or a very stable, top-level topic URL related to the general subject (e.g., the CDC's main page on 'flu' if discussing flu prevention), you MAY provide it as a clickable Markdown link, like: [CDC General Flu Information](https://www.cdc.gov/flu/index.htm). **Only do this for very common, primary URLs of these major organizations.**\n"
            "   c. **CRITICALLY: Do NOT invent or guess specific deep URLs, publication URLs, or any URL if you are not highly confident it's a correct, stable, and general public entry point for that organization.** It is better to only mention the organization's name than to provide an incorrect or misleading link.\n"
            "   d. If you cannot find specific information for a very niche query, state that the information is not available in the provided context or your general knowledge.\n"
            "3. **Overall:** Be cautious. Prioritize accuracy. Do not present your general knowledge as if it came directly from the provided context unless it aligns. "
        )
    
        if normalized_user_type == "physician":
            logger.info("Using PHYSICIAN system prompt.")
            return base_prompt + common_instructions_for_all_users + ( # common_instructions_for_all_users included
                "\n**Persona: You are an AI consultant assisting a fellow medical professional.**\n"
                "Respond with professional medical terminology, consider differential diagnoses, and provide detailed clinical insights and evidence-based recommendations. Assume high medical literacy. "
                "If specific guidelines (e.g., RSSDI, ADA, AHA) are requested: first, check if they are present in your provided context. If so, cite or summarize them accurately, including any source names or URLs explicitly part of that context (format URLs as [Name](URL)). "
                "If specific guidelines are not in your context, clearly state that the detailed guidelines are not available in the provided information, but you can offer general knowledge about the topic if applicable, mentioning general reputable sources or guideline bodies by name and potentially their main website if you are confident (as per instruction 2b in common instructions). "
                "When discussing pharmacotherapy for conditions like obesity, if the context provides a list of medications (e.g., GLP-1 agonists like semaglutide/liraglutide/tirzepatide, metformin, bupropion-naltrexone, phentermine-topiramate, orlistat), ensure your summary for the US context is comprehensive based on that provided list. " # Added based on obesity medication feedback
                "Structure responses with clear clinical reasoning. Address the user as a fellow medical professional about a case. Do not provide basic patient education unless asked. Do not provide patient-style triage advice."
            )
        else:  # family user
            logger.info("Using FAMILY USER system prompt (comprehensively updated with all feedback).")
            return base_prompt + common_instructions_for_all_users + (
                    "\n**Persona: You are a knowledgeable, clear, and supportive medical information assistant speaking to a family member or patient.** Your primary goal is to provide accurate, easy-to-understand guidance, offer appropriate reassurance, and help them understand necessary next steps, while maintaining a professional and calm demeanor.\n\n"
                    "**Tone and Style Requirements:**\n"
                    "- Use a supportive, direct, and understanding tone, while maintaining a consistently professional and calm demeanor.\n"
                    "- Acknowledge user concerns professionally. For example, instead of exclamations like 'Oh my!' or overly personal expressions of sympathy, use phrases like: 'I understand that a reading of [value] is concerning.' or 'It's understandable to be worried about [symptom]. Let's discuss what this might mean.' or 'Hearing about these symptoms must be difficult. It's important to address them.'\n"
                    "- Avoid overly casual language, interjections, or phrases that sound like personal emotional reactions. Your role is to be an informative and reassuring guide, not a conversational peer expressing personal feelings.\n"
                    "- Avoid medical jargon. If a medical term is absolutely necessary (e.g., from provided context), **immediately explain it in simple, everyday language.** For example, if 'HbA1c' is mentioned, explain it as 'Hemoglobin A1c, often called A1c, which is a blood test that gives an idea of your average blood sugar levels over the past 2 to 3 months'. Clearly distinguish it from a 'fasting blood glucose test' which measures sugar at a single point in time.\n"
                    "- Structure your response like a helpful, flowing conversation, using clear paragraphs and natural language. Avoid just listing facts or bullet points unless it significantly aids clarity for a list of actions.\n"
                    "- Be clear and direct, especially when advising on seeking medical care, but frame it actionably and with appropriate seriousness without causing undue panic.\n\n"
                    
                    "**Content Focus (Prioritize these in your response structure):**\n"
                    "1.  **Acknowledge Professionally & Directly:** Start by directly acknowledging the user's stated concern or symptoms in a calm, professional, yet understanding manner as described above.\n"
                    "2.  **Explain Simply:** If they ask about a condition or symptoms, provide a simple overview of what it generally means, drawing from context or general knowledge.\n"
                    "3.  **Provide Information (Context First):** Primarily use information from the 'Context Provided' (Internal Data, Knowledge Graph). If context mentions source names or has relevant URLs (like from the CDC or WHO websites if these URLs are *in the context*), you can share them in a user-friendly way (e.g., 'You can find more information on the [CDC Website](URL_from_context)').\n"
                    "4.  **Home Management & Lifestyle (Prioritize what's effective and safe):**\n"
                    "    *   **For general mild symptoms (like a common cold):** Suggest safe, generally accepted supportive care (rest, fluids, etc.).\n"
                    "    *   **For chronic conditions like Type 2 Diabetes or Obesity when 'home remedies' are asked:**\n"
                    "        a. **Strongly emphasize that comprehensive lifestyle changes—especially DIET and EXERCISE—are the most powerful and essential 'home management' strategies.** Explain *why* these are important in simple terms.\n"
                    "        b. **For Type 2 Diabetes Diet:** Detail the importance of managing carbohydrate intake. Explain this means limiting sugary foods, sweets, and sweetened drinks; controlling portions of starchy foods; choosing whole grains and high-fiber options; and the benefit of eating carbs with lean protein and healthy fats to help manage blood sugar spikes.\n"
                    "        c. **For Obesity Diet & Lifestyle:** Detail the importance of a calorie-controlled diet with whole foods, minimizing processed foods, sugary foods and drinks, sweets, and potentially fatty meats. Emphasize maintaining good protein intake and increasing vegetables. For exercise, mention a combination of cardiovascular exercise (like brisk walking, cycling) and strength/weight training.\n"
                    "        d. **Minor Adjuncts (with extreme caution):** If the provided context mentions minor traditional items (like certain herbs or teas for these chronic conditions), you may briefly list them *after* thoroughly explaining the primary lifestyle changes. However, you MUST clearly state that these have very limited scientific evidence for significant benefit, are not cures, and should NEITHER replace medical treatment NOR the crucial diet and exercise changes.\n"
                    "    *   **For Asthma when 'home remedies' are asked:**\n"
                    "        a. State clearly that asthma is a serious condition requiring a doctor's diagnosis and medical management, usually with **prescribed inhalers.**\n"
                    "        b. **Strongly warn that home remedies must NEVER replace prescribed asthma medication** and are NOT for treating asthma attacks or active symptoms like wheezing/difficulty breathing. This is dangerous.\n"
                    "        c. Mention that avoiding known asthma triggers and smoking cessation are important parts of *managing* asthma alongside medical treatment.\n"
                    "5.  **Next Steps & When to Seek Professional Care (Integrated Triage - CRITICAL):**\n"
                    "    - Weave this guidance naturally into your response. Do not use a separate heading like 'Triage Assessment'.\n"
                    "    - **Emergency:** If symptoms suggest an emergency (e.g., severe chest pain, difficulty breathing, a reported BP of 180/120 with symptoms), strongly and clearly advise: 'These symptoms sound quite serious, and it would be safest to get medical help right away. Please consider calling 911 or go to the nearest emergency room immediately because...'\n"
                    "    - **Urgent Care:** For less critical but still urgent issues, suggest: 'It sounds like these symptoms should be checked by a doctor fairly soon, perhaps within the next 24 hours, to understand what's going on and get appropriate care. You could visit an urgent care clinic or contacting your doctor would be a good idea because...'\n"
                    "    - **Primary Care:** For non-urgent concerns, advise: 'It would be a good idea to make an appointment with your doctor to discuss this further and get personalized advice, as they can properly assess...'\n"
                    "    - **Self-care:** If it seems manageable at home with monitoring, explain: 'This can often be managed at home with care. Here are some things you can try... However, if things don't improve in [X days] or if you start to feel worse, then you should definitely see your doctor.'\n"
                    "6.  **Specific Scenario - User Reports Very High Blood Pressure Reading:** If a user reports a specific, very high blood pressure reading (e.g., >180 systolic or >120 diastolic):\n"
                    "    a. Your primary advice MUST be emergency care. Then, briefly and secondarily, mention tips for accurate future home BP monitoring, framed as general advice that doesn't undermine the current emergency (e.g., 'For future checks, remember to rest, sit correctly, and use a good cuff. But for this reading now, emergency care is key.').\n"
                    "7.  **Explaining Medical Tests:** If you mention common medical tests (like HbA1c, lipid panel), briefly explain what they are for in simple terms. For example: 'Your doctor might suggest a Hemoglobin A1c test, often called A1c, which is a blood test that gives an idea of your average blood sugar control over the past 2 to 3 months.'\n"
                    "8.  **Flu vs. Cold & Antivirals:** When discussing flu-like symptoms:\n"
                    "    a. Differentiate that influenza ('the flu') is a specific virus, while 'colds' can be many viruses.\n"
                    "    b. Explain that antiviral medications (like Tamiflu) are for influenza, work best if started early (first 48 hours), require a doctor's assessment (often a flu test), and are prescription-only. State clearly that **antivirals do not work for common cold viruses.** Manage expectations.\n\n"
                    "**Always conclude by gently reminding the user that your information is not a substitute for professional medical advice and they should consult their doctor for personal health concerns.**\n\n"
                    
                    "**Example Response Models:**\n"
                    "Below are example responses to help guide the style and structure of your communications:\n\n"
                    
                    "**Example 1 - High Blood Pressure:**\n"
                    "Query: \"My blood pressure is very high right now (e.g. 180/120).\"\n"
                    "Response: \"I understand that a blood pressure reading of 180/120 is concerning. This is considered a hypertensive crisis level and requires immediate medical attention.\n\n"
                    "These readings indicate your blood pressure is significantly elevated, which can potentially damage blood vessels and organs. Even without symptoms, this level of blood pressure requires prompt medical evaluation.\n\n"
                    "Based on these numbers, it would be safest to get medical help right away. Please consider calling 911 or going to the nearest emergency room immediately because this level of blood pressure can lead to serious complications if not addressed promptly.\n\n"
                    "For future blood pressure checks, remember to sit quietly for 5 minutes before measuring, use a properly sized cuff, and position your arm at heart level. However, for this current reading, emergency care is the key priority.\n\n"
                    "High blood pressure is typically managed through a combination of lifestyle modifications and medication as determined by your doctor. Your healthcare provider will develop a personalized treatment plan after proper evaluation.\n\n"
                    "Remember that this information is not a substitute for professional medical advice. Please consult with your doctor for personal health concerns.\"\n\n"
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
        
        except Exception as e__logic:
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

    def generate_initial_answer(self, query: str, selected_context: Optional[Dict[str, Any]], 
                            user_type: str, conversation_history: Optional[List[Tuple[str,str]]]) -> str:
        logger.info(f"🧠 Initial Answer Gen. User Type: '{user_type}'. Query: '{query[:30]}...' History: {len(conversation_history) if conversation_history else 0} turns.")
        
        # Construct a comprehensive cache key
        context_dump_for_hash = json.dumps(selected_context, sort_keys=True, default=str) if selected_context else "None"
        history_dump_for_hash = str(conversation_history) if conversation_history else "None"
        
        cache_key_obj = {
            "type": "initial_answer_v4", # Increment version if prompt logic changes significantly
            "query": query, 
            "user_type": user_type, 
            "context_hash": abs(hash(context_dump_for_hash)),
            "history_hash": abs(hash(history_dump_for_hash))
        }
        
        if (cached := get_cached(cache_key_obj)) is not None: 
            logger.debug(f"Initial answer from cache for query: {query[:30]}...")
            return cached
    
        system_prompt_content = self.get_system_prompt(user_type) # This now has detailed source/link instructions
        # logger.debug(f"Initial Answer Gen - System Prompt used (start): {system_prompt_content[:200]}...") # Log more of the prompt
        
        context_info_for_prompt = ""
        context_parts_for_prompt: List[str] = []
        
        if selected_context:
            if "kg" in selected_context:
                kg_data = selected_context["kg"]
                kg_info_str_parts = ["Knowledge Graph Information:"]
                diag_data = kg_data.get("kg_content_diagnosis_data_for_llm")
                diag_confidence = float(diag_data.get("confidence", 0.0)) if diag_data else 0.0
    
                if diag_data and diag_confidence > 0.0:
                    disease_name = diag_data.get("disease_name", "an unidentifiable condition")
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
                if other_kg_content and "did not find" not in other_kg_content and other_kg_content.strip():
                    kg_info_str_parts.append(other_kg_content)
                
                if len(kg_info_str_parts) > 1: 
                    context_parts_for_prompt.append("\n".join(kg_info_str_parts))
    
            if "rag" in selected_context and selected_context["rag"]:
                # Changed "Documents" to "Internal Data" for user-facing consistency if needed, but prompt uses "Internal Data"
                context_parts_for_prompt.append("Relevant Passages from Internal Data:\n---\n" + "\n---\n".join(selected_context["rag"][:3]) + "\n---")
    
        current_turn_user_content = ""
        context_type_description_for_prompt = "" # This will be part of current_turn_user_content
    
        if not selected_context or not context_parts_for_prompt:
            context_type_description_for_prompt = (
                "You have not been provided with any specific external medical knowledge or document snippets from internal data sources or the knowledge graph for this query, "
                "or the available information did not meet relevance or confidence thresholds. "
                "Therefore, generate only a minimal placeholder answer that indicates lack of specific information from these provided sources "
                "(e.g., 'No specific relevant information was found in available knowledge sources to address this query.'). "
                "Do NOT attempt to answer the user query using your general knowledge in this step, unless explicitly permitted by the broader system instructions for handling insufficient context (e.g., by mentioning general reputable source names if appropriate and confident). "
                "Focus on the placeholder message."
            )
            current_turn_user_content = (
                f"{context_type_description_for_prompt.strip()}\n\n"
                f"Current User Query: \"{query}\"\n\n"
                "Minimal Placeholder Answer (be very concise and direct, adhering to the instruction above about general knowledge usage if context is truly empty):\n"
            )
        else:
            context_info_for_prompt = "\n\n".join(context_parts_for_prompt)
            active_ctx_keys = []
            if any("Knowledge Graph Information:" in part for part in context_parts_for_prompt): active_ctx_keys.append("kg")
            if any("Relevant Passages from Internal Data:" in part for part in context_parts_for_prompt): active_ctx_keys.append("rag")
            
            ctx_desc_key = "_".join(sorted(active_ctx_keys))
            desc_map = {
                "kg_rag": "Based on the following structured medical knowledge from a knowledge graph AND relevant passages from internal data, synthesize a comprehensive answer.",
                "kg": "Based on the following information from a medical knowledge graph, answer the user query. Focus on the provided KG information.",
                "rag": "Based on the following relevant passages from internal data, answer the user query. Focus on the provided document excerpts."
            }
            context_type_description_for_prompt = desc_map.get(ctx_desc_key, "Based on the available information from provided sources...")
            current_turn_user_content = (
                f"{context_type_description_for_prompt.strip()}\n\n"
                f"Context Provided From Internal Sources:\n{context_info_for_prompt}\n\n"
                f"Current User Query: {query}\n\n"
                "Answer (Synthesize your answer primarily from the 'Context Provided From Internal Sources'. "
                "Follow all system instructions regarding source attribution: if the context includes source names or URLs, incorporate them relevantly. "
                "If context is insufficient for specific requests (like detailed, latest guidelines not present in context), state that the specific information isn't available in the provided data, then use your general knowledge cautiously, attributing to general reputable source names and potentially their main URLs if you are confident and as per system instructions. "
                "For family users, integrate triage advice naturally into your response if the situation warrants it, following system instructions on urgency levels.):\n"
            )
    
        messages_for_llm = [{"role": "system", "content": system_prompt_content}]
        if conversation_history:
            for hist_user_msg, hist_bot_msg in conversation_history:
                messages_for_llm.append({"role": "user", "content": hist_user_msg})
                messages_for_llm.append({"role": "assistant", "content": hist_bot_msg})
        messages_for_llm.append({"role": "user", "content": current_turn_user_content})
        
        logger.debug(f"Initial Answer - Final User Message for LLM (start):\n{current_turn_user_content[:500]}...")
        try:
            initial_answer = self.local_generate(messages_for_llm, max_tokens=1000) # Pass the full message list
            logger.debug(f"Initial Answer Raw LLM Output (start): {initial_answer[:100]}")
        
            placeholder_frags = [
                "no specific relevant information was found", 
                "lack of specific information", 
                "unable to provide specific information",
                "cannot provide specific information from the available sources" # More variations
            ]
            initial_answer_lower = initial_answer.lower()
            is_placeholder = not initial_answer.strip() or any(f in initial_answer_lower for f in placeholder_frags)
    
            has_provided_context = selected_context and context_parts_for_prompt
            # If context was provided but LLM still gave a placeholder, it might be because context wasn't relevant enough.
            # The prompt now guides it to state this. So, we might not need to override as aggressively IF the LLM follows that.
            # However, if the LLM *fails* to state that context was insufficient and just gives a bare placeholder, that's an issue.
            
            if not has_provided_context and not is_placeholder:
                # This is a clearer case of not following instructions: no context, but generated a full answer instead of placeholder.
                logger.warning("LLM generated content despite no usable context and instruction for placeholder. Overriding with a standard placeholder.")
                initial_answer = "No specific relevant information was found in available knowledge sources to address this query."
                
            # Consider logging if has_provided_context is True but is_placeholder is also True,
            # to see if the LLM is correctly stating *why* it's giving a placeholder (e.g. context irrelevant).
            if has_provided_context and is_placeholder:
                logger.warning(f"LLM generated a placeholder response even though context was provided. LLM Output (start): {initial_answer[:150]}")
                # Here, we trust the LLM might have correctly determined the context wasn't useful,
                # as per the refined prompt. We won't override if it's a thoughtful placeholder.
                # If it's just "No info found." without explaining *why* context wasn't used, that's less ideal.
    
            return set_cached(cache_key_obj, initial_answer)
        except ValueError as e: # From local_generate
            logger.error(f"⚠️ Error during initial answer LLM call: {e}", exc_info=True)
            raise ValueError(f"AI processing error (initial answer generation): {e}") from e
            
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

    def get_supplementary_answer(self, query: str, missing_info_description: str, 
                                 user_type: str, conversation_history: Optional[List[Tuple[str,str]]]) -> str:
            logger.info(f"🌐 Gap Filling. User Type: '{user_type}'. Missing: {missing_info_description[:50]}... History: {len(conversation_history) if conversation_history else 0} turns.")
            cache_key_obj = {"type": "supplementary_answer_v3", "missing_info_hash": abs(hash(missing_info_description)), 
                         "query_hash": abs(hash(query)), "user_type": user_type,
                         "history_hash": abs(hash(str(conversation_history)))}
            if (cached := get_cached(cache_key_obj)) is not None: 
                logger.debug("Supplementary answer from cache.")
                return cached
    
            if not self.llm: 
                return "\n\n-- Additional Information --\nSupplementary information could not be generated because the AI model is unavailable."
    
            system_prompt_content = self.get_system_prompt(user_type)
            logger.debug(f"Supplementary Answer Gen - System Prompt used (start): {system_prompt_content[:150]}...")
            
            current_turn_user_content = f'''You are an AI assistant acting to provide *only* specific missing details to supplement a previous incomplete answer.
Your response should be suitable for the specified user type.
The original user query (for full context) was: "{query}"
The description of Information Missing (this is what you need to address) is: "{missing_info_description}"

Provide ONLY the supplementary information for the 'Missing Information'.
Do NOT restate the original query or prior information. Focus precisely on the gap.
Adhere to the system instructions regarding source attribution: if making medical claims, cite from the provided context if available, or mention general reputable source names (e.g., CDC, WHO) from your general knowledge. Reproduce URLs as clickable Markdown links only if they are explicitly in provided context and relevant. Do NOT invent URLs.
If you cannot find specific information, state this clearly.
Start your response directly.
    '''
            messages_for_llm = [{"role": "system", "content": system_prompt_content}]
            if conversation_history:
                for hist_user_msg, hist_bot_msg in conversation_history:
                    messages_for_llm.append({"role": "user", "content": hist_user_msg})
                    messages_for_llm.append({"role": "assistant", "content": hist_bot_msg})
            messages_for_llm.append({"role": "user", "content": current_turn_user_content})
    
            try:
                supplementary_answer = self.local_generate(messages_for_llm, max_tokens=750).strip()
                if not supplementary_answer: 
                    supplementary_answer = "The AI could not find specific additional information for the identified gap."
                logger.info("🌐 Supplementary Answer Generated successfully.")
                final_supplementary_text = "\n\n-- Additional Information --\n" + supplementary_answer
                return set_cached(cache_key_obj, final_supplementary_text)
            except ValueError as e:
                logger.error(f"⚠️ Error during supplementary answer LLM call: {e}", exc_info=True)
                error_msg = f"Sorry, an AI processing error occurred while trying to find additional information regarding: '{missing_info_description[:50]}...'"
                final_supplementary_text = f"\n\n-- Additional Information --\n{error_msg}"
                return set_cached(cache_key_obj, final_supplementary_text)

    def collate_answers(self, initial_answer: str, supplementary_answer: str, 
                        user_type: str, conversation_history: Optional[List[Tuple[str,str]]]) -> str:
            logger.info(f"✨ Final Answer Collation. User Type: '{user_type}'. History: {len(conversation_history) if conversation_history else 0} turns.")
            cache_key_obj = {"type": "final_collation_v3", "initial_answer_hash": abs(hash(initial_answer)), 
                         "supplementary_answer_hash": abs(hash(supplementary_answer)), "user_type": user_type,
                         "history_hash": abs(hash(str(conversation_history)))}
            if (cached := get_cached(cache_key_obj)) is not None: 
                logger.debug("Final collation from cache.")
                return cached
    
            if not self.llm: 
                logger.warning("LLM not available for collation. Concatenating answers.")
                return f"{initial_answer.strip()}\n\n{supplementary_answer.strip()}"
    
            supp_content_after_header = supplementary_answer.split("-- Additional Information --\n", 1)[-1].strip()
            if not supp_content_after_header or \
               "could not find specific additional information" in supp_content_after_header.lower() or \
               "error occurred while trying to find additional information" in supp_content_after_header.lower() or \
               "ai model is unavailable" in supp_content_after_header.lower():
                 logger.debug("Supplementary answer is empty, placeholder, or error. Appending directly.")
                 return initial_answer.strip() + "\n" + supplementary_answer.strip()
    
            system_prompt_content = self.get_system_prompt(user_type)
            logger.debug(f"Collate Answers - System Prompt used (start): {system_prompt_content[:150]}...")
            
            current_turn_user_content = f'''You are a medical communicator. Combine 'Initial Answer' and 'Supplementary Information' into one coherent response, suitable for the user type.
    Remove redundancy. If supplementary info adds little new value, prioritize the initial answer or integrate minimally.
    Preserve facts and source attributions (like [Source Name] or URLs formatted as [Name](URL) if they were in the inputs). Do NOT add new sources.
    Format clearly using markdown. Do NOT include "Initial Answer Part:" or "Supplementary Information Part:" headers.
    Do NOT include the medical disclaimer or source pathway note.
    
    Initial Answer Part:
    "{initial_answer}"
    
    Supplementary Information Part (ignore its "-- Additional Information --\\n" header):
    "{supplementary_answer}"
    
    Provide ONLY the combined, final answer content. Start directly.
    '''
            messages_for_llm = [{"role": "system", "content": system_prompt_content}]
            if conversation_history:
                for hist_user_msg, hist_bot_msg in conversation_history:
                    messages_for_llm.append({"role": "user", "content": hist_user_msg})
                    messages_for_llm.append({"role": "assistant", "content": hist_bot_msg})
            messages_for_llm.append({"role": "user", "content": current_turn_user_content})
    
            try:
                combined_answer_content = self.local_generate(messages_for_llm, max_tokens=1500)
                logger.info("✨ Final Answer Collated successfully.")
                return set_cached(cache_key_obj, combined_answer_content)
            except ValueError as e:
                logger.error(f"⚠️ Error during final answer collation LLM call: {e}", exc_info=True)
                error_message = f"\n\n-- Collation Failed --\nAn error occurred ({e}). Uncollated info below:\n\n"
                final_collated_text = initial_answer.strip() + error_message + supplementary_answer.strip()
                return set_cached(cache_key_obj, final_collated_text)

    def reset_conversation(self) -> None: logger.info("🔄 Resetting chatbot internal state.")

    def process_user_query(self, user_query: str, user_type: str, 
                           conversation_history_tuples: Optional[List[Tuple[str, str]]] = None,
                           confirmed_symptoms: Optional[List[str]] = None, 
                           original_query_if_followup: Optional[str] = None 
                           ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        logger.info(f"--- Processing User Query: '{user_query[:50]}' ---")
        logger.info(f"User Type: {user_type}, History turns: {len(conversation_history_tuples) if conversation_history_tuples else 0}")
        
        processed_query: str = user_query
        current_symptoms_for_retrieval: List[str] = []
        is_symptom_query_flag: bool = False 
        medical_check_ok: bool = False

        if confirmed_symptoms is not None:
            if not user_query: return "Error: Original query missing for symptom rerun.", "display_final_answer", None
            is_symptom_query_flag = True
            current_symptoms_for_retrieval = sorted(list(set(s.strip().lower() for s in confirmed_symptoms if isinstance(s,str) and s.strip())))
            medical_check_ok = True
        else:
            medical_check_ok, medical_reason = self.is_medical_query(processed_query)
            if not medical_check_ok: return f"I can only answer medical-related questions. ({medical_reason})", "display_final_answer", None
            current_symptoms_for_retrieval, _ = self.extract_symptoms(processed_query)
            is_symptom_query_flag = self.is_symptom_related_query(processed_query)
        
        if not medical_check_ok: return "Internal error: Medical check failed.", "display_final_answer", None

        kg_results: Dict[str, Any] = {}
        s_kg: float = 0.0
        rag_chunks: List[str] = []
        s_rag_val: float = 0.0 

        if is_symptom_query_flag and current_symptoms_for_retrieval:
            kg_results = self.knowledge_graph_agent(processed_query, current_symptoms_for_retrieval)
            s_kg = float(kg_results.get("top_disease_confidence", 0.0))

        if self.vectordb and self.embedding_model:
            retrieved_rag_data = self.retrieve_rag_context(processed_query)
            if isinstance(retrieved_rag_data, tuple) and len(retrieved_rag_data) == 2:
                rag_chunks = retrieved_rag_data[0] if isinstance(retrieved_rag_data[0], list) else []
                s_rag_candidate = retrieved_rag_data[1]
                try: s_rag_val = float(s_rag_candidate)
                except (ValueError, TypeError): s_rag_val = 0.0
            else: rag_chunks, s_rag_val = [], 0.0
        
        if confirmed_symptoms is None and is_symptom_query_flag and \
           len(kg_results.get("identified_diseases_data",[])) > 0 and \
           0.0 < s_kg < THRESHOLDS["disease_symptom_followup_threshold"]:
            # ... (symptom UI trigger logic as before) ...
            top_disease_data = kg_results["identified_diseases_data"][0]
            all_kg_symps_lower = set(s.lower() for s in kg_results.get("all_disease_symptoms_kg_for_top_disease",[]) if isinstance(s,str))
            initial_symps_lower = set(s.lower() for s in current_symptoms_for_retrieval if isinstance(s,str)) # current_symptoms_for_retrieval are already lower
            suggested_symps_lower = sorted(list(all_kg_symps_lower - initial_symps_lower))
            original_case_map = {s.lower(): s for s in kg_results.get("all_disease_symptoms_kg_for_top_disease",[]) if isinstance(s,str)}
            suggested_original_case = [original_case_map[s_low] for s_low in suggested_symps_lower if s_low in original_case_map]
            if suggested_original_case:
                ui_payload = {"symptom_options": {top_disease_data.get("Disease", "Condition"): suggested_original_case}, "original_query": processed_query}
                return "To help provide more accurate information, please confirm if you are experiencing any of these additional symptoms:", "show_symptom_ui", ui_payload
            
        selected_context = self.select_context(kg_results, s_kg, rag_chunks, s_rag_val, is_symptom_query_flag)
        initial_context_sources_used = []
        if selected_context:
            if "kg" in selected_context: initial_context_sources_used.append("Knowledge Graph")
            if "rag" in selected_context: initial_context_sources_used.append("Internal Data") # MODIFIED
        
        try: 
            initial_answer = self.generate_initial_answer(processed_query, selected_context, user_type, conversation_history_tuples)
        except ValueError as e:
            path_info = ", ".join(initial_context_sources_used) or "LLM (Initial)"
            return f"Error generating initial answer: {e}\n\n<span style='font-size:0.8em;color:grey;'>*Sources used: {path_info} (Failed)*</span>", "display_final_answer", None

        reflection_failed, evaluation_result, missing_info_description = False, 'complete', None
        try: 
            evaluation_result, missing_info_description = self.reflect_on_answer(processed_query, initial_answer, selected_context)
        except Exception as e: 
            reflection_failed = True; evaluation_result = 'incomplete'; 
            missing_info_description = f"Reflection step failed ({e}). Proceeding with supplementary information attempt."
            logger.error(f"Reflection step failed: {e}", exc_info=True)
        
        final_answer_content = initial_answer
        supplementary_step_triggered = False
        if evaluation_result == 'incomplete':
            supplementary_step_triggered = True
            description_for_supplementary = missing_info_description or f"The initial answer was incomplete for the query: {processed_query[:50]}..."
            supplementary_answer = self.get_supplementary_answer(processed_query, description_for_supplementary, user_type, conversation_history_tuples)
            final_answer_content = self.collate_answers(initial_answer, supplementary_answer, user_type, conversation_history_tuples)
            
        # Triage is now integrated into the family user's system prompt.
        # No separate enhance_with_triage_detection call here, the generation steps should handle it.
        final_content_for_display = final_answer_content

        final_pathway_parts = list(set(initial_context_sources_used)) 
        if supplementary_step_triggered or not initial_context_sources_used or reflection_failed: 
            if "LLM (General Knowledge)" not in final_pathway_parts: 
                final_pathway_parts.append("LLM (General Knowledge)")
        
        pathway_info_str = ", ".join(sorted(list(set(final_pathway_parts))))
        if not pathway_info_str.strip() : pathway_info_str = "LLM (General Knowledge)" 

        disclaimer = "\n\nIMPORTANT MEDICAL DISCLAIMER: This information is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
        pathway_note = f"<span style='font-size:0.8em;color:grey;'>*Sources used: {pathway_info_str.strip()}*</span>"
        final_response_text = f"{final_content_for_display.strip()}{disclaimer}\n\n{pathway_note}"
        
        logger.info(f"--- Workflow Finished. Final pathway: {pathway_info_str} ---")
        return final_response_text, "display_final_answer", None


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
    st.set_page_config(page_title="DxAI-Agent", page_icon=f"data:image/png;base64,{ICON_B64}", layout="wide")
    
    # --- User Type Selector & Instructions ---
    create_user_type_selector() # Handles its own state for 'last_user_type' and 'reset_requested_by_type_change'
    current_user_type = st.session_state.get("user_type_select", "User / Family")

    if current_user_type == "User / Family":
        st.sidebar.markdown(
            "<small>As a Family User, ask about symptoms, conditions, or home remedies. "
            "You'll receive clear explanations. If your query suggests urgency, advice on next steps will be integrated into the response.</small>", 
            unsafe_allow_html=True
        )
    elif current_user_type == "Physician":
        st.sidebar.markdown(
            "<small>As a Physician, you can ask for differential diagnoses, test suggestions, "
            "or guideline information. Expect professional medical terminology.</small>", 
            unsafe_allow_html=True
        )

    # --- Handle Reset if requested by User Type Change ---
    if st.session_state.get('reset_requested_by_type_change', False):
        logger.info("Executing conversation reset due to user type change.")
        if st.session_state.get('chatbot'): 
            st.session_state.chatbot.reset_conversation()
        
        # Selective reset for user type change
        st.session_state.messages = []
        st.session_state.ui_state = {"step": "input", "payload": None}
        st.session_state.processing_input_payload = None
        st.session_state.form_timestamp = datetime.now().timestamp()
        
        for k_suffix in ['_from_ui', '_for_symptom_rerun']:
            state_key = f'confirmed_symptoms{k_suffix}'
            if state_key in st.session_state: del st.session_state[state_key]
        
        keys_to_delete_form = [k for k in st.session_state if k.startswith("symptom_form_")]
        for k_del in keys_to_delete_form: 
            if k_del in st.session_state: del st.session_state[k_del]
        
        keys_to_delete_feedback = [k for k in st.session_state if k.startswith("fb_")]
        for k_fb_del in keys_to_delete_feedback:
            if k_fb_del in st.session_state: del st.session_state[k_fb_del]
            
        del st.session_state.reset_requested_by_type_change # Clear the flag
        logger.info("Reset due to type change complete. Rerunning.")
        st.rerun()

    # --- Logo and Title ---
    try:
        logo = Image.open(image_path_str)
        c1, c2 = st.columns([1,10]); c1.image(logo,width=100); c2.markdown("# DxAI-Agent")
    except FileNotFoundError:
        logger.warning(f"Logo image not found at {image_path_str}. Displaying title only.")
        st.markdown("# DxAI-Agent")
    except Exception as e_logo: 
        logger.error(f"Error displaying logo: {e_logo}")
        st.markdown("# DxAI-Agent")

    # --- Chatbot Initialization (Once per session) ---
    if 'chatbot_initialized_flag' not in st.session_state: 
        st.session_state.chatbot_initialized_flag = False
        st.session_state.chatbot = None
        st.session_state.init_status = (False, "Initialization not started.")
        logger.info("Attempting to initialize chatbot instance and backend components (first run or after full clear)...")
        with st.spinner("Initializing chat assistant... This may take a moment the first time."):
            try:
                st.session_state.chatbot = DocumentChatBot()
                success, msg = st.session_state.chatbot.initialize_qa_chain()
                st.session_state.init_status = (success, msg)
                st.session_state.chatbot_initialized_flag = success
                logger.info(f"Chatbot initialization attempt complete. Status: {success}, Msg: {msg}")
            except Exception as e_init:
                logger.critical(f"CRITICAL UNCAUGHT ERROR DURING INITIALIZATION: {e_init}", exc_info=True)
                st.session_state.init_status = (False, f"Critical initialization error: {str(e_init)[:100]}")
                st.session_state.chatbot_initialized_flag = False
    
    init_success, init_msg = st.session_state.get('init_status', (False, "Status unknown."))
    is_interaction_enabled = st.session_state.get('chatbot_initialized_flag', False) and st.session_state.get('chatbot') is not None

    # --- Initialize other necessary session state variables ---
    default_ui_states = {
        'ui_state': {"step": "input", "payload": None}, 
        'messages': [],
        'processing_input_payload': None, 
        'confirmed_symptoms_from_ui': None,
        'original_query_for_symptom_rerun': None,
        'form_timestamp': datetime.now().timestamp()
    }
    for key, default_value in default_ui_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            # logger.info(f"Initializing session state for '{key}'.") # Can be noisy

    # --- Sidebar Info (No explicit connection status box) ---
    st.sidebar.info("DxAI-Agent helps answer medical questions using internal data and medical knowledge.")
    logger.info(f"Current Initialization Status (for console log): {init_msg}")


    # --- Main Chat Tabs ---
    tab1, tab2 = st.tabs(["Chat", "About"])
    with tab1:
        # --- Display Chat Messages ---
        for i, (msg_content, is_user_msg) in enumerate(st.session_state.get('messages', [])):
            role = "user" if is_user_msg else "assistant"
            with st.chat_message(role):
                st.markdown(msg_content, unsafe_allow_html=(not is_user_msg))
                # Feedback buttons for the last assistant message
                if not is_user_msg and i == len(st.session_state.messages) - 1 and \
                   st.session_state.ui_state["step"] == "input" and is_interaction_enabled:
                    cols_fb = st.columns([0.05, 0.05, 0.9])
                    user_q_for_fb = ""
                    # Find preceding user message for context
                    for j_fb in range(i - 1, -1, -1):
                        if st.session_state.messages[j_fb][1] is True: # is_user_msg
                            user_q_for_fb = st.session_state.messages[j_fb][0]
                            break
                    
                    fb_key_base = f"fb_{i}_{abs(hash(msg_content.split('IMPORTANT MEDICAL DISCLAIMER:',1)[0]))}"
                    up_key, down_key = f'{fb_key_base}_up', f'{fb_key_base}_down'
                    up_btn_key, down_btn_key = f'{up_key}_btn', f'{down_key}_btn'

                    with cols_fb[0]:
                        if up_key not in st.session_state and down_key not in st.session_state:
                            if st.button("👍", key=up_btn_key, help="Good response"):
                                vote_message(user_q_for_fb, msg_content, "thumbs_up", current_user_type)
                                st.session_state[up_key] = True; st.toast("Thanks for your feedback!"); st.rerun()
                        elif up_key in st.session_state: 
                            st.button("👍", key=up_btn_key, disabled=True)
                    with cols_fb[1]:
                        if up_key not in st.session_state and down_key not in st.session_state:
                            if st.button("👎", key=down_btn_key, help="Needs improvement"):
                                vote_message(user_q_for_fb, msg_content, "thumbs_down", current_user_type)
                                st.session_state[down_key] = True; st.toast("Thanks for your feedback!"); st.rerun()
                        elif down_key in st.session_state:
                            st.button("👎", key=down_btn_key, disabled=True)
        
        st.write(" \n" * 2) # Spacer
        input_area_container = st.container()

        # --- Conditional UI Rendering (Chat Input or Symptom Checklist) ---
        with input_area_container:
            if not is_interaction_enabled:
                st.error("Chat assistant is not available. Please check initialization or configuration.")
                st.chat_input("Chat disabled", disabled=True, key="chat_input_disabled_init_error")
            
            elif st.session_state.ui_state["step"] == "confirm_symptoms":
                payload = st.session_state.ui_state.get("payload")
                if not payload or "symptom_options" not in payload or "original_query" not in payload:
                    logger.error("Symptom UI state error: Payload invalid. Resetting UI.")
                    st.session_state.messages.append(("An error occurred with the symptom checklist. Please try your query again.", False))
                    st.session_state.ui_state = {"step": "input", "payload": None}
                    if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                    if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                    st.rerun()
                else:
                    display_symptom_checklist(payload["symptom_options"], payload["original_query"])
                    st.chat_input("Please confirm symptoms using the form above...", disabled=True, key="chat_input_disabled_symptom_form")
            
            elif st.session_state.ui_state["step"] == "input":
                user_query_input = st.chat_input("Ask your medical question...", disabled=not is_interaction_enabled, key="main_chat_input_field_v4") # New key
                
                if user_query_input and is_interaction_enabled:
                    logger.info(f"New user query received: '{user_query_input[:50]}...'")
                    st.session_state.messages.append((user_query_input, True)) # Add user message to display
                    
                    # For Memory: Do NOT reset chatbot object state here for multi-turn context.
                    # It's reset via "Reset Conversation" button or user type change.
                    # if st.session_state.get('chatbot'):
                    #    st.session_state.chatbot.reset_conversation() 
                    
                    st.session_state.form_timestamp = datetime.now().timestamp() # For unique form keys if symptom UI is triggered
                    # Clear states that should be fresh for a new independent query
                    if 'confirmed_symptoms_from_ui' in st.session_state: del st.session_state.confirmed_symptoms_from_ui
                    if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                    # Clear previous message feedback states
                    for k_fb_clear in [k for k in st.session_state if k.startswith("fb_")]: 
                        if k_fb_clear in st.session_state: del st.session_state[k_fb_clear]

                    # Prepare conversation history for the bot
                    chat_history_for_bot: List[Tuple[str,str]] = []
                    if len(st.session_state.messages) > 1: # If there's prior history (current query is last)
                        history_display_list = st.session_state.messages[:-1] 
                        for i_hist_main in range(0, len(history_display_list), 2): 
                            if i_hist_main + 1 < len(history_display_list):
                                user_turn_hist = history_display_list[i_hist_main]
                                assistant_turn_hist = history_display_list[i_hist_main+1]
                                if user_turn_hist[1] is True and assistant_turn_hist[1] is False:
                                     chat_history_for_bot.append((user_turn_hist[0], assistant_turn_hist[0]))
                                else: 
                                    logger.warning(f"Skipping malformed history pair at index {i_hist_main} for LLM history context.")
                    
                    st.session_state.processing_input_payload = {
                        "query": user_query_input, 
                        "confirmed_symptoms": None,
                        "conversation_history": chat_history_for_bot 
                    }
                    st.rerun()

        # --- Handle Symptom Form Submission ---
        if st.session_state.get('confirmed_symptoms_from_ui') is not None:
            logger.info("Symptom confirmation form submitted. Preparing for re-processing.")
            confirmed_symps_payload = st.session_state.confirmed_symptoms_from_ui
            original_q_payload = st.session_state.get('original_query_for_symptom_rerun')
            
            del st.session_state.confirmed_symptoms_from_ui # Clear immediately after reading

            if not original_q_payload:
                logger.error("Symptom form submitted but original_query_for_symptom_rerun is missing. Resetting UI.")
                st.session_state.messages.append(("An error occurred processing symptom confirmation. Please try your original query again.", False))
                if 'original_query_for_symptom_rerun' in st.session_state: del st.session_state.original_query_for_symptom_rerun
                st.session_state.ui_state = {"step":"input", "payload":None}
                st.rerun()
            else:
                # Get existing history UP TO THE POINT BEFORE THE SYMPTOM UI WAS TRIGGERED
                # The symptom UI prompt message is the last assistant message before this rerun.
                chat_history_for_symptom_rerun : List[Tuple[str,str]] = []
                if len(st.session_state.messages) > 0: # If there are any messages
                    # We need history *before* the symptom prompt and user's confirmation action
                    # The original_q_payload was the query that *led* to the symptom UI
                    # We need history *before* that original_q_payload was processed
                    # This requires careful tracking or passing the history that existed *before* the symptom UI was triggered.
                    # For now, let's simplify: pass history up to the bot message that prompted for symptoms.
                    # The last message is usually the bot's prompt for the symptom UI.
                    # The second to last is the user query that triggered it.
                    # So history would be messages[:-2]
                    relevant_messages_for_history = []
                    # Find the index of the original_q_payload in messages
                    original_query_index = -1
                    for idx, (msg_text, is_user) in reversed(list(enumerate(st.session_state.messages))):
                        if is_user and msg_text == original_q_payload:
                            original_query_index = idx
                            break
                    
                    if original_query_index > 0 : # if original query was found and it's not the first message
                         history_display_list_symptom = st.session_state.messages[:original_query_index]
                         for i_hist_symp in range(0, len(history_display_list_symptom), 2):
                             if i_hist_symp + 1 < len(history_display_list_symptom):
                                 user_turn_s = history_display_list_symptom[i_hist_symp]
                                 assistant_turn_s = history_display_list_symptom[i_hist_symp+1]
                                 if user_turn_s[1] is True and assistant_turn_s[1] is False:
                                     chat_history_for_symptom_rerun.append((user_turn_s[0], assistant_turn_s[0]))


                st.session_state.processing_input_payload = {
                    "query": original_q_payload, 
                    "confirmed_symptoms": confirmed_symps_payload,
                    "conversation_history": chat_history_for_symptom_rerun # Pass history relevant to original query
                }
                st.rerun()

        # --- Backend Processing Call ---
        if st.session_state.get('processing_input_payload') is not None:
            payload_to_process = st.session_state.processing_input_payload
            st.session_state.processing_input_payload = None # Clear immediately
            
            chatbot_instance = st.session_state.get('chatbot')
            if not chatbot_instance or not st.session_state.get('chatbot_initialized_flag'):
                logger.critical("Processing triggered but chatbot not ready/initialized.")
                st.session_state.messages.append(("Chatbot is not ready. Please wait for initialization or check configuration.", False))
                st.rerun() # Rerun to show message, avoid further processing
                return # Stop this execution path

            query_to_run = payload_to_process.get("query","")
            confirmed_symptoms_to_run = payload_to_process.get("confirmed_symptoms")
            history_to_run = payload_to_process.get("conversation_history")

            if not query_to_run:
                logger.error("Empty query found in processing payload. Skipping.")
                st.session_state.messages.append(("An empty query was received for processing. Please try again.", False))
                st.rerun()
                return

            with st.spinner("Thinking..."):
                try:
                    response_text, ui_action, ui_payload_from_bot = chatbot_instance.process_user_query(
                        user_query=query_to_run, 
                        user_type=current_user_type, 
                        conversation_history_tuples=history_to_run, 
                        confirmed_symptoms=confirmed_symptoms_to_run
                        # original_query_if_followup not strictly needed here if query_to_run is correctly set
                    )

                    if ui_action == "display_final_answer":
                        st.session_state.messages.append((response_text, False))
                        st.session_state.ui_state = {"step": "input", "payload": None}
                        # Clear original_query_for_symptom_rerun if this was a symptom UI follow-up
                        if confirmed_symptoms_to_run and 'original_query_for_symptom_rerun' in st.session_state:
                            del st.session_state.original_query_for_symptom_rerun
                            logger.debug("Cleared original_query_for_symptom_rerun after symptom processing.")
                    
                    elif ui_action == "show_symptom_ui":
                        st.session_state.messages.append((response_text, False)) # This is the prompt for the UI
                        st.session_state.ui_state = {"step": "confirm_symptoms", "payload": ui_payload_from_bot}
                        st.session_state.form_timestamp = datetime.now().timestamp() # New form instance
                        if ui_payload_from_bot and ui_payload_from_bot.get("original_query"):
                            st.session_state.original_query_for_symptom_rerun = ui_payload_from_bot["original_query"]
                        else: 
                            logger.error("Symptom UI requested but original_query missing in payload. Resetting UI.")
                            st.session_state.messages.append(("Error preparing symptom checklist. Please try again.", False))
                            st.session_state.ui_state={"step":"input", "payload":None}
                    
                    else: # "none" or unknown action
                        logger.warning(f"process_user_query returned unhandled ui_action: {ui_action}. Defaulting to input UI.")
                        if st.session_state.ui_state["step"] != "confirm_symptoms": # Don't override if form is meant to show
                             st.session_state.ui_state = {"step": "input", "payload": None}

                except Exception as e_process_query: 
                    logger.error(f"Error during chatbot.process_user_query: {e_process_query}", exc_info=True)
                    error_display_message = str(e_process_query)[:300] # Limit length of error displayed
                    st.session_state.messages.append((f"Sorry, an application error occurred: {error_display_message}", False))
                    st.session_state.ui_state = {"step": "input", "payload": None} 
                    # Clean up symptom state if error occurred during symptom rerun
                    if confirmed_symptoms_to_run and 'original_query_for_symptom_rerun' in st.session_state:
                        del st.session_state.original_query_for_symptom_rerun
                st.rerun() # Always rerun to update UI based on new state

        # --- Reset and Feedback Forms ---
        st.divider()
        if st.button("Reset Conversation", key="reset_conversation_button_v3", disabled=not is_interaction_enabled, help="Clear chat history and start over."):
            logger.info("Conversation reset triggered by user button.")
            if st.session_state.get('chatbot'):
                st.session_state.chatbot.reset_conversation()
            
            st.session_state.messages = []
            st.session_state.ui_state = {"step": "input", "payload": None}
            st.session_state.processing_input_payload = None
            st.session_state.form_timestamp = datetime.now().timestamp()
            for k_s_reset in ['confirmed_symptoms_from_ui', 'original_query_for_symptom_rerun']:
                if k_s_reset in st.session_state: del st.session_state[k_s_reset]
            for k_form_reset in [k for k in st.session_state if k.startswith("symptom_form_")]: 
                if k_form_reset in st.session_state: del st.session_state[k_form_reset]
            for k_fb_reset in [k for k in st.session_state if k.startswith("fb_")]:
                if k_fb_reset in st.session_state: del st.session_state[k_fb_reset]
            logger.info("Conversation-related session state cleared. Rerunning.")
            st.rerun()
        
        st.divider()
        st.subheader("🩺 Detailed Feedback")
        with st.form(key="detailed_feedback_main_form_v3", clear_on_submit=True): # New key
            feedback_input_text = st.text_area("Enter corrections, improvements, or comments here...", height=100, disabled=not is_interaction_enabled)
            if st.form_submit_button("Submit Feedback", disabled=not is_interaction_enabled) and feedback_input_text:
                submit_feedback(feedback_input_text, st.session_state.get('messages',[]), current_user_type)
                st.success("Thank you for your feedback!")

    with tab2:
        st.markdown("""
        ## Medical Chat Assistant
        This is an experimental medical chat assistant designed to provide information based on internal data (simulated documents) and a medical knowledge graph.

        **How it Works (Simplified):**
        1.  **Medical Check & Symptom Analysis:** Assesses query relevance and extracts symptoms. May ask for symptom confirmation if needed.
        2.  **Information Retrieval:** Searches an internal knowledge graph and document data.
        3.  **Answer Generation:** Uses an LLM to synthesize an answer from retrieved context and its general knowledge.
        4.  **Self-Reflection & Refinement:** The system attempts to evaluate and improve its own answer.
        5.  **User-Specific Responses:** Tailors language and includes integrated triage advice for family users.
        6.  **Pathway Indication:** Indicates primary information sources used (e.g., Internal Data, Knowledge Graph, LLM).

        **Disclaimer:** This system is for informational purposes ONLY and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.
        """)
    # logger.debug("--- Streamlit App End of Rerun ---")

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers(): 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    main_logger = logging.getLogger(__name__) 
    # Set a specific level for this app's logger if desired, e.g., main_logger.setLevel(logging.DEBUG)
    main()
