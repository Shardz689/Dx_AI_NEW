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

# Get environment variables with fallback to placeholder values
# IMPORTANT: Replace 'YOUR_GEMINI_API_KEY' with your actual key or ensure .env is set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBv-I8Ld-k09Lxu9Yi7HPffZHKXIqGSdHU")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://1b47920f.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "eCqDfyhDcuGMLzbYfiqL6jsvjH3LIXr86xQGAEKmY8Y")

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

# Configure logging
# Log messages to the console where streamlit is run
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load and convert the image to base64
def get_image_as_base64(file_path):
    """Converts an image file to a base64 string."""
    if not os.path.exists(file_path):
        logger.warning(f"Image file not found at {file_path}")
        # Return a tiny valid base64 image as a fallback
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        logger.error(f"Error encoding image {file_path} to base64: {e}")
        # Return a tiny valid base64 image as a fallback
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


# Option 1: If your image is stored locally
image_path = "Zoom My Life.jpg"  # Update with your actual path
icon = get_image_as_base64(image_path)

# Cache for expensive operations
CACHE = {}

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
    return value


# Hardcoded PDF files to use
# Update with your actual local PDF paths
HARDCODED_PDF_FILES = [
    "rawdata.pdf",
    # Add more PDF paths here if needed
]

# For testing purposes - add more relevant known diseases if possible
# These aren't strictly used in the new logic, but useful for context/dummy data
known_diseases = ["hypertension", "type 2 diabetes mellitus", "respiratory infections", "obesity", "cardiovascular disease", "common cold", "influenza", "strep throat", "anxiety", "acid reflux", "costochondritis", "angina"]


# DocumentChatBot class
class DocumentChatBot:
    def __init__(self):
        logger.info("DocumentChatBot initializing...")
        self.qa_chain: Optional[ConversationalRetrievalChain] = None # Langchain QA chain
        self.vectordb: Optional[FAISS] = None # Langchain Vectorstore
        self.chat_history: List[Tuple[str, str]] = [] # Stores (user_msg, bot_msg) tuples for display and history
        # Tracks if the single allowed LLM follow-up has been asked (0 or 1) for the current thread
        self.followup_context = {"round": 0}

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
            # Test the embedding model (important check)
            try:
                test_embedding = self.embedding_model.embed_query("test query")
                if test_embedding and len(test_embedding) > 0:
                     logger.info("Embedding model initialized and tested successfully.")
                else:
                    raise ValueError("Test embedding was empty.")
            except Exception as test_e:
                 logger.warning(f"Embedding model test failed: {test_e}. Setting embedding_model to None.")
                 self.embedding_model = None # Set to None if test fails

        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Could not initialize embedding model: {e}")
            self.embedding_model = None # Ensure it's None on failure


        self.llm: Optional[ChatGoogleGenerativeAI] = None # LLM for general generation and specific prompts

        # Initialize KG driver connection status
        self.kg_driver = None
        self.kg_connection_ok = False
        self._init_kg_connection()
        logger.info("DocumentChatBot initialization finished.")


    def _init_kg_connection(self):
        """Attempts to connect to the Neo4j database."""
        logger.info("Attempting to connect to Neo4j...")
        try:
            # Use a small timeout for the connection test
            self.kg_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, connection_timeout=5.0)
            self.kg_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
            self.kg_connection_ok = True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}. KG features will be unavailable.")
            self.kg_driver = None
            self.kg_connection_ok = False


    def create_vectordb(self):
            """Create vector database from hardcoded PDF documents."""
            logger.info("Creating vector database...")
            pdf_files = [Path(pdf_file) for pdf_file in HARDCODED_PDF_FILES if Path(pdf_file).exists()]

            if not pdf_files:
                logger.warning("No PDF files found. Cannot create vector database.")
                return None, "No PDF files found at the specified paths. Cannot create vector database."

            loaders = []
            for pdf_file in pdf_files:
                try:
                    loaders.append(PyPDFLoader(str(pdf_file)))
                    # logger.debug(f"Loaded PDF loader for: {pdf_file}")
                except Exception as e:
                    logger.error(f"Error creating loader for {pdf_file}: {e}")
                    # Continue with other files

            if not loaders:
                 logger.warning("No valid PDF loaders could be created.")
                 return None, "No valid PDF loaders could be created."

            pages = []
            for loader in loaders:
                try:
                    loaded_pages = loader.load()
                    pages.extend(loaded_pages)
                    logger.info(f"Loaded {len(loaded_pages)} pages from {loader.file_path}.")
                except Exception as e:
                    logger.error(f"Error loading pages from PDF {loader.file_path}: {e}")
                    # Continue loading other files


            if not pages:
                 logger.warning("No pages were loaded from the PDFs.")
                 return None, "No pages were loaded from the PDFs."

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # Adjust chunk size if needed
                chunk_overlap=100 # Adjust overlap if needed
            )
            splits = text_splitter.split_documents(pages)
            logger.info(f"Split {len(pages)} pages into {len(splits)} chunks.")

            if not splits:
                logger.warning("No text chunks were created from the PDF pages.")
                return None, "No text chunks were created from the PDF pages."

            # Use the initialized embedding model
            if self.embedding_model is None:
                 logger.warning("Embedding model is not initialized. Cannot create vector database.")
                 return None, "Embedding model is not initialized. Cannot create vector database."

            try:
                logger.info("Creating FAISS vectorstore from documents using the embedding model...")
                # Use the real FAISS
                vectordb = FAISS.from_documents(splits, self.embedding_model)
                logger.info("FAISS vectorstore created.")
                return vectordb, "Vector database created successfully."
            except Exception as e:
                logger.error(f"Error creating FAISS vector database: {e}")
                return None, f"Failed to create vector database: {str(e)}"


    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        """
        Check if the query is relevant to the medical domain using LLM.
        Returns a tuple of (is_relevant, reason)
        """
        cache_key = {"type": "medical_relevance", "query": query}
        cached = get_cached(cache_key)
        if cached:
            return cached

        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform medical relevance check reliably.")
             # Fallback if LLM is not available - simple keyword check
             medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose", "sick", "doctor"]
             if any(keyword in query.lower() for keyword in medical_keywords):
                  return (True, "Fallback heuristic match")
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

        try:
            response = self.local_generate(medical_relevance_prompt, max_tokens=150) # Reduced max_tokens for speed
            json_match = re.search(r'\{[\s\S]*\}', response)

            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    is_medical = data.get("is_medical", False)
                    confidence = data.get("confidence", 0.0)
                    reasoning = data.get("reasoning", "")

                    # Use a confidence threshold
                    if is_medical and confidence >= THRESHOLDS.get("medical_relevance", 0.6):
                        result = (True, "")
                    else:
                        result = (False, reasoning)

                    set_cached(cache_key, result)
                    return result
                except json.JSONDecodeError:
                    logger.warning("Could not parse medical relevance JSON from LLM response.")
                    # Fallback if JSON parsing fails
                    medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose"]
                    if any(keyword in query.lower() for keyword in medical_keywords):
                         return (True, "Fallback heuristic match")
                    return (False, "Fallback: JSON parsing failed and no keywords found")

        except Exception as e:
            logger.error(f"Error checking medical relevance: {e}")
            # Fallback if LLM call fails
            medical_keywords = ["symptom", "disease", "health", "medical", "pain", "cough", "fever", "treatment", "diagnose"]
            if any(keyword in query.lower() for keyword in medical_keywords):
                 return (True, "Fallback heuristic match")

        # Default fallback - treat as non-medical if everything else fails
        return (False, "Default: Check failed")


    def initialize_qa_chain(self):
        """Initialize the QA chain with Gemini Flash 1.5 and vector database."""
        logger.info("Initializing QA chain...")
        # This method sets up self.llm, self.embedding_model, self.vectordb, and self.qa_chain
        # self.embedding_model is already initialized in __init__
        # self.kg_driver is initialized in __init__

        if self.qa_chain is None or self.llm is None: # Only initialize if not already done

            # --- 1. Initialize LLM ---
            llm_init_success = False
            llm_init_message = "LLM initialization skipped." # Default message
            if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                logger.warning("Gemini API key not set or invalid. LLM will not be initialized.")
                self.llm = None # Explicitly set LLM to None
                llm_init_message = "Gemini API key not found or invalid."
            else:
                try:
                    logger.info("Initializing Gemini Flash 1.5...")
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
                         test_response = self.llm.invoke("Hello, are you ready?")
                         if test_response.content:
                             logger.info(f"Gemini says: {test_response.content[:50]}...") # Print start of response
                             logger.info("Successfully connected to Gemini Flash 1.5")
                             llm_init_success = True
                             llm_init_message = "Gemini Flash 1.5 initialized."
                         else:
                            raise ValueError("LLM test response was empty.")
                    except Exception as test_e:
                         logger.warning(f"Initial Gemini test failed: {test_e}. LLM may not be functional.")
                         self.llm = None # Set back to None if test fails
                         llm_init_success = False
                         llm_init_message = f"Gemini LLM test failed: {test_e}"


                except Exception as e:
                    logger.error(f"Failed to initialize Gemini Flash 1.5: {e}")
                    self.llm = None # Ensure LLM is None on failure
                    llm_init_success = False
                    llm_init_message = f"Failed to initialize Gemini Flash 1.5: {str(e)}"


            # --- 2. Create Vector Database (Requires Embedding Model) ---
            # This uses the embedding model initialized in __init__
            vdb_message = "Vector database initialization skipped." # Default message
            if self.embedding_model is None:
                 logger.warning("Embedding model not initialized. Cannot create vector database.")
                 self.vectordb = None
                 vdb_message = "Embedding model not initialized."
            else:
                 # Create the vector database
                 self.vectordb, vdb_message = self.create_vectordb()
                 if self.vectordb is None:
                      logger.warning(f"Vector DB creation failed: {vdb_message}")
                 else:
                      logger.info("Vector database initialized.")


            # --- 3. Create Retrieval QA Chain ---
            chain_message = "Retrieval chain initialization skipped." # Default message
            try:
                 # The real ConversationalRetrievalChain requires a functional LLM and a retriever from a vectorstore
                 if self.llm is not None and self.vectordb is not None:
                      logger.info("Creating Real Conversational Retrieval Chain.")
                      memory = ConversationBufferMemory(
                         memory_key="chat_history", # This key must match the chain's memory key
                         output_key='answer', # Output key from the chain
                         return_messages=True # Keep messages in list format
                      )

                      self.qa_chain = ConversationalRetrievalChain.from_llm(
                          self.llm, # Pass the real LLM
                          retriever=self.vectordb.as_retriever(search_kwargs={"k": 5}), # Pass real retriever
                          chain_type="stuff", # Use the 'stuff' chain type
                          memory=memory, # Pass memory
                          return_source_documents=True, # Ensure source documents are returned
                          verbose=False, # Set to True for detailed logs from Langchain
                      )
                      chain_message = "Real Conversational Retrieval Chain initialized."
                 else:
                      logger.warning("Cannot create real Retrieval Chain: LLM or Vector DB not initialized.")
                      self.qa_chain = None # Ensure qa_chain is None if creation fails
                      chain_message = "Retrieval chain requires both LLM and Vector DB."


            except Exception as e:
                logger.error(f"Failed to create Retrieval Chain: {e}")
                # Set qa_chain to None to indicate failure
                self.qa_chain = None
                chain_message = f"Failed to create Retrieval Chain: {str(e)}"

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
            overall_success = self.llm is not None # Assume LLM is the minimal requirement for any useful response


            logger.info(f"QA Chain Initialization Result: Success={overall_success}, Message='{overall_message}'")
            return overall_success, overall_message

        # If already initialized
        logger.info("QA Chain is already initialized.")
        return True, "Chat assistant is already initialized."
    
    def local_generate(self, prompt, max_tokens=500):
        """Generate text using Gemini Flash 1.5"""
        if self.llm is None:
            raise ValueError("LLM is not initialized")

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating with Gemini: {e}")
            # Fallback direct generation using genai
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                result = model.generate_content(prompt)
                return result.text
            except Exception as inner_e:
                print(f"Error in fallback generation: {inner_e}")
                return f"Error generating response. Please try again."


    def generate_llm_answer(self, query: str, kg_content: Optional[str] = None, rag_content: Optional[str] = None, initial_combined_answer: Optional[str] = None, missing_elements: Optional[List[str]] = None) -> str:
        """
        Generates an LLM answer, synthesizing information from KG and RAG,
        and potentially focusing on missing elements identified.
        This is the core synthesis step in Path 2.
        Instructs the LLM to include source mentions/links if available.
        """
        logger.info("➡️ LLM Synthesis Step (with source/link emphasis)")

        if self.llm is None:
            logger.warning("LLM not initialized. Skipping synthesis.")
            return "I'm currently unable to synthesize a complete answer. Please consult a healthcare professional."

        # Format reliable medical URLs for the prompt
        reliable_sources_prompt_text = "Reliable Medical Information Sources:\n"
        for topic, urls in RELIABLE_MEDICAL_URLS.items():
            reliable_sources_prompt_text += f"- {topic}:\n"
            for url in urls:
                reliable_sources_prompt_text += f"  - {url}\n"
        reliable_sources_prompt_text += "---\n"


        prompt_parts = [
            "You are a helpful medical AI assistant providing a comprehensive answer based on the provided information.",
            f"USER QUESTION: {query}",
            "When providing information, **prioritize using the sources provided below** (Knowledge Graph, Document Search Draft, Raw KG/RAG content).",
            reliable_sources_prompt_text, # Add the list of reliable URLs to the prompt
        ]

        # Provide the initial combined answer draft to the LLM
        if initial_combined_answer and initial_combined_answer.strip() != "" and initial_combined_answer.strip() != "I found limited specific information regarding your query from my knowledge sources.":
             prompt_parts.append(f"Available Information Draft (from Knowledge Graph and Document Search):\n---\n{initial_combined_answer}\n---")
        # If no draft (e.g., error in combination), provide raw content if available and meaningful
        elif kg_content or rag_content:
            if kg_content and kg_content.strip() != "" and kg_content.strip() != "Knowledge Graph information on treatments or remedies is unavailable." and kg_content.strip() != "Knowledge Graph did not find specific relevant information on treatments or remedies.":
                 prompt_parts.append(f"Available Medical Knowledge Graph Information:\n---\n{kg_content}\n---")
            if rag_content and rag_content.strip() != "" and rag_content.strip() != "An error occurred while retrieving information from documents." and rag_content.strip() != "Document search is currently unavailable." and rag_content.strip() != "I searched my documents but couldn't find specific information for that.":
                 prompt_parts.append(f"Available Retrieved Information (Document Search):\n---\n{rag_content}\n---")
            else:
                 prompt_parts.append("No specific relevant information was found from the provided knowledge sources.")
        else:
             # If no context or draft, rely on LLM's general knowledge but explicitly state sources are missing
             prompt_parts.append("No specific information was found from the provided knowledge sources. Providing a general, safe response based on general medical knowledge.")


        prompt_parts.append("Please synthesize the available information (if any) to provide a helpful, accurate, and comprehensive answer to the USER QUESTION.")

        if missing_elements:
            missing_desc = []
            if "duration" in missing_elements: missing_desc.append("how long the symptoms have lasted")
            if "severity" in missing_elements: missing_desc.append("how severe the symptoms are")
            if "location" in missing_elements: missing_desc.append("the location of symptoms (e.g., pain location)")
            if "frequency" in missing_elements: missing_desc.append("how often symptoms occur")
            if "onset" in missing_elements: missing_desc.append("when the symptoms started")
            # Add other specific element descriptions

            if missing_desc:
                focus_text = f"""
                **IMMEDIATE PRIORITY:** Based on the *provided sources*, ensure your answer directly addresses and incorporates details related to the following crucial aspects that were potentially missing from the initial draft: {", ".join(missing_desc)}.
                If the information needed to cover these aspects is present in the provided sources, prioritize including it in your synthesized answer and **mention the source you used for that specific piece of information** (e.g., [Source: Document X Page Y] or [Source: Knowledge Graph]). If the information is NOT present in the sources, state that the information is not available but still synthesize the rest of the answer based on what *is* available from the sources.
                """
                prompt_parts.append(focus_text)

        # Add instructions about using the *provided list* of reliable sources and linking
        prompt_parts.append("""
        When discussing information related to diseases like Hypertension, Cardiovascular Disease, Obesity, Respiratory Infections, or Type 2 Diabetes Mellitus, or general health information:
        1.  Prioritize using information from the "Available Medical Knowledge Graph Information" or "Available Retrieved Information (Document Search)". Use the [Source: ...] format for these.
        2.  If you discuss general information about the diseases or health topics mentioned in the "Reliable Medical Information Sources" list provided in this prompt, **you may include one or more of the URLs from that provided list**.
        3.  Format these URLs as clickable links in markdown: `[Link Text Here, e.g., CDC Website](URL_from_the_provided_list)`. Use descriptive link text.
        4.  **IMPORTANT:** ONLY use the URLs explicitly listed in the "Reliable Medical Information Sources" section of *this prompt*. Do NOT invent URLs or source names.
        5.  **Disclaimer:** Include a statement in the final answer that these external links are for general information and are not a substitute for professional medical advice, and may not directly correspond to every specific detail in the answer.
        """)

        prompt_parts.append("Include appropriate medical disclaimers about consulting healthcare professionals for diagnosis and treatment.")
        prompt_parts.append("Format your answer clearly and concisely using markdown.")


        prompt = "\n\n".join(prompt_parts)

        try:
            response = self.local_generate(prompt, max_tokens=1500) # Increased max_tokens slightly
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating LLM synthesis answer: {e}")
            return "I'm sorry, but I couldn't synthesize a complete answer to your question at this moment. Please consult a healthcare professional for personalized advice."

    def format_kg_diagnosis_with_llm(self, disease_name: str, symptoms_list: List[str], confidence: float) -> str:
        """
        Uses LLM to format the KG-identified disease and symptoms into a user-friendly statement for Path 1.
        """
        logger.info("➡️ LLM Formatting KG Diagnosis Step")
        if self.llm is None:
            logger.warning("LLM not initialized. Skipping KG diagnosis formatting.")
            fallback_symptoms_str = ", ".join(symptoms_list) if symptoms_list else "your symptoms"
            return f"Based on {fallback_symptoms_str}, **{disease_name}** is a potential condition. This is not a definitive diagnosis and requires professional medical evaluation."


        # Use the symptoms list that was *used to query KG* for this diagnosis
        symptoms_str = ", ".join(symptoms_list) if symptoms_list else "the symptoms you've reported"

        prompt = f"""
        You are a medical assistant tasked with explaining a potential medical condition based on symptoms.
        Given a highly probable disease based on the reported symptoms from a knowledge graph, write a concise, single-paragraph statement for a user.
        The statement should mention the symptoms and state that they are most likely associated with the identified disease.
        Crucially, include a clear disclaimer that this is NOT a definitive diagnosis and professional medical advice is necessary.
        Do NOT add references, bullet points for treatments/remedies, or other detailed information here. Keep it focused on the diagnosis possibility.

        Identified disease: {disease_name}
        Symptoms considered: {symptoms_str}
        Confidence score (for internal context, do not explicitly state in answer): {confidence:.2f}

        Example format: "Based on the symptoms you've reported, like [symptoms], these are most likely associated with [disease name]. However, this is not a definitive diagnosis and requires professional medical evaluation."

        Write the statement now:
        """
        try:
            response = self.local_generate(prompt, max_tokens=300) # Keep it concise
            return response.strip()
        except Exception as e:
            logger.error(f"Error formatting KG diagnosis with LLM: {e}")
            # Fallback manual format if LLM call fails
            fallback_symptoms_str = ", ".join(symptoms_list) if symptoms_list else "your symptoms"
            return f"Based on {fallback_symptoms_str}, **{disease_name}** is a potential condition. This is not a definitive diagnosis and requires professional medical evaluation."


    def identify_missing_info(self, user_query: str, generated_answer: str, conversation_history: List[Tuple[str, str]]) -> Tuple[bool, List[str]]:
        """
        Identifies what CRITICAL medical information is still missing from the GENERATED ANSWER
        relative to the USER QUERY, using conversation context.
        This is used for the FINAL completeness check in Path 2.
        """
        logger.info("🕵️ Identifying missing info from generated answer (Final Check)...")
    
        if self.llm is None:
            logger.warning("LLM not initialized. Cannot perform final completeness check.")
            return (False, [])  # Cannot check completeness without LLM
    
        # Convert conversation history to a string for context
        # Include a few recent exchanges for better understanding
        context = ""
        history_limit = 6  # Include last 3 exchanges (user+bot)
        recent_history = conversation_history[-history_limit:]
        
        # Track previously asked questions to avoid redundancy
        previously_asked_questions = []
        
        for i, entry in enumerate(recent_history):
            # Ensure the entry is a tuple of length 2
            if isinstance(entry, tuple) and len(entry) == 2:
                user_msg, bot_msg = entry
    
                # Safely get string representation of user_msg
                user_msg_str = str(user_msg) if user_msg is not None else ""
                context += f"User: {user_msg_str}\n"
    
                # Safely get string representation of bot_msg before formatting
                if isinstance(bot_msg, str):
                    truncated_bot_msg = bot_msg[:300] + "..." if len(bot_msg) > 300 else bot_msg
                    context += f"Assistant: {truncated_bot_msg}\n"
                    
                    # Check for questions in the bot's message to track previously asked questions
                    question_patterns = [
                        r"(?<!\w)how (?:long|much|often|severe|many|would|could|should|do|does|did|is|are|was|were)(?!\w).*\?",
                        r"(?<!\w)what (?:are|is|was|were|do|does|did|should|would|could|will|might)(?!\w).*\?",
                        r"(?<!\w)when (?:did|do|does|is|are|was|were|will|would|should|could|might)(?!\w).*\?",
                        r"(?<!\w)where (?:is|are|was|were|do|does|did|will|would|should|could|might)(?!\w).*\?",
                        r"(?<!\w)why (?:is|are|was|were|do|does|did|will|would|should|could|might)(?!\w).*\?",
                        r"(?<!\w)which (?:is|are|was|were|do|does|did|will|would|should|could|might)(?!\w).*\?",
                        r"(?<!\w)can you (?:tell|describe|explain|clarify|specify|indicate)(?!\w).*\?",
                        r"(?<!\w)could you (?:tell|describe|explain|clarify|specify|indicate)(?!\w).*\?",
                        r"(?<!\w)have you (?:ever|had|been|experienced|noticed|observed)(?!\w).*\?",
                        r"(?<!\w)do you (?:have|feel|experience|notice|observe)(?!\w).*\?",
                        r"(?<!\w)are you (?:experiencing|feeling|having|noticing|observing)(?!\w).*\?",
                        r"\?$"  # Catch any remaining questions ending with ?
                    ]
                    
                    for pattern in question_patterns:
                        matches = re.finditer(pattern, truncated_bot_msg, re.IGNORECASE)
                        for match in matches:
                            # Add any questions found to our tracking list
                            question = match.group(0).strip()
                            if len(question) > 10:  # Avoid very short fragments
                                previously_asked_questions.append(question)
                    
                elif bot_msg is not None:
                    # Log if it's not a string but not None (unexpected type)
                    logger.warning(f"Unexpected type in chat_history bot message at index {i}. Type: {type(bot_msg)}. Value: {bot_msg}. Appending placeholder.")
                    context += f"Assistant: [Non-string response of type {type(bot_msg)}]\n"
                # else: bot_msg is None, do not add Assistant line
            else:
                # Log if an entry in history is not a tuple of length 2 or not a tuple at all
                logger.warning(f"Unexpected format in chat_history entry at index {i}. Entry: {entry}. Skipping entry or adding placeholder.")
                context += f"[Invalid history entry at index {i}]\n"
    
        # Enhance the prompt to explicitly highlight previously asked questions
        previously_asked_str = "\n".join(previously_asked_questions) if previously_asked_questions else "None detected."
    
        MISSING_INFO_PROMPT = '''
        You are a medical AI assistant analyzing a patient's conversation history and the latest generated answer.
        Your primary goal is to help provide a safe and comprehensive answer to the user's initial medical question.
        After reviewing the entire conversation history and the most recent generated answer, determine if there is any *absolutely critical* piece of medical information still missing from the *latest generated answer itself* that is essential for providing a safe and minimally helpful response.
    
        Conversation history (for context, includes previous turns and the latest generated answer):
        ---
        {context}
        ---
    
        PREVIOUSLY ASKED QUESTIONS (DO NOT ASK THESE AGAIN):
        ---
        {previously_asked}
        ---
    
        USER'S INITIAL QUESTION: "{user_query}" # Refer to the initial question for the core intent
        LATEST GENERATED ANSWER: "{generated_answer}" # Evaluate the completeness of this specific answer
    
        **CRITICAL EVALUATION:**
        Based on the ENTIRE CONVERSATION HISTORY and the LATEST GENERATED ANSWER:
        1.  Does the latest answer directly address the core medical question posed by the user, using all available information in the history?
        2.  Does the answer include necessary safety disclaimers for personal medical queries?
        3.  Are there any obvious gaps regarding *critical safety information* (e.g., symptoms requiring urgent care) that are missing from the latest answer, given what the user has described throughout the conversation?
        4.  **Review the Conversation History Carefully:** Has a similar or the same type of critical follow-up question *already been clearly asked* by the Assistant in a previous turn that the user responded to or did not provide the requested information for?
        5.  **Considering all information in the history and the latest answer:** Is there *one single most critical piece* of information still needed from the user to proceed safely or provide a meaningfully better answer?
    
        STRICT REQUIREMENTS:
        1. You must ONLY identify absolutely critical missing information that is essential for safety.
        2. You must NEVER ask a follow-up question that is semantically similar to anything in the "PREVIOUSLY ASKED QUESTIONS" section.
        3. You must return EXACTLY ONE follow-up question at most - never multiple questions.
        4. If no critical information is missing or a similar question was already asked, you must set "needs_followup" to false.
    
        Return your answer in this exact JSON format:
        {{
            "needs_followup": true/false,
            "reasoning": "brief explanation of why more information is needed from the answer or why the answer is sufficient, referencing the history review and critical gaps",
            "missing_info_questions": [
                {{"question": "specific follow-up question 1"}}
            ]
        }}
    
        Only include the "missing_info_questions" array if "needs_followup" is true, and limit it to exactly 1 question. If "needs_followup" is true but you cannot formulate the *single most critical* specific question based on your evaluation, still return "needs_followup": true but with an empty "missing_info_questions" array (though try hard to formulate one if needed).
        '''.format(
            context=context,
            user_query=user_query,
            generated_answer=generated_answer,
            previously_asked=previously_asked_str
        )
    
        try:
            # Use local_generate for this LLM call
            response = self.local_generate(MISSING_INFO_PROMPT, max_tokens=500).strip()
    
            # Attempt to parse JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    needs_followup_llm = data.get("needs_followup", False)  # LLM's opinion
                    
                    # IMPORTANT FIX: Strictly enforce the limit of one question
                    missing_info_questions = []
                    if needs_followup_llm and "missing_info_questions" in data:
                        # Only take the first question if any exist
                        questions_list = data.get("missing_info_questions", [])
                        if questions_list and len(questions_list) > 0:
                            if isinstance(questions_list[0], dict) and "question" in questions_list[0]:
                                question = questions_list[0]["question"]
                                
                                # Check if the proposed question is similar to previously asked questions
                                is_redundant = any(
                                    self.calculate_similarity(question, prev_q) > 0.7  # Using a hypothetical similarity function
                                    for prev_q in previously_asked_questions
                                )
                                
                                if not is_redundant:
                                    missing_info_questions.append(question)
                                else:
                                    logger.info("⚠️ Proposed follow-up question is too similar to a previously asked question. Skipping.")
                                    needs_followup_llm = False  # Override the LLM's decision since the question is redundant
                    
                    reasoning = data.get("reasoning", "Answer is missing critical information.")
    
                    if needs_followup_llm and missing_info_questions:
                        logger.info(f"❓ Critical Information Missing from Final Answer (LLM opinion): {missing_info_questions}. Reasoning: {reasoning}")
                        return (True, missing_info_questions)  # Return True and questions
    
                    else:
                        logger.info("✅ Final Answer appears sufficient (LLM opinion) or no questions provided.")
                        return (False, [])  # Return False and no questions
    
                except json.JSONDecodeError:
                    logger.warning("Could not parse final missing info JSON from LLM response.")
                    # Fallback: Assume no critical info is missing if JSON parsing fails
                    return (False, [])
                except Exception as e:
                    logger.error(f"Error processing LLM response structure in identify_missing_info: {e}", exc_info=True)
                    return (False, [])  # Fallback on structure error
    
            else:
                logger.warning("LLM response did not contain expected JSON format.")
                # Fallback: Assume no critical info is missing if no JSON is found
                return (False, [])
    
        except Exception as e:
            logger.error(f"⚠️ Error during LLM call in identify_missing_info: {e}", exc_info=True)
            # Fallback: Assume no critical info is missing if LLM call fails
            return (False, [])


    def knowledge_graph_agent(self, user_query: str, all_symptoms: List[str]) -> Dict[str, Any]:
        """
        Knowledge Graph Agent - Extracts symptoms (done before calling),
        identifies diseases, and finds treatments/remedies.
        Returns a dictionary of KG results.
        Updated to return symptom associations for the UI step.
        """
        logger.info("📚 Knowledge Graph Agent Initiated")

        kg_results: Dict[str, Any] = {
            "extracted_symptoms": all_symptoms, # Symptoms used for KG query
            "identified_diseases_data": [], # List of {disease, conf, matched_symp, all_kg_symp} - used internally & for UI step
            "top_disease_confidence": 0.0, # Confidence of the highest match
            "kg_matched_symptoms": [], # Symptoms from input that matched for the top disease
            "kg_treatments": [],
            "kg_treatment_confidence": 0.0,
            "kg_home_remedies": [],
            "kg_remedy_confidence": 0.0,
            "kg_content_diagnosis_data_for_llm": { # Always provide data for LLM formatting fallback
                 "disease_name": "an unidentifiable condition",
                 "symptoms_list": all_symptoms,
                 "confidence": 0.0
            },
            "kg_content_other": "Medical Knowledge Graph information on treatments or remedies is unavailable.", # Default message
        }

        if not self.kg_connection_ok or self.kg_driver is None:
             logger.warning("📚 KG Agent: Connection not OK. Skipping KG queries.")
             kg_results["kg_content_other"] = "Medical Knowledge Graph is currently unavailable."
             # KG results remain empty/default
             return kg_results

        try:
            with self.kg_driver.session() as session:
                # Task 1: Identify Diseases from Symptoms
                if all_symptoms:
                    logger.info(f"📚 KG Task: Identify Diseases from symptoms: {all_symptoms}")
                    # query_disease_from_symptoms now uses the session to run queries
                    # It returns the list of dicts
                    disease_data_from_kg: List[Dict[str, Any]] = self._query_disease_from_symptoms_with_session(session, all_symptoms)

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

                        logger.info(f"✔️ Diseases Identified: {[(d['Disease'], d['Confidence']) for d in disease_data_from_kg]} (Top Confidence: {top_disease_conf:.4f})")

                        # Task 2 & 3: Find Treatments/Remedies (if a primary disease was identified with decent confidence)
                        # Query treatments/remedies for the TOP identified disease
                        if top_disease_conf >= THRESHOLDS.get("knowledge_graph_general", 0.6): # Use a general threshold for finding related info
                            logger.info(f"📚 KG Tasks: Find Treatments & Remedies for {top_disease_name}")

                            kg_treatments, kg_treatment_confidence = self._query_treatments_with_session(session, top_disease_name)
                            kg_results["kg_treatments"] = kg_treatments
                            kg_results["kg_treatment_confidence"] = kg_treatment_confidence
                            logger.info(f"✔️ Treatments found: {kg_treatments} (Confidence: {kg_treatment_confidence:.4f})")

                            kg_remedies, kg_remedy_confidence = self._query_home_remedies_with_session(session, top_disease_name)
                            kg_results["kg_home_remedies"] = kg_remedies
                            kg_results["kg_remedy_confidence"] = kg_remedy_confidence
                            logger.info(f"✔️ Home Remedies found: {kg_remedies} (Confidence: {kg_remedy_confidence:.4f})")
                        else:
                            logger.info("📚 KG Tasks: Treatments/Remedies skipped - Top disease confidence below threshold.")
                else:
                     logger.info("📚 KG Task: Identify Diseases skipped - No symptoms provided.")


                # Prepare data needed for the LLM formatting step if Path 1 is chosen
                # This data should be prepared even if no diseases were found, for the fallback phrasing
                kg_results["kg_content_diagnosis_data_for_llm"] = {
                      "disease_name": kg_results["identified_diseases_data"][0]["Disease"] if kg_results["identified_diseases_data"] else "an unidentifiable condition", # Use top disease or fallback
                      "symptoms_list": all_symptoms, # Use all input/confirmed symptoms for phrasing
                      "confidence": kg_results["top_disease_confidence"] # Use top confidence or 0.0
                }


                # Other KG content part (treatments/remedies) for Path 2 combination
                other_parts: List[str] = []
                if kg_results["kg_treatments"]:
                     other_parts.append("## Recommended Treatments (from KG)")
                     for treatment in kg_results["kg_treatments"]:
                          other_parts.append(f"- {treatment}")
                     other_parts.append("") # Add empty line for separation

                if kg_results["kg_home_remedies"]:
                     other_parts.append("## Home Remedies (from KG)")
                     for remedy in kg_results["kg_home_remedies"]:
                          remedy_text = remedy # Assume it's already a string
                          # Add source if available? KG typically doesn't have per-remedy sources unless modeled
                          other_parts.append(f"- {remedy_text}")
                     other_parts.append("") # Add empty line for separation

                kg_results["kg_content_other"] = "\n".join(other_parts).strip()
                # Only set default message if no treatments *or* remedies were actually found
                if not kg_results["kg_content_other"] and not kg_results["kg_treatments"] and not kg_results["kg_home_remedies"]:
                    kg_results["kg_content_other"] = "Medical Knowledge Graph did not find specific relevant information on treatments or remedies."


                logger.info("📚 Knowledge Graph Agent Finished successfully.")
                return kg_results

        except Exception as e:
            logger.error(f"⚠️ Error within KG Agent: {e}", exc_info=True) # Log traceback
            # Populate with error/empty info on failure
            kg_results["kg_content_diagnosis_data_for_llm"] = {
                 "disease_name": "an unidentifiable condition",
                 "symptoms_list": all_symptoms,
                 "confidence": 0.0
            } # Still provide data for LLM formatting fallback
            kg_results["kg_content_other"] = f"An error occurred while querying the Medical Knowledge Graph: {str(e)}"
            kg_results["top_disease_confidence"] = 0.0
            return kg_results


    # Helper methods to query KG with a session (reduces repetitive session handling)
    # These methods are called *by* the kg_agent
    def _query_disease_from_symptoms_with_session(self, session, symptoms: List[str]) -> List[Dict[str, Any]]:
         """Queries KG for diseases based on symptoms using an existing session."""
         if not symptoms:
              return []

         # Use a cache key based on the sorted list of symptoms
         cache_key = {"type": "disease_matching_v2", "symptoms": tuple(sorted([s.lower() for s in symptoms]))} # Cache lowercase sorted symptoms
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
              CASE WHEN total_disease_symptoms_count = 0 THEN 0
                     ELSE matching_symptoms_count * 1.0 / total_disease_symptoms_count
                END AS confidence_score
                WHERE matching_symptoms_count > 0 // Only return diseases with at least one matching symptom from input
         RETURN Disease, confidence_score AS Confidence, matched_symptoms_from_input AS MatchedSymptoms, all_disease_symptoms_in_kg AS AllDiseaseSymptomsKG
         ORDER BY confidence_score DESC
         LIMIT 5 // Limit potential diseases shown for performance/relevance
         """

         try:
              result = session.run(cypher_query, symptomNames=[s.lower() for s in symptoms if s]) # Pass parameter
              records = list(result)

              disease_data = [
                   {
                        "Disease": rec["Disease"],
                        "Confidence": float(rec["Confidence"]),
                        "MatchedSymptoms": rec["MatchedSymptoms"], # List of symptom strings (KG case)
                        "AllDiseaseSymptomsKG": rec["AllDiseaseSymptomsKG"] # List of symptom strings (KG case)
                   }
                   for rec in records
              ]

              logger.debug(f"🦠 Executed KG Disease Query, found {len(disease_data)} results.")
              set_cached(cache_key, disease_data)
              return disease_data

         except Exception as e:
              logger.error(f"⚠️ Error executing KG query for diseases: {e}")
              return [] # Return empty list on failure


    def _query_treatments_with_session(self, session, disease: str) -> Tuple[List[str], float]:
         """Queries KG for treatments using an existing session."""
         if not disease:
              return [], 0.0

         cache_key = {"type": "treatment_query_kg", "disease": disease.lower()}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("🧠 Using cached KG treatments.")
             return cached

         cypher_query = """
         MATCH (d:disease)-[r:TREATED_BY]->(t:treatment)
         WHERE toLower(d.Name) = toLower($diseaseName)
         WITH t, COUNT(r) as rel_count
         RETURN t.Name as Treatment,
                CASE WHEN rel_count > 3 THEN 0.9
                     WHEN rel_count > 1 THEN 0.8
                     ELSE 0.7
                END as Confidence
         ORDER BY Confidence DESC
         """ # Use parameter $diseaseName

         try:
              result = session.run(cypher_query, diseaseName=disease) # Pass parameter
              records = list(result)

              treatments_list: List[str] = []
              avg_confidence = 0.0

              if records:
                   treatments = [(rec["Treatment"], float(rec["Confidence"])) for rec in records]
                   treatments_list = [t[0] for t in treatments]
                   avg_confidence = sum(t[1] for t in treatments) / len(treatments) if treatments else 0.0

              logger.debug(f"💊 Executed KG Treatment Query for {disease}, found {len(treatments_list)} treatments.")
              result = (treatments_list, avg_confidence)
              set_cached(cache_key, result)
              return result
         except Exception as e:
              logger.error(f"⚠️ Error executing KG query for treatments: {e}")
              return [], 0.0


    def _query_home_remedies_with_session(self, session, disease: str) -> Tuple[List[str], float]:
         """Queries KG for home remedies using an existing session."""
         if not disease:
             return [], 0.0

         cache_key = {"type": "remedy_query_kg", "disease": disease.lower()}
         cached = get_cached(cache_key)
         if cached:
             logger.debug("🧠 Using cached KG home remedies.")
             return cached

         cypher_query = """
         MATCH (d:disease)-[r:HAS_HOMEREMEDY]->(h:homeremedy)
         WHERE toLower(d.Name) = toLower($diseaseName)
         WITH h, COUNT(r) as rel_count
         RETURN h.Name as HomeRemedy,
                CASE WHEN rel_count > 2 THEN 0.85
                     WHEN rel_count > 1 THEN 0.75
                     ELSE 0.65
                END as Confidence
         ORDER BY Confidence DESC
         """ # Use parameter $diseaseName

         try:
             result = session.run(cypher_query, diseaseName=disease) # Pass parameter
             records = list(result)

             remedies_list: List[str] = []
             avg_confidence = 0.0

             if records:
                 remedies = [(rec["HomeRemedy"], float(rec["Confidence"])) for rec in records]
                 remedies_list = [r[0] for r in remedies]
                 avg_confidence = sum(r[1] for r in remedies) / len(remedies) if remedies else 0.0

             logger.debug(f"🏡 Executed KG Remedy Query for {disease}, found {len(remedies_list)} remedies.")
             result = (remedies_list, avg_confidence)
             set_cached(cache_key, result)
             return result
         except Exception as e:
             logger.error(f"⚠️ Error executing KG query for home remedies: {e}")
             return [], 0.0


    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        """Extract symptoms from user query with confidence scores using LLM."""
        cache_key = {"type": "symptom_extraction", "query": user_query}
        cached = get_cached(cache_key)
        if cached:
            logger.debug("🧠 Using cached symptom extraction.")
            return cached

        if self.llm is None:
             logger.warning("LLM not initialized. Cannot perform LLM symptom extraction.")
             # Fallback to keyword extraction if LLM is not available
             fallback_symptoms = []
             common_symptom_keywords = ["fever", "cough", "headache", "sore throat", "nausea", "dizziness", "chest pain", "shortness of breath", "fatigue", "body aches", "runny nose", "congestion", "chills", "sweats", "joint pain", "muscle aches", "rash", "swelling", "pain", "ache", "burning", "itching", "numbness", "tingling", "diarrhea", "vomiting", "difficulty breathing", "difficulty swallowing"]
             query_lower = user_query.lower()
             for symptom in common_symptom_keywords:
                 if symptom in query_lower:
                     fallback_symptoms.append(symptom.capitalize()) # Capitalize for consistency

             logger.info(f"🔍 Fallback Extracted Symptoms (LLM failed): {fallback_symptoms} (confidence: 0.4)")
             result = (fallback_symptoms, 0.4) # Low confidence for fallback
             set_cached(cache_key, result)
             return result


        # Use LLM for extraction first
        # Use local_generate which wraps self.llm
        SYMPTOM_PROMPT = '''
        You are a medical assistant.
        Extract all medical symptoms mentioned in the following user query.
        For each symptom, assign a confidence score between 0.0 and 1.0 indicating how certain you are that it is a symptom.
        Be strict and only extract actual symptoms or medical signs.
        **Important:** Return your answer in exactly the following format:
        Extracted Symptoms: [{{"symptom": "symptom1", "confidence": 0.9}}, {{"symptom": "symptom2", "confidence": 0.8}}, ...]

        User Query: "{}"
        '''.format(user_query)

        llm_symptoms = []
        llm_avg_confidence = 0.0

        try:
            # Use local_generate for the LLM call
            response = self.local_generate(SYMPTOM_PROMPT, max_tokens=500).strip()
            # logger.debug(f"\nRaw Symptom Extraction Response:\n{response}")

            # Parse JSON format response with regex
            match = re.search(r"Extracted Symptoms:\s*(\[.*?\])", response, re.DOTALL)
            if match:
                try:
                    symptom_data = json.loads(match.group(1))
                    # Filter symptoms based on confidence threshold
                    llm_symptoms_confident = [item["symptom"].strip() # Keep original casing for KG matching attempt later
                                            for item in symptom_data
                                            if item.get("confidence", 0) >= THRESHOLDS["symptom_extraction"]]

                    # Calculate average confidence for all symptoms returned by LLM before thresholding
                    if symptom_data:
                        llm_avg_confidence = sum(item.get("confidence", 0) for item in symptom_data) / len(symptom_data)
                    else:
                        llm_avg_confidence = 0.0

                    llm_symptoms = llm_symptoms_confident # Use the thresholded list
                    # logger.debug(f"🔍 LLM Extracted Symptoms (confident): {llm_symptoms} (avg raw confidence: {llm_avg_confidence:.4f})")

                except json.JSONDecodeError:
                    logger.warning("Could not parse symptom JSON from LLM response")
            else:
                 logger.warning("Could not find 'Extracted Symptoms: [...]: in LLM response.")

        except Exception as e:
            logger.error(f"Error in LLM symptom extraction: {e}")

        # Fallback/Enhancement with Keyword Matching
        # If LLM extraction failed or returned nothing, use keyword matching as the sole result.
        # If LLM extraction succeeded, combine with keyword matching.
        fallback_symptoms_from_keywords = []
        common_symptom_keywords = ["fever", "cough", "headache", "sore throat", "nausea", "dizziness", "chest pain", "shortness of breath", "fatigue", "body aches", "runny nose", "congestion", "chills", "sweats", "joint pain", "muscle aches", "rash", "swelling", "pain", "ache", "burning", "itching", "numbness", "tingling", "diarrhea", "vomiting", "difficulty breathing", "difficulty swallowing"]
        query_lower = user_query.lower()

        for symptom in common_symptom_keywords:
            # Simple keyword presence check
            if symptom in query_lower:
                fallback_symptoms_from_keywords.append(symptom.capitalize()) # Capitalize for consistency with KG case

        if not llm_symptoms and fallback_symptoms_from_keywords:
             # LLM failed/found nothing, rely on keywords
             combined_symptoms = list(set(fallback_symptoms_from_keywords))
             final_confidence = 0.4 # Low confidence for keyword-only extraction
             logger.info(f"🔍 Keyword Fallback Extracted Symptoms: {combined_symptoms} (confidence: {final_confidence:.4f})")
        else:
             # Combine LLM symptoms (confident ones) and keyword symptoms
             combined_symptoms = list(set(llm_symptoms + fallback_symptoms_from_keywords)) # Use set to deduplicate

             # Assign a confidence score. If LLM provided confident symptoms, use its average. Otherwise, use a fallback confidence.
             if llm_symptoms:
                 final_confidence = llm_avg_confidence # Use LLM's average if it found anything
             elif combined_symptoms: # If keywords found something but LLM didn't
                  final_confidence = 0.4 # Low confidence for combined if LLM failed
             else: # Nothing found
                 final_confidence = 0.0

             logger.info(f"🔍 Final Extracted Symptoms: {combined_symptoms} (confidence: {final_confidence:.4f})")


        result = (combined_symptoms, final_confidence)
        set_cached(cache_key, result)
        return result


    def is_disease_identification_query(self, query: str) -> bool:
        """Improved check for queries primarily focused on identifying a disease from symptoms."""
        query_lower = query.lower()

        # Keywords/patterns that strongly suggest disease identification
        disease_keywords = [
            r"what disease", r"what condition", r"what could be causing",
            r"what might be causing", r"possible disease", r"possible condition",
            r"diagnosis", r"diagnose", r"what causes", r"what is causing",
            r"what do i have", r"what do they have", r"could this be", r"is it possible i have",
            r"what's wrong with me", r"what does this mean", r"identify (?:a )?(?:condition|disease)", # Added more phrases
            r"symptoms.*mean", r"what does .* symptom.* indicate", r"what is .* symptom of",
            r"what about .* symptoms", r"what could be .* (?:illness|sickness)"
        ]

        # Check for symptom mentions (using a broader keyword list or rely on symptom extraction)
        # Rely on successful symptom extraction as a strong indicator
        extracted_symptoms, _ = self.extract_symptoms(query)
        has_symptoms = len(extracted_symptoms) > 0


        # Check for disease identification intent using regex patterns
        is_asking_for_disease = any(re.search(pattern, query_lower) for pattern in disease_keywords)

        # It's a disease identification query if it explicitly asks for one (and symptoms were found),
        # or if it's a personal query structure combined with symptoms found,
        # or if it's a query that looks like a list of symptoms and asks implicitly.
        is_symptom_query_pattern_v2 = re.search(r"i have .* (?:and|with) .*\.? what could (?:it|they|this) be", query_lower) is not None
        is_personal_symptoms_query = (
             ("i have" in query_lower or "my symptoms are" in query_lower or "i'm experiencing" in query_lower or "these are my symptoms" in query_lower or "my health issue is" in query_lower) and has_symptoms
        )
        # Check if the query is primarily a list of symptoms ending with a question mark
        # Use the extracted symptoms to make this check more robust
        is_symptom_list_query = (
            has_symptoms and # Must contain symptom keywords
            query_lower.strip().endswith('?') and # Must end with a question mark
            # Check if most of the query consists of potential symptom phrases (simple heuristic using extracted symptoms)
            len([word for word in query_lower.split() if any(s.lower() in word.lower() for s in extracted_symptoms)]) / max(len(query_lower.split()), 1) > 0.3 # Lower threshold
        )


        # Combine conditions
        is_disease_query = is_asking_for_disease or \
                           is_symptom_query_pattern_v2 or \
                           is_personal_symptoms_query or \
                           is_symptom_list_query

        # logger.debug(f"Is disease identification query ('{query}')? {is_disease_query}")
        return is_disease_query


    def identify_missing_elements(self, user_query: str, generated_answer: str) -> List[str]:
        """
        Identifies high-level concepts (like duration, severity, specific history points)
        that *might* be missing from the answer relative to the query's potential intent.
        This is a simpler check than identify_missing_info.
        Used *before* LLM synthesis to tell the LLM what to focus on.
        """
        logger.debug("🔍 Identifying high-level potential missing elements for LLM focus...")
        missing = set()
        query_lower = user_query.lower()
        answer_lower = generated_answer.lower()

        # Simple rule-based checks based on common medical info needs
        # Check if query mentions personal symptoms ("i have", "my symptoms") or looks like a personal case
        is_personal_symptom_query = ("i have" in query_lower or "my symptoms are" in query_lower or "i'm experiencing" in query_lower or self.is_disease_identification_query(user_query))

        if is_personal_symptom_query:
            # Look for common ways duration is mentioned in the answer
            duration_keywords_in_answer = [" duration", "days", "weeks", "months", "how long", "since", "for X time"] # Added 'since'
            if not any(kw in answer_lower for kw in duration_keywords_in_answer):
                missing.add("duration")

            # Look for common ways severity is mentioned in the answer
            severity_keywords_in_answer = [" severity", "mild", "moderate", "severe", "how severe", "intense", "level of pain", "scale of"] # Added more terms
            if not any(kw in answer_lower for kw in severity_keywords_in_answer):
                 missing.add("severity")

            # Add other checks as needed, e.g., "location", "frequency", "relieved by", "worsened by", "onset"
            # Location: "where", "location", "area", "left side", "right side", specific body parts mentioned in query?
            location_keywords_in_answer = [" location", "where", "area", "on the left", "on the right", "in the chest", "in the abdomen", "radiating"] # Example
            # Only add location if the query mentioned a potentially localized symptom like pain, rash etc.
            if any(symptom in query_lower for symptom in ["pain", "ache", "rash", "swelling", "bruise", "tenderness"]):
                 if not any(kw in answer_lower for kw in location_keywords_in_answer):
                      missing.add("location")


            # Frequency: "how often", "frequency", "intermittent", "constant", "sporadic"
            frequency_keywords_in_answer = [" frequency", "how often", "intermittent", "constant", "sporadic", "comes and goes"]
            # Only add frequency if the query mentioned symptoms that could be episodic
            if any(symptom in query_lower for symptom in ["pain", "headache", "dizziness", "nausea", "palpitations"]):
                 if not any(kw in answer_lower for kw in frequency_keywords_in_answer):
                    missing.add("frequency")

            # Onset: "when it started", "onset", "began", "started"
            onset_keywords_in_answer = [" onset", "started", "began", "when it happened"]
            # Onset is generally useful for any personal symptom query
            if not any(kw in answer_lower for kw in onset_keywords_in_answer):
                 missing.add("onset")


        # This is a heuristic, not a definitive check like identify_missing_info
        missing_list = list(missing)
        # logger.debug(f"Identified potential missing elements (for LLM focus): {missing_list}")
        return missing_list


    def combine_initial_answer_draft(self, kg_diagnosis_component: Optional[str], kg_content_other: str, rag_content: str) -> str:
         """Combines the Path 1 KG diagnosis component (if any), other KG content, and RAG content."""
         logger.info("Merging KG (diagnosis & other) and RAG results into initial draft...")
         combined_parts: List[str] = []

         # Add KG diagnosis component if generated
         if kg_diagnosis_component and kg_diagnosis_component.strip() != "":
              combined_parts.append(kg_diagnosis_component.strip())

         # Add other KG content (treatments/remedies) if found and is not the default empty message
         if kg_content_other and kg_content_other.strip() != "" and kg_content_other.strip() != "Medical Knowledge Graph information on treatments or remedies is unavailable." and kg_content_other.strip() != "Knowledge Graph did not find specific relevant information on treatments or remedies.":
              # Add other KG content (treatments/remedies), separate if diagnosis is present
              if combined_parts:
                   combined_parts.append("\n\n" + kg_content_other.strip())
              else:
                   combined_parts.append(kg_content_other.strip())


         # Add RAG content if found and is not the default empty message
         if rag_content and rag_content.strip() != "" and rag_content.strip() != "An error occurred while retrieving information from documents." and rag_content.strip() != "Document search is currently unavailable." and rag_content.strip() != "I searched my documents but couldn't find specific information for that.":
              # Add RAG content, potentially separated
              if combined_parts:
                   combined_parts.append("\n\n## Additional Information from Document Search\n")
                   combined_parts.append(rag_content.strip())
              else:
                   combined_parts.append(rag_content.strip())
         elif not combined_parts: # If neither KG diagnosis, other KG, nor RAG had useful content
               combined_parts.append("I found limited specific information regarding your query from my knowledge sources.")


         initial_combined_answer = "\n".join(combined_parts).strip()
         logger.debug("Initial combined answer draft created.")
         return initial_combined_answer


    # --- Main Response Generation Function (Orchestrator) ---

    # Updated signature to accept confirmed_symptoms and original_query_if_followup
    # Returns: (response_text, sources_list, action_flag, ui_data)
    # action_flag: "final_answer", "llm_followup_prompt", "symptom_ui_prompt", "none" (no response/action needed)
    # ui_data: None or Dict { "symptom_options": {disease_label: [symptoms]}, "original_query": str } for "symptom_ui_prompt"
    
    def generate_response(self, user_input: str, user_type: str = "User / Family", confirmed_symptoms: Optional[List[str]] = None, original_query_if_followup: Optional[str] = None) -> Tuple[str, List[str], str, Optional[Dict]]:
        """
        Generate response using orchestration based on Path 1 / Path 2 logic,
        prioritizing internal gap filling before user follow-up.
        """
        global RELIABLE_MEDICAL_SOURCES # Declare intent to use the global variable
        logger.info(f"--- Generating Response for Input: '{user_input}' ---")
        logger.info(f"   Confirmed symptoms from UI: {confirmed_symptoms}")
        logger.info(f"   Original query if follow-up: '{original_query_if_followup}'")
        logger.info(f"   Current followup_context: {self.followup_context}")
        logger.info(f"   Current chat_history length: {len(self.chat_history)}")

        core_query_for_processing = original_query_if_followup if original_query_if_followup is not None else user_input
        logger.info(f"   Core query for processing logic: '{core_query_for_processing}'")

        if not core_query_for_processing.strip() and confirmed_symptoms is None:
             logger.info("Empty core query and no confirmed symptoms. Skipping.")
             return "", [], "none", None

        # --- Initialization Check ---
        if self.llm is None or self.qa_chain is None or not self.kg_connection_ok:
            logger.info("Chatbot is not fully initialized. Attempting re-initialization...")
            success, message = self.initialize_qa_chain()
            if not success:
                error_message = f"Error processing request: Assistant failed to initialize fully ({message}). Some features may be unavailable. Please check your configuration and try again later."
                self.log_orchestration_decision(core_query_for_processing, f"SELECTED_STRATEGY: INIT_ERROR\nREASONING: Re-initialization failed: {message}", 0.0, 0.0)
                if self.llm is None:
                     logger.critical("LLM is still not initialized after re-attempt. Cannot generate any response.")
                     return error_message, [], "final_answer", None
                else:
                    logger.warning("Initialization partially successful (LLM available). Proceeding with limited features (No RAG or KG).")


        is_response_to_llm_followup = original_query_if_followup is not None and self.followup_context["round"] == 1
        if is_response_to_llm_followup:
             logger.info(f"Detected response to LLM follow-up (round {self.followup_context['round']}). Processing '{user_input}' in context of '{original_query_if_followup}'.")


        # --- Step 2: Extract Symptoms ---
        extracted_symptoms_from_core_query, extracted_conf_core = self.extract_symptoms(core_query_for_processing)
        all_symptoms: List[str] = []
        symptom_confidence = 0.0

        if is_response_to_llm_followup:
            extracted_symptoms_from_response, response_conf = self.extract_symptoms(user_input)
            all_symptoms = list(set(extracted_symptoms_from_response + extracted_symptoms_from_core_query))
            symptom_confidence = max(response_conf, extracted_conf_core)
            logger.info(f"Combined symptoms from response and original query: {all_symptoms}")
        elif confirmed_symptoms is not None:
            extracted_symptoms_from_input, extracted_conf_input = self.extract_symptoms(user_input)
            all_symptoms = list(set(extracted_symptoms_from_input + confirmed_symptoms))
            symptom_confidence = max(extracted_conf_input, 0.9)
            logger.info(f"Using symptoms from UI confirmation: {confirmed_symptoms}. Combined with input: {all_symptoms}")
        else:
            all_symptoms = extracted_symptoms_from_core_query
            symptom_confidence = extracted_conf_core
            logger.info(f"Extracted symptoms from input: {all_symptoms}")


        # --- Step 3: KG Processing ---
        logger.info("📚 Processing with Knowledge Graph...")
        t_start_kg = datetime.now()
        kg_data = self.knowledge_graph_agent(core_query_for_processing, all_symptoms)
        top_disease_confidence = kg_data.get("top_disease_confidence", 0.0)
        kg_diagnosis_data_for_llm = kg_data.get("kg_content_diagnosis_data_for_llm")
        kg_content_other = kg_data.get("kg_content_other", "")
        logger.info(f"📊 KG Top Disease Confidence: {top_disease_confidence:.4f} (took {(datetime.now() - t_start_kg).total_seconds():.2f}s)")


        # --- Step 4: Path 1 - Diagnosis Focus & Symptom Follow-up UI (Decision Point 1) ---
        is_disease_query = self.is_disease_identification_query(core_query_for_processing)
        kg_found_diseases = len(kg_data.get("identified_diseases_data", [])) > 0

        # Condition to trigger Symptom Follow-up UI:
        if is_disease_query and \
           kg_found_diseases and \
           top_disease_confidence < THRESHOLDS["disease_symptom_followup_threshold"] and \
           confirmed_symptoms is None and \
           any(d.get("AllDiseaseSymptomsKG") for d in kg_data.get("identified_diseases_data", [])[:5]) and \
           self.followup_context["round"] == 0 and \
           self.llm is not None and \
           self.kg_connection_ok:

            logger.info(f"❓ Disease query ('{core_query_for_processing}') with low KG confidence ({top_disease_confidence:.4f}). Triggering Symptom Follow-up UI.")
            symptom_options_for_ui: Dict[str, List[str]] = {}
            relevant_diseases_for_ui = [d for d in kg_data["identified_diseases_data"][:5] if d.get("AllDiseaseSymptomsKG")]
            for disease_data in relevant_diseases_for_ui:
                 disease_label = f"{disease_data['Disease']} (Confidence: {disease_data['Confidence']:.2f})"
                 symptoms_list = sorted(list(set(str(s) for s in disease_data.get("AllDiseaseSymptomsKG", []) if isinstance(s, str))))
                 if symptoms_list:
                    symptom_options_for_ui[disease_label] = symptoms_list

            follow_up_prompt_text = f"""
            Thank you for sharing your symptoms. Based on what you've told me, I found some potential conditions.
            To help narrow it down and provide a more relevant response, please confirm which of the associated symptoms from my knowledge base you are also experiencing.
            """
            self.log_orchestration_decision(
                core_query_for_processing,
                f"SELECTED_STRATEGY: SYMPTOM_UI_FOLLOWUP\nREASONING: Disease query with low KG confidence ({top_disease_confidence:.2f}). Presenting symptom checklist for confirmation.",
                top_disease_confidence, 0.0
            )
            logger.info("Returning Symptom UI prompt.")
            return follow_up_prompt_text.strip(), [], "symptom_ui_prompt", {"symptom_options": symptom_options_for_ui, "original_query": core_query_for_processing}


        # --- Step 5: Path 1 - Direct KG Diagnosis Component (if high confidence or after symptom confirmation) ---
        path1_kg_diagnosis_component = None
        is_high_conf_kg_diagnosis = is_disease_query and kg_found_diseases and top_disease_confidence >= THRESHOLDS["disease_symptom_followup_threshold"]
        is_post_symptom_confirmation = confirmed_symptoms is not None

        if (is_high_conf_kg_diagnosis or is_post_symptom_confirmation) and kg_diagnosis_data_for_llm:
            if self.llm is not None:
                 logger.info(f"✅ Path 1: High confidence KG diagnosis ({top_disease_confidence:.4f}) OR received symptom confirmation. Formatting KG diagnosis answer with LLM.")
                 path1_kg_diagnosis_component = self.format_kg_diagnosis_with_llm(
                      kg_diagnosis_data_for_llm["disease_name"],
                      all_symptoms,
                      kg_diagnosis_data_for_llm["confidence"]
                  )
                 logger.info(f"   --- KG Diagnosis Component Generated ---")
            else:
                 logger.warning("⚠️ LLM not available for formatting KG diagnosis. Using manual format.")
                 disease_name = kg_diagnosis_data_for_llm["disease_name"]
                 symptoms_str = ", ".join(all_symptoms) if all_symptoms else "your symptoms"
                 path1_kg_diagnosis_component = f"Based on {symptoms_str}, **{disease_name}** is a potential condition. This is not a definitive diagnosis and requires professional medical evaluation."
        elif (is_high_conf_kg_diagnosis or is_post_symptom_confirmation):
             logger.info("⚠️ KG found no diseases even after symptom input. Proceeding without specific KG diagnosis component.")
             path1_kg_diagnosis_component = "Based on the symptoms provided, I couldn't find a specific medical condition matching them in my knowledge base."


        # --- Step 6: Path 2 - RAG Processing ---
        logger.info("📚 Processing with RAG...")
        t_start_rag = datetime.now()
        rag_content = ""
        rag_source_docs = []
        rag_confidence = 0.0

        if self.llm is not None and self.qa_chain is not None:
            try:
                 logger.info(f"Invoking RAG chain with question: {core_query_for_processing}")
                 rag_response = self.qa_chain.invoke({"question": core_query_for_processing})
                 rag_content = rag_response.get("answer", "").strip()
                 if "Helpful Answer:" in rag_content:
                      rag_content = rag_content.split("Helpful Answer:", 1)[-1].strip()
                 rag_source_docs = rag_response.get("source_documents", [])
                 if rag_source_docs:
                      rag_confidence = 0.3 + min(len(rag_source_docs), 5) / 5.0 * 0.4
                 logger.info(f"📊 RAG Confidence: {rag_confidence:.4f} (took {(datetime.now() - t_start_rag).total_seconds():.2f}s)")
            except Exception as e:
                 logger.error(f"⚠️ Error during RAG processing: {e}", exc_info=True)
                 rag_content = "An error occurred while retrieving information from documents."
                 rag_source_docs = []
                 rag_confidence = 0.0
        else:
             logger.warning("Warning: RAG chain (or necessary components) not initialized. Skipping RAG processing.")
             rag_content = "Document search is currently unavailable."


        # --- Step 7: Initial Combination of Path 1 Component and RAG ---
        initial_combined_answer = self.combine_initial_answer_draft(
             path1_kg_diagnosis_component,
             kg_content_other,
             rag_content
        )

        # --- Step 8: Reflection Agent (Main Check - Initial Draft) ---
        logger.info("🧠 Initiating Reflection Agent (Main Check - Initial Draft)...")
        needs_followup_initial, missing_info_questions_initial = (False, [])
        if self.llm is not None:
             # Use identify_missing_info to check the INITIAL combined draft
             needs_followup_initial, missing_info_questions_initial = self.identify_missing_info(
                  core_query_for_processing, initial_combined_answer, self.chat_history
             )
        else:
             logger.warning("LLM not available for initial reflection check.")


        final_core_answer = initial_combined_answer # Start with the initial draft
        final_reflection_needed = True # Assume we need a final reflection after potential gap filling

        # --- NEW: Conditional Internal Gap Filling / Synthesis ---
        # If the initial draft is insufficient according to reflection AND LLM is available
        if needs_followup_initial and missing_info_questions_initial and self.llm is not None:
            logger.info("➡️ Initial draft insufficient. Initiating Internal Gap Filling / Synthesis.")

            # Formulate a prompt for the LLM to specifically address the missing areas
            gap_filling_prompt_parts = [
                "You are a medical AI assistant tasked with providing missing information based on provided sources.",
                f"USER'S ORIGINAL QUESTION: {core_query_for_processing}",
                "Based on the available sources below, please provide information that directly addresses the following specific areas:"
            ]
            # List the questions derived from the missing info areas
            for i, question_data in enumerate(missing_info_questions_initial, 1):
                 gap_filling_prompt_parts.append(f"{i}. {question_data}")

            # Include all available source contexts
            gap_filling_prompt_parts.append("\n**Available Medical Knowledge Graph Information:**")
            gap_filling_prompt_parts.append("---\n" + (kg_content_other.strip() if kg_content_other.strip() != "Knowledge Graph information on treatments or remedies is unavailable." else "No relevant KG info found.") + "\n---") # Use other KG content

            # Pass raw RAG source content, not the RAG answer summary, for detailed info
            rag_raw_content = "\n---\n".join([doc.page_content for doc in rag_source_docs]) if rag_source_docs else "No relevant document content found."
            gap_filling_prompt_parts.append("\n**Available Raw Document Content (from search):**")
            gap_filling_prompt_parts.append("---\n" + rag_raw_content + "\n---")
            
            reliable_sources_prompt_section_local = "\n\n**Additional Medically Reliable External Sources (Use for relevant information and link directly):**\n"
            if RELIABLE_MEDICAL_SOURCES: # Use the global constant RELIABLE_MEDICAL_SOURCES
                for src in RELIABLE_MEDICAL_SOURCES:
                    reliable_sources_prompt_section_local += f"- [{src['description']}]({src['url']}) (Relevant to: {', '.join(src['diseases'])})\n"
                reliable_sources_prompt_section_local += "When using information related to the diseases listed with these links, include the corresponding clickable Markdown link directly in your answer where appropriate, e.g., `[Description](URL)`.\n"
            else:
                reliable_sources_prompt_section_local += "No specific external reliable sources are available in the current context.\n"
            # Add the external reliable sources
            gap_filling_prompt_parts.append("\n" + reliable_sources_prompt_section) # Use the same section content

            gap_filling_prompt_parts.append("\nProvide concise answers to the requested areas using ONLY the provided sources. If information for an area is not in the sources, state that. Attribute information using the specified source formats ([Source:...], clickable link for external sources). Do NOT invent information or sources.")
            gap_filling_prompt_parts.append("Format your answers clearly.")

            gap_filling_prompt = "\n\n".join(gap_filling_prompt_parts)

            # Call the LLM to generate text specifically for the gaps
            gap_filling_text = ""
            try:
                 # Use local_generate for the gap filling call
                 gap_filling_text = self.local_generate(gap_filling_prompt, max_tokens=800).strip() # Limit tokens for this part
                 logger.info("✔️ Internal Gap Filling Synthesis completed.")
            except Exception as e:
                 logger.error(f"Error during Internal Gap Filling LLM call: {e}", exc_info=True)
                 gap_filling_text = "An error occurred while trying to find more information from sources."

            # Combine the Initial Draft with the Gap Filling text
            # This is a crucial combination step - need to avoid redundancy
            # A simple concatenation might work, but a smarter merge is better
            # For simplicity, let's just append the gap filling text if it's substantial
            combined_with_gap_filling = initial_combined_answer.strip()
            if gap_filling_text and gap_filling_text.strip() != "An error occurred while trying to find more information from sources.":
                 combined_with_gap_filling += "\n\n" + gap_filling_text.strip()

            final_core_answer = combined_with_gap_filling # This is the new candidate answer
            logger.info("Combined initial draft with gap-filling text.")
            # No need for a second reflection check *here*; the next step is the final reflection on *this* answer.
            # final_reflection_needed = True # Already true by default


        # --- Step 9: Final Reflection Agent (Final Check) ---
        # This reflection happens whether gap filling occurred or not, or if the initial draft was deemed sufficient.
        logger.info("🧠 Initiating Final Reflection Agent (Final Check)...")
        # Use identify_missing_info to check the FINAL core answer candidate
        needs_final_followup, missing_questions_list = (False, [])
        if self.llm is not None: # Only check if LLM is available
             needs_final_followup, missing_questions_list = self.identify_missing_info(
                  core_query_for_processing, final_core_answer, self.chat_history # Pass FINAL candidate answer
             )
        else:
             logger.warning("LLM not available for final reflection check.")
             # If LLM is not available, we cannot ask a follow-up. The answer is effectively final.
             needs_final_followup = False
             missing_questions_list = [] # Ensure list is empty


        # --- Step 10: Final LLM Follow-up Decision / Return Final Answer ---
        # Decide if the ONE allowed LLM follow-up should be asked now.
        # Trigger if LLM *thinks* a follow-up is needed (from the FINAL check)
        # AND we haven't asked it yet (round == 0)
        # AND there's actually a question to ask
        # AND LLM is available to phrase the prompt.
        if needs_final_followup and self.followup_context["round"] == 0 and missing_questions_list and self.llm is not None:
             logger.info("❓ Final Reflection indicates missing critical info, asking the one allowed LLM follow-up.")
             follow_up_question_text = missing_questions_list[0] # Use the first question recommended

             llm_follow_up_prompt_text = f"""
             Thank you. To ensure I provide the most helpful response, I need a bit more information:

             {follow_up_question_text}
             """
             self.followup_context["round"] = 1 # Mark that the one LLM follow-up has been asked for this thread

             self.log_orchestration_decision(
                 core_query_for_processing,
                 f"SELECTED_STRATEGY: LLM_FINAL_FOLLOWUP\nREASONING: Final answer completeness check failed (LLM opinion) after internal synthesis. Asking for critical missing info (round 1).",
                 kg_data.get("top_disease_confidence", 0.0), rag_confidence
             )

             # Return prompt text and action flag
             logger.info("Returning LLM Follow-up prompt.")
             # We don't add this to chat history here. Let the UI manage adding the user's message and this prompt.
             return llm_follow_up_prompt_text.strip(), [], "llm_followup_prompt", None

        else:
            logger.info("✅ Final Reflection indicates answer is sufficient or single LLM follow-up already asked.")
            # This is the end of the processing path, return the final answer.

            # Collect all sources (RAG docs + KG mentions + Reliable External)
            all_sources: List[str] = []
            if rag_source_docs: # Add RAG sources
                for doc in rag_source_docs:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        source_name = doc.metadata["source"]
                        page_num = doc.metadata.get("page", "N/A")
                        source_snippet = doc.page_content[:100].replace('\n', ' ') + '...' if doc.page_content else ''
                        all_sources.append(f"[Source: {source_name}{f', Page {page_num}' if page_num != 'N/A' else ''}] {source_snippet}".strip())
                    else:
                         all_sources.append(f"[Source: Document]")


            if self.kg_connection_ok and (kg_data.get("identified_diseases_data") or kg_data.get("kg_treatments") or kg_data.get("kg_home_remedies")):
                 kg_parts_mentioned = []
                 if kg_data.get("identified_diseases_data"): kg_parts_mentioned.append("Diagnosis Data")
                 if kg_data.get("kg_treatments"): kg_parts_mentioned.append("Treatment Data")
                 if kg_data.get("kg_home_remedies"): kg_parts_mentioned.append("Home Remedy Data")

                 if kg_parts_mentioned:
                    all_sources.append(f"[Source: Medical Knowledge Graph ({', '.join(kg_parts_mentioned)})]")
                 else:
                     all_sources.append(f"[Source: Medical Knowledge Graph]")


            # Add the reliable external sources to the list (always include if LLM synthesis happened?)
            # Let's include them if LLM was used for synthesis, as it had access to them.
            if self.llm is not None:
                 for src in RELIABLE_MEDICAL_SOURCES:
                      # Add the source as a clickable link for the final reference list
                      all_sources.append(f"[{src['description']}]({src['url']})")


            # Deduplicate and clean up source strings for final display
            all_sources_unique = sorted(list(set(s.strip() for s in all_sources if s.strip())))


            final_response_text = final_core_answer

            # Add references section if not already included by LLM synthesis
            has_reference_section_pattern = re.compile(r"##?\s*(?:References|Sources)[:\n]", re.IGNORECASE | re.DOTALL)
            has_reference_section = has_reference_section_pattern.search(final_response_text) is not None

            if not has_reference_section and all_sources_unique:
                references_section = "\n\n## References:\n"
                for i, src_display in enumerate(all_sources_unique, 1):
                     references_section += f"\n{i}. {src_display}"

                final_response_text += references_section


            # Add standard medical disclaimer if not already present or if needed for safety
            has_disclaimer_pattern = re.compile(r"This information is not a substitute for professional medical advice", re.IGNORECASE | re.DOTALL)
            has_disclaimer = has_disclaimer_pattern.search(final_response_text) is not None

            if not has_disclaimer:
                disclaimer = "\n\n---\nIMPORTANT MEDICAL DISCLAIMER:\nThis information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. "
                query_lower = core_query_for_processing.lower()
                mentions_serious_symptoms = any(symptom in query_lower for symptom in ["chest pain", "shortness of breath", "difficulty breathing"]) or \
                                           any(any(symptom.lower() in d['Disease'].lower() for symptom in ["heart attack", "angina", "pulmonary embolism", "pneumonia"]) for d in kg_data.get("identified_diseases_data", []))

                if mentions_serious_symptoms:
                     disclaimer += "Chest pain and shortness of breath can be symptoms of serious conditions requiring immediate medical attention. Please seek emergency medical care if you experience severe symptoms, or if symptoms are sudden or worsening. "

                disclaimer += "Always consult with a qualified healthcare provider for any questions you may have regarding a medical condition. Never disregard professional medical advice or delay seeking it because of something you have read here."
                final_response_text += disclaimer

            self.log_orchestration_decision(
                core_query_for_processing,
                f"SELECTED_STRATEGY: FINAL_ANSWER\nREASONING: Answer deemed sufficient after synthesis and final check or single LLM follow-up already asked.",
                kg_data.get("top_disease_confidence", 0.0), rag_confidence
            )

            self.chat_history.append((user_input, final_response_text.strip()))
            logger.info("Returning Final Answer.")
            return final_response_text.strip(), all_sources_unique, "final_answer", None

        # The exception handler at the very bottom of generate_response catches any uncaught errors


    # --- Existing and Adjusted Logging Functions ---

    def log_response_metrics(self, metrics):
        """Log response generation metrics to CSV for analysis."""
        # Note: This function isn't explicitly called in the current generate_response
        # as the primary logging is done via log_orchestration_decision.
        # You could add calls here if needed, e.g., after a final_answer is generated.
        try:
            log_file = "response_metrics.csv"
            file_exists = os.path.isfile(log_file)

            metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metrics["query"] = metrics["query"].replace("\n", " ") # Sanitize query for CSV

            with open(log_file, mode='a', newline='', encoding='utf-8') as file:
                fieldnames = ['timestamp', 'query', 'kg_confidence', 'rag_confidence',
                              'strategy', 'response_length', 'processing_time', 'source_count']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(metrics)

        except Exception as e:
            logger.error(f"⚠️ Error logging response metrics: {e}")

    def log_orchestration_decision(self, query, orchestration_result, kg_confidence, rag_confidence):
        """
        Analyzes and logs the orchestration decision for monitoring and improvement.
        Extracts the strategy and reasoning from the orchestration result.
        """
        try:
            # Extract strategy and reasoning
            strategy = "UNKNOWN"
            reasoning = "Not provided"

            if "SELECTED_STRATEGY:" in orchestration_result:
                strategy_part = orchestration_result.split("SELECTED_STRATEGY:")[1].split("\n", 1)[0].strip() # Take first line after strategy
                strategy = strategy_part

            if "REASONING:" in orchestration_result:
                # Capture everything between REASONING: and RESPONSE: or end of string
                reasoning_match = re.search(r"REASONING:(.*?)(?:RESPONSE:|$)", orchestration_result, re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()

            # Truncate reasoning if too long for CSV
            max_reasoning_len = 200
            if len(reasoning) > max_reasoning_len:
                 reasoning = reasoning[:max_reasoning_len] + "..."


            # Log the decision
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query.replace("\n", " "), # Sanitize query for CSV
                "strategy": strategy,
                "reasoning": reasoning,
                "kg_confidence": kg_confidence,
                "rag_confidence": rag_confidence
            }

            # Print logging information
            logger.info(f"📊 Orchestration Decision:")
            logger.info(f"   Query: {log_entry['query']}")
            logger.info(f"   Strategy: {strategy}")
            logger.info(f"   KG Confidence: {kg_confidence:.4f}")
            logger.info(f"   RAG Confidence: {rag_confidence:.4f}")
            logger.info(f"   Reasoning: {reasoning}")

            # Save to CSV file for analysis
            log_file = "orchestration_log.csv"
            file_exists = os.path.isfile(log_file)

            with open(log_file, mode='a', newline='', encoding='utf-8') as file:
                fieldnames = ['timestamp', 'query', 'strategy', 'reasoning', 'kg_confidence', 'rag_confidence']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(log_entry)

            return strategy

        except Exception as e:
            logger.error(f"⚠️ Error logging orchestration decision: {e}")
            return "ERROR"


    def reset_conversation(self):
      """Reset the conversation history and follow-up context"""
      logger.info("🔄 Resetting conversation.")
      self.chat_history = []
      self.followup_context = {"round": 0} # Reset round counter
      # Reset Streamlit session state variables from the UI if they are managed externally
      # This needs to be done in the Streamlit main function, not here.
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
    st.subheader("Confirm Your Symptoms")
    st.info(f"Based on your query: '{original_query}' and initial analysis, please confirm the symptoms you are experiencing from the list below to help narrow down possibilities.")

    # Use a unique key for the form based on the original query and a timestamp
    form_key = f"symptom_confirmation_form_{abs(hash(original_query))}_{st.session_state.get('form_timestamp', datetime.now().timestamp())}"

    # Initialize a local set to store confirmed symptoms during the current form interaction
    local_confirmed_symptoms_key = f'{form_key}_confirmed_symptoms_local'
    if local_confirmed_symptoms_key not in st.session_state:
        st.session_state[local_confirmed_symptoms_key] = set()
        logger.debug(f"Initialized local symptom set for form {form_key}")
    else:
         logger.debug(f"Using existing local symptom set for form {form_key} with {len(st.session_state[local_confirmed_symptoms_key])} items.")


    # --- New Logic: Combine all symptoms into a single list ---
    all_unique_symptoms = set()
    for disease_label, symptoms_list in symptom_options.items():
        # Ensure symptoms_list is iterable and contains strings before adding to the set
        if isinstance(symptoms_list, list):
            for symptom in symptoms_list:
                 if isinstance(symptom, str):
                      all_unique_symptoms.add(symptom.strip()) # Add stripped symptom, keep original case for display

    # Sort the unique symptoms alphabetically for consistent display order
    sorted_all_symptoms = sorted(list(all_unique_symptoms))
    # --- End New Logic ---


    with st.form(form_key):
        st.markdown("Please check all symptoms that apply to you from the list below:")

        if not sorted_all_symptoms:
            st.info("No specific symptoms were found in the knowledge graph for the potential conditions. You can use the box below to add any symptoms you are experiencing.")
        else:
            # --- Modified Display Logic: Iterate through the combined list ---
            cols = st.columns(4) # Arrange checkboxes in columns (adjust number of columns as needed)
            for i, symptom in enumerate(sorted_all_symptoms):
                col = cols[i % 4]
                # Use a unique key for each checkbox
                checkbox_key = f"{form_key}_checkbox_{symptom}" # Key based on form and symptom name
                # Check if this symptom was previously selected in this form render cycle (using lower case for comparison)
                initial_state = symptom.strip().lower() in st.session_state[local_confirmed_symptoms_key]

                if col.checkbox(symptom, key=checkbox_key, value=initial_state):
                     # Add the lower case stripped symptom to the local set on click
                     st.session_state[local_confirmed_symptoms_key].add(symptom.strip().lower())
                else:
                     # Remove from the local set if unchecked (using lower case for comparison)
                     st.session_state[local_confirmed_symptoms_key].discard(symptom.strip().lower())
            # --- End Modified Display Logic ---


        # Add an "Other" symptom input field (remains the same)
        st.markdown("**Other Symptoms (if any):**")
        other_symptoms_text = st.text_input("Enter additional symptoms here (comma-separated)", key=f"{form_key}_other_symptoms_input")
        if other_symptoms_text:
             other_symptoms_list = [s.strip().lower() for s in other_symptoms_text.split(',') if s.strip()]
             st.session_state[local_confirmed_symptoms_key].update(other_symptoms_list) # Add lower case stripped symptoms


        # When the form is submitted, save the *final* state of the local set
        submit_button = st.form_submit_button("Confirm and Continue")

        if submit_button:
            logger.info(f"Symptom confirmation form submitted. Final confirmed symptoms: {st.session_state[local_confirmed_symptoms_key]}")
            # Store the confirmed symptoms from the local set into the *main* session state variable
            # Store as a list of lowercase strings for consistent processing later
            st.session_state.confirmed_symptoms_from_ui = sorted(list(st.session_state[local_confirmed_symptoms_key]))

            # Clear the UI state flags *after* submitting the form, but before the rerun.
            st.session_state.awaiting_symptom_confirmation = False
            # Keep original_query_for_followup; generate_response needs it to know which query to process.

            # Set a new form timestamp for the next potential checklist interaction
            st.session_state.form_timestamp = datetime.now().timestamp()

            # Set the original query into current_input_to_process to trigger its processing in the next rerun
            st.session_state.current_input_to_process = original_query # This is the prompt that triggered the UI

            st.rerun() # Trigger rerun to process the confirmed symptoms


# Create chatbot instance (globally or in main)
# This should only be done once per session
# Using session state for the chatbot instance
# chatbot = DocumentChatBot() # Moved to session state


# --- Main Streamlit App Function ---
def main():
    # Set page title and favicon
    try:
        st.set_page_config(
            page_title="DxAI-Agent",
            page_icon=f"data:image/png;base64,{icon}", # Use base64 encoded icon
            layout="wide"
        )
    except Exception as e:
        logger.error(f"Error setting page config: {e}") # Log potential errors
        st.set_page_config(page_title="DxAI-Agent", layout="wide") # Fallback config

    # Title and description
    try:
        logo = Image.open(image_path)
        col1, col2 = st.columns([1, 10])
        with col1:
            st.image(logo, width=100)  # Adjust width as needed
        with col2:
            st.markdown("# DxAI-Agent")
    except FileNotFoundError:
         st.markdown("# DxAI-Agent") # Fallback title if logo not found
    except Exception as e:
         logger.error(f"Error displaying logo: {e}")
         st.markdown("# DxAI-Agent")


    # Initialize session state variables if they don't exist
    # The chatbot instance itself lives in session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DocumentChatBot()
        # Trigger initialization on first load
        with st.spinner("Initializing chat assistant..."):
             success, init_message = st.session_state.chatbot.initialize_qa_chain()
             if not success:
                  st.error(f"Initialization failed: {init_message}")
                  st.session_state.init_failed = True
             else:
                  st.success(init_message) # Show success message briefly
                  st.session_state.init_failed = False # Ensure this is false on success

    # --- UI State Variables ---
    # These variables control how the UI behaves in the current rerun cycle
    if 'messages' not in st.session_state:
        st.session_state.messages = [] # List of (content, is_user) tuples for UI display

    # UI State variables for symptom confirmation and follow-up management
    if 'awaiting_symptom_confirmation' not in st.session_state:
        st.session_state.awaiting_symptom_confirmation = False
    if 'symptom_options_for_ui' not in st.session_state:
        st.session_state.symptom_options_for_ui = {} # Dict {disease_label: [symptom_names]} for UI
    if 'confirmed_symptoms_from_ui' not in st.session_state:
        st.session_state.confirmed_symptoms_from_ui = None # List of confirmed symptoms after form submit
    if 'original_query_for_followup' not in st.session_state:
        st.session_state.original_query_for_followup = "" # Stores the query that triggered a follow-up prompt/UI
    if 'init_failed' not in st.session_state:
         st.session_state.init_failed = False # Flag to track initialization status
    # Add a timestamp for the symptom confirmation form key to ensure uniqueness across reruns
    if 'form_timestamp' not in st.session_state:
         st.session_state.form_timestamp = datetime.now().timestamp()

    # Variable to hold input that needs processing in the next rerun cycle
    # Used for example clicks and symptom form submissions
    if 'current_input_to_process' not in st.session_state:
         st.session_state.current_input_to_process = None


    # User type selection dropdown in sidebar
    user_type = st.sidebar.selectbox(
        "Who is asking?",
        ["User / Family", "Physician"],
        index=0
    )

    # Add sidebar info
    st.sidebar.info("DxAI-Agent helps answer medical questions using our medical knowledge base.")
    # Display LLM follow-up attempt count from the chatbot instance
    st.sidebar.markdown(f"LLM Final Follow-up Attempts: {st.session_state.chatbot.followup_context['round']} / 1")

    # Display initialization status in sidebar
    if st.session_state.init_failed:
         st.sidebar.error("Initialization Failed. Check console logs.")
    else:
         # Display detailed initialization status from the chatbot
         init_status_success, init_status_msg = st.session_state.chatbot.initialize_qa_chain()
         st.sidebar.markdown(f"Status: {init_status_msg}")


    # Tabs
    tab1, tab2 = st.tabs(["Chat", "About"])

    with tab1:
        # Examples section (keep as is, but ensure they trigger the main logic flow)
        st.subheader("Try these examples")
        examples = [
            "What are treatments for cough and cold?",
            "I have a headache and sore throat. What could it be?", # Example that might trigger Path 1 / low conf
            "What home remedies help with flu symptoms?",
            "I have chest pain and shortness of breath. What could i do?" # Example that should trigger symptom UI
        ]

        cols = st.columns(len(examples))
        # Check if initialization failed or symptom UI is active before enabling example buttons
        examples_disabled = st.session_state.init_failed or st.session_state.awaiting_symptom_confirmation
        for i, col in enumerate(cols):
            if col.button(examples[i], key=f"example_{i}", disabled=examples_disabled):
                # On example click, clear relevant UI state and trigger bot response
                # We keep the chat history for the example flow
                st.session_state.awaiting_symptom_confirmation = False
                st.session_state.symptom_options_for_ui = {}
                st.session_state.confirmed_symptoms_from_ui = None # Clear confirmed symptoms state
                st.session_state.original_query_for_followup = "" # Clear original query for follow-up
                st.session_state.chatbot.followup_context = {"round": 0} # Reset LLM follow-up round for a new thread
                st.session_state.form_timestamp = datetime.now().timestamp() # Reset form timestamp

                prompt = examples[i]
                # Add the example text to messages as user input
                st.session_state.messages.append((prompt, True)) # True means user message

                # Set the input into session state to be processed in the next rerun cycle
                st.session_state.current_input_to_process = prompt
                st.rerun() # Trigger rerun


        # --- Chat Messages Display ---
        # Iterate through messages state for display
        for i, (msg_content, is_user) in enumerate(st.session_state.messages):
            if is_user:
                with st.chat_message("user"):
                    st.write(msg_content)
            else: # Assistant message
                with st.chat_message("assistant"):
                    st.write(msg_content)

                    # Add feedback buttons only for assistant messages that are *not* prompts or UI triggers
                    # Check if the message looks like a prompt (heuristic) or UI trigger
                    is_prompt_or_ui = (
                        st.session_state.awaiting_symptom_confirmation or # If we are awaiting symptom UI, the *last* assistant message was the prompt
                        # Check if this message IS the follow-up prompt message based on the original query flag and round count
                        # This check uses the *previous* user message (st.session_state.messages[i-1]) and the *current* bot message (msg_content)
                        (i > 0 and st.session_state.messages[i-1][0] == st.session_state.original_query_for_followup and st.session_state.chatbot.followup_context["round"] == 1) or
                        ("could you please" in msg_content.lower()) or
                        ("please tell me" in msg_content.lower()) or
                        ("please confirm" in msg_content.lower()) or
                        ("enter additional symptoms" in msg_content.lower()) or
                        ("additional details" in msg_content.lower()) or
                        ("to help me answer better" in msg_content.lower()) or
                        ("this will help me provide the most accurate information" in msg_content.lower()) # Add phrase from prompt
                    )
                    # Add feedback buttons if it doesn't look like a prompt/UI trigger
                    if not is_prompt_or_ui:
                        # Add feedback buttons using callback keys
                        col = st.container()
                        with col:
                            # Add hash of content and index for unique keys per message
                            feedback_key_up = f"thumbs_up_{i}_{abs(hash(msg_content))}"
                            feedback_key_down = f"thumbs_down_{i}_{abs(hash(msg_content))}"

                            b1, b2 = st.columns([0.05, 0.95]) # Adjust column width
                            with b1:
                                 if st.button("👍", key=feedback_key_up):
                                     # Find the preceding user message for context
                                     user_msg_content = ""
                                     # Look back for the nearest user message
                                     for j in range(i - 1, -1, -1):
                                         if st.session_state.messages[j][1] is True:
                                             user_msg_content = st.session_state.messages[j][0]
                                             break

                                     feedback_result = vote_message(
                                         user_msg_content, msg_content, # Pass user/bot pair
                                         "thumbs_up", user_type
                                     )
                                     st.toast(feedback_result)
                            with b2:
                                 if st.button("👎", key=feedback_key_down):
                                    user_msg_content = ""
                                    # Look back for the nearest user message
                                    for j in range(i - 1, -1, -1):
                                        if st.session_state.messages[j][1] is True:
                                            user_msg_content = st.session_state.messages[j][0]
                                            break
                                    feedback_result = vote_message(
                                        user_msg_content, msg_content, # Pass user/bot pair
                                        "thumbs_down", user_type
                                    )
                                    st.toast(feedback_result)


        # --- Input Area ---
        st.write("  \n" * 5) # Add space pusher at the end of the tab

        # Determine the input source and relevant state for this rerun
        current_user_text_input = None
        confirmed_symps_to_pass = None
        original_query_context = None # Represents the query being followed up on

        # 1. Check if symptom confirmation form was just submitted
        # This happens if confirmed_symptoms_from_ui is set and we are NOT currently awaiting confirmation
        if st.session_state.confirmed_symptoms_from_ui is not None and not st.session_state.awaiting_symptom_confirmation:
             logger.info("Detected symptom confirmation form submission via state.")
             # Get the confirmed symptoms and the original query that triggered the UI
             confirmed_symps_to_pass = st.session_state.confirmed_symptoms_from_ui
             original_query_context = st.session_state.original_query_for_followup # Use the stored original query as context

             # The text input for the bot's process is the original query text
             current_user_text_input = original_query_context

             # Clear the UI state variables after capturing them
             st.session_state.confirmed_symptoms_from_ui = None
             # original_query_for_followup and awaiting_symptom_confirmation are cleared by generate_response action flags

        # 2. Check if a new message was submitted in the chat input box
        # This happens if st.chat_input returns a non-empty string AND we are NOT awaiting symptom confirmation UI
        elif (prompt := st.chat_input("Ask your medical question...", disabled=st.session_state.init_failed or st.session_state.awaiting_symptom_confirmation, key="main_chat_input")):
             logger.info(f"Detected chat input submission: '{prompt}'")
             current_user_text_input = prompt
             # For a new chat input, confirmed_symps_to_pass remains None

             # Check if this chat input is a response to a previous LLM follow-up prompt
             # This is true if original_query_for_followup is NOT empty from the previous turn
             if st.session_state.original_query_for_followup != "":
                 logger.info("Detected chat input is a response to a prior LLM follow-up prompt.")
                 original_query_context = st.session_state.original_query_for_followup # Set context to the query that received the prompt
                 # The user's response text is in current_user_text_input (the 'prompt' variable)
             else:
                 # This is a brand new thread start
                 logger.info("Detected chat input is the start of a new thread.")
                 # Clear any old follow-up state explicitly for a new thread start
                 st.session_state.chatbot.followup_context = {"round": 0}
                 st.session_state.original_query_for_followup = "" # Ensure clear
                 st.session_state.symptom_options_for_ui = {} # Clear symptom UI options
                 st.session_state.confirmed_symptoms_from_ui = None # Clear confirmed symptoms

             # Add the user message to the UI messages state immediately
             st.session_state.messages.append((current_user_text_input, True))


        # --- Call generate_response if there is input to process ---
        # This call happens only if current_user_text_input was set by one of the above blocks
        if current_user_text_input is not None:
            with st.spinner("Thinking..."):
                 response_text, sources, action_flag, ui_data = st.session_state.chatbot.generate_response(
                      current_user_text_input, user_type,
                      confirmed_symptoms=confirmed_symps_to_pass,
                      original_query_if_followup=original_query_context # Pass the original query context
                 )

            # --- Process the action flag returned by generate_response ---
            logger.info(f"generate_response returned action_flag: {action_flag}")
            if action_flag == "symptom_ui_prompt":
                 # Update UI state to show the symptom checklist next rerun
                 st.session_state.awaiting_symptom_confirmation = True
                 st.session_state.symptom_options_for_ui = ui_data["symptom_options"]
                 st.session_state.original_query_for_followup = ui_data["original_query"] # Store query that triggered UI
                 st.session_state.form_timestamp = datetime.now().timestamp() # Set new timestamp for the form key

                 # Add the prompt message for the UI to messages (it's the response_text)
                 if not st.session_state.messages or st.session_state.messages[-1] != (response_text, False):
                      st.session_state.messages.append((response_text, False))

            elif action_flag == "llm_followup_prompt":
                 # Add the LLM prompt message to messages (it's the response_text)
                 if not st.session_state.messages or st.session_state.messages[-1] != (response_text, False):
                    st.session_state.messages.append((response_text, False))

                 # Store the original query so generate_response knows the *next* user input is a response to *this* query thread's prompt
                 # The 'original_query_for_followup' for the *next* turn should be the 'core_query_for_processing' of *this* turn
                 # which was passed as 'original_query_context' to generate_response.
                 st.session_state.original_query_for_followup = current_user_text_input # Store the query that 

            elif action_flag == "final_answer":
                 # Add the final answer message to messages (it's the response_text)
                  if not st.session_state.messages or st.session_state.messages[-1] != (response_text, False):
                       st.session_state.messages.append((response_text, False))

                   # Clear original_query_for_followup as the thread is concluded or follow-up limit reached
                  st.session_state.original_query_for_followup = ""
                  # Clear symptom UI specific state too just in case
                  st.session_state.awaiting_symptom_confirmation = False
                  st.session_state.symptom_options_for_ui = {}


            elif action_flag == "none":
                 # No action needed, do not add to messages
                 pass # generate_response already logged why it skipped

            # Force a rerun to update the UI based on state/messages
            # This rerun is necessary to display the *new* message(s) and potentially the symptom form
            st.rerun()


        # --- Reset Conversation Button ---
        # This button is outside the input area conditional rendering
        st.divider() # Add a visual separator
        if st.button("Reset Conversation", key="reset_conversation_button"):
            st.session_state.chatbot.reset_conversation() # Resets internal history and followup_context
            st.session_state.messages = [] # Clear UI messages
            # Also reset UI state variables
            st.session_state.awaiting_symptom_confirmation = False
            st.session_state.symptom_options_for_ui = {}
            st.session_state.confirmed_symptoms_from_ui = None
            st.session_state.original_query_for_followup = ""
            st.session_state.form_timestamp = datetime.now().timestamp() # Reset form timestamp
            # Clear any pending input that might have been in the chat_input buffer before reset clicked
            # This might require a custom chat input component or clearing its value via state if possible.
            # For now, rely on the logic that current_user_text_input will be None on the next rerun if the chat input is empty.
            logger.info("Conversation reset triggered by user.")
            st.rerun()

        # Physician feedback section (keep as is, uses chatbot's internal chat_history)
        st.divider()
        st.subheader("🩺 Detailed Feedback")
        with st.form("feedback_form"):
            feedback_text = st.text_area(
                "Enter corrections, improvements, or comments here...",
                height=100
            )
            submit_feedback_btn = st.form_submit_button("Submit Feedback")

            if submit_feedback_btn and feedback_text:
                # Use the chatbot's internal chat_history for feedback context
                history_for_feedback = st.session_state.chatbot.chat_history
                feedback_result = submit_feedback(feedback_text, history_for_feedback, user_type)
                st.success(feedback_result)

        # Reset conversation button
        if st.button("Reset Conversation"):
            st.session_state.chatbot.reset_conversation() # Resets internal history and followup_context
            st.session_state.messages = [] # Clear UI messages
            # Also reset UI state variables
            st.session_state.awaiting_symptom_confirmation = False
            st.session_state.symptom_options_for_ui = {}
            st.session_state.confirmed_symptoms_from_ui = None
            st.session_state.original_query_for_followup = ""
            st.session_state.current_input_to_process = None # Clear any pending input
            st.session_state.form_timestamp = datetime.now().timestamp() # Reset form timestamp
            st.rerun()

    with tab2:
        st.markdown("""
        ## Medical Chat Assistant

        **Powered by:**
        - **Symptom Extraction:** Identifies symptoms mentioned in your query using an LLM and keyword matching.
        - **Knowledge Graph (KG):** Queries a medical database (Neo4j) for diseases potentially related to symptoms, as well as treatments and home remedies for identified conditions. Requires Neo4j connection.
        - **Retrieval-Augmented Generation (RAG):** Searches internal medical documents (PDFs) using embeddings and an LLM to retrieve and summarize relevant information. Requires PDF files, embedding model, and LLM.
        - **LLM Synthesis:** Uses a large language model (Gemini Flash 1.5) to synthesize information from KG and RAG into a coherent, user-friendly answer, focusing on addressing the original query and filling any identified information gaps. Requires LLM.
        - **Reflection/Completeness Check:** Evaluates the final generated answer using an LLM to see if any critical information is still missing that warrants a follow-up question to the user. Requires LLM.

        **Workflow:**
        The system orchestrates these components following two main paths:

        **Path 1: Diagnosis Focus**
        1.  Extracts symptoms from your query.
        2.  Queries the Knowledge Graph for potential diseases indicated by these symptoms.
        3.  If KG confidence for potential diseases is below a threshold, it presents you with a list of symptoms associated with the top candidate diseases (from the KG) to confirm via checkboxes (UI Interaction). This is the first type of follow-up. (Requires LLM, KG).
        4.  After you confirm symptoms, KG confidence is recalculated based on all symptoms (initial + confirmed).
        5.  The highest confidence KG disease is selected. An LLM formats a concise diagnosis-focused statement based on this KG result, including a disclaimer. (Requires LLM).
        6.  This KG Diagnosis component is then passed to Path 2 for potential combination with RAG and further synthesis.

        **Path 2: General & Comprehensive Answer**
        1.  This path is followed if Path 1 symptom confirmation wasn't triggered, or after the KG Diagnosis component is ready from Path 1.
        2.  It performs a RAG search using your original query and conversation history. (Requires RAG chain - LLM, Embeddings, Vector DB, PDFs).
        3.  It combines the (optional) KG Diagnosis component from Path 1 and the RAG results into an initial draft.
        4.  An LLM synthesizes this draft into a final user-facing answer, focusing on addressing the original query and any identified gaps identified in the draft. (Requires LLM).
        5.  A final check evaluates the completeness of the LLM-synthesized answer against the original query using an LLM. (Requires LLM).
        6.  If critical information is still missing from the *final answer* according to the LLM check, AND this is the first time asking for clarification in *this specific conversation thread* (controlled by a counter), the system asks *one* targeted follow-up question generated by an LLM. This is the second type of follow-up (LLM prompt). (Requires LLM).
        7.  Otherwise, it provides the final answer, including references from both KG and RAG sources, and a standard medical disclaimer.

        **Note:** An LLM follow-up question in Path 2 is limited to a single question per conversation thread to avoid excessive questioning. The symptom confirmation in Path 1 is a separate UI interaction that does not count towards this LLM follow-up limit. Features requiring LLM, Embeddings, Vector DB, or KG will be unavailable if initialization fails.

        **Modes:**
        - **User / Family:** Provides a comprehensive answer including potential conditions, treatments, and remedies, with strong disclaimers.
        - **Physician:** *(Note: Current implementation provides similar output regardless of mode, but this is a placeholder for future differentiation)*. The intention is to focus more on diagnostic possibilities and evidence for physicians.

        **Disclaimer:** This system is informational only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.
        """)


# Add a main guard for execution
if __name__ == "__main__":
    # When running with `streamlit run app.py`, streamlit calls main()
    main()
