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


    def local_generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using Gemini Flash 1.5 (direct call, not part of QA chain), with fallback."""
        if self.llm is None:
            # Fallback if LLM wasn't initialized successfully
            logger.warning("Real LLM not initialized, using simple string fallback for local_generate.")
            # Provide a very basic, safe fallback
            return "I'm unable to generate a specific response right now due to a technical issue. Please try again later."

        try:
            response = self.llm.invoke(prompt, max_tokens=max_tokens, temperature=0.3, top_p=0.95, top_k=40) 
            # Access the content attribute for the generated text
            logger.debug(f"LLM local_generate successful. Response length: {len(response.content)}")
            return response.content
        except Exception as e:
            logger.error(f"Error generating with Gemini (local_generate): {e}")
            # Fallback to a simple string on error during generation
            return f"Error generating response: {str(e)}. Please try again."


    def generate_llm_answer(self, query: str, kg_content: Optional[str] = None, rag_content: Optional[str] = None, initial_combined_answer: Optional[str] = None, missing_elements: Optional[List[str]] = None) -> str:
        """
        Generates an LLM answer, synthesizing information from KG and RAG,
        and potentially focusing on missing elements identified.
        This is the core synthesis step in Path 2.
        """
        logger.info("➡️ LLM Synthesis Step")

        if self.llm is None:
            logger.warning("LLM not initialized. Skipping synthesis.")
            # Fallback response if LLM isn't available
            return "I'm currently unable to synthesize a complete answer. Please consult a healthcare professional."


        prompt_parts = [
            "You are a helpful medical AI assistant providing a comprehensive answer based on the provided information.",
            f"USER QUESTION: {query}"
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
                 # If raw sources were also unhelpful, explicitly state limited info
                 prompt_parts.append("No specific relevant information was found from knowledge sources.")
        else:
             # If no context or draft, rely on LLM's general knowledge but with caution
             prompt_parts.append("No specific information was found from knowledge sources. Please provide a general, safe response based on your medical knowledge.")


        prompt_parts.append("Please synthesize the available information (if any) to provide a helpful, accurate, and comprehensive answer to the USER QUESTION.")

        if missing_elements:
            # Refine missing elements list to be more descriptive for the LLM prompt
            missing_desc = []
            if "duration" in missing_elements: missing_desc.append("how long the symptoms have lasted")
            if "severity" in missing_elements: missing_desc.append("how severe the symptoms are")
            if "location" in missing_elements: missing_desc.append("the location of symptoms (e.g., pain location)")
            if "frequency" in missing_elements: missing_desc.append("how often symptoms occur")
            if "onset" in missing_elements: missing_desc.append("when the symptoms started")
            # Add other specific element descriptions

            if missing_desc:
                focus_text = "Ensure your answer addresses the user's question and attempts to incorporate details related to: " + ", ".join(missing_desc)
                prompt_parts.append(focus_text)

        prompt_parts.append("Include appropriate medical disclaimers about consulting healthcare professionals for diagnosis and treatment.")
        prompt_parts.append("Format your answer clearly and concisely using markdown.")


        prompt = "\n\n".join(prompt_parts)

        try:
            # Use the local_generate method for this specific LLM call
            # Increase max_tokens for synthesis, but avoid excessively large responses
            response = self.local_generate(prompt, max_tokens=1200) # Adjusted max_tokens
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
                 return (False, []) # Cannot check completeness without LLM


            # Convert conversation history to a string for context
            # Include a few recent exchanges for better understanding
            context = ""
            history_limit = 6 # Include last 3 exchanges (user+bot)
            recent_history = conversation_history[-history_limit:]
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
                    elif bot_msg is not None:
                        # Log if it's not a string but not None (unexpected type)
                        logger.warning(f"Unexpected type in chat_history bot message at index {i}. Type: {type(bot_msg)}. Value: {bot_msg}. Appending placeholder.")
                        context += f"Assistant: [Non-string response of type {type(bot_msg)}]\n"
                    # else: bot_msg is None, do not add Assistant line
                else:
                    # Log if an entry in history is not a tuple of length 2 or not a tuple at all
                    logger.warning(f"Unexpected format in chat_history entry at index {i}. Entry: {entry}. Skipping entry or adding placeholder.")
                    context += f"[Invalid history entry at index {i}]\n"


            # The `generate_response` function will check `self.followup_context["round"]`
            # This function just needs to determine *if* a follow-up is logically required based on completeness.

            MISSING_INFO_PROMPT = '''
            You are a medical assistant analyzing a patient query and a generated answer.
            Your task is to determine if any ABSOLUTELY CRITICAL medical information is still missing *from the GENERATED ANSWER ITSELF* to adequately address the USER QUESTION, considering the conversation context.
            
            Conversation history (for context, possibly truncated):
            ---
            {context}
            ---
            
            USER QUESTION: "{user_query}"
            GENERATED ANSWER (the answer being evaluated): "{generated_answer}"
            
            CRITICALLY EVALUATE the GENERATED ANSWER: Does it provide a safe and reasonably comprehensive response to the USER QUESTION given the context?
            
            Rules for determining if critical information is missing:
            1. Does the answer directly respond to the core medical question asked by the user?
            2. If the query is about personal symptoms, does the answer include appropriate medical disclaimers? (The answer *should* already have this, but assess if its absence makes the answer critically incomplete or unsafe).
            3. Are there obvious critical gaps that make the answer potentially unsafe or unhelpful? (e.g., failing to mention seeking immediate care for red-flag symptoms like severe chest pain if the query implied them, and the answer didn't cover this safety warning).
            4. ONLY identify missing information if it's ABSOLUTELY NECESSARY to ask the user ONE follow-up question to make the answer minimally helpful or safe. Avoid asking for more detail that is not crucial.
            5. If the GENERATED ANSWER indicates it cannot answer fully (e.g., due to lack of information), is that explanation sufficient, or is a clarifying question still needed from the user?
            6. If the GENERATED ANSWER is very short or generic and the USER QUESTION was specific (e.g., listed symptoms), does it warrant asking for clarification from the user to provide a better response?
            
            If critical information is missing *from the generated answer itself* that requires asking the user for input, formulate ONE clear, specific follow-up question to obtain the MOST critical missing piece.
            
            Return your answer in this exact JSON format:
            {{
                "needs_followup": true,
                "reasoning": "brief explanation of why more information is needed from the answer or why the answer is sufficient",
                "missing_info_questions": [
                    {{"question": "specific follow-up question 1"}}
                ]
            }}
            
            Only include the "missing_info_questions" array if "needs_followup" is true, and limit it to exactly 1 question. If needs_followup is true but you cannot formulate a specific question based on the answer's gaps, still return needs_followup: true but with an empty "missing_info_questions" array (though try hard to formulate one).
            '''.format(
                context=context,
                user_query=user_query,
                generated_answer=generated_answer
            )

            try:
                # Use local_generate for this LLM call
                response = self.local_generate(MISSING_INFO_PROMPT, max_tokens=500).strip()
                # logger.debug(f"\nRaw Missing Info Evaluation (Final Check):\n{response}")

                # Attempt to parse JSON
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        data = json.loads(json_str)
                        needs_followup_llm = data.get("needs_followup", False) # LLM's opinion
                        missing_info_questions = [item["question"] for item in data.get("missing_info_questions", []) if isinstance(item, dict) and "question" in item] # Safely get questions
                        reasoning = data.get("reasoning", "Answer is missing critical information.")

                        if needs_followup_llm and missing_info_questions:
                             logger.info(f"❓ Critical Information Missing from Final Answer (LLM opinion): {missing_info_questions}. Reasoning: {reasoning}")
                             return (True, missing_info_questions) # Return True and questions

                        else:
                             logger.info("✅ Final Answer appears sufficient (LLM opinion) or no questions provided.")
                             return (False, []) # Return False and no questions

                    except json.JSONDecodeError:
                        logger.warning("Could not parse final missing info JSON from LLM response.")
                        # Fallback: Assume no critical info is missing if JSON parsing fails
                        return (False, [])
                    except Exception as e:
                         logger.error(f"Error processing LLM response structure in identify_missing_info: {e}", exc_info=True)
                         return (False, []) # Fallback on structure error

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
        logger.info(f"--- Generating Response for Input: '{user_input}' ---")
        logger.info(f"   Confirmed symptoms from UI: {confirmed_symptoms}")
        logger.info(f"   Original query if follow-up: '{original_query_if_followup}'")
        logger.info(f"   Current followup_context: {self.followup_context}")
        logger.info(f"   Current chat_history length: {len(self.chat_history)}")


        # Determine the core query being processed in this turn
        # If confirmed_symptoms is provided, the "real" query is the one that triggered the UI (`original_query_if_followup`).
        # If original_query_if_followup is provided (and confirmed_symptoms is None), the "real" query is the one that triggered the LLM prompt.
        # Otherwise, the current user_input is the start of a new thread.
        core_query_for_processing = original_query_if_followup if original_query_if_followup is not None else user_input
        logger.info(f"   Core query for processing logic: '{core_query_for_processing}'")

        if not core_query_for_processing.strip() and confirmed_symptoms is None:
             logger.info("Empty core query and no confirmed symptoms. Skipping.")
             return "", [], "none", None # No action needed for empty input


        # --- Initialization Check ---
        # Check if *critical* components are missing. LLM is needed for basic function. RAG needs LLM + VDB + Chain. KG needs KG driver.
        # Re-attempt initialization if LLM or RAG chain is None
        if self.llm is None or self.qa_chain is None or not self.kg_connection_ok:
            logger.info("Chatbot is not fully initialized. Attempting re-initialization...")
            success, message = self.initialize_qa_chain() # Re-attempt initialization
            if not success:
                error_message = f"Error processing request: Assistant failed to initialize fully ({message}). Some features may be unavailable. Please check your configuration and try again later."
                self.log_orchestration_decision(core_query_for_processing, f"SELECTED_STRATEGY: INIT_ERROR\nREASONING: Re-initialization failed: {message}", 0.0, 0.0)
                if self.llm is None:
                     # If LLM is still none after re-attempt, we cannot do anything useful
                     logger.critical("LLM is still not initialized after re-attempt. Cannot generate any response.")
                     return error_message, [], "final_answer", None
                else:
                    # If LLM is available but RAG/KG failed, proceed with limited features
                    logger.warning("Initialization partially successful (LLM available). Proceeding with limited features (No RAG or KG).")
                    # Continue processing below, but RAG/KG calls will gracefully handle missing components


        # --- Step 0.1: Handle User Response to Prior LLM Follow-up ---
        # Check if this input is a response to the single allowed LLM follow-up prompt.
        # This is indicated by `original_query_if_followup` being present AND `self.followup_context["round"] == 1`.
        is_response_to_llm_followup = original_query_if_followup is not None and self.followup_context["round"] == 1
        if is_response_to_llm_followup:
             logger.info(f"Detected response to LLM follow-up (round {self.followup_context['round']}). Processing '{user_input}' in context of '{original_query_if_followup}'.")
             # The `core_query_for_processing` is already set to `original_query_if_followup` correctly.
             # The user_input (the response) will be implicitly included in the RAG history due to Langchain memory.


        # --- Step 2: Extract Symptoms ---
        # Combine symptoms extracted from the *core query* with any *confirmed symptoms* from UI.
        # If processing a response to an LLM follow-up, extract symptoms from the *response* text AND combine with symptoms from the *original query*.
        extracted_symptoms_from_core_query, extracted_conf_core = self.extract_symptoms(core_query_for_processing)
        all_symptoms: List[str] = []
        symptom_confidence = 0.0

        if is_response_to_llm_followup:
            logger.info(f"Extracting symptoms from LLM follow-up response: '{user_input}'")
            extracted_symptoms_from_response, response_conf = self.extract_symptoms(user_input)
            all_symptoms = list(set(extracted_symptoms_from_response + extracted_symptoms_from_core_query))
            symptom_confidence = max(response_conf, extracted_conf_core) # Simple confidence merge
            logger.info(f"Combined symptoms from response and original query: {all_symptoms}")

        elif confirmed_symptoms is not None:
            logger.info(f"Using symptoms from UI confirmation: {confirmed_symptoms}")
            # Extract symptoms from the current input text as well, just in case user added more
            extracted_symptoms_from_input, extracted_conf_input = self.extract_symptoms(user_input)
            all_symptoms = list(set(extracted_symptoms_from_input + confirmed_symptoms)) # Combine and deduplicate
            # Boost confidence significantly if user confirmed via UI
            symptom_confidence = max(extracted_conf_input, 0.9) # Assume high confidence if user confirmed


        else: # Standard initial input
            all_symptoms = extracted_symptoms_from_core_query
            symptom_confidence = extracted_conf_core
            logger.info(f"Extracted symptoms from input: {all_symptoms}")

        # --- Step 3: KG Processing ---
        # Run KG agent with all available symptoms
        logger.info("📚 Processing with Knowledge Graph...")
        t_start_kg = datetime.now()
        # Pass the core query as context to KG agent if needed internally (optional)
        kg_data = self.knowledge_graph_agent(core_query_for_processing, all_symptoms)
        top_disease_confidence = kg_data.get("top_disease_confidence", 0.0)
        kg_diagnosis_data_for_llm = kg_data.get("kg_content_diagnosis_data_for_llm") # Data for LLM Path 1 formatting
        kg_content_other = kg_data.get("kg_content_other", "") # Treatments/Remedies

        logger.info(f"📊 KG Top Disease Confidence: {top_disease_confidence:.4f} (took {(datetime.now() - t_start_kg).total_seconds():.2f}s)")

        # --- Step 4: Path 1 - Diagnosis Focus & Symptom Follow-up UI (Decision Point 1) ---
        is_disease_query = self.is_disease_identification_query(core_query_for_processing)
        kg_found_diseases = len(kg_data.get("identified_diseases_data", [])) > 0

        # Condition to trigger Symptom Follow-up UI:
        # 1. It's a disease identification query.
        # 2. KG found *at least one* disease.
        # 3. Top disease confidence is below the threshold for direct Path 1 conclusion.
        # 4. We are *not* currently processing a response *from* the symptom confirmation UI (confirmed_symptoms is None).
        # 5. KG returned potential disease-symptom associations for the UI step (`identified_diseases_data` has 'AllDiseaseSymptomsKG' for top diseases).
        # 6. We haven't asked the single LLM follow-up yet (round == 0).
        # 7. LLM is initialized (needed to process the UI response effectively later).
        # 8. KG connection is OK (otherwise we can't get relevant symptom options from KG).
        if is_disease_query and \
           kg_found_diseases and \
           top_disease_confidence < THRESHOLDS["disease_symptom_followup_threshold"] and \
           confirmed_symptoms is None and \
           any(d.get("AllDiseaseSymptomsKG") for d in kg_data.get("identified_diseases_data", [])[:5]) and \
           self.followup_context["round"] == 0 and \
           self.llm is not None and \
           self.kg_connection_ok: # Symptom UI requires KG is available

            logger.info(f"❓ Disease query ('{core_query_for_processing}') with low KG confidence ({top_disease_confidence:.4f}). Triggering Symptom Follow-up UI.")
            # Prepare data for the UI checklist
            symptom_options_for_ui: Dict[str, List[str]] = {}
            # Get symptoms associated with top N diseases (e.g., top 3-5) for the UI
            # Ensure we only include diseases that actually have associated symptoms in KG
            relevant_diseases_for_ui = [d for d in kg_data["identified_diseases_data"][:5] if d.get("AllDiseaseSymptomsKG")]
            for disease_data in relevant_diseases_for_ui:
                 disease_label = f"{disease_data['Disease']} (Confidence: {disease_data['Confidence']:.2f})"
                 # Ensure symptom names are strings and unique within the disease list
                 symptoms_list = sorted(list(set(str(s) for s in disease_data.get("AllDiseaseSymptomsKG", []) if isinstance(s, str))))
                 if symptoms_list:
                    symptom_options_for_ui[disease_label] = symptoms_list

            follow_up_prompt_text = f"""
            Thank you for sharing your symptoms. Based on what you've told me, I found some potential conditions.
            To help narrow it down and provide a more relevant response, please confirm which of the associated symptoms from my knowledge base you are also experiencing.
            """
            # The Streamlit UI will display the checkboxes below this message based on symptom_options_for_ui

            # Log decision
            self.log_orchestration_decision(
                core_query_for_processing,
                f"SELECTED_STRATEGY: SYMPTOM_UI_FOLLOWUP\nREASONING: Disease query with low KG confidence ({top_disease_confidence:.2f}). Presenting symptom checklist for confirmation.",
                top_disease_confidence, # Use specific KG conf here
                0.0 # RAG skipped in this path
            )

            # Return data indicating UI action is needed
            # Pass the core_query_for_processing back so Streamlit can resubmit it with confirmed symptoms
            logger.info("Returning Symptom UI prompt.")
            return follow_up_prompt_text.strip(), [], "symptom_ui_prompt", {"symptom_options": symptom_options_for_ui, "original_query": core_query_for_processing}


        # --- Step 5: Path 1 - Direct KG Diagnosis Component (if high confidence or after symptom confirmation) ---
        # This step is reached IF confirmed_symptoms is NOT None (user responded to UI)
        # OR IF confirmed_symptoms IS None BUT top_disease_confidence was already >= threshold on first check
        # AND it is a disease identification query.
        path1_kg_diagnosis_component = None
        is_high_conf_kg_diagnosis = is_disease_query and kg_found_diseases and top_disease_confidence >= THRESHOLDS["disease_symptom_followup_threshold"]
        is_post_symptom_confirmation = confirmed_symptoms is not None # Flag indicates user completed the UI step

        if (is_high_conf_kg_diagnosis or is_post_symptom_confirmation) and kg_diagnosis_data_for_llm:
            if self.llm is not None: # LLM needed for formatting
                 logger.info(f"✅ Path 1: High confidence KG diagnosis ({top_disease_confidence:.4f}) OR received symptom confirmation. Formatting KG diagnosis answer with LLM.")

                 # Use LLM to format the KG diagnosis into a user-friendly statement
                 # Pass all available symptoms (extracted + confirmed) to the formatter for phrasing
                 path1_kg_diagnosis_component = self.format_kg_diagnosis_with_llm(
                      kg_diagnosis_data_for_llm["disease_name"],
                      all_symptoms, # Use the combined list of symptoms for phrasing
                      kg_diagnosis_data_for_llm["confidence"]
                  )

                 # Log decision for the KG Diagnosis component (internal log)
                 logger.info(f"   --- KG Diagnosis Component Generated ---")

            else:
                 # Fallback to manual formatting if LLM is not available but KG found data
                 logger.warning("⚠️ LLM not available for formatting KG diagnosis. Using manual format.")
                 disease_name = kg_diagnosis_data_for_llm["disease_name"]
                 symptoms_str = ", ".join(all_symptoms) if all_symptoms else "your symptoms"
                 path1_kg_diagnosis_component = f"Based on {symptoms_str}, **{disease_name}** is a potential condition. This is not a definitive diagnosis and requires professional medical evaluation."

        elif (is_high_conf_kg_diagnosis or is_post_symptom_confirmation):
             # KG found no diseases even after potential confirmation, provide a statement
             logger.info("⚠️ KG found no diseases even after symptom input. Proceeding to Path 2 without specific KG diagnosis component.")
             path1_kg_diagnosis_component = "Based on the symptoms provided, I couldn't find a specific medical condition matching them in my knowledge base."


        # --- NEW: Decision Point 2 - Conclude with KG-only answer for high-confidence diagnosis query ---
        # Only do this if LLM is available to ensure the formatted answer is good
        if (is_high_conf_kg_diagnosis or is_post_symptom_confirmation) and path1_kg_diagnosis_component is not None and self.llm is not None:
             # Check if the original query *primarily* asked for a diagnosis and not other things explicitly.
             # This check is heuristic, refine as needed.
             query_lower = core_query_for_processing.lower()
             asks_for_treatment = any(kw in query_lower for kw in ["treat", "medication", "cure", "what to do", "how to manage", "resolve"])
             asks_for_remedy = any(kw in query_lower for kw in ["remedy", "home", "natural", "relief"])


             # Conclude with KG-only if it's a diagnosis query AND (high conf OR post-confirmation)
             # AND it doesn't seem to explicitly ask for treatments/remedies in the *original* phrasing.
             # Note: The "what could i do?" query in your log *does* imply asking for action/treatment,
             # so this logic *shouldn't* trigger for that specific example, and it should proceed to RAG/Path 2.
             # If you want "what could i do?" to ALSO trigger KG-only when diagnosis is high confidence,
             # you might adjust the `asks_for_treatment` condition or add a specific check.
             # Let's keep the check strict for now based on explicit keywords.
             # Adding a condition: If the initial query has *no* symptom keywords, it's likely a general info query,
             # so we shouldn't conclude with KG-only diagnosis unless symptoms were added via UI.
             # Check if the core query contained symptom keywords initially
             extracted_symptoms_initial, _ = self.extract_symptoms(user_input if original_query_if_followup is None else original_query_if_followup)
             core_query_had_symptoms_initially = len(extracted_symptoms_initial) > 0


             if (is_high_conf_kg_diagnosis or is_post_symptom_confirmation) and not (asks_for_treatment or asks_for_remedy or not core_query_had_symptoms_initially): # Don't KG-only if original query had no symptoms unless it's post-confirmation
                 logger.info(f"✅ Decision Point 2: Concluding with KG-only answer for high-confidence diagnosis ({top_disease_confidence:.4f}). Query did not explicitly ask for treatment/remedy AND it had symptoms initially.")

                 # The formatted diagnosis component includes the disclaimer
                 final_response_text = path1_kg_diagnosis_component

                 # Collect KG sources for this diagnosis component
                 all_sources: List[str] = []
                 if self.kg_connection_ok:
                      all_sources.append(f"[Source: Medical Knowledge Graph (Diagnosis Data)]")

                 # Log the orchestration decision
                 self.log_orchestration_decision(
                     core_query_for_processing,
                     f"SELECTED_STRATEGY: KG_DIAGNOSIS_ONLY\nREASONING: Disease query with high KG confidence ({top_disease_confidence:.2f}) or post-symptom confirmation, and query did not explicitly ask for treatment/remedy.",
                     top_disease_confidence,
                     0.0 # RAG was skipped
                 )

                 # Add to chat history *before* returning
                 # Use the user_input (what the user actually typed this turn) and the final formatted response
                 self.chat_history.append((user_input, final_response_text.strip()))


                 logger.info("Returning KG-only Final Answer.")
                 # Return the formatted KG diagnosis answer as the final answer
                 return final_response_text.strip(), all_sources, "final_answer", None

             # Else: It's a high-confidence diagnosis query, BUT it also asked for treatment/remedy,
             # OR it was the result of symptom confirmation which should always proceed to Path 2
             # to combine with RAG, OR the original query had no symptoms. Proceed to Path 2.
             logger.info("✅ Proceeding to Path 2 with KG Diagnosis Component (if generated), as query asked for treatments/remedies or it's post-symptom confirmation/general query.")


        # --- Step 6: Path 2 - RAG Processing ---
        # This step is reached if Path 1 Symptom UI was not triggered,
        # or if Path 1 Diagnosis resulted in a component but didn't conclude the answer.
        logger.info("📚 Processing with RAG...")
        t_start_rag = datetime.now()

        rag_content = ""
        rag_source_docs = []
        rag_confidence = 0.0

        # Only attempt RAG if both the LLM and QA Chain (which includes VectorDB/Embeddings) are initialized
        if self.llm is not None and self.qa_chain is not None:
            try:
                 # The qa_chain handles chat history internally via its memory
                 # Pass the core_query_for_processing to the RAG chain
                 logger.info(f"Invoking RAG chain with question: {core_query_for_processing}")
                 rag_response = self.qa_chain.invoke({"question": core_query_for_processing})

                 rag_content = rag_response.get("answer", "").strip()
                 # Clean up potential prefixes or unwanted phrases from RAG LLM step
                 if "Helpful Answer:" in rag_content:
                      rag_content = rag_content.split("Helpful Answer:", 1)[-1].strip()

                 # Extract RAG sources
                 rag_source_docs = rag_response.get("source_documents", [])
                 if rag_source_docs:
                      # Calculate RAG confidence (simplified - could use retrieval scores if available)
                      # Base score 0.3, adds up to 0.7 based on up to 5 docs
                      rag_confidence = 0.3 + min(len(rag_source_docs), 5) / 5.0 * 0.4


                 logger.info(f"📊 RAG Confidence: {rag_confidence:.4f} (took {(datetime.now() - t_start_rag).total_seconds():.2f}s)")

            except Exception as e:
                 logger.error(f"⚠️ Error during RAG processing: {e}", exc_info=True) # Log traceback
                 rag_content = "An error occurred while retrieving information from documents."
                 rag_source_docs = []
                 rag_confidence = 0.0
                 # Note: The RAG chain might fail if the LLM fails during the synthesis step *within* the chain.
                 # This is one reason why the LLM is critical.
        else:
             logger.warning("Warning: RAG chain (or necessary components) not initialized. Skipping RAG processing.")
             rag_content = "Document search is currently unavailable." # Indicate RAG skipped


        # --- Step 7: Initial Combination of Path 1 Component and RAG ---
        initial_combined_answer = self.combine_initial_answer_draft(
             path1_kg_diagnosis_component,
             kg_content_other,
             rag_content
        )

        # --- Step 8: Identify Missing Elements for LLM Focus ---
        # Evaluate the *initial combined answer* to guide the LLM synthesis
        missing_elements_for_llm = self.identify_missing_elements(core_query_for_processing, initial_combined_answer)


        # --- Step 9: LLM Synthesis ---
        logger.info("➡️ Initiating LLM Synthesis Step...")
        llm_synthesized_answer = ""
        if self.llm is not None: # Only attempt synthesis if LLM is available
            # Provide the initial combined answer draft and the original query to the synthesis LLM
            llm_synthesized_answer = self.generate_llm_answer(
                core_query_for_processing, # Original user query context
                initial_combined_answer=initial_combined_answer, # The combined draft
                missing_elements=missing_elements_for_llm # Tell LLM to focus on these
            ).strip()
        else:
            logger.warning("LLM not initialized. Skipping synthesis. Using initial combined answer directly.")
            llm_synthesized_answer = initial_combined_answer.strip() # Use draft directly as fallback

        final_core_answer = llm_synthesized_answer # The LLM synthesis (or draft) is the core final answer


        # --- Step 10: Final Reflection Check for LLM Follow-up (Decision Point 2) ---
        logger.info("🧠 Initiating Final Reflection Check (for LLM Follow-up)...")
        # Check completeness of the LLM-synthesized answer against the original query
        # Do NOT check if the LLM itself failed or if we are already processing a response to an LLM follow-up
        needs_final_followup_llm_opinion = False
        missing_questions_list = []
        # Only check if LLM is available AND we are NOT processing a response to an LLM follow-up
        if self.llm is not None and not is_response_to_llm_followup:
            needs_final_followup_llm_opinion, missing_questions_list = self.identify_missing_info(
                 core_query_for_processing, final_core_answer, self.chat_history # Pass current chat history state
            )
        else:
             logger.info("Skipping final reflection check (LLM not available or currently processing follow-up response).")


        # --- Step 11: Final LLM Follow-up Decision / Return Final Answer ---
        # Decide if the ONE allowed LLM follow-up should be asked now.
        # Trigger if LLM *thinks* a follow-up is needed AND we haven't asked it yet (round == 0).
        # Also ensure there's actually a question to ask AND LLM is available to phrase the prompt.
        if needs_final_followup_llm_opinion and self.followup_context["round"] == 0 and missing_questions_list and self.llm is not None:
             logger.info("❓ Final Reflection indicates missing critical info, asking the one allowed LLM follow-up.")
             # Construct the final LLM follow-up prompt (using the question from identify_missing_info)
             follow_up_question_text = missing_questions_list[0] # Use the first question recommended

             llm_follow_up_prompt_text = f"""
             Thank you. To ensure I provide the most helpful response, I need a bit more information:

             {follow_up_question_text}
             """
             self.followup_context["round"] = 1 # Mark that the one LLM follow-up has been asked for this thread

             # Log decision
             self.log_orchestration_decision(
                 core_query_for_processing,
                 f"SELECTED_STRATEGY: LLM_FINAL_FOLLOWUP\nREASONING: Final answer completeness check failed (LLM opinion). Asking for critical missing info (round 1).",
                 kg_data.get("top_disease_confidence", 0.0), # Use specific KG conf if available
                 rag_confidence
             )

             # Return prompt text and action flag
             # We don't add this to chat history here. Let the UI manage adding the user's message and this prompt.
             logger.info("Returning LLM Follow-up prompt.")
             return llm_follow_up_prompt_text.strip(), [], "llm_followup_prompt", None

        else:
            logger.info("✅ Final Reflection indicates answer is sufficient or single LLM follow-up already asked.")
            # This is the end of the processing path, return the final answer.

            # Collect all sources (RAG docs + KG mentions)
            all_sources: List[str] = []
            if rag_source_docs: # Add RAG sources
                for doc in rag_source_docs:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        source_name = doc.metadata["source"]
                        page_num = doc.metadata.get("page", "N/A")
                        # Include a snippet if available, but prioritize clean source string
                        source_snippet = doc.page_content[:100].replace('\n', ' ') + '...' if doc.page_content else ''
                        all_sources.append(f"[Source: {source_name}{f', Page {page_num}' if page_num != 'N/A' else ''}] {source_snippet}".strip())
                    # Else: Unknown source, could add a generic "[Source: Document]" if desired


            # Add KG source mentions if KG contributed specific sections and KG connection was ok
            if self.kg_connection_ok and (kg_data.get("identified_diseases_data") or kg_data.get("kg_treatments") or kg_data.get("kg_home_remedies")):
                 # Check which KG components actually had data to mention
                 kg_parts_mentioned = []
                 if kg_data.get("identified_diseases_data"): kg_parts_mentioned.append("Diagnosis Data")
                 if kg_data.get("kg_treatments"): kg_parts_mentioned.append("Treatment Data")
                 if kg_data.get("kg_home_remedies"): kg_parts_mentioned.append("Home Remedy Data")

                 if kg_parts_mentioned:
                    all_sources.append(f"[Source: Medical Knowledge Graph ({', '.join(kg_parts_mentioned)})]")
                 else:
                    # Fallback just mention KG if connection ok but no specific data extracted
                     all_sources.append(f"[Source: Medical Knowledge Graph]")


            # Deduplicate and clean up source strings for final display
            # Sort for consistent ordering
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
                # Add specific warning for chest pain/shortness of breath if relevant
                query_lower = core_query_for_processing.lower()
                # Check if the original query or the identified diseases mentioned these serious symptoms
                mentions_serious_symptoms = any(symptom in query_lower for symptom in ["chest pain", "shortness of breath", "difficulty breathing"]) or \
                                           any(any(symptom.lower() in d['Disease'].lower() for symptom in ["heart attack", "angina", "pulmonary embolism", "pneumonia"]) for d in kg_data.get("identified_diseases_data", [])) # Check KG diseases known to cause this

                if mentions_serious_symptoms:
                     disclaimer += "Chest pain and shortness of breath can be symptoms of serious conditions requiring immediate medical attention. Please seek emergency medical care if you experience severe symptoms, or if symptoms are sudden or worsening. "

                disclaimer += "Always consult with a qualified healthcare provider for any questions you may have regarding a medical condition. Never disregard professional medical advice or delay seeking it because of something you have read here."
                final_response_text += disclaimer

            # Log the final orchestration decision
            self.log_orchestration_decision(
                core_query_for_processing,
                f"SELECTED_STRATEGY: FINAL_ANSWER\nREASONING: Answer deemed sufficient after synthesis and final check or single LLM follow-up already asked.",
                kg_data.get("top_disease_confidence", 0.0),
                rag_confidence
            )

            # Add the conversation turn to the chat history *before* returning
            # Use the user_input (what the user *actually* typed this turn) and the final formatted response
            self.chat_history.append((user_input, final_response_text.strip()))

            logger.info("Returning Final Answer.")
            # Return response text, sources list, and action flag
            source_strings_for_display = all_sources_unique # Use the cleaned unique list
            return final_response_text.strip(), source_strings_for_display, "final_answer", None

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
    """Streamlit UI component to display symptom checkboxes per disease."""
    st.subheader("Confirm Your Symptoms")
    st.info(f"Based on your query: '{original_query}' and initial analysis, please confirm the symptoms you are experiencing from the list below to help narrow down possibilities.")

    # Use a unique key for the form based on the original query and a timestamp
    # This ensures a new form is rendered if the query changes or if the process is retried
    form_key = f"symptom_confirmation_form_{abs(hash(original_query))}_{st.session_state.get('form_timestamp', datetime.now().timestamp())}"

    # Initialize a local set to store confirmed symptoms during the current form interaction
    # This set will be updated by the checkboxes
    # Use the specific form key for the local state variable
    local_confirmed_symptoms_key = f'{form_key}_confirmed_symptoms_local'
    if local_confirmed_symptoms_key not in st.session_state:
        # Initialize empty for a new form
        st.session_state[local_confirmed_symptoms_key] = set()
        logger.debug(f"Initialized local symptom set for form {form_key}")
    else:
         logger.debug(f"Using existing local symptom set for form {form_key} with {len(st.session_state[local_confirmed_symptoms_key])} items.")


    with st.form(form_key):
        st.markdown("Please check all symptoms that apply to you:")
        # Sort diseases alphabetically for consistent display
        sorted_diseases = sorted(symptom_options.keys())
        for disease_label in sorted_diseases:
            symptoms = symptom_options[disease_label]
            st.markdown(f"**Possible Symptoms Associated with {disease_label.split(' (Confidence:')[0].strip()}:**") # Show clean disease name

            # Sort symptoms alphabetically within each disease list
            sorted_symptoms = sorted(symptoms)

            cols = st.columns(4) # Arrange checkboxes in columns (adjust number of columns as needed)
            for i, symptom in enumerate(sorted_symptoms):
                col = cols[i % 4]
                # Use a unique key for each checkbox
                checkbox_key = f"{form_key}_checkbox_{disease_label}_{symptom}"
                # Check if this symptom was previously selected in this form render cycle
                # This helps preserve selections if the app reruns while the form is displayed
                initial_state = symptom.strip().lower() in st.session_state[local_confirmed_symptoms_key]
                if col.checkbox(symptom, key=checkbox_key, value=initial_state):
                     # Add to the local set on click
                     st.session_state[local_confirmed_symptoms_key].add(symptom.strip().lower())
                else:
                     # Remove from the local set if unchecked
                     st.session_state[local_confirmed_symptoms_key].discard(symptom.strip().lower())


        # Add an "Other" symptom input field
        st.markdown("**Other Symptoms (if any):**")
        other_symptoms_text = st.text_input("Enter additional symptoms here (comma-separated)", key=f"{form_key}_other_symptoms_input")
        if other_symptoms_text:
             other_symptoms_list = [s.strip().lower() for s in other_symptoms_text.split(',') if s.strip()]
             st.session_state[local_confirmed_symptoms_key].update(other_symptoms_list)


        # When the form is submitted, save the *final* state of the local set
        # This is done implicitly by Streamlit when the form is submitted and a rerun happens.
        # The submit button triggers the rerun.
        submit_button = st.form_submit_button("Confirm and Continue")

        if submit_button:
            logger.info(f"Symptom confirmation form submitted. Final confirmed symptoms: {st.session_state[local_confirmed_symptoms_key]}")
            # Store the confirmed symptoms from the local set into the *main* session state variable
            st.session_state.confirmed_symptoms_from_ui = list(st.session_state[local_confirmed_symptoms_key])

            # Clear the UI state flags *after* submitting the form, but before the rerun.
            st.session_state.awaiting_symptom_confirmation = False
            # Keep original_query_for_followup; generate_response needs it to know which query to process.
            # Do NOT clear confirmed_symptoms_from_ui here; generate_response needs it in the next rerun.

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
        # This container helps position the input at the bottom
        input_container = st.container()
        st.write("  \n" * 5) # Add space pusher at the end of the tab


        # --- Process Input (Triggered by Rerun) ---
        # This block runs on every rerun. It checks session state and widgets
        # to determine if a response needs to be generated.

        # Initialize variables to hold the input and context for the chatbot
        current_user_text_input = None
        confirmed_symps_to_pass = None
        original_query_to_pass = None

        # Check if the symptom confirmation form was just submitted
        # This happens if confirmed_symptoms_from_ui is set AND we are NOT currently awaiting confirmation (meaning the form was submitted)
        if st.session_state.confirmed_symptoms_from_ui is not None and not st.session_state.awaiting_symptom_confirmation:
             logger.info("Detected symptom confirmation form submission.")
             # Get the confirmed symptoms and the original query that triggered the UI
             confirmed_symps_to_pass = st.session_state.confirmed_symptoms_from_ui
             original_query_to_pass = st.session_state.original_query_for_followup # Use the stored original query for processing context
             # The user's actual response text in this turn is not relevant, the confirmed symptoms are the input.
             # We use the original query as the "user_input" for generate_response's first argument in this case
             current_user_text_input = original_query_to_pass # Pass the original query text as user input for the bot's process

             # Clear the UI state variables after capturing them
             st.session_state.confirmed_symptoms_from_ui = None
             # original_query_for_followup is cleared by generate_response based on its action flag
             st.session_state.awaiting_symptom_confirmation = False # Ensure this is off

             # Add the original query to messages as if the user typed it if it's not already the last user message
             # This makes the conversation flow look correct
             if not st.session_state.messages or st.session_state.messages[-1] != (original_query_to_pass, True):
                  st.session_state.messages.append((original_query_to_pass, True))


        # Check if a new message was submitted in the chat input box
        # This happens if st.chat_input returns a non-empty string AND we are NOT awaiting symptom confirmation
        elif (prompt := st.chat_input("Ask your medical question...", disabled=st.session_state.init_failed or st.session_state.awaiting_symptom_confirmation)):
             logger.info(f"Detected chat input submission: '{prompt}'")
             current_user_text_input = prompt
             # For a new chat input, confirmed_symps_to_pass and original_query_to_pass remain None

             # Add the user message to the UI messages state immediately
             st.session_state.messages.append((current_user_text_input, True))

             # Clear any previous follow-up state for a brand new query thread
             st.session_state.chatbot.followup_context = {"round": 0} # Reset LLM follow-up round
             st.session_state.original_query_for_followup = "" # Clear original query for follow-up
             # confirmed_symptoms_from_ui should already be None here


        # --- Call generate_response if there is input to process ---
        # This call happens only if one of the above blocks set current_user_text_input
        if current_user_text_input is not None:
            with st.spinner("Thinking..."):
                 response_text, sources, action_flag, ui_data = st.session_state.chatbot.generate_response(
                      current_user_text_input, user_type,
                      confirmed_symptoms=confirmed_symps_to_pass,
                      original_query_if_followup=original_query_to_pass # Pass the original query context if available
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
                 # The 'original_query_if_followup' for the *next* turn should be the 'core_query_for_processing' of *this* turn
                 st.session_state.original_query_for_followup = core_query_for_processing


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
            st.rerun()


        # --- Reset Conversation Button ---
        st.divider() # Add a visual separator
        if st.button("Reset Conversation", key="reset_conversation_button"):
            st.session_state.chatbot.reset_conversation() # Resets internal history and followup_context
            st.session_state.messages = [] # Clear UI messages
            # Also reset UI state variables
            st.session_state.awaiting_symptom_confirmation = False
            st.session_state.symptom_options_for_ui = {}
            st.session_state.confirmed_symptoms_from_ui = None
            st.session_state.original_query_for_followup = ""
            st.session_state.current_input_to_process = None # Clear any pending input
            st.session_state.form_timestamp = datetime.now().timestamp() # Reset form timestamp
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
