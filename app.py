
import streamlit as st
from pathlib import Path
import csv
import os
import re
import torch
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import base64
import logging
from PIL import Image
import io
# Import Gemini API
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from neo4j import GraphDatabase
# Configuration
# Gemini API key
# Add this near the top after imports
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables with fallback to hardcoded values
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBv-I8Ld-k09Lxu9Yi7HPffZHKXIqGSdHU")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://1b47920f.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "eCqDfyhDcuGMLzbYfiqL6jsvjH3LIXr86xQGAEKmY8Y")

# Update the NEO4J_AUTH variable to use environment variables
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# Threshold settings
THRESHOLDS = {
    "symptom_extraction": 0.6,
    "disease_matching": 0.5,
    "": 0.6,
}

# Load and convert the image to base64
def get_image_as_base64(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: Image file not found at {file_path}")
        return ""  # Return empty string if file not found
    
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Option 1: If your image is stored locally
image_path = "Zoom My Life.jpg"  # Update with your actual path
icon = get_image_as_base64(image_path)

# Cache for expensive operations
CACHE = {}

# Hardcoded PDF files to use
HARDCODED_PDF_FILES = [
    "rawdata.pdf",  # Update with local paths
]

# For testing purposes - update with your actual list
known_diseases = ["hypertension", "type 2 diabetes mellitus", "respiratory infections", "obesity", "cardiovascular disease"]

# Execution state enum
class ExecutionState:
    ACTIVE = "active"
    HALTED = "halted"

def get_cached(key):
    """Get cached result if it exists"""
    key_str = str(key)
    if key_str in CACHE:
        return CACHE[key_str]
    return None

def set_cached(key, value):
    """Set cache for a key"""
    key_str = str(key)
    CACHE[key_str] = value
    return value

# DocumentChatBot class
class DocumentChatBot:
    def __init__(self):
        self.qa_chain = None
        self.vectordb = None
        self.chat_history = []
        self.identified_diseases = []
        self.disease_confidence = 0.0
        # Initialize embedding model during initialization
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache'),  # Without sentence-transformers/ prefix
                cache_folder='./cache',
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"Warning: Could not initialize embedding model: {e}")
            self.embedding_model = None
            
        self.llm = None
        self.followup_context = {"asked": False, "round": 0}

    def create_vectordb(self):
            """Create vector database from hardcoded PDF documents"""
            pdf_files = [Path(pdf_file) for pdf_file in HARDCODED_PDF_FILES if Path(pdf_file).exists()]
        
            if not pdf_files:
                return None, "No PDF files found at the specified paths."
        
            loaders = [PyPDFLoader(str(pdf_file)) for pdf_file in pdf_files]
            pages = []
            for loader in loaders:
                pages.extend(loader.load())
        
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=64
            )
            splits = text_splitter.split_documents(pages)
        
            # Initialize embedding model with fix for meta tensor issue
            try:
                # Try SentenceTransformer directly instead of HuggingFaceEmbeddings
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                print(f"Loading embedding model: {model_name}")
                
                # Fix for the meta tensor issue
                import torch
                
                # Check torch version - newer versions may need different approach
                torch_version = torch.__version__
                print(f"Using torch version: {torch_version}")
                
                # Use a device-specific loading approach
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"Using device: {device}")
                
                # Method 1: Try loading with explicit device
                try:
                    sentence_model = SentenceTransformer(model_name, cache_folder='./cache', device=device)
                except Exception as e1:
                    print(f"First loading method failed: {e1}. Trying alternative approach...")
                    # Method 2: Try loading with empty initialization and then moving to device
                    if hasattr(torch.nn.Module, 'to_empty'):
                        # For newer PyTorch versions with to_empty support
                        sentence_model = SentenceTransformer(model_name, cache_folder='./cache')
                        sentence_model = sentence_model.to_empty(device=device)
                        # Now load the state dict
                        sentence_model.load_state_dict(torch.load(f"./cache/{model_name.replace('/', '_')}/pytorch_model.bin"))
                    else:
                        # Fallback for older PyTorch versions
                        sentence_model = SentenceTransformer(model_name, cache_folder='./cache')
                        # Force CPU if needed
                        sentence_model = sentence_model.to('cpu')
                
                # Create a wrapper to make it compatible with LangChain
                from langchain.embeddings.base import Embeddings
                
                class SentenceTransformerEmbeddings(Embeddings):
                    def __init__(self, model):
                        self.model = model
                        
                    def embed_documents(self, texts):
                        return self.model.encode(texts, normalize_embeddings=True).tolist()
                        
                    def embed_query(self, text):
                        return self.model.encode(text, normalize_embeddings=True).tolist()
                
                embeddings = SentenceTransformerEmbeddings(sentence_model)
                vectordb = FAISS.from_documents(splits, embeddings)
                return vectordb, "Vector database created successfully."
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                return None, f"Failed to load embeddings model: {str(e)}"
        
    def is_medical_query(self, query: str) -> Tuple[bool, str]:
        """
        Check if the query is relevant to the medical domain.
        Returns a tuple of (is_relevant, reason)
        """
        cache_key = {"type": "medical_relevance", "query": query}
        cached = get_cached(cache_key)
        if cached:
            return cached
    
        medical_relevance_prompt = f"""
        Determine if the following query is related to medical topics, health concerns, symptoms, treatments, 
        diseases, or other healthcare topics. 
        
        Query: "{query}"
        
        Return ONLY a JSON object with this format:
        {{
            "is_medical": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.local_generate(medical_relevance_prompt, max_tokens=250)
            json_match = re.search(r'\{[\s\S]*\}', response)
            
            if json_match:
                data = json.loads(json_match.group(0))
                is_medical = data.get("is_medical", False)
                confidence = data.get("confidence", 0.0)
                reasoning = data.get("reasoning", "")
                
                if is_medical and confidence >= 0.6:
                    result = (True, "")
                else:
                    result = (False, reasoning)
                
                set_cached(cache_key, result)
                return result
        except Exception as e:
            print(f"Error checking medical relevance: {e}")
        
        # Default fallback - allow the query 
        return (True, "")

    def initialize_qa_chain(self):
        """Initialize the QA chain with Gemini Flash 1.5 and vector database"""
        if self.qa_chain is None:
            self.vectordb, message = self.create_vectordb()
            if self.vectordb is None:
                return False, message

            # Check for API key
            if not GEMINI_API_KEY:
                return False, "Gemini API key not found. Please set your API key."

            try:
                # Initialize Gemini Flash 1.5
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.3,
                    top_p=0.95,
                    top_k=40,
                    convert_system_message_to_human=True
                )

                # Test the model with a simple prompt
                test_response = self.llm.invoke("Hello")
                print(f"Successfully connected to Gemini Flash 1.5")

                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    output_key='answer',
                    return_messages=True
                )

                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    self.llm,
                    retriever=self.vectordb.as_retriever(),
                    chain_type="stuff",
                    memory=memory,
                    return_source_documents=True,
                    verbose=True,
                )

                return True, "Chat assistant initialized successfully with Gemini Flash 1.5 and knowledge graph capabilities!"
            except Exception as e:
                return False, f"Failed to connect to Gemini Flash 1.5: {str(e)}"

        return True, "Chat assistant is already initialized."

    def format_chat_history(self):
        """Format chat history for the LLM"""
        formatted_history = []
        for user_msg, bot_msg in self.chat_history:
            formatted_history.append(f"User: {user_msg}")
            formatted_history.append(f"Assistant: {bot_msg}")
        return formatted_history

    def calculate_kg_completeness(self, state: Dict) -> float:
        """Improved completeness scoring for diagnostic queries"""
        base_score = 0.0
        
        if state.get("disease"):
            base_score = 0.7  # Base score for having a disease
            
            # Bonus for symptom matches
            if state.get("matched_symptoms"):
                base_score += min(0.3, len(state["matched_symptoms"]) * 0.1)
                
            # Bonus for direct mention
            if state.get("direct_disease_mention"):
                base_score = min(1.0, base_score + 0.2)
        
        return min(1.0, base_score)
    
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

    def generate_llm_answer(self, query, kg_missing=None, rag_missing=None, llm_client=None):
        """
        Generates an LLM answer focused on addressing missing elements.
        
        Args:
            query: Original user query
            kg_missing: Elements missing from KG answer
            rag_missing: Elements missing from RAG answer
            llm_client: Optional LLM client (not used in this implementation)
            
        Returns:
            str: LLM-generated answer
        """
        # Construct focus areas from missing elements
        focus_areas = set()
        if kg_missing:
            focus_areas.update(kg_missing)
        if rag_missing:
            focus_areas.update(rag_missing)
        
        focus_text = ""
        if focus_areas:
            focus_text = "Focus particularly on these aspects: " + ", ".join(focus_areas)
        
        prompt = f"""
        You are a medical AI assistant providing information about a health question.
        
        USER QUESTION: {query}
        
        {focus_text}
        
        Provide a helpful, accurate answer to the user's question.
        Include appropriate disclaimers about consulting healthcare professionals.
        """
        
        try:
            response = self.local_generate(prompt, max_tokens=1000)
            return response.strip()
        except Exception as e:
            print(f"Error generating LLM answer: {e}")
            return "I'm sorry, but I couldn't generate a specific answer to your question. Please consult a healthcare professional for personalized advice."
        
    def identify_missing_info(self, user_query: str, conversation_history: List[Tuple[str, str]]) -> List[str]:
            """Identifies what critical medical information is missing from user query with conversation context"""
        
            # Convert conversation history to a string to provide context
            context = ""
            # Only include the last 3 exchanges for context
            recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            for user_msg, bot_msg in recent_history:
                context += f"User: {user_msg}\n"
                if bot_msg:
                    context += f"Assistant: {bot_msg}\n"
        
            # Increment the follow-up round counter
            self.followup_context["round"] += 1
        
            # If we've already asked follow-up questions twice, don't ask more
            if self.followup_context["asked"] and self.followup_context["round"] >= 2:
                print("⚠️ Already asked follow-ups twice, proceeding with available information")
                return []
            
            # Check if we have enough context already
            if len(conversation_history) >= 2:
                # Don't ask follow-up if we've had multiple exchanges already
                return []
        
            cache_key = {"type": "missing_info", "query": user_query, "context": context}
            cached = get_cached(cache_key)
            if cached:
                print("🧠 Using cached missing info assessment.")
                return cached
            
            # First check if we have knowledge graph or RAG results
            has_results = False
            try:
                # Process with KG to see if we get results
                kg_data = self.process_with_knowledge_graph(user_query)
                if kg_data.get("disease") or kg_data.get("symptoms"):
                    has_results = True
                    print("✅ Knowledge graph returned results, no follow-up needed")
                
                # If we have results, don't ask follow-up questions
                if has_results:
                    return []
            except Exception:
                pass  # Continue to LLM evaluation if KG fails
        
            MISSING_INFO_PROMPT = (
                    f"You are a medical assistant analyzing a patient query and conversation.\n"
                    f"Determine if any CRITICAL information is still missing to properly assess their situation.\n\n"
                    f"Conversation history:\n{context}\n\n"
                    f"Latest patient input: \"{user_query}\"\n\n"
                    "CRITICALLY EVALUATE if you have enough information to provide a reasonable medical assessment.\n"
                    "Only ask follow-up questions if ABSOLUTELY NECESSARY for basic assessment.\n\n"
                    "Rules for determining if information is sufficient:\n"
                    "1. If the patient has provided symptoms, duration, and basic severity, that's usually enough\n"
                    "2. If you've already asked follow-up questions once, avoid asking more unless critical\n"
                    "3. If a general assessment can be made with current information, proceed without more questions\n"
                    "4. ONLY ask about truly essential missing information\n"
                    "5. If the query is about general medical information (not about a specific case), NO follow-up needed\n"
                    "6. If the question is about treatments, medication, or general information, NO follow-up needed\n\n"
                    "Return your answer in this exact JSON format:\n"
                    "{{\n"
                    '    "needs_followup": true/false,\n'
                    '    "reasoning": "brief explanation of why more information is needed or not",\n'
                    '    "missing_info": [\n'
                    '        {{"question": "specific follow-up question 1"}}\n'
                    "    ]\n"
                    "}}"
                )
        
            try:
                response = self.local_generate(MISSING_INFO_PROMPT, max_tokens=500).strip()
                print("\nRaw Missing Info Evaluation:\n", response)
        
                # Parse JSON format response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        data = json.loads(json_str)
                        needs_followup = data.get("needs_followup", False)
                        
                        if not needs_followup:
                            print("✅ Sufficient information available, no follow-up needed")
                            return []
                        
                        missing_info = [item["question"] for item in data.get("missing_info", [])]
                        reasoning = data.get("reasoning", "Need more specific information")
                        
                        if missing_info:
                            print(f"❓ Missing Information Identified: {missing_info}")
                            print(f"Reasoning: {reasoning}")
                            # Mark that we've asked follow-up questions
                            self.followup_context["asked"] = True
                            set_cached(cache_key, missing_info)
                            return missing_info
                    except json.JSONDecodeError:
                        print("⚠️ Could not parse missing info JSON from LLM response")
            except Exception as e:
                print(f"⚠️ Error identifying missing information: {e}")
            
            return []
                                 
    def extract_disease_from_query(self, user_query: str) -> Tuple[Optional[str], float]:
        """Extract disease names from user query with confidence scores"""
        # Input validation
        if not user_query or not isinstance(user_query, str):
            print(f"⚠️ Invalid user query for disease extraction: {type(user_query)}")
            return None, 0.0
        
        cache_key = {"type": "disease_extraction", "query": user_query}
        cached = get_cached(cache_key)
        if cached:
            print("🧠 Using cached disease extraction.")
            return cached
    
        # Common symptom phrases that should not be classified as diseases
        symptom_phrases = [
            "chest pain", "shortness of breath", "headache", "fever", "cough",
            "sore throat", "fatigue", "nausea", "dizziness", "pain", "ache",
            "sweating", "vomiting", "chills", "weakness", "discomfort"
        ]
        
        # Check if the user is asking about symptoms rather than mentioning a disease
        query_lower = user_query.lower()
        is_symptom_query = False
        
        # Don't classify symptom phrases as diseases
        for phrase in symptom_phrases:
            if phrase in query_lower:
                print(f"⚠️ Found symptom phrase '{phrase}' - this is not a disease mention")
                is_symptom_query = True
    
        # If user is just asking about symptoms, don't extract as disease
        symptom_question_patterns = [
            "what disease", "what condition", "what could be causing", 
            "what might be causing", "possible disease", "possible condition",
            "what is causing", "why do i have", "reasons for"
        ]
        
        for pattern in symptom_question_patterns:
            if pattern in query_lower:
                print(f"⚠️ Query is asking about causes of symptoms, not mentioning a disease directly")
                is_symptom_query = True
        
        if is_symptom_query:
            # This is asking about symptoms, not mentioning a disease
            return None, 0.0
    
        # First check if any known diseases are directly mentioned
        known_diseases = ["cardiovascular disease", "hypertension", "obesity", 
                         "respiratory infections", "type 2 diabetes mellitus",
                         "diabetes", "heart disease", "high blood pressure"]
        
        for disease in known_diseases:
            if disease.lower() in query_lower:
                print(f"🔍 Found direct disease mention: {disease} (confidence: 0.95)")
                return disease, 0.95
    
        # If no direct match, use LLM to extract
        DISEASE_PROMPT = (
            f"You are a medical assistant.\n"
            f"Extract and identify any disease, medical condition, or health disorder mentioned in the following user query.\n"
            f"Assign a confidence score between 0.0 and 1.0 indicating how certain you are.\n"
            f"DO NOT classify symptoms like \"chest pain\", \"shortness of breath\", \"headache\", etc. as diseases.\n\n"
            "**Important:** Return your answer in exactly the following format:\n"
            "Extracted Disease: {{\"disease\": \"disease_name\", \"confidence\": 0.9}}\n\n"
            "If no disease is mentioned, respond with:\n"
            "Extracted Disease: {{\"disease\": null, \"confidence\": 0.0}}\n\n"
            f"User Query: \"{user_query}\""
        )
            
        try:
            response = self.local_generate(DISEASE_PROMPT, max_tokens=250).strip()
            print("\nRaw Disease Extraction Response:\n", response)
    
            # Parse JSON format response with regex
            match = re.search(r"Extracted Disease:\s*(\{.*?\})", response, re.DOTALL)
            if match:
                try:
                    disease_data = json.loads(match.group(1))
                    disease = disease_data.get("disease")
                    confidence = disease_data.get("confidence", 0.0)
    
                    # Validation check - don't allow symptom phrases to be classified as diseases
                    if disease:
                        for symptom in symptom_phrases:
                            if symptom.lower() in disease.lower():
                                print(f"⚠️ Rejected disease extraction: '{disease}' contains symptom phrase '{symptom}'")
                                return None, 0.0
    
                    if disease and confidence >= 0.7:
                        print(f"🔍 Extracted Disease: {disease} (confidence: {confidence:.4f})")
                        result = (disease, confidence)
                        set_cached(cache_key, result)
                        return result
                    else:
                        print(f"🔍 No disease extracted with sufficient confidence")
                        return None, 0.0
                except json.JSONDecodeError:
                    print("⚠️ Could not parse disease JSON from LLM response")
        except Exception as e:
            print(f"⚠️ Error in disease extraction: {e}")
    
        return None, 0.0
    
    def is_treatment_query(self, query: str) -> bool:
        """Determine if the query is asking about treatments or remedies for a disease"""
        query_lower = query.lower()
        
        # Keywords related to treatments
        treatment_keywords = [
            "treatment", "medication", "medicine", "drug", "therapy", "prescription",
            "cure", "remedy", "therapeutic", "pharmacological", "manage", "management",
            "how to treat", "how to manage", "how to cure", "what helps", "how to control",
            "home remedy", "natural remedy", "alternative treatment"
        ]
        
        return any(keyword in query_lower for keyword in treatment_keywords)  
                          
    def extract_diseases_from_kg_answer(self, kg_answer):
        """
        Extracts disease information from a formatted KG answer string.
        """
        self.identified_diseases = []
        
        if not kg_answer:
            return
            
        # Try to extract diseases from structured KG answer
        if "Possible Diseases:" in kg_answer:
            # Find the section with diseases
            disease_section = kg_answer.split("Possible Diseases:")[1].split("\n\n")[0]
            # Extract disease names (assuming they're listed with bullet points or numbers)
            import re
            disease_matches = re.findall(r'[•*\-\d.]\s*([^•*\-\n]+)', disease_section)
            if disease_matches:
                self.identified_diseases = [d.strip() for d in disease_matches if d.strip()]
        elif "Disease:" in kg_answer:
            # Extract single disease
            disease_section = kg_answer.split("Disease:")[1].split("\n")[0]
            if disease_section.strip():
                self.identified_diseases = [disease_section.strip()]
                          
    def get_kg_symptoms(self) -> Tuple[List[str], np.ndarray]:
            """Get all symptoms from Neo4j knowledge graph"""
            cache_key = "kg_symptoms"
            cached = get_cached(cache_key)
            if cached:
                print("🧠 Using cached symptoms list.")
                return cached
        
            try:
                with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
                    query = "MATCH (s:symptom) RETURN DISTINCT s.Name"
                    result = driver.session().run(query)
                    kg_symptoms = [record["s.Name"] for record in result]
        
                if self.embedding_model is None:
                    # Initialize with the correct model name
                    self.embedding_model = HuggingFaceEmbeddings(
                        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache'),  # Without sentence-transformers/ prefix
                        cache_folder='./cache',
                        encode_kwargs={'normalize_embeddings': True}
                    )
        
                embeddings = self.embedding_model.embed_documents(kg_symptoms)
                result = (kg_symptoms, np.array(embeddings))
                set_cached(cache_key, result)
                return result
            except Exception as e:
                print(f"⚠️ Error querying Neo4j for symptoms: {e}")
                return [], np.array([])
                      
    def extract_symptoms(self, user_query: str) -> Tuple[List[str], float]:
        """Extract symptoms from user query with confidence scores"""
        # Input validation
        if not user_query or not isinstance(user_query, str):
            print(f"⚠️ Invalid user query for symptom extraction: {type(user_query)}")
            return [], 0.0
    
        # Cache check with proper string query
        cache_key = {"type": "symptom_extraction", "query": user_query}
        cached = get_cached(cache_key)
        if cached:
            print("🧠 Using cached symptom extraction.")
            return cached
    
        # Known symptom keywords for fallback detection
        known_symptoms = [
            "chest pain", "shortness of breath", "headache", "fever", "cough", 
            "sore throat", "fatigue", "nausea", "dizziness", "pain", "runny nose",
            "congestion", "sweating", "vomiting", "chills", "weakness"
        ]
    
        SYMPTOM_PROMPT = (
                f"You are a medical assistant.\n"
                f"Extract and correct all symptoms mentioned in the following user query.\n"
                f"For each symptom, assign a confidence score between 0.0 and 1.0 indicating how certain you are.\n"
                "**Important:** Return your answer in exactly the following format:\n"
                "Extracted Symptoms: [{{\"symptom\": \"symptom1\", \"confidence\": 0.9}}, {{\"symptom\": \"symptom2\", \"confidence\": 0.8}}, ...]\n\n"
                f"User Query:\n"
                f"\"{user_query}\""
            )
    
        try:
            response = self.local_generate(SYMPTOM_PROMPT, max_tokens=500).strip()
            print("\nRaw Symptom Extraction Response:\n", response)
    
            # Parse JSON format response with regex
            match = re.search(r"Extracted Symptoms:\s*(\[.*?\])", response, re.DOTALL)
            if match:
                try:
                    symptom_data = json.loads(match.group(1))
                    # Filter symptoms based on confidence threshold
                    confident_symptoms = [item["symptom"].strip().lower()
                                        for item in symptom_data
                                        if item.get("confidence", 0) >= THRESHOLDS["symptom_extraction"]]
    
                    # Calculate average confidence for symptom extraction
                    if symptom_data:
                        avg_confidence = sum(item.get("confidence", 0) for item in symptom_data) / len(symptom_data)
                    else:
                        avg_confidence = 0.0
    
                    if confident_symptoms:
                        print(f"🔍 Extracted Symptoms: {confident_symptoms} (avg confidence: {avg_confidence:.4f})")
                        result = (confident_symptoms, avg_confidence)
                        set_cached(cache_key, result)
                        return result
                except json.JSONDecodeError:
                    print("⚠️ Could not parse symptom JSON from LLM response")
        except Exception as e:
            print(f"⚠️ Error in symptom extraction: {e}")
    
        # Fallback keyword extraction
        print("⚠️ Using fallback keyword extraction for symptoms.")
        fallback_symptoms = []
    
        query_lower = user_query.lower()
        for symptom in known_symptoms:
            if symptom in query_lower:
                fallback_symptoms.append(symptom)
    
        fallback_confidence = 0.6 if fallback_symptoms else 0.4  # Increased confidence for fallback extraction
        print(f"🔍 Fallback Extracted Symptoms: {fallback_symptoms} (confidence: {fallback_confidence:.4f})")
        
        result = (fallback_symptoms, fallback_confidence)
        set_cached(cache_key, result)
        return result


    def query_disease_from_symptoms(self, symptoms: List[str]) -> Tuple[Optional[str], float, List[str], List[Tuple[str, float]]]:
        """Query for diseases based on symptoms"""
        if not symptoms:
            print("⚠️ No symptoms provided to query diseases")
            return None, 0.0, [], []
    
        cache_key = {"type": "disease_matching", "symptoms": tuple(symptoms)}
        cached = get_cached(cache_key)
        if cached:
            print("🧠 Using cached disease match.")
            return cached
    
        # Validate symptoms against known symptoms
        valid_symptoms = []
        for symptom in symptoms:
            # Normalize symptom format
            normalized = symptom.lower().strip()
            valid_symptoms.append(normalized)
        
        if not valid_symptoms:
            print("⚠️ No valid symptoms after normalization")
            return None, 0.0, [], []
    
        print(f"🔍 Querying diseases for symptoms: {valid_symptoms}")
    
        # Check for direct match with known diseases
        for s in valid_symptoms:
            if s in known_diseases:
                detected = s.capitalize()
                print(f"Using extracted symptom as disease: {detected} (confidence: 0.95)")
                return detected, 0.95, [s], [(detected, 0.95)]
    
        # Construct a more sophisticated Cypher query with confidence scoring
        cypher_query = f"""
        MATCH (s:symptom)-[r:INDICATES]->(d:disease)
        WHERE LOWER(s.Name) IN {str(valid_symptoms)}
        WITH d, COUNT(DISTINCT s) AS matching_symptoms,
             COLLECT(DISTINCT LOWER(s.Name)) AS matched_symptoms
        WITH d, matching_symptoms, matched_symptoms,
             matching_symptoms * 1.0 / {max(1, len(valid_symptoms))} AS confidence_score
        WHERE confidence_score >= {THRESHOLDS["disease_matching"]}
        RETURN d.Name AS Disease, confidence_score AS Confidence, matched_symptoms AS MatchedSymptoms
        ORDER BY confidence_score DESC
        LIMIT 3
        """
    
        try:
            with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
                session = driver.session()
                result = session.run(cypher_query)
                records = list(result)
    
                if records:
                    all_diseases = [(rec["Disease"], float(rec["Confidence"]), rec["MatchedSymptoms"]) for rec in records]
                    
                    # For backward compatibility and logging
                    top_disease = records[0]["Disease"]
                    confidence = float(records[0]["Confidence"])
                    matched_symptoms = records[0]["MatchedSymptoms"]
                
                    print(f"🦠 Detected Diseases: {[d[0] for d in all_diseases]} (top confidence: {confidence:.4f})")
                    print(f"Top matched symptoms: {matched_symptoms}")
                
                    # Return all diseases as primary result
                    result = (all_diseases, confidence, matched_symptoms, all_diseases)
                    set_cached(cache_key, result)
                    return result
                else:
                    print("⚠️ No diseases matched the provided symptoms in the knowledge graph")
    
        except Exception as e:
            print(f"⚠️ Error querying Neo4j for diseases: {e}")
            print(f"⚠️ Query attempted: {cypher_query}")
    
        # If no matches in Knowledge Graph, use LLM as fallback
        print("⚠️ Using LLM fallback for symptom-to-disease mapping")
        
        DISEASE_INFERENCE_PROMPT = f"""
        You are a medical assistant. Based on the following symptoms, identify the most likely diseases or conditions.
        For each disease, provide a confidence score between 0 and 1.
        
        Symptoms: {', '.join(valid_symptoms)}
        
        Format your response as a JSON array:
        [
          {{"disease": "Disease 1", "confidence": 0.8, "matched_symptoms": ["symptom1", "symptom2"]}},
          {{"disease": "Disease 2", "confidence": 0.6, "matched_symptoms": ["symptom1"]}}
        ]
        
        Focus on these common conditions: Cardiovascular Disease, Hypertension, Obesity, Respiratory Infections, Type 2 Diabetes Mellitus
        """
        
        try:
            response = self.local_generate(DISEASE_INFERENCE_PROMPT, max_tokens=500)
            json_match = re.search(r'\[[\s\S]*\]', response)
            
            if json_match:
                try:
                    diseases_data = json.loads(json_match.group(0))
                    if diseases_data:
                        # Filter and format diseases above threshold
                        filtered_diseases = [
                            (d["disease"], d["confidence"], d.get("matched_symptoms", []))
                            for d in diseases_data
                            if d.get("confidence", 0) >= THRESHOLDS["disease_matching"]
                        ]
                        
                        if filtered_diseases:
                            # Sort by confidence
                            filtered_diseases.sort(key=lambda x: x[1], reverse=True)
                            top_disease = filtered_diseases[0][0]
                            confidence = filtered_diseases[0][1]
                            matched_symptoms = filtered_diseases[0][2]
                            
                            print(f"🦠 LLM Fallback - Detected Diseases: {[d[0] for d in filtered_diseases]} (top confidence: {confidence:.4f})")
                            return (filtered_diseases, confidence, matched_symptoms, filtered_diseases)
                except json.JSONDecodeError:
                    print("⚠️ Could not parse disease inference JSON from LLM response")
        except Exception as e:
            print(f"⚠️ Error in LLM disease inference: {e}")
    
        return None, 0.0, [], []

    def query_treatments(self, disease: str) -> Tuple[List[str], float]:
        """Query for treatments with confidence scores"""
        if not disease:
            return [], 0.0

        cache_key = {"type": "treatment_query", "disease": disease.lower()}
        cached = get_cached(cache_key)
        if cached:
            print("🧠 Using cached treatments.")
            return cached

        cypher_query = f"""
        MATCH (d:disease)-[r:TREATED_BY]->(t:treatment)
        WHERE LOWER(d.Name) = '{disease.lower()}'
        WITH t, COUNT(r) as rel_count
        RETURN t.Name as Treatment,
               CASE WHEN rel_count > 3 THEN 0.9
                    WHEN rel_count > 1 THEN 0.8
                    ELSE 0.7
               END as Confidence
        ORDER BY Confidence DESC
        """

        try:
            with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
                session = driver.session()
                result = session.run(cypher_query)
                records = list(result)

                if records:
                    treatments = [(rec["Treatment"], float(rec["Confidence"])) for rec in records]
                    treatments_list = [t[0] for t in treatments]
                    avg_confidence = sum(t[1] for t in treatments) / len(treatments)

                    print(f"💊 Available Treatments: {treatments_list} (avg confidence: {avg_confidence:.4f})")
                    result = (treatments_list, avg_confidence)
                    set_cached(cache_key, result)
                    return result
        except Exception as e:
            print(f"⚠️ Error querying Neo4j for treatments: {e}")

        return [], 0.0

    def query_home_remedies(self, disease: str) -> Tuple[List[str], float]:
        """Query for home remedies with confidence scores"""
        if not disease:
            return [], 0.0

        cache_key = {"type": "remedy_query", "disease": disease.lower()}
        cached = get_cached(cache_key)
        if cached:
            print("🧠 Using cached home remedies.")
            return cached

        cypher_query = f"""
        MATCH (d:disease)-[r:HAS_HOMEREMEDY]->(h:homeremedy)
        WHERE LOWER(d.Name) = '{disease.lower()}'
        WITH h, COUNT(r) as rel_count
        RETURN h.Name as HomeRemedy,
               CASE WHEN rel_count > 2 THEN 0.85
                    WHEN rel_count > 1 THEN 0.75
                    ELSE 0.65
               END as Confidence
        ORDER BY Confidence DESC
        """

        try:
            with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
                session = driver.session()
                result = session.run(cypher_query)
                records = list(result)

                if records:
                    remedies = [(rec["HomeRemedy"], float(rec["Confidence"])) for rec in records]
                    remedies_list = [r[0] for r in remedies]
                    avg_confidence = sum(r[1] for r in remedies) / len(remedies)

                    print(f"🏡 Available Home Remedies: {remedies_list} (avg confidence: {avg_confidence:.4f})")
                    result = (remedies_list, avg_confidence)
                    set_cached(cache_key, result)
                    return result
        except Exception as e:
            print(f"⚠️ Error querying Neo4j for home remedies: {e}")

        return [], 0.0

    def is_disease_identification_query(self, query):
        query_lower = query.lower()
        
        # Keywords that suggest disease identification
        disease_keywords = ["what disease", "what condition", "what could be causing", 
                            "what might be causing", "possible disease", "possible condition",
                            "diagnosis", "diagnose", "what causes", "what is causing"]
        
        # Check for symptom mentions
        has_symptoms = any(symptom in query_lower for symptom in [
            "chest pain", "shortness of breath", "cough", "fever", "headache", 
            "fatigue", "dizziness", "symptom"
        ])
        
        # Check for disease identification intent
        is_asking_for_disease = any(keyword in query_lower for keyword in disease_keywords)
        
        return has_symptoms and (is_asking_for_disease or "?" in query)
            
    def format_disease_list_answer(self, query, disease_list, confidence=None):
            """
            Formats a list of diseases into a user-friendly answer.
            
            Args:
                query: The original user question
                disease_list: List of diseases identified
                confidence: Confidence score (if available)
                
            Returns:
                str: Formatted answer with the disease list
            """
            symptoms_mentioned = self.extract_symptoms_from_query(query)
            symptoms_text = ", ".join(symptoms_mentioned) if symptoms_mentioned else "your symptoms"
            
            # Start with a clinical introduction
            answer = f"Based on {symptoms_text}, the following conditions could potentially be causing your symptoms:\n\n"
            
            # Add each disease with a bullet point
            for i, disease in enumerate(disease_list, 1):
                answer += f"{i}. {disease}\n"
            
            # Add important medical disclaimer
            answer += "\nIMPORTANT: This is not a definitive diagnosis. These are potential conditions that can cause the symptoms you described. "
            answer += "Chest pain and shortness of breath can be symptoms of serious conditions that require immediate medical attention. "
            answer += "Please consult with a healthcare provider as soon as possible for proper evaluation, diagnosis, and treatment."
            
            return answer 
        
    def check_answer_covers_aspects(self, answer, query_aspects):
        """Check if an answer addresses all required aspects of a query"""
        if not answer:
            return False
            
        answer_lower = answer.lower()
        
        # Check for each required aspect
        aspects_coverage = {aspect: False for aspect, required in query_aspects.items() if required}
        
        # Disease identification coverage
        if query_aspects.get("disease_identification"):
            disease_indicators = ["disease", "condition", "diagnosis", "disorder"]
            aspects_coverage["disease_identification"] = any(indicator in answer_lower for indicator in disease_indicators)
        
        # Treatment coverage
        if query_aspects.get("treatment"):
            treatment_indicators = ["treatment", "therapy", "medication", "manage", "drug", "prescription"]
            aspects_coverage["treatment"] = any(indicator in answer_lower for indicator in treatment_indicators)
        
        # Prevention coverage
        if query_aspects.get("prevention"):
            prevention_indicators = ["prevent", "prevention", "avoid", "reduce risk", "lifestyle change"]
            aspects_coverage["prevention"] = any(indicator in answer_lower for indicator in prevention_indicators)
        
        # Causes coverage
        if query_aspects.get("causes"):
            cause_indicators = ["cause", "reason", "etiology", "factor", "lead to", "result from"]
            aspects_coverage["causes"] = any(indicator in answer_lower for indicator in cause_indicators)
        
        # All aspects covered?
        return all(aspects_coverage.values())
    
    def knowledge_graph_agent(self, state: Dict) -> Dict:
        """
        Knowledge Graph Agent
        Extracts symptoms, identifies diseases, and recommends treatments using the knowledge graph
        Implements optimized symptom-first logic with improved validation and separation of concerns
        """
        print("Knowledge Graph Agent - Optimized Symptom-First Logic")
    
        if state.get("halt_execution") == ExecutionState.HALTED:
            return state
    
        new_state = {**state}
        query = state["user_query"].lower()
        subtasks = state.get("subtasks", [])
        subtask_results = state.get("subtask_results", {})
        updated_results = {**subtask_results}
        
        # Extract query aspects for multi-part handling
        query_aspects = self.segment_query(query)
        is_multi_part = sum(1 for aspect in query_aspects.values() if aspect) > 1
        
        if is_multi_part:
            print(f"Multi-part query detected with aspects: {[k for k, v in query_aspects.items() if v]}")
            new_state.update({
                "query_aspects": query_aspects,
                "is_multi_part": True,
                "prevention_info": [],
                "cause_info": []
            })
    
        # 1. Enhanced symptom extraction - always do this first
        symptoms, symptom_conf = self.extract_symptoms(query)
        if symptoms and symptom_conf >= THRESHOLDS["symptom_extraction"]:
            new_state.update({
                "symptoms": symptoms,
                "symptom_confidence": symptom_conf
            })
            print(f"✔️ Primary symptom extraction: {symptoms} (conf: {symptom_conf:.2f})")
    
        # 2. Improved disease identification
        direct_disease, disease_conf = self.extract_disease_from_query(query)
        if direct_disease and disease_conf >= THRESHOLDS["disease_extraction"]:
            new_state.update({
                "direct_disease_mention": direct_disease,
                "direct_disease_confidence": disease_conf,
                "disease": direct_disease,
                "diseases": [direct_disease],
                "disease_confidence": disease_conf,
                "matched_symptoms": new_state.get("symptoms", ["mentioned directly"])
            })
            print(f"✔️ Direct disease mention: {direct_disease} (conf: {disease_conf:.2f})")
        elif new_state.get("symptoms"):
            # Only query diseases if no direct mention exists
            disease_data = self.query_disease_from_symptoms(new_state["symptoms"])
            if disease_data[1] >= THRESHOLDS["disease_matching"]:
                new_state.update({
                    "disease": disease_data[0],
                    "diseases": [disease_data[0]],
                    "disease_confidence": disease_data[1],
                    "matched_symptoms": disease_data[2]
                })
                print(f"✔️ Disease matched from symptoms: {disease_data[0]} (conf: {disease_data[1]:.2f})")
    
        # Process tasks selectively based on identified information
        processed_tasks = 0
        successful_tasks = 0
    
        for task in subtasks:
            if task.get("data_source") != "KG" or task.get("id") in updated_results:
                continue
    
            processed_tasks += 1
            task_id = task.get("id")
            task_query = task.get("subtask_query", task.get("description", ""))
            task_desc = task_query.lower()
    
            try:
                task_result = {
                    "id": task_id,
                    "subtask_query": task_query,
                    "subtask_answer": None,
                    "confidence": 0.0,
                    "status": "failed",
                    "data_source": "KG"
                }
    
                # Symptom extraction task
                if "symptom" in task_desc:
                    if new_state.get("symptoms"):
                        task_result.update({
                            "subtask_answer": ", ".join(new_state["symptoms"]),
                            "confidence": new_state["symptom_confidence"],
                            "status": "completed"
                        })
                        successful_tasks += 1
    
                # Disease identification task
                elif "disease" in task_desc:
                    if new_state.get("disease"):
                        task_result.update({
                            "subtask_answer": new_state["disease"],
                            "confidence": new_state["disease_confidence"],
                            "status": "completed"
                        })
                        successful_tasks += 1
    
                # Treatment/remedy tasks only if disease identified
                elif ("treatment" in task_desc or "remedy" in task_desc) and new_state.get("disease"):
                    disease = new_state.get("disease")
                    if "treatment" in task_desc:
                        treatments, conf = self.query_treatments(disease)
                        if conf >= THRESHOLDS["knowledge_graph"]:
                            task_result.update({
                                "subtask_answer": ", ".join(treatments),
                                "confidence": conf,
                                "status": "completed"
                            })
                            new_state["treatments"] = treatments
                            successful_tasks += 1
                            
                            # Preserve multi-part handling
                            if is_multi_part:
                                new_state["prevention_info"] = [t for t in treatments 
                                    if any(kw in t.lower() for kw in ["prevent", "avoid"])]
                                new_state["cause_info"] = [t for t in treatments 
                                    if any(kw in t.lower() for kw in ["cause", "factor"])]
                    else:  # remedy
                        remedies, conf = self.query_home_remedies(disease)
                        if conf >= THRESHOLDS["knowledge_graph"]:
                            task_result.update({
                                "subtask_answer": ", ".join(remedies),
                                "confidence": conf,
                                "status": "completed"
                            })
                            new_state["home_remedies"] = remedies
                            successful_tasks += 1
    
                updated_results[task_id] = task_result
    
            except Exception as e:
                print(f"⚠️ Error processing task {task_id}: {e}")
                updated_results[task_id] = {
                    "id": task_id,
                    "subtask_query": task_query,
                    "subtask_answer": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "status": "failed",
                    "data_source": "KG"
                }
    
        # Update completion metrics
        completion_rate = successful_tasks / max(processed_tasks, 1)
        new_state.update({
            "subtask_results": updated_results,
            "kg_completion_rate": completion_rate,
            "kg_completeness": self.calculate_kg_completeness(new_state)
        })
    
        # Generate comprehensive answer
        if new_state.get("disease") or new_state.get("symptoms"):
            new_state["kg_answer"] = self.format_kg_diagnostic_answer(new_state)
            new_state["kg_confidence"] = self.calculate_kg_confidence(new_state)
        
        return new_state
        
    def process_with_knowledge_graph(self, user_query: str) -> Dict:
        """Process user query with knowledge graph agent"""
        print(f"Processing with knowledge graph: {user_query}")
        
        # Initialize state
        state = {
            "user_query": user_query,
            "halt_execution": ExecutionState.ACTIVE,
            "subtasks": [],
            "subtask_results": {},
            "kg_answer": "",  # Initialize empty kg_answer
            "kg_confidence": 0.0  # Initialize kg_confidence
        }
        
        # Check if query directly mentions a disease
        extracted_disease, disease_confidence = self.extract_disease_from_query(user_query)
        if extracted_disease and disease_confidence >= 0.7:
            print(f"📊 Query mentions disease: {extracted_disease} (confidence: {disease_confidence:.4f})")
            state["direct_disease_mention"] = extracted_disease
            state["direct_disease_confidence"] = disease_confidence
            
            # Check if it's a treatment query
            if self.is_treatment_query(user_query):
                state["is_treatment_query"] = True
        
        # Extract symptoms directly
        symptoms, symptom_confidence = self.extract_symptoms(user_query)
        if symptoms and symptom_confidence >= 0.5:
            print(f"📊 Extracted symptoms: {symptoms} (confidence: {symptom_confidence:.4f})")
            state["direct_symptom_mention"] = symptoms
            state["direct_symptom_confidence"] = symptom_confidence
        
        # Create subtasks for the knowledge graph
        subtasks = [
            {"id": "kg_symptom", "description": "Extract symptoms from user query", "data_source": "KG"},
            {"id": "kg_disease", "description": "Identify disease from symptoms", "data_source": "KG"},
            {"id": "kg_treatment", "description": "Find treatments for the disease", "data_source": "KG"},
            {"id": "kg_remedy", "description": "Find home remedies for the disease", "data_source": "KG"}
        ]
        state["subtasks"] = subtasks
        
        # Process knowledge graph tasks
        result = self.knowledge_graph_agent(state)
        
        # Ensure we have kg_answer field even if knowledge_graph_agent doesn't set it
        if "kg_answer" not in result:
            result["kg_answer"] = ""
        if "kg_confidence" not in result:
            result["kg_confidence"] = 0.0
        
        # Check for two types of queries:
        
        # 1. Disease identification query
        if self.is_disease_identification_query(user_query) and (result.get("diseases") or result.get("disease")):
            # Format a more comprehensive symptom-to-disease answer
            diseases = result.get("diseases", [])
            if not diseases and result.get("disease"):
                diseases = [result.get("disease")]
                
            disease_confidences = result.get("disease_confidences", [0.0] * len(diseases))
            if len(disease_confidences) < len(diseases):
                disease_confidences = disease_confidences + [0.7] * (len(diseases) - len(disease_confidences))
            
            # Create a more focused disease identification answer
            disease_answer = "# Possible Diseases Based on Symptoms\n\n"
            
            # Add symptoms section
            symptoms = result.get("symptoms", [])
            if symptoms:
                disease_answer += f"## Symptoms Identified\n{', '.join(symptoms)}\n\n"
            
            # Add each disease with details
            for i, disease in enumerate(diseases):
                conf = disease_confidences[i] if i < len(disease_confidences) else 0.7
                disease_answer += f"## {disease}\n"
                disease_answer += f"**Confidence:** {conf:.2f}\n"
                
                # Add matched symptoms for this disease if available
                if result.get("matched_symptoms"):
                    disease_answer += f"**Matching Symptoms:** {', '.join(result.get('matched_symptoms'))}\n\n"
                
                # Add treatments if available
                treatments = result.get("treatments", [])
                if treatments:
                    disease_answer += "**Recommended Treatments:**\n"
                    for treatment in treatments:
                        disease_answer += f"- {treatment}\n"
                    disease_answer += "\n"
                
                # Add home remedies if available
                remedies = result.get("home_remedies", [])
                if remedies:
                    disease_answer += "**Home Remedies:**\n"
                    for remedy in remedies:
                        disease_answer += f"- {remedy}\n"
                    disease_answer += "\n"
            
            # Update the kg_answer with this more focused format
            result["kg_answer"] = disease_answer
            
            # Boost confidence for disease identification queries to maximum
            result["kg_confidence"] = 1.0
            print("✅ Disease identification query detected, formatted comprehensive KG response with maximum confidence")
        
        # 2. Treatment query for specific disease
        elif (result.get("is_treatment_query", False) or self.is_treatment_query(user_query)) and \
             (result.get("direct_disease_mention") or result.get("disease")) and \
             (result.get("treatments") or result.get("home_remedies")):
            
            disease = result.get("direct_disease_mention", result.get("disease", "this condition"))
            
            treatment_answer = f"# Treatment Options for {disease.title()}\n\n"
            
            # Add treatments if available
            if result.get("treatments"):
                treatment_answer += "## Medical Treatments\n"
                for treatment in result.get("treatments", []):
                    treatment_answer += f"- {treatment}\n"
                treatment_answer += "\n"
            
            # Add home remedies if available
            if result.get("home_remedies"):
                treatment_answer += "## Home Remedies\n"
                for remedy in result.get("home_remedies", []):
                    treatment_answer += f"- {remedy}\n"
                treatment_answer += "\n"
            
            # Add general information
            treatment_answer += "## General Information\n"
            treatment_answer += f"Treatment for {disease} typically involves a combination of medical interventions and lifestyle adjustments. "
            treatment_answer += "The specific approach depends on the severity of the condition, the patient's overall health, and other individual factors. "
            treatment_answer += "Always consult a healthcare provider for personalized medical advice.\n"
            
            # Override the existing kg_answer with the treatment-focused answer
            result["kg_answer"] = treatment_answer
            
            # Boost confidence to ensure KG-only route
            result["kg_confidence"] = 1.0
            print(f"✅ Treatment query detected for {disease}, formatted response with maximum confidence")
        
        return result
                    
    def reflection_agent(self, user_query, kg_answer, rag_answer, kg_confidence=None):
        """
        Comprehensive reflection agent that analyzes and combines knowledge sources
        to produce the optimal answer for each query type.
        
        Routes: KG-only, RAG-only, LLM-only, KG+RAG, KG+LLM, RAG+LLM, KG+RAG+LLM
        """
        # Start detailed logging
        print(f"\n===== REFLECTION AGENT ANALYSIS =====")
        print(f"Query: {user_query}")
        
        # 1. QUERY ANALYSIS
        # Identify query aspects
        query_aspects = self.segment_query(user_query)
        is_multi_part = sum(1 for aspect in query_aspects.values() if aspect) > 1
        
        if is_multi_part:
            print(f"Multi-part query detected with aspects: {[k for k, v in query_aspects.items() if v]}")
        
        is_disease_identification = self.is_disease_identification_query(user_query)
        is_treatment_query = self.is_treatment_query(user_query)
        
        print(f"Query types: Disease identification: {is_disease_identification}, Treatment: {is_treatment_query}")
        print(f"KG Confidence: {kg_confidence:.4f}" if kg_confidence is not None else "KG Confidence: None")
        
        # Extract diseases for disease-based queries
        if not self.identified_diseases and kg_answer:
            self.extract_diseases_from_kg_answer(kg_answer)
        
        # 2. THRESHOLD DEFINITIONS
        # Define thresholds for different decisions
        KG_HIGH_THRESHOLD = 0.70    # Extremely high confidence for KG
        KG_MEDIUM_THRESHOLD = 0.60  # Good confidence for KG
        COMPLETENESS_THRESHOLD = 0.50  # Basic threshold for answer completeness
        
        # 3. INDIVIDUAL SOURCE EVALUATION
        # Evaluate the completeness of each individual source
        print("\n--- Individual Source Evaluation ---")
        kg_completeness = self.evaluate_answer_completeness(user_query, kg_answer)
        rag_completeness = self.evaluate_answer_completeness(user_query, rag_answer)
        
        print(f"KG completeness: {kg_completeness:.4f}")
        print(f"RAG completeness: {rag_completeness:.4f}")
        
        # Check if individual sources cover all aspects for multi-part queries
        kg_covers_all_aspects = self.check_answer_covers_aspects(kg_answer, query_aspects) if is_multi_part else True
        rag_covers_all_aspects = self.check_answer_covers_aspects(rag_answer, query_aspects) if is_multi_part else True
        
        print(f"KG covers all aspects: {kg_covers_all_aspects}")
        print(f"RAG covers all aspects: {rag_covers_all_aspects}")
        
        # Identify what's missing from each source
        kg_missing = self.identify_missing_elements(user_query, kg_answer)
        rag_missing = self.identify_missing_elements(user_query, rag_answer)
        
        if kg_missing:
            print(f"KG missing elements: {kg_missing}")
        if rag_missing:
            print(f"RAG missing elements: {rag_missing}")
        
        # Initialize source contributions tracking
        source_contributions = {
            "KG": False,
            "RAG": False,
            "LLM": False
        }
        
        # CRITICAL CHECK: Validate KG extraction success
        has_valid_kg_data = False
        if kg_answer and hasattr(self, 'identified_diseases') and self.identified_diseases:
            has_valid_kg_data = True
            print(f"✅ KG extraction validated with identified diseases: {self.identified_diseases}")
        elif kg_answer and "Disease:" in kg_answer or "Possible Diseases:" in kg_answer:
            has_valid_kg_data = True
            print(f"✅ KG extraction validated with disease information in answer")
        else:
            print(f"⚠️ KG extraction validation failed - no valid disease data found")
        
        # 4. DECISION TREE FOR SOURCE SELECTION
        # Follows priority order: KG > RAG > Combined > LLM
        
        # 4.1 KG-ONLY ROUTE (Highest Priority)
        # Cases where we should use KG-only
        if kg_answer and has_valid_kg_data:
            # Case 1: Very high confidence KG answers
            if kg_confidence is not None and kg_confidence >= KG_HIGH_THRESHOLD and kg_covers_all_aspects:
                print("Reflection decision: KG_ONLY - Very high confidence KG answer")
                source_contributions["KG"] = True
                return kg_answer, "KG_ONLY", source_contributions
            
            # Case 2: Disease identification queries with good KG results
            if is_disease_identification and (
                "Possible Diseases:" in kg_answer or 
                "Disease:" in kg_answer or 
                "# Possible Diagnoses" in kg_answer or
                hasattr(self, 'identified_diseases') and self.identified_diseases
            ):
                # For multi-part queries, check if KG covers all aspects
                if not is_multi_part or (is_multi_part and kg_covers_all_aspects):
                    print("Reflection decision: KG_ONLY - Disease identification query with KG results")
                    source_contributions["KG"] = True
                    return kg_answer, "KG_ONLY", source_contributions
            
            # Case 3: Treatment queries with good KG results
            if is_treatment_query and (
                "Treatment Options for" in kg_answer or 
                "Medical Treatments" in kg_answer or 
                "Home Remedies" in kg_answer or
                "Treatments:" in kg_answer
            ):
                # For multi-part queries, check if KG covers all aspects
                if not is_multi_part or (is_multi_part and kg_covers_all_aspects):
                    print("Reflection decision: KG_ONLY - Treatment query with KG results")
                    source_contributions["KG"] = True
                    return kg_answer, "KG_ONLY", source_contributions
            
            # Case 4: Medium confidence KG answers that are complete
            if kg_confidence is not None and kg_confidence >= KG_MEDIUM_THRESHOLD and kg_completeness >= COMPLETENESS_THRESHOLD:
                print(f"Reflection decision: KG_ONLY - Good confidence KG answer ({kg_confidence:.4f})")
                source_contributions["KG"] = True
                return kg_answer, "KG_ONLY", source_contributions
        
        # 4.2 RAG-ONLY ROUTE (Second Priority)
        # When RAG provides a highly complete answer
        if rag_answer and rag_completeness >= COMPLETENESS_THRESHOLD and rag_covers_all_aspects:
            print("Reflection decision: RAG_ONLY - RAG answer is complete")
            source_contributions["RAG"] = True
            return rag_answer, "RAG_ONLY", source_contributions
        
        # 4.3 COMBINED SOURCE ROUTES
        print("\n--- Combination Evaluation ---")
        # First try combining KG and RAG
        if kg_answer and rag_answer:
            kg_rag_combined = self.combine_answers(kg_answer, rag_answer, kg_missing)
            kg_rag_completeness = self.evaluate_answer_completeness(user_query, kg_rag_combined)
            print(f"KG+RAG completeness: {kg_rag_completeness:.4f}")
            
            if kg_rag_completeness >= COMPLETENESS_THRESHOLD:
                print("Reflection decision: KG_RAG_COMBINED - Combined KG and RAG answer is complete")
                source_contributions["KG"] = True
                source_contributions["RAG"] = True
                return kg_rag_combined, "KG_RAG_COMBINED", source_contributions
        
        # 4.4 LLM ROUTES
        # Generate an LLM answer focused on the missing elements
        print("\n--- LLM Generation ---")
        llm_answer = self.generate_llm_answer(user_query, kg_missing, rag_missing)
        llm_completeness = self.evaluate_answer_completeness(user_query, llm_answer)
        print(f"LLM completeness: {llm_completeness:.4f}")
        
        # Check if LLM alone is sufficient
        if llm_completeness >= COMPLETENESS_THRESHOLD:
            print("Reflection decision: LLM_ONLY - LLM answer is complete")
            source_contributions["LLM"] = True
            return llm_answer, "LLM_ONLY", source_contributions
        
        # Try KG + LLM combination
        if kg_answer:
            kg_llm_combined = self.combine_answers(kg_answer, llm_answer, kg_missing)
            kg_llm_completeness = self.evaluate_answer_completeness(user_query, kg_llm_combined)
            print(f"KG+LLM completeness: {kg_llm_completeness:.4f}")
            
            if kg_llm_completeness >= COMPLETENESS_THRESHOLD:
                print("Reflection decision: KG_LLM_COMBINED - Combined KG and LLM answer is complete")
                source_contributions["KG"] = True
                source_contributions["LLM"] = True
                return kg_llm_combined, "KG_LLM_COMBINED", source_contributions
        
        # Try RAG + LLM combination
        if rag_answer:
            rag_llm_combined = self.combine_answers(rag_answer, llm_answer, rag_missing)
            rag_llm_completeness = self.evaluate_answer_completeness(user_query, rag_llm_combined)
            print(f"RAG+LLM completeness: {rag_llm_completeness:.4f}")
            
            if rag_llm_completeness >= COMPLETENESS_THRESHOLD:
                print("Reflection decision: RAG_LLM_COMBINED - Combined RAG and LLM answer is complete")
                source_contributions["RAG"] = True
                source_contributions["LLM"] = True
                return rag_llm_combined, "RAG_LLM_COMBINED", source_contributions
        
        # Try KG + RAG + LLM combination (all three sources)
        if kg_answer and rag_answer:
            all_combined = self.combine_all_answers(kg_answer, rag_answer, llm_answer)
            all_combined_completeness = self.evaluate_answer_completeness(user_query, all_combined)
            print(f"All sources combined completeness: {all_combined_completeness:.4f}")
            
            if all_combined_completeness >= COMPLETENESS_THRESHOLD:
                print("Reflection decision: KG_RAG_LLM_COMBINED - All sources combined answer is complete")
                source_contributions["KG"] = True
                source_contributions["RAG"] = True
                source_contributions["LLM"] = True
                return all_combined, "KG_RAG_LLM_COMBINED", source_contributions
        
        # 5. FALLBACK TO BEST AVAILABLE
        # If no combination meets the threshold, choose the highest scoring option
        print("\n--- Fallback Selection ---")
        best_completeness = max(
            kg_completeness if kg_answer else 0.0,
            rag_completeness if rag_answer else 0.0,
            llm_completeness,
            kg_rag_completeness if 'kg_rag_completeness' in locals() else 0.0,
            kg_llm_completeness if 'kg_llm_completeness' in locals() else 0.0,
            rag_llm_completeness if 'rag_llm_completeness' in locals() else 0.0,
            all_combined_completeness if 'all_combined_completeness' in locals() else 0.0
        )
        
        print(f"Best completeness: {best_completeness:.4f}")
        
        # Return the option with the highest completeness score
        if kg_answer and best_completeness == kg_completeness and has_valid_kg_data:
            print("Reflection decision: KG_ONLY - Best available answer")
            source_contributions["KG"] = True
            return kg_answer, "KG_ONLY", source_contributions
        elif rag_answer and best_completeness == rag_completeness:
            print("Reflection decision: RAG_ONLY - Best available answer")
            source_contributions["RAG"] = True
            return rag_answer, "RAG_ONLY", source_contributions
        elif best_completeness == llm_completeness:
            print("Reflection decision: LLM_ONLY - Best available answer")
            source_contributions["LLM"] = True
            return llm_answer, "LLM_ONLY", source_contributions
        elif 'kg_rag_completeness' in locals() and best_completeness == kg_rag_completeness:
            print("Reflection decision: KG_RAG_COMBINED - Best available answer")
            source_contributions["KG"] = True
            source_contributions["RAG"] = True
            return kg_rag_combined, "KG_RAG_COMBINED", source_contributions
        elif 'kg_llm_completeness' in locals() and best_completeness == kg_llm_completeness:
            print("Reflection decision: KG_LLM_COMBINED - Best available answer")
            source_contributions["KG"] = True
            source_contributions["LLM"] = True
            return kg_llm_combined, "KG_LLM_COMBINED", source_contributions
        elif 'rag_llm_completeness' in locals() and best_completeness == rag_llm_completeness:
            print("Reflection decision: RAG_LLM_COMBINED - Best available answer")
            source_contributions["RAG"] = True
            source_contributions["LLM"] = True
            return rag_llm_combined, "RAG_LLM_COMBINED", source_contributions
        else:
            print("Reflection decision: KG_RAG_LLM_COMBINED - Best available answer")
            source_contributions["KG"] = True
            source_contributions["RAG"] = True
            source_contributions["LLM"] = True
            return all_combined, "KG_RAG_LLM_COMBINED", source_contributions
            
    def segment_query(self, user_query: str) -> Dict[str, bool]:
        """Identify different aspects of the query"""
        query_lower = user_query.lower()
        
        # Initialize query aspects
        aspects = {
            "disease_identification": False,
            "treatment": False,
            "prevention": False,
            "causes": False,
            "symptoms": False
        }
        
        # Check for disease identification aspect
        if self.is_disease_identification_query(query_lower):
            aspects["disease_identification"] = True
        
        # Check for treatment aspect
        if self.is_treatment_query(query_lower):
            aspects["treatment"] = True
        
        # Check for prevention aspect
        prevention_keywords = [
            "prevent", "prevention", "avoid", "reducing risk", "risk factor", 
            "how to stop", "how to reduce", "protective", "preventative", "prophylaxis",
            "lifestyle change", "primary prevention", "secondary prevention", "risk reduction"
        ]
        if any(keyword in query_lower for keyword in prevention_keywords):
            aspects["prevention"] = True
        
        # Check for causes aspect
        cause_keywords = [
            "cause", "causing", "reason for", "etiology", "why", 
            "what leads to", "pathophysiology", "mechanism", "origin"
        ]
        if any(keyword in query_lower for keyword in cause_keywords):
            aspects["causes"] = True
        
        # Check for symptom description
        symptom_keywords = [
            "symptom", "sign", "indication", "manifestation", 
            "feel", "feeling", "experiencing", "having", "suffering from"
        ]
        if any(keyword in query_lower for keyword in symptom_keywords):
            aspects["symptoms"] = True
        
        return aspects
    
    def evaluate_answer_completeness(self, query, answer, llm_client=None):
        """
        Evaluates how completely an answer addresses all aspects of the user query.
        """
        if not answer or answer.strip() == "":
            return 0.0
        
        # Identify query aspects
        query_aspects = self.segment_query(query)
        
        # Create a more specific evaluation prompt
        prompt = f"""
        You are evaluating how completely a medical answer addresses a multi-part health-related question.
        
        USER QUESTION: {query}
        
        Query aspects that need to be addressed:
        - Disease identification: {'Yes' if query_aspects['disease_identification'] else 'No'}
        - Treatment information: {'Yes' if query_aspects['treatment'] else 'No'}
        - Prevention strategies: {'Yes' if query_aspects['prevention'] else 'No'}
        - Cause/etiology information: {'Yes' if query_aspects['causes'] else 'No'}
        - Symptom information: {'Yes' if query_aspects['symptoms'] else 'No'}
        
        ANSWER TO EVALUATE: {answer}
        
        Evaluation Guidelines:
        - Check if EACH aspect of the query is addressed thoroughly
        - A complete answer should address ALL aspects marked as 'Yes' above
        - If any aspect is completely missing, the score should be significantly reduced
        - Factual correctness is essential for a high score
        
        On a scale of 0.0 to 1.0, how completely does this answer address ALL aspects of the user's question?
        0.0: Does not address the question at all
        0.3: Addresses only a minor portion of the question's aspects
        0.5: Addresses roughly half of the important aspects
        0.7: Addresses most aspects but may be missing some details on one aspect
        0.9: Addresses all aspects thoroughly
        1.0: Addresses all aspects perfectly with comprehensive information
        
        Provide only a single number as your response.
        """
        
        try:
            response_text = self.local_generate(prompt, max_tokens=100)
            
            # Extract just the number using regex
            import re
            score_match = re.search(r'([0-9]*\.?[0-9]+)', response_text)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0.0 and 1.0
            return 0.5  # Default to middle score if parsing fails
        except Exception as e:
            print(f"Error evaluating answer completeness: {e}")
            return 0.5  # Default to middle score on error
    
    def identify_missing_elements(self, query, answer, llm_client=None):
        """
        Identifies what aspects of the query are not addressed in the answer.
        
        Args:
            query: The original user question
            answer: The current answer
            llm_client: Optional LLM client (not used in this implementation)
            
        Returns:
            list: List of missing elements or aspects
        """
        if not answer or answer.strip() == "":
            return ["The entire query is unanswered"]
        
        # Improved prompt with medical domain-specific guidance
        prompt = f"""
        You are analyzing a medical answer to identify what aspects of the user's question remain unaddressed.
        
        USER QUESTION: {query}
        
        CURRENT ANSWER: {answer}
        
        Medical Answer Evaluation Guidelines:
        - If the answer comes from a Knowledge Graph and contains structured medical information, it should be considered highly reliable
        - Focus on clinically significant missing information, not minor details
        - For disease questions: core symptoms, diagnostic criteria, and primary treatments are essential
        - For medication questions: main uses, dosage ranges, and major side effects are essential
        - For procedural questions: purpose, general process, and recovery expectations are essential
        - If the answer states limitations clearly (e.g., "consult a doctor for specific advice"), this is appropriate
        - If the answer provides the main information but lacks minor details, consider it mostly complete
        
        What specific aspects or parts of the user's question are NOT adequately addressed in this answer?
        List each CLINICALLY SIGNIFICANT missing element in a separate line. 
        If the answer covers all essential medical information, respond with "COMPLETE".
        """
        
        try:
            response_text = self.local_generate(prompt, max_tokens=500)
            
            if "COMPLETE" in response_text:
                return []
            
            # Parse the missing elements (one per line)
            missing_elements = [line.strip() for line in response_text.split('\n') if line.strip()]
            return missing_elements
        except Exception as e:
            print(f"Error identifying missing elements: {e}")
            return ["Unable to identify specific missing elements"]
                      
    def is_kg_answer_high_quality(self, user_query, kg_answer, kg_confidence=None):
        """
        Evaluates if the KG answer is high quality enough to use on its own.
        This function acts as an additional check focused specifically on KG answer quality.
        
        Args:
            user_query: The original user question
            kg_answer: Answer from Knowledge Graph
            kg_confidence: Optional confidence score from KG
            
        Returns:
            bool: True if the KG answer is high quality, False otherwise
        """
        # Add detailed logging
        print(f"Evaluating KG answer quality for query: {user_query}")
        
        # Handle various input formats for kg_answer
        content = kg_answer
        confidence = kg_confidence
        symptoms = []
        causes = []
        
        if isinstance(kg_answer, dict):
            content = kg_answer.get('content', kg_answer)
            confidence = kg_answer.get('confidence', kg_confidence)
            symptoms = kg_answer.get('symptoms', [])
            causes = kg_answer.get('causes', [])
        
        # Check if content is empty
        if not content or (isinstance(content, str) and content.strip() == ""):
            print("KG answer rejected: Empty content")
            return False
            
        # Log KG metrics 
        print(f"KG evaluation - Confidence: {confidence}, Symptoms: {len(symptoms)}, Causes: {len(causes)}")
        
        # Improvement #2: Adjusted KG Answer Quality Checks - Reduce threshold to 0.7
        QUALITY_THRESHOLD = 0.7  # Reduced from 0.9
        
        # If confidence score is available and high, consider it high quality
        if confidence is not None and confidence >= QUALITY_THRESHOLD:
            print(f"KG answer accepted: High confidence score {confidence}")
            return True
            
        # Improvement #2: Check for at least 1 symptom OR 1 cause
        if len(symptoms) >= 1 or len(causes) >= 1:
            print(f"KG answer accepted: Has sufficient symptoms ({len(symptoms)}) or causes ({len(causes)})")
            return True
        
        # Check for indicators of a high-quality KG answer
        indicators = [
            "According to medical knowledge",
            "Clinical guidelines recommend",
            "Medical research indicates",
            "Symptoms include",
            "Treatment options include",
            "Diagnostic criteria include"
        ]
        
        # Count how many quality indicators are present
        indicator_count = sum(1 for indicator in indicators if isinstance(content, str) and indicator.lower() in content.lower())
        
        # Check for structured information patterns
        has_structured_info = False
        if isinstance(content, str):
            has_structured_info = any([
                ":" in content,  # Key-value pairs
                "\n-" in content,  # Bulleted lists
                "1." in content,   # Numbered lists
                "•" in content     # Bullet points
            ])
        
        # Create a prompt to evaluate KG quality
        if isinstance(content, str):
            prompt = f"""
            You are evaluating whether a Knowledge Graph-based medical answer is high quality enough to use on its own.
            
            USER QUESTION: {user_query}
            
            KG ANSWER: {content}
            
            A high-quality Knowledge Graph answer should:
            1. Provide specific, factual medical information directly relevant to the query
            2. Present information in a structured, organized manner
            3. Include specifics like symptoms, causes, treatments, or definitions as appropriate
            4. Not contain hedging language like "I'm not sure" or "I think"
            5. Present information authoritatively based on medical knowledge
            
            Is this KG answer high quality enough to use on its own? Respond with only "YES" or "NO".
            """
            
            try:
                response_text = self.local_generate(prompt, max_tokens=50)
                is_high_quality = "YES" in response_text.upper()
                
                # Improvement #2: Relaxed quality conditions
                # If the answer has multiple quality indicators OR structured information AND the LLM says it's high quality
                result = (indicator_count >= 1 or has_structured_info) and is_high_quality  # Reduced from indicator_count >= 2
                
                if result:
                    print(f"KG answer accepted: Quality indicators: {indicator_count}, Structured: {has_structured_info}, LLM: {is_high_quality}")
                else:
                    print(f"KG answer rejected: Quality indicators: {indicator_count}, Structured: {has_structured_info}, LLM: {is_high_quality}")
                
                return result
            except Exception as e:
                print(f"Error evaluating KG answer quality: {e}")
                return False
        else:
            # If content is not a string and we can't evaluate with the LLM,
            # fall back to checking for symptoms or causes
            print("KG answer evaluated using fallback method")
            return len(symptoms) >= 1 or len(causes) >= 1
                       
    def combine_answers(self, primary_answer, secondary_answer, missing_elements=None, prioritize_kg=False, llm_client=None):
        """
        Intelligently combines two answers to provide a more complete response.
        
        Args:
            primary_answer: The answer from the primary/preferred source
            secondary_answer: The answer from the secondary source
            missing_elements: Optional list of elements missing from primary answer
            prioritize_kg: Boolean flag indicating if KG should be prioritized
            llm_client: Optional LLM client for more sophisticated combining
            
        Returns:
            str: Combined answer with proper attribution and flow
        """
        # Log the combination process
        print(f"Combining answers with prioritize_kg={prioritize_kg}")
        
        # Handle the case when primary_answer is None or empty
        if not primary_answer or (isinstance(primary_answer, str) and primary_answer.strip() == ""):
            return secondary_answer
        
        # Handle the case when secondary_answer is None or empty
        if not secondary_answer or (isinstance(secondary_answer, str) and secondary_answer.strip() == ""):
            return primary_answer
    
        # If we're prioritizing KG (improvement #4)
        if prioritize_kg:
            # Create a formatted answer that clearly prioritizes KG content
            if isinstance(primary_answer, dict):
                primary_content = primary_answer.get('content', str(primary_answer))
            else:
                primary_content = primary_answer
                
            if isinstance(secondary_answer, dict):
                secondary_content = secondary_answer.get('content', str(secondary_answer))
            else:
                secondary_content = secondary_answer
                
            # Format the combined answer to emphasize KG content
            formatted_answer = f"""
    {primary_content}
    
    **Additional Information:**
    {secondary_content}
    """
            return formatted_answer.strip()
            
        missing_elements_text = ""
        if missing_elements and len(missing_elements) > 0:
            missing_elements_text = "The primary answer is missing these elements: " + ", ".join(missing_elements)
        
        prompt = f"""
        You are creating a comprehensive medical answer by combining information from multiple sources.
        
        PRIMARY ANSWER (SOURCE A): {primary_answer}
        
        SECONDARY ANSWER (SOURCE B): {secondary_answer}
        
        {missing_elements_text}
        
        Create a unified answer that:
        1. Preserves all relevant information from both sources
        2. Eliminates redundancy
        3. Maintains a coherent, flowing narrative
        4. Ensures all medical information is accurate
        5. Uses clear and patient-friendly language
        
        IMPORTANT: In your answer, DO NOT explicitly label which parts came from which source.
        However, ensure the combined answer retains the key elements from each source.
        """
        
        try:
            combined = self.local_generate(prompt, max_tokens=1000)
            return combined.strip()
        except Exception as e:
            print(f"Error combining answers: {e}")
            # Fallback: Simple concatenation with separator
            return f"{primary_answer}\n\nAdditional information:\n{secondary_answer}"
    
    def combine_all_answers(self, kg_answer, rag_answer, llm_answer, llm_client=None):
        """
        Combines answers from all three sources into a single comprehensive response.
        
        Args:
            kg_answer: Answer from Knowledge Graph
            rag_answer: Answer from RAG
            llm_answer: Answer from LLM
            llm_client: Optional LLM client (not used in this implementation)
            
        Returns:
            str: Fully combined answer
        """
        # First combine KG and RAG
        kg_rag_combined = self.combine_answers(kg_answer, rag_answer, llm_client=llm_client)
        return self.combine_answers(kg_rag_combined, llm_answer, llm_client=llm_client)
    
    def generate_followup_request(self, kg_missing=None, rag_missing=None):
        """
        Generates a request for the user to provide more information.
        
        Args:
            kg_missing: Elements missing from KG answer
            rag_missing: Elements missing from RAG answer
            
        Returns:
            str: Follow-up request message
        """
        # Combine missing elements
        missing_elements = set()
        if kg_missing:
            missing_elements.update(kg_missing)
        if rag_missing:
            missing_elements.update(rag_missing)
        
        base_message = "To provide you with better guidance, I need some additional information."
        
        if missing_elements:
            specifics = "\n\nSpecifically, could you please provide more details about:\n"
            for element in missing_elements:
                specifics += f"- {element}\n"
            
            return base_message + specifics
        else:
            return base_message + "\n\nCould you provide more context about your symptoms, when they started, and any other medical conditions or medications you're taking?"
                            
    def generate_response(self, user_input, user_type="User / Family"):
        """
        Generate response using reflection agent to evaluate and combine multiple knowledge sources.
        """
        if not user_input.strip():
            return "", []
    
        if self.qa_chain is None:
            success, message = self.initialize_qa_chain()
            if not success:
                return message, []
    
        try:
            print(f"\n🔍 Processing query: {user_input}")
            
            # Check if this is a response to a follow-up question
            if self.followup_context["asked"]:
                print("🔍 User responded to follow-up question")
    
            # Check if we need follow-up questions
            missing_info = self.identify_missing_info(user_input, self.chat_history)
    
            # If critical information is missing, ask follow-up questions
            if missing_info and len(missing_info) > 0:
                follow_up_prompt = f"""
                I need a bit more information to provide a helpful response. Could you please tell me:
    
                {missing_info[0]}
    
                This will help me give you a more accurate assessment.
                """
                self.chat_history.append((user_input, follow_up_prompt))
                return follow_up_prompt.strip(), []
    
            # Reset follow-up context for next query
            self.followup_context = {"asked": False, "round": 0}
    
            # Process with Knowledge Graph
            print("📚 Processing with Knowledge Graph...")
            t_start = datetime.now()
            
            # IMPORTANT: Set identified_diseases to empty list before processing
            self.identified_diseases = []
            
            kg_data = self.process_with_knowledge_graph(user_input)
            kg_content = kg_data.get("kg_answer", "")
            kg_confidence = kg_data.get("kg_confidence", 0.0)
            
            # Capture key KG extraction results for validation
            disease = kg_data.get("disease", "")
            diseases = kg_data.get("diseases", [])
            symptoms = kg_data.get("symptoms", [])
            treatments = kg_data.get("treatments", [])
            home_remedies = kg_data.get("home_remedies", [])
            
            # Log key KG extraction metrics
            print(f"📊 KG Extraction Results:")
            print(f"   Diseases: {diseases if diseases else disease if disease else 'None'}")
            print(f"   Symptoms: {symptoms if symptoms else 'None'}")
            print(f"   Treatments: {treatments if treatments else 'None'}")
            print(f"   KG Confidence: {kg_confidence:.4f} (took {(datetime.now() - t_start).total_seconds():.2f}s)")
            
            # Process with RAG
            print("📚 Processing with RAG...")
            t_start = datetime.now()
            formatted_history = self.format_chat_history()
            
            # Get RAG response
            rag_response = self.qa_chain.invoke({
                "question": user_input,
                "chat_history": formatted_history
            })
            
            rag_content = rag_response["answer"]
            if "Helpful Answer:" in rag_content:
                rag_content = rag_content.split("Helpful Answer:")[-1]
            
            # Extract RAG sources
            rag_sources = []
            source_texts = []
            for doc in rag_response["source_documents"][:3]:
                source_text = doc.page_content
                source_texts.append(source_text)
                
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    source_name = doc.metadata["source"]
                    if "dxbook" in source_name.lower() or "rawdata.pdf" in source_name:
                        page_num = doc.metadata.get("page", "N/A")
                        rag_sources.append(f"Internal Data: DxBook, Page {page_num} - {source_text}")
                    else:
                        rag_sources.append(source_text)
                else:
                    rag_sources.append(source_text)
            
            # Calculate RAG confidence based on relevance of retrieved documents
            rag_relevance_scores = []
            for text in source_texts:
                # Simple keyword matching
                query_keywords = set(word.lower() for word in user_input.split() if len(word) > 3)
                content_lower = text.lower()
                matches = sum(1 for kw in query_keywords if kw in content_lower)
                score = min(matches / max(len(query_keywords), 1), 1.0)
                rag_relevance_scores.append(score)
            
            # Average relevance score
            rag_confidence = sum(rag_relevance_scores) / max(len(rag_relevance_scores), 1)
            print(f"📊 RAG Confidence: {rag_confidence:.4f} (took {(datetime.now() - t_start).total_seconds():.2f}s)")
            
            # Log confidence scores and query
            print(f"📊 Orchestration Decision:")
            print(f"   Query: {user_input}")
            print(f"   KG Confidence: {kg_confidence:.4f}")
            print(f"   RAG Confidence: {rag_confidence:.4f}")
            
            # VALIDATE KG DATA
            # Check if meaningful KG data was extracted
            has_kg_data = False
            if kg_data.get("disease") or kg_data.get("diseases") or kg_data.get("symptoms") or self.identified_diseases:
                has_kg_data = True
                print("✅ KG data validation passed - found useful content")
            else:
                print("⚠️ KG data validation failed - no meaningful content extracted")
                # If KG has no meaningful data but confidence is still high (which happens with some bugs),
                # reset it to indicate that KG processing was not useful
                if kg_confidence > 0.3:
                    print(f"⚠️ Resetting KG confidence from {kg_confidence:.4f} to 0.0 due to lack of data")
                    kg_confidence = 0.0
                    
            # Use the reflection agent to evaluate and combine answers
            print("🧠 Using reflection agent to evaluate and combine answers...")
            t_start = datetime.now()
            
            # Use the reflection agent to evaluate and combine answers
            final_answer, strategy_used, source_contributions = self.reflection_agent(user_input, kg_content, rag_content, kg_confidence)
            
            print(f"📊 Reflection Strategy: {strategy_used}")
            print(f"✅ Response generated (took {(datetime.now() - t_start).total_seconds():.2f}s)")
            
            # Format source information including the routing strategy
            routing_info = f"Route: {strategy_used}"
            
            # Check the actual source contributions and update if needed
            # This adds an additional validation step
            if "KG_ONLY" in strategy_used and not has_kg_data:
                # Strategy claims KG but no KG data found - correct it
                print("⚠️ Strategy correction: KG route selected but no KG data found - adjusting to LLM_ONLY")
                strategy_used = "LLM_ONLY"
                routing_info = f"Route: {strategy_used}"
                source_contributions = {"KG": False, "RAG": False, "LLM": True}
            
            # Prepare to collect appropriate sources for the reference section
            formatted_sources = []
            
            # Add sources based on the contribution flags
            if source_contributions["KG"]:
                # Add KG sources
                if disease:
                    formatted_sources.append(f"- Knowledge Graph: Used for disease identification ({disease})")
                elif diseases:
                    formatted_sources.append(f"- Knowledge Graph: Used for disease identification ({', '.join(diseases)})")
                elif symptoms:
                    formatted_sources.append(f"- Knowledge Graph: Used for symptom analysis ({', '.join(symptoms)})")
                else:
                    formatted_sources.append("- Knowledge Graph: Used for medical entity extraction")
            
            if source_contributions["RAG"]:
                # Add RAG document sources
                formatted_sources.append("- Document Retrieval: Used for detailed medical information")
                
                # Include the specific document references
                for i, src in enumerate(rag_sources, 1):
                    # Format the source reference nicely
                    if "Internal Data:" in src:
                        # Extract just the page number and a snippet
                        page_match = re.search(r'Page (\d+)', src)
                        page_num = page_match.group(1) if page_match else "N/A"
                        
                        # Limit the text snippet to a reasonable length
                        content = src.split(" - ", 1)[1] if " - " in src else src
                        content = content[:100] + "..." if len(content) > 100 else content
                        
                        formatted_sources.append(f"  • Internal Data: DxBook, Page {page_num}")
                    else:
                        # External source or unknown format
                        formatted_sources.append(f"  • {src[:100]}...")
            
            if source_contributions["LLM"]:
                # Add LLM contribution
                formatted_sources.append("- Medical Knowledge: Used to supplement or synthesize information not found in structured sources")
            
            # Construct references section
            references = "\n\n## References:\n"
            references += f"{routing_info}\n"
            for src in formatted_sources:
                references += f"{src}\n"
            
            # Add disclaimer
            disclaimer = "\n\nThis information is not a substitute for professional medical advice. If symptoms persist or worsen, please consult with a qualified healthcare provider."
            
            # Check if response already has a references section
            if "## References:" in final_answer:
                # Replace existing references section
                parts = final_answer.split("## References:")
                final_response = parts[0].strip() + references + disclaimer
            else:
                # Add new references section
                final_response = final_answer.strip() + references + disclaimer
            
            # Log the orchestration decision for analysis
            self.log_orchestration_decision(
                user_input,
                f"SELECTED_STRATEGY: {strategy_used}\nREASONING: Determined by reflection agent based on answer completeness evaluation.\nSOURCES: {','.join(k for k, v in source_contributions.items() if v)}\nRESPONSE: {final_answer[:100]}...",
                kg_confidence,
                rag_confidence
            )
            self.last_strategy = strategy_used  # Store for testing
            
            # Add to chat history
            self.chat_history.append((user_input, final_response))
            
            # Collect all sources for reference
            all_sources = []
            if source_contributions["KG"]:
                all_sources.append(f"[KG] Knowledge Graph: {disease if disease else ', '.join(diseases) if diseases else 'Medical Entity Analysis'}")
            all_sources.extend(rag_sources)
            
            # Return response and sources
            return final_response, all_sources
            
        except Exception as e:
            import traceback
            print(f"⚠️ Error in generate_response: {e}")
            print(traceback.format_exc())
            return f"Error generating response: {str(e)}", []
            
    def log_response_metrics(self, metrics):
        """Log response generation metrics to CSV for analysis."""
        try:
            log_file = "response_metrics.csv"
            file_exists = os.path.isfile(log_file)
            
            metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(log_file, mode='a', newline='', encoding='utf-8') as file:
                fieldnames = ['timestamp', 'query', 'kg_confidence', 'rag_confidence', 
                              'strategy', 'response_length', 'processing_time', 'source_count']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(metrics)
                
        except Exception as e:
            print(f"⚠️ Error logging response metrics: {e}")
                            
    def log_orchestration_decision(self, query, orchestration_result, kg_confidence, rag_confidence):
        """
        Analyzes and logs the orchestration decision for monitoring and improvement.
        Extracts the strategy and reasoning from the orchestration result.
        """
        try:
            # Extract strategy and reasoning
            strategy = "UNKNOWN"
            reasoning = "Not provided"
            sources_used = []
            
            if "SELECTED_STRATEGY:" in orchestration_result:
                strategy_part = orchestration_result.split("SELECTED_STRATEGY:")[1].split("\n")[0].strip()
                strategy = strategy_part
                
            if "REASONING:" in orchestration_result:
                reasoning_parts = orchestration_result.split("REASONING:")[1].split("SOURCES:")[0].strip()
                reasoning = reasoning_parts.strip()
                
            if "SOURCES:" in orchestration_result:
                sources_part = orchestration_result.split("SOURCES:")[1].split("RESPONSE:")[0].strip()
                sources_used = sources_part.split(",")
            
            # Log the decision
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "strategy": strategy,
                "reasoning": reasoning,
                "kg_confidence": kg_confidence,
                "rag_confidence": rag_confidence,
                "kg_used": "KG" in sources_used,
                "rag_used": "RAG" in sources_used,
                "llm_used": "LLM" in sources_used
            }
            
            # Print logging information
            print(f"📊 Orchestration Decision:")
            print(f"   Query: {query}")
            print(f"   Strategy: {strategy}")
            print(f"   Sources: {sources_used}")
            print(f"   KG Confidence: {kg_confidence:.4f}")
            print(f"   RAG Confidence: {rag_confidence:.4f}")
            print(f"   Reasoning: {reasoning}")
            
            # Save to CSV file for analysis
            log_file = "orchestration_log.csv"
            file_exists = os.path.isfile(log_file)
            
            with open(log_file, mode='a', newline='', encoding='utf-8') as file:
                fieldnames = ['timestamp', 'query', 'strategy', 'reasoning', 'kg_confidence', 'rag_confidence', 
                             'kg_used', 'rag_used', 'llm_used']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(log_entry)
            
            return strategy
            
        except Exception as e:
            print(f"⚠️ Error logging orchestration decision: {e}")
            return "ERROR"
                      
    def reset_conversation(self):
      """Reset the conversation history"""
      self.chat_history = []
      self.followup_context = {"asked": False, "round": 0}
      return "Conversation has been reset."

    

# Create chatbot instance
chatbot = DocumentChatBot()

# Feedback storage
feedback_storage = []


                  
def main():
    # Set page title and favicon
    st.set_page_config(
        page_title="DxAI-Agent",
        page_icon=f"data:image/jpeg;base64,{icon}",
        layout="wide"
    )
                  
    # Title and description
    logo = Image.open(image_path)

    # Display logo and title side by side
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image(logo, width=1000)  # Adjust width as needed
    with col2:
        st.markdown("# DxAI-Agent")
    
    # Initialize session state variables if they don't exist
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DocumentChatBot()
        # Initialize silently without displaying status message
        st.session_state.chatbot.initialize_qa_chain()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # User type selection dropdown in sidebar
    user_type = st.sidebar.selectbox(
        "Who is asking?",
        ["User / Family", "Physician"],
        index=0
    )
    
    # Add sidebar info without initialization status
    st.sidebar.info("DxAI-Agent helps answer medical questions using our medical knowledge base.")

    # Tabs
    tab1, tab2 = st.tabs(["Chat", "About"])

    with tab1:
        # Examples section at the top
        st.subheader("Try these examples")
        examples = [
            "What are treatments for cough and cold?",
            "I have a headache and sore throat, what could it be?",
            "What home remedies help with flu symptoms?"
        ]

        # Create columns for example buttons
        cols = st.columns(len(examples))
        for i, col in enumerate(cols):
            if col.button(examples[i], key=f"example_{i}"):
                # Add the example text to messages as user input
                st.session_state.messages.append((examples[i], True))
                
                # Generate response for the example
                is_medical, reason = st.session_state.chatbot.is_medical_query(examples[i])
                
                # Process as a medical query (we know these examples are medical)
                with st.spinner("Generating response..."):
                    bot_response, sources = st.session_state.chatbot.generate_response(examples[i], user_type)
                    
                    # Format sources
                    source_info = "\n\nReferences:\n" + "\n\n".join([f"- {src}" for src in sources]) if sources else ""
                    full_response = bot_response + source_info
                    
                    # Add to state and chat history
                    st.session_state.messages.append((full_response, False))
                    st.session_state.chatbot.chat_history.append((examples[i], full_response))
                
                # Force rerun to show the updated conversation
                st.rerun()
        
        # Display chat messages
        for i, (msg, is_user) in enumerate(st.session_state.messages):
            if is_user:
                st.chat_message("user").write(msg)
            else:
                with st.chat_message("assistant"):
                    st.write(msg)
        
                    # Feedback buttons in a single line using HTML and callback keys
                    col = st.container()
                    with col:
                        # Create two buttons inline with form to detect clicks
                        feedback_key_up = f"thumbs_up_{i}"
                        feedback_key_down = f"thumbs_down_{i}"
        
                        # Use columns trick for proper spacing
                        b1, b2 = st.columns([1, 1])
                        with b1:
                            if st.button("👍", key=feedback_key_up):
                                feedback_result = vote_message(
                                    [(st.session_state.messages[i-1][0], msg)] if i > 0 else [("", msg)],
                                    "thumbs_up",
                                    0,
                                    user_type
                                )
                                st.toast(feedback_result)
                        with b2:
                            if st.button("👎", key=feedback_key_down):
                                feedback_result = vote_message(
                                    [(st.session_state.messages[i-1][0], msg)] if i > 0 else [("", msg)],
                                    "thumbs_down",
                                    0,
                                    user_type
                                )
                                st.toast(feedback_result)
                                               
        input_container = st.container()
    
        # This creates space to push the input to the bottom
        st.write("  \n" * 2)  

        # Message input - now will appear at the bottom
        if prompt := st.chat_input("Ask your medical question..."):
            # First, check if this is a medical query
            st.session_state.messages.append((prompt, True))                               
            is_medical, reason = st.session_state.chatbot.is_medical_query(prompt)
            
            if not is_medical:
                st.session_state.messages.append((prompt, True))
                st.chat_message("user").write(prompt)
                
                non_medical_response = (
                    f"I'm a medical assistant focused on health-related questions. "
                    f"Your question doesn't appear to be medical in nature. "
                    f"Please ask me about health conditions, symptoms, treatments, or other medical topics."
                )
                
                st.session_state.messages.append((non_medical_response, False))
                st.chat_message("assistant").write(non_medical_response)
            else:
                # Process the medical query with spinner
               with st.spinner("Thinking..."):
                    bot_response, sources = st.session_state.chatbot.generate_response(prompt, user_type)
                    
                    # Check if the LLM already included a references section
                    has_references = "## References:" in bot_response
                    
                    if not has_references and sources:
                        # Format sources only if they're not already included
                        formatted_sources = []
                        
                        for src in sources:
                            # Knowledge graph references
                            if "knowledge graph" in src.lower() or "neo4j" in src.lower():
                                formatted_sources.append(f"[KG] Knowledge Graph reference")
                                
                            # Internal data/DxBook references
                            elif "rawdata.pdf" in src or "dxbook" in src.lower():
                                # Extract page number if available
                                page_match = re.search(r'page[_\s]?(\d+)', src, re.IGNORECASE)
                                page_num = page_match.group(1) if page_match else "N/A"
                                
                                # Extract the referenced paragraph
                                paragraph_match = re.search(r'page[_\s]?\d+\s*[:-]?\s*(.*?)(?=$|\n|page)', src, re.IGNORECASE)
                                paragraph = paragraph_match.group(1).strip() if paragraph_match else src
                                
                                # Format as internal data with page reference and paragraph
                                formatted_sources.append(f"[Internal Data: DxBook, Page {page_num}] {paragraph}")
                                
                            # External links/URLs
                            elif re.search(r'https?://[^\s]+', src):
                                url_match = re.search(r'(https?://[^\s]+)', src)
                                if url_match:
                                    url = url_match.group(1)
                                    # Make the URL clickable in markdown
                                    formatted_sources.append(f"[External Source]({url})")
                                else:
                                    formatted_sources.append(src)
                                    
                            # Other sources
                            else:
                                formatted_sources.append(src)
                        
                        # Deduplicate sources while preserving order
                        seen = set()
                        formatted_sources = [x for x in formatted_sources if not (x in seen or seen.add(x))]
                        
                        # Add formatted sources to response
                        if formatted_sources:
                            source_info = "\n\n## References:\n"
                            for i, src in enumerate(formatted_sources, 1):
                                source_info += f"{i}. {src}\n\n"
                            
                            # Handle medical disclaimer
                            disclaimer_pattern = r'(This information is not a substitute.*?provider\.)'
                            disclaimer_match = re.search(disclaimer_pattern, bot_response, re.DOTALL)
                            
                            if disclaimer_match:
                                # Extract the disclaimer
                                disclaimer = disclaimer_match.group(1)
                                
                                # Remove the existing disclaimer
                                bot_response = re.sub(disclaimer_pattern, '', bot_response, flags=re.DOTALL)
                                
                                # Add references and then disclaimer
                                full_response = bot_response.strip() + source_info + "\n\n" + disclaimer
                            else:
                                # Just add references if no disclaimer
                                full_response = bot_response.strip() + source_info
                        else:
                            full_response = bot_response
                    else:
                        # Use response as is if it already has references or there are no sources
                        full_response = bot_response
                    
                    # Add to state and chat history
                    st.session_state.messages.append((full_response, False))
                    st.session_state.chatbot.chat_history.append((prompt, full_response))
                
            # Force a rerun to update the UI
            st.rerun()

        # Physician feedback section
        st.divider()
        st.subheader("🩺 Detailed Feedback")
        with st.form("feedback_form"):
            feedback_text = st.text_area(
                "Enter corrections, improvements, or comments here...",
                height=100
            )
            submit_feedback_btn = st.form_submit_button("Submit Feedback")

            if submit_feedback_btn and feedback_text:
                # Format the chat history for feedback
                history_for_feedback = []
                for i in range(0, len(st.session_state.messages), 2):
                    if i+1 < len(st.session_state.messages):
                        history_for_feedback.append((st.session_state.messages[i][0],
                                                    st.session_state.messages[i+1][0]))

                feedback_result = submit_feedback(feedback_text, history_for_feedback, user_type)
                st.success(feedback_result)

        # Reset conversation button
        if st.button("Reset Conversation"):
            st.session_state.chatbot.reset_conversation()
            st.session_state.messages = []
            st.rerun()

    with tab2:
        st.markdown("""
        ## Medical Chat Assistant

        **Powered by:**
        - RAG: Retrieval from PDFs
        - KG: Knowledge Graph (Neo4j)
        - LLM: Gemini Flash 1.5
        - Reflection: Analyzes response completeness

        **Modes:**
        - **User / Family:** Full answer (diagnosis, treatment, remedies)
        - **Physician:** Only probable diagnosis provided

        **Disclaimer:** This system is informational only, not a substitute for professional advice.
        """)

    

# --- Feedback Handler ---
def submit_feedback(feedback_text, chat_history, user_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if feedback_text and feedback_text.strip():
        feedback_text = feedback_text.strip()

        # Format chat history for storage
        formatted_chat = []
        for msg in chat_history:
            if len(msg) == 2:  # Standard format [user_msg, bot_msg]
                formatted_chat.append({"user": msg[0], "bot": msg[1]})

        # Create feedback entry
        feedback_entry = {
            "timestamp": timestamp,
            "user_type": user_type,
            "feedback_text": feedback_text,
            "chat_history": formatted_chat
        }

        feedback_storage.append(feedback_entry)
        print(f"📝 New feedback received at {timestamp} from {user_type}: {feedback_text}")

        # Save feedback to CSV with extended information
        feedback_file = "physician_feedback.csv"
        file_exists = os.path.isfile(feedback_file)

        with open(feedback_file, mode='a', newline='', encoding='utf-8') as file:
            fieldnames = ['timestamp', 'user_type', 'feedback_text', 'chat_history']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Convert chat history to string for CSV storage
            chat_json = json.dumps(formatted_chat)
            writer.writerow({
                'timestamp': timestamp,
                'user_type': user_type,
                'feedback_text': feedback_text,
                'chat_history': chat_json
            })

        return "✅ Thank you for your feedback!"
    else:
        return "⚠️ Please enter feedback before submitting."

# --- Vote Message Handler ---
def vote_message(history, vote_type, index, user_type):
    """Record a vote (thumbs up/down) for a specific message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if index < len(history) and index >= 0:
        # Extract the specific message being voted on
        user_msg = history[index][0]
        bot_msg = history[index][1]

        # Save the vote to CSV
        feedback_file = "message_feedback.csv"
        file_exists = os.path.isfile(feedback_file)

        with open(feedback_file, mode='a', newline='', encoding='utf-8') as file:
            fieldnames = ['timestamp', 'user_type', 'vote_type', 'user_message', 'bot_message']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'timestamp': timestamp,
                'user_type': user_type,
                'vote_type': vote_type,
                'user_message': user_msg,
                'bot_message': bot_msg
            })

        print(f"👍👎 Message feedback: {vote_type} for message {index} from {user_type}")
        return f"✅ {vote_type} feedback recorded. Thank you!"

    return "⚠️ Could not record feedback for this message."

# --- Reset Handler ---
def reset_chat():
    chatbot.reset_conversation()
    return None

# --- Respond Handler ---
def respond(message, history, user_type_value):
    bot_response, sources = chatbot.generate_response(message, user_type_value)
    source_info = "\n\nReferences:\n" + "\n\n".join([f"- {src}" for src in sources]) if sources else ""
    full_response = bot_response + source_info

    # Return the response to be displayed in the chatbot
    return full_response

# --- Model Init ---
def init_model():
    try:
        success, message = chatbot.initialize_qa_chain()
        return f"Model initialization: {message}"
    except Exception as e:
        return f"Error initializing model: {str(e)}"

if __name__ == "__main__":
    main()
