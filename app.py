
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
        
            MISSING_INFO_PROMPT = f"""
            You are a medical assistant analyzing a patient query and conversation.
            Determine if any CRITICAL information is still missing to properly assess their situation.
        
            Conversation history:
            {context}
        
            Latest patient input: "{user_query}"
        
            CRITICALLY EVALUATE if you have enough information to provide a reasonable medical assessment.
            Only ask follow-up questions if ABSOLUTELY NECESSARY for basic assessment.
        
            Rules for determining if information is sufficient:
            1. If the patient has provided symptoms, duration, and basic severity, that's usually enough
            2. If you've already asked follow-up questions once, avoid asking more unless critical
            3. If a general assessment can be made with current information, proceed without more questions
            4. ONLY ask about truly essential missing information
            5. If the query is about general medical information (not about a specific case), NO follow-up needed
            6. If the question is about treatments, medication, or general information, NO follow-up needed
        
            Return your answer in this exact JSON format:
            {{
                "needs_followup": true/false,
                "reasoning": "brief explanation of why more information is needed or not",
                "missing_info": [
                    {{"question": "specific follow-up question 1"}}
                ]
            }}
        
            Only include 1 question if follow-up is needed.
            """
        
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
        cache_key = {"type": "symptom_extraction", "query": user_query}
        cached = get_cached(cache_key)
        if cached:
            print("🧠 Using cached symptom extraction.")
            return cached

        SYMPTOM_PROMPT = f"""
        You are a medical assistant.
        Extract and correct all symptoms mentioned in the following user query.
        For each symptom, assign a confidence score between 0.0 and 1.0 indicating how certain you are.
        **Important:** Return your answer in exactly the following format:
        Extracted Symptoms: [{{"symptom": "symptom1", "confidence": 0.9}}, {{"symptom": "symptom2", "confidence": 0.8}}, ...]

        User Query: "{user_query}"
        """

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
        common_symptoms = ["fever", "cough", "headache", "sore throat", "nausea", "dizziness"]
        query_lower = user_query.lower()

        for symptom in common_symptoms:
            if symptom in query_lower:
                fallback_symptoms.append(symptom)

        fallback_confidence = 0.4  # Low confidence for fallback extraction
        print(f"🔍 Fallback Extracted Symptoms: {fallback_symptoms} (confidence: {fallback_confidence:.4f})")
        result = (fallback_symptoms, fallback_confidence)
        set_cached(cache_key, result)
        return result

    def query_disease_from_symptoms(self, symptoms: List[str]) -> Tuple[Optional[str], float, List[str], List[Tuple[str, float]]]:
        """Query for diseases based on symptoms"""
        if not symptoms:
            return None, 0.0, [], []

        cache_key = {"type": "disease_matching", "symptoms": tuple(symptoms)}
        cached = get_cached(cache_key)
        if cached:
            print("🧠 Using cached disease match.")
            return cached

        # Check for direct match with known diseases
        for s in symptoms:
            if s in known_diseases:
                detected = s.capitalize()
                print(f"Using extracted symptom as disease: {detected} (confidence: 0.95)")
                return detected, 0.95, [s], [(detected, 0.95)]

        # Construct a more sophisticated Cypher query with confidence scoring
        cypher_query = f"""
        MATCH (s:symptom)-[r:INDICATES]->(d:disease)
        WHERE LOWER(s.Name) IN {str(symptoms)}
        WITH d, COUNT(DISTINCT s) AS matching_symptoms,
             COLLECT(DISTINCT LOWER(s.Name)) AS matched_symptoms
        WITH d, matching_symptoms, matched_symptoms,
             matching_symptoms * 1.0 / {max(1, len(symptoms))} AS confidence_score
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

        except Exception as e:
            print(f"⚠️ Error querying Neo4j for diseases: {e}")

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
                   
    def knowledge_graph_agent(self, state: Dict) -> Dict:
        """
        Knowledge Graph Agent
        Extracts symptoms, identifies diseases, and recommends treatments using the knowledge graph
        """
        print("Knowledge Graph Agent")
    
        # Use proper enum comparison
        if state.get("halt_execution") == ExecutionState.HALTED:
            return state
    
        subtasks = state.get("subtasks", [])
        subtask_results = state.get("subtask_results", {})
        updated_results = {**subtask_results}  # Clone to avoid modifying the original
    
        # Create a new state to collect all updates
        new_state = {**state}
    
        # Track progress
        processed_tasks = 0
        successful_tasks = 0
    
        # Loop through the tasks
        for task in subtasks:
            if task.get("data_source") != "KG" or task.get("id") in updated_results:
                continue
    
            processed_tasks += 1
            task_id = task.get("id")
            task_query = task.get("subtask_query", task.get("description", ""))
            print(f"📚 Processing KG task: {task_id} - {task_query}")
    
            task_desc = task_query.lower()
    
            try:
                # Initialize task result with standard structure
                task_result = {
                    "id": task_id,
                    "subtask_query": task_query,
                    "subtask_answer": None,
                    "confidence": 0.0,
                    "status": "failed",  # Default status
                    "data_source": "KG"
                }
    
                if "symptom" in task_desc:
                    symptoms, conf = self.extract_symptoms(state["user_query"])
    
                    # Update task result
                    task_result["subtask_answer"] = ", ".join(symptoms) if symptoms else None
                    task_result["confidence"] = conf
                    task_result["status"] = "completed" if (conf >= THRESHOLDS["symptom_extraction"] and symptoms) else "failed"
    
                    # Update main state if successful
                    if task_result["status"] == "completed":
                        new_state["symptoms"] = symptoms
                        new_state["symptom_confidence"] = conf
                        successful_tasks += 1
                        print(f"✔️ Symptoms extraction successful with confidence {conf:.4f}")
    
                elif "disease" in task_desc:
                    # Get symptoms from either earlier subtask results or state
                    symptoms = []
                    for r in updated_results.values():
                        if r.get("status") == "completed" and "symptom" in r.get("subtask_query", "").lower():
                            if r.get("subtask_answer"):
                                symptoms = [s.strip() for s in r.get("subtask_answer").split(",")]
    
                    if not symptoms:
                        symptoms = state.get("symptoms", [])
    
                    if symptoms:
                        disease, conf, matched, alt = self.query_disease_from_symptoms(symptoms)
    
                        # Update task result
                        task_result["subtask_answer"] = disease
                        task_result["confidence"] = conf
                        task_result["status"] = "completed" if conf >= THRESHOLDS["disease_matching"] else "failed"
    
                        # Update main state if successful
                        if task_result["status"] == "completed":
                            # Handle the case where we receive a list of diseases
                            if isinstance(disease, list):
                                new_state["diseases"] = [d[0] for d in disease]  # Store all disease names
                                new_state["disease_confidences"] = [d[1] for d in disease]  # Store all confidences
                                new_state["disease"] = disease[0][0] if disease else None  # For backward compatibility
                                new_state["disease_confidence"] = conf
                                new_state["matched_symptoms"] = matched
                                successful_tasks += 1
                                print(f"✔️ Diseases identified: {new_state['diseases']} with top confidence {conf:.4f}")
                            else:
                                # Backward compatibility with existing code
                                new_state["disease"] = disease
                                new_state["diseases"] = [disease] if disease else []
                                new_state["disease_confidence"] = conf
                                new_state["disease_confidences"] = [conf] if disease else []
                                new_state["matched_symptoms"] = matched
                                new_state["alternative_diseases"] = alt
                                successful_tasks += 1
                                print(f"✔️ Disease identified: {disease} with confidence {conf:.4f}")
                    else:
                        task_result["status"] = "failed"
                        task_result["subtask_answer"] = "Could not identify disease - no symptoms provided"
    
                elif "treatment" in task_desc:
                    # Get disease from earlier subtask results or state
                    disease = None
                    for r in updated_results.values():
                        if r.get("status") == "completed" and "disease" in r.get("subtask_query", "").lower():
                            disease = r.get("subtask_answer")
    
                    if not disease:
                        disease = state.get("disease")
    
                    if disease:
                        treatments, conf = self.query_treatments(disease)
    
                        # Update task result
                        task_result["subtask_answer"] = ", ".join(treatments) if treatments else None
                        task_result["confidence"] = conf
                        task_result["status"] = "completed" if (conf >= THRESHOLDS["knowledge_graph"] and treatments) else "failed"
    
                        # Update main state if successful
                        if task_result["status"] == "completed":
                            new_state["treatments"] = treatments
                            new_state["treatment_confidence"] = conf
                            successful_tasks += 1
                            print(f"✔️ Treatments found for {disease}: {treatments} with confidence {conf:.4f}")
                    else:
                        task_result["status"] = "failed"
                        task_result["subtask_answer"] = "Could not find treatments - no disease identified"
    
                elif "remedy" in task_desc:
                    # Get disease from earlier subtask results or state
                    disease = None
                    for r in updated_results.values():
                        if r.get("status") == "completed" and "disease" in r.get("subtask_query", "").lower():
                            disease = r.get("subtask_answer")
    
                    if not disease:
                        disease = state.get("disease")
    
                    if disease:
                        remedies, conf = self.query_home_remedies(disease)
    
                        # Update task result
                        task_result["subtask_answer"] = ", ".join(remedies) if remedies else None
                        task_result["confidence"] = conf
                        task_result["status"] = "completed" if (conf >= THRESHOLDS["knowledge_graph"] and remedies) else "failed"
    
                        # Update main state if successful
                        if task_result["status"] == "completed":
                            new_state["home_remedies"] = remedies
                            new_state["remedy_confidence"] = conf
                            successful_tasks += 1
                            print(f"✔️ Home remedies for {disease}: {remedies} with confidence {conf:.4f}")
                    else:
                        task_result["status"] = "failed"
                        task_result["subtask_answer"] = "Could not find home remedies - no disease identified"
    
                # Save task result
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
    
        # Calculate completion rate
        completion_rate = successful_tasks / max(processed_tasks, 1)
        print(f"KG Task Completion Rate: {completion_rate:.2f}")
    
        # Update state with all collected changes
        new_state["subtask_results"] = updated_results
        new_state["kg_completion_rate"] = completion_rate
    
        # Set kg_answer for reflection agent
        if successful_tasks > 0:
            kg_answers = []
            if new_state.get("diseases") and len(new_state.get("diseases", [])) > 0:
                # For multiple diseases, list them with their confidence
                if len(new_state.get("diseases", [])) > 1:
                    diseases_list = []
                    for i, (disease, conf) in enumerate(zip(
                            new_state.get("diseases", []), 
                            new_state.get("disease_confidences", [0.7] * len(new_state.get("diseases", [])))
                        )):
                        diseases_list.append(f"{disease} (Confidence: {conf:.2f})")
                    kg_answers.append(f"Possible Diseases: {', '.join(diseases_list)}")
                else:
                    # Single disease case
                    kg_answers.append(f"Disease: {new_state['diseases'][0]}")
            elif new_state.get("disease"):  # Backward compatibility
                kg_answers.append(f"Disease: {new_state['disease']}")
            
            if new_state.get("symptoms"):
                kg_answers.append(f"Symptoms: {', '.join(new_state['symptoms'])}")
            if new_state.get("treatments"):
                kg_answers.append(f"Treatments: {', '.join(new_state['treatments'])}")
            if new_state.get("home_remedies"):
                kg_answers.append(f"Home Remedies: {', '.join(new_state['home_remedies'])}")
    
            if kg_answers:
                new_state["kg_answer"] = "\n".join(kg_answers)
                new_state["kg_confidence"] = max(
                    new_state.get("disease_confidence", 0.0),
                    new_state.get("treatment_confidence", 0.0),
                    new_state.get("remedy_confidence", 0.0)
                )
        else:
            # Make sure we have at least an empty kg_answer field for the reflection agent
            new_state["kg_answer"] = ""
            new_state["kg_confidence"] = 0.0
            
        query = new_state.get("user_query", "").lower()
        disease_patterns = [
            r"what (could|might|can) (i|my|the patient|they) have",
            r"what (is|are) (the )?(possible|potential) (disease|diagnosis|condition)",
            r"what (disease|condition|diagnosis) (matches|is indicated by|could cause)",
            r"what (might|could) (be|cause) (these|the|my|their) symptoms"
        ]
        
        is_disease_query = any(re.search(pattern, query) for pattern in disease_patterns)
        
        # If it's a disease query and we found diseases, boost confidence
        if is_disease_query and (new_state.get("diseases") or new_state.get("disease")):
            # Format the KG answer for disease identification specifically
            symptoms = new_state.get("symptoms", [])
            
            disease_answer = "# Possible Diagnoses Based on Symptoms\n\n"
            
            if symptoms:
                disease_answer += f"## Symptoms Identified\n{', '.join(symptoms)}\n\n"
            
            # List all diseases with their details
            diseases = new_state.get("diseases", [])
            if not diseases and new_state.get("disease"):
                diseases = [new_state.get("disease")]
            
            for i, disease in enumerate(diseases):
                conf = new_state.get("disease_confidences", [0.7])[i] if i < len(new_state.get("disease_confidences", [])) else 0.7
                disease_answer += f"## {disease}\n"
                disease_answer += f"**Confidence:** {conf:.2f}\n"
                
                if new_state.get("matched_symptoms"):
                    disease_answer += f"**Matching Symptoms:** {', '.join(new_state.get('matched_symptoms'))}\n\n"
                
                if new_state.get("treatments"):
                    disease_answer += "**Recommended Treatments:**\n"
                    for treatment in new_state.get("treatments", []):
                        disease_answer += f"- {treatment}\n"
                    disease_answer += "\n"
                
                if new_state.get("home_remedies"):
                    disease_answer += "**Home Remedies:**\n"
                    for remedy in new_state.get("home_remedies", []):
                        disease_answer += f"- {remedy}\n"
                    disease_answer += "\n"
            
            # Override the existing kg_answer with the formatted one
            new_state["kg_answer"] = disease_answer
            
            # Boost confidence to ensure KG-only route
            new_state["kg_confidence"] = max(0.9, new_state.get("kg_confidence", 0.0))
            print("✅ Disease identification query detected, boosting KG confidence")
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
        
        # Check if this is a disease identification query
        is_disease_query = False
        disease_patterns = [
            r"what (could|might|can) (i|my|the patient|they) have",
            r"what (is|are) (the )?(possible|potential) (disease|diagnosis|condition)",
            r"what (disease|condition|diagnosis) (matches|is indicated by|could cause)",
            r"what (might|could) (be|cause) (these|the|my|their) symptoms"
        ]
        
        for pattern in disease_patterns:
            if re.search(pattern, user_query.lower()):
                is_disease_query = True
                break
        
        # If it's a disease query and we found diseases, format the answer and boost confidence
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
            
            # Boost confidence for disease identification queries
            result["kg_confidence"] = max(0.85, result.get("kg_confidence", 0.0))
            print("✅ Disease identification query detected, formatted comprehensive KG response")
        
        return result
                    
    def reflection_agent(self, user_query, kg_answer, rag_answer, kg_confidence=None):
        """
        Evaluates all possible combinations of knowledge sources to provide the most
        complete answer to the user query.
        
        Args:
            user_query: The original user question
            kg_answer: Answer from Knowledge Graph (refined by LLM)
            rag_answer: Answer from RAG
            kg_confidence: Confidence score from the KG system (if available)
            
        Returns:
            tuple: (final_answer, strategy_used)
        """
        if not self.identified_diseases and kg_answer:
            self.extract_diseases_from_kg_answer(kg_answer)
                            
        # Special case for disease identification with KG results - ENHANCED
        if self.is_disease_identification_query(user_query):
            if isinstance(self.identified_diseases, list) and len(self.identified_diseases) > 0:
                # Format the diseases directly from the list
                formatted_answer = self.format_disease_list_answer(user_query, self.identified_diseases, self.disease_confidence)
                print(f"Reflection decision: KG_ONLY - Disease identification with {len(self.identified_diseases)} diseases")
                return formatted_answer, "KG_ONLY"
            elif "Possible Diseases:" in kg_answer or "Disease:" in kg_answer:
                print("Reflection decision: KG_ONLY - Disease identification query with KG results")
                return kg_answer, "KG_ONLY"
        
        # Check if KG confidence score is available and high enough
        if kg_confidence is not None and kg_confidence >= 0.65:
            print(f"Reflection decision: KG_ONLY - High confidence KG answer ({kg_confidence})")
            return kg_answer, "KG_ONLY"
            
        # Special case for disease identification with KG results
        if self.is_disease_identification_query(user_query) and ("Possible Diseases:" in kg_answer or "Disease:" in kg_answer):
            print("Reflection decision: KG_ONLY - Disease identification query with KG results")
            return kg_answer, "KG_ONLY"
            
        # Set threshold for considering an answer complete
        COMPLETENESS_THRESHOLD = 0.65
        
        # Start logging the reflection process
        print(f"Reflection agent analyzing query: {user_query}")
        
        # Evaluate completeness of individual sources
        kg_completeness = self.evaluate_answer_completeness(user_query, kg_answer)
        rag_completeness = self.evaluate_answer_completeness(user_query, rag_answer)
        
        print(f"KG completeness: {kg_completeness}, RAG completeness: {rag_completeness}")
        
        # Evaluate what's missing from each source
        kg_missing = self.identify_missing_elements(user_query, kg_answer)
        rag_missing = self.identify_missing_elements(user_query, rag_answer)
        
        # Case 1: KG only - sufficient
        if kg_completeness >= COMPLETENESS_THRESHOLD:
            print("Reflection decision: KG_ONLY - KG answer is complete")
            return kg_answer, "KG_ONLY"
        
        # Case 2: RAG only - sufficient
        if rag_completeness >= COMPLETENESS_THRESHOLD:
            print("Reflection decision: RAG_ONLY - RAG answer is complete")
            return rag_answer, "RAG_ONLY"
        
        # Case 3: KG + RAG combination
        kg_rag_combined = self.combine_answers(kg_answer, rag_answer, kg_missing)
        kg_rag_completeness = self.evaluate_answer_completeness(user_query, kg_rag_combined)
        
        print(f"KG+RAG completeness: {kg_rag_completeness}")
        
        if kg_rag_completeness >= COMPLETENESS_THRESHOLD:
            print("Reflection decision: KG_RAG_COMBINED - Combined KG and RAG answer is complete")
            return kg_rag_combined, "KG_RAG_COMBINED"
        
        # Generate LLM answer for missing parts
        llm_answer = self.generate_llm_answer(user_query, kg_missing, rag_missing)
        llm_completeness = self.evaluate_answer_completeness(user_query, llm_answer)
        
        print(f"LLM completeness: {llm_completeness}")
        
        # Case 4: LLM only - sufficient
        if llm_completeness >= COMPLETENESS_THRESHOLD:
            print("Reflection decision: LLM_ONLY - LLM answer is complete")
            return llm_answer, "LLM_ONLY"
        
        # Case 5: KG + LLM combination
        kg_llm_combined = self.combine_answers(kg_answer, llm_answer, kg_missing)
        kg_llm_completeness = self.evaluate_answer_completeness(user_query, kg_llm_combined)
        
        print(f"KG+LLM completeness: {kg_llm_completeness}")
        
        if kg_llm_completeness >= COMPLETENESS_THRESHOLD:
            print("Reflection decision: KG_LLM_COMBINED - Combined KG and LLM answer is complete")
            return kg_llm_combined, "KG_LLM_COMBINED"
        
        # Case 6: RAG + LLM combination
        rag_llm_combined = self.combine_answers(rag_answer, llm_answer, rag_missing)
        rag_llm_completeness = self.evaluate_answer_completeness(user_query, rag_llm_combined)
        
        print(f"RAG+LLM completeness: {rag_llm_completeness}")
        
        if rag_llm_completeness >= COMPLETENESS_THRESHOLD:
            print("Reflection decision: RAG_LLM_COMBINED - Combined RAG and LLM answer is complete")
            return rag_llm_combined, "RAG_LLM_COMBINED"
        
        # Case 7: KG + RAG + LLM combination
        all_combined = self.combine_all_answers(kg_answer, rag_answer, llm_answer)
        all_combined_completeness = self.evaluate_answer_completeness(user_query, all_combined)
        
        print(f"All sources combined completeness: {all_combined_completeness}")
        
        if all_combined_completeness >= COMPLETENESS_THRESHOLD:
            print("Reflection decision: KG_RAG_LLM_COMBINED - All sources combined answer is complete")
            return all_combined, "KG_RAG_LLM_COMBINED"
        
        # Case 8: Best available answer (even if below threshold)
        best_completeness = max(kg_completeness, rag_completeness, llm_completeness, 
                               kg_rag_completeness, kg_llm_completeness, 
                               rag_llm_completeness, all_combined_completeness)
        
        # If KG answer is already quite good (above 0.7), prioritize it for simplicity
        if kg_completeness >= 0.7 and kg_completeness == best_completeness:
            print("Reflection decision: KG_ONLY - Good enough KG answer prioritized")
            return kg_answer, "KG_ONLY"
            
        # Choose the best answer based on completeness scores
        if best_completeness == kg_completeness:
            print("Reflection decision: KG_ONLY - Best available answer")
            return kg_answer, "KG_ONLY"
        elif best_completeness == rag_completeness:
            print("Reflection decision: RAG_ONLY - Best available answer")
            return rag_answer, "RAG_ONLY"
        elif best_completeness == llm_completeness:
            print("Reflection decision: LLM_ONLY - Best available answer")
            return llm_answer, "LLM_ONLY"
        elif best_completeness == kg_rag_completeness:
            print("Reflection decision: KG_RAG_COMBINED - Best available answer")
            return kg_rag_combined, "KG_RAG_COMBINED"
        elif best_completeness == kg_llm_completeness:
            print("Reflection decision: KG_LLM_COMBINED - Best available answer")
            return kg_llm_combined, "KG_LLM_COMBINED"
        elif best_completeness == rag_llm_completeness:
            print("Reflection decision: RAG_LLM_COMBINED - Best available answer")
            return rag_llm_combined, "RAG_LLM_COMBINED"
        else:
            print("Reflection decision: KG_RAG_LLM_COMBINED - Best available answer")
            return all_combined, "KG_RAG_LLM_COMBINED"
                   
    def evaluate_answer_completeness(self, query, answer, llm_client=None):
        """
        Evaluates how completely an answer addresses the user query.
        
        Args:
            query: The original user question
            answer: The answer to evaluate
            llm_client: Optional LLM client for evaluation
            
        Returns:
            float: Score between 0.0 and 1.0 representing completeness
        """
        if not answer or answer.strip() == "":
            return 0.0
        
        # Create an improved prompt with domain-specific guidance for medical question evaluation
        prompt = f"""
        You are evaluating how completely a medical answer addresses a user's health-related question.
        
        USER QUESTION: {query}
        
        ANSWER TO EVALUATE: {answer}
        
        Evaluation Guidelines:
        - Focus on medical accuracy and clinical relevance of the information
        - Consider whether all aspects of the query are addressed (symptoms, causes, treatments, prevention, etc.)
        - For disease-related questions, check if relevant diagnoses, symptoms, and treatments are covered
        - For medication questions, check if dosage, side effects, and contraindications are addressed when relevant
        - For procedural questions, check if preparation, process, and recovery information is provided when relevant
        - Factual correctness is more important than comprehensiveness
        - If the answer says it doesn't have enough information but provides what is known, consider this appropriate
        
        On a scale of 0.0 to 1.0, how completely does this answer address all aspects of the user's question?
        - 0.0: Does not address the question at all
        - 0.3: Addresses a minor portion of the question
        - 0.5: Addresses roughly half of the important aspects
        - 0.7: Addresses most important aspects but may be missing some details
        - 0.8: Addresses nearly all aspects with minimal omissions
        - 0.9: Addresses all medical aspects with high quality information
        - 1.0: Addresses all aspects perfectly with comprehensive information
        
        Provide only a single number as your response.
        """
        
        try:
            # Use the local_generate method to evaluate with Gemini
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
                      
    def is_kg_answer_high_quality(self, user_query, kg_answer):
        """
        Evaluates if the KG answer is high quality enough to use on its own.
        This function acts as an additional check focused specifically on KG answer quality.
        
        Args:
            user_query: The original user question
            kg_answer: Answer from Knowledge Graph
            
        Returns:
            bool: True if the KG answer is high quality, False otherwise
        """
        if not kg_answer or kg_answer.strip() == "":
            return False
        
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
        indicator_count = sum(1 for indicator in indicators if indicator.lower() in kg_answer.lower())
        
        # Check for structured information patterns
        has_structured_info = any([
            ":" in kg_answer,  # Key-value pairs
            "\n-" in kg_answer,  # Bulleted lists
            "1." in kg_answer,   # Numbered lists
            "•" in kg_answer     # Bullet points
        ])
        
        # Create a prompt to evaluate KG quality
        prompt = f"""
        You are evaluating whether a Knowledge Graph-based medical answer is high quality enough to use on its own.
        
        USER QUESTION: {user_query}
        
        KG ANSWER: {kg_answer}
        
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
            
            # If the answer has multiple quality indicators OR structured information AND the LLM says it's high quality
            return (indicator_count >= 2 or has_structured_info) and is_high_quality
        except Exception as e:
            print(f"Error evaluating KG answer quality: {e}")
            return False
                       
    def combine_answers(self, primary_answer, secondary_answer, missing_elements=None, llm_client=None):
        """
        Intelligently combines two answers to provide a more complete response.
        
        Args:
            primary_answer: The answer from the primary/preferred source
            secondary_answer: The answer from the secondary source
            missing_elements: Optional list of elements missing from primary answer
            
        Returns:
            str: Combined answer with proper attribution and flow
        """
        if not primary_answer or primary_answer.strip() == "":
            return secondary_answer
        
        if not secondary_answer or secondary_answer.strip() == "":
            return primary_answer
        
        missing_elements_text = ""
        if missing_elements and len(missing_elements) > 0:
            missing_elements_text = "The primary answer is missing these elements: " + ", ".join(missing_elements)
        
        prompt = f"""
        You are creating a comprehensive medical answer by combining information from multiple sources.
        
        PRIMARY ANSWER: {primary_answer}
        
        SECONDARY ANSWER: {secondary_answer}
        
        {missing_elements_text}
        
        Create a unified answer that:
        1. Preserves all relevant information from both sources
        2. Eliminates redundancy
        3. Maintains a coherent, flowing narrative
        4. Ensures all medical information is accurate
        5. Uses clear and patient-friendly language
        
        The combined answer should be comprehensive but concise.
        """
        
        try:
            combined = self.local_(prompt, max_tokens=1000)
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
        
        # Then add LLM content
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
                kg_data = self.process_with_knowledge_graph(user_input)
                kg_content = kg_data.get("kg_answer", "")
                kg_confidence = kg_data.get("kg_confidence", 0.0)
                
                disease = kg_data.get("disease", "")
                symptoms = kg_data.get("symptoms", [])
                treatments = kg_data.get("treatments", [])
                home_remedies = kg_data.get("home_remedies", [])
                
                print(f"📊 KG Confidence: {kg_confidence:.4f} (took {(datetime.now() - t_start).total_seconds():.2f}s)")
                
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
                
                # Use reflection agent to determine the best answer
                print("🧠 Using reflection agent to evaluate and combine answers...")
                t_start = datetime.now()
                
                # Use the reflection agent to evaluate and combine answers
                final_answer, strategy_used = self.reflection_agent(user_input, kg_content, rag_content)
                
                print(f"📊 Reflection Strategy: {strategy_used}")
                print(f"✅ Response generated (took {(datetime.now() - t_start).total_seconds():.2f}s)")
                
                # Combine all sources for reference
                all_sources = rag_sources
                if disease:
                    all_sources.append(f"[KG] Knowledge Graph: {disease}")
                
                # Log the orchestration decision for analysis
                self.log_orchestration_decision(
                    user_input,
                    f"SELECTED_STRATEGY: {strategy_used}\nREASONING: Determined by reflection agent based on answer completeness evaluation.\nRESPONSE: {final_answer[:100]}...",
                    kg_confidence,
                    rag_confidence
                )
                self.last_strategy = strategy_used  # Store for testing
                
                # Add references if not included
                if "## References:" not in final_answer and all_sources:
                    references = "\n\n## References:\n"
                    for i, src in enumerate(all_sources, 1):
                        references += f"{i}. {src}\n\n"
                    
                    # Check if response already has disclaimer
                    if "This information is not a substitute" in final_answer:
                        # Add references before disclaimer
                        disclaimer_pattern = r'(This information is not a substitute.*?provider\.)'
                        disclaimer_match = re.search(disclaimer_pattern, final_answer, re.DOTALL)
                        if disclaimer_match:
                            disclaimer = disclaimer_match.group(1)
                            final_response = re.sub(disclaimer_pattern, '', final_answer, flags=re.DOTALL)
                            final_response = final_response.strip() + "\n\n" + references + "\n\n" + disclaimer
                        else:
                            final_response = final_answer.strip() + "\n\n" + references
                    else:
                        # Add references and standard disclaimer
                        disclaimer = "This information is not a substitute for professional medical advice. If symptoms persist or worsen, please consult with a qualified healthcare provider."
                        final_response = final_answer.strip() + "\n\n" + references + "\n\n" + disclaimer
                else:
                    final_response = final_answer
                
                # Collect response metrics
                metrics = {
                    "query": user_input,
                    "kg_confidence": kg_confidence,
                    "rag_confidence": rag_confidence,
                    "strategy": strategy_used,
                    "response_length": len(final_response),
                    "processing_time": (datetime.now() - t_start).total_seconds(),
                    "source_count": len(all_sources)
                }
                
                # Log metrics to CSV
                self.log_response_metrics(metrics)
                
                # Add to chat history
                self.chat_history.append((user_input, final_response))
                
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
            
            if "SELECTED_STRATEGY:" in orchestration_result:
                strategy_part = orchestration_result.split("SELECTED_STRATEGY:")[1].split("\n")[0].strip()
                strategy = strategy_part
                
            if "REASONING:" in orchestration_result:
                reasoning_parts = orchestration_result.split("REASONING:")[1].split("RESPONSE:")[0].strip()
                reasoning = reasoning_parts.strip()
            
            # Log the decision
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "strategy": strategy,
                "reasoning": reasoning,
                "kg_confidence": kg_confidence,
                "rag_confidence": rag_confidence
            }
            
            # Print logging information
            print(f"📊 Orchestration Decision:")
            print(f"   Query: {query}")
            print(f"   Strategy: {strategy}")
            print(f"   KG Confidence: {kg_confidence:.4f}")
            print(f"   RAG Confidence: {rag_confidence:.4f}")
            print(f"   Reasoning: {reasoning}")
            
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
