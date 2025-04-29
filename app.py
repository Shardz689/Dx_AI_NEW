
import streamlit as st
from pathlib import Path
import csv
import os
import re
import torch._C 
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

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from neo4j import GraphDatabase

# Import Gemini API
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import json

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://your_instance.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")

# Update the NEO4J_AUTH variable to use environment variables
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# Threshold settings
THRESHOLDS = {
    "symptom_extraction": 0.6,
    "disease_matching": 0.5,
    "knowledge_graph": 0.6,
}

# Load and convert the image to base64
def get_image_as_base64(file_path):
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
known_diseases = ["covid-19", "flu", "common cold", "migraine", "diabetes"]

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

        # Initialize embedding model
        try:
            import sentence_transformers
            from sentence_transformers import SentenceTransformer
            self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return None, f"Failed to load embeddings model: {str(e)}"
        vectordb = FAISS.from_documents(splits, self.embedding_model)
        return vectordb, "Vector database created successfully."

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
                self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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
                    top_disease = records[0]["Disease"]
                    confidence = float(records[0]["Confidence"])
                    matched_symptoms = records[0]["MatchedSymptoms"]

                    print(f"🦠 Detected Disease: {top_disease} (confidence: {confidence:.4f})")
                    print(f"Matched symptoms: {matched_symptoms}")

                    # Store all potential diseases for fallback
                    all_diseases = [(rec["Disease"], float(rec["Confidence"])) for rec in records]

                    result = (top_disease, confidence, matched_symptoms, all_diseases)
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
                            new_state["disease"] = disease
                            new_state["disease_confidence"] = conf
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
            if new_state.get("disease"):
                kg_answers.append(f"Disease: {new_state['disease']}")
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

        return new_state

    def process_with_knowledge_graph(self, user_query: str) -> Dict:
        """Process user query with knowledge graph agent"""
        print(f"Processing with knowledge graph: {user_query}")

        # Initialize state
        state = {
            "user_query": user_query,
            "halt_execution": ExecutionState.ACTIVE,
            "subtasks": [],
            "subtask_results": {}
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
        return self.knowledge_graph_agent(state)

    def reflection_agent(self, user_query, rag_response, kg_data):
        """
        Reflection agent that analyzes if combined RAG+KG responses fully answer the user query.
        Removed supplementary information generation.
        """
        # Skip reflection if either source is missing
        if not rag_response or not kg_data:
            return None, []

        reflection_prompt = f"""
        You are a medical reflection agent. Analyze if the available information fully answers the user's query:

        USER QUERY: {user_query}

        INFORMATION FROM DOCUMENT SEARCH:
        {rag_response}

        INFORMATION FROM KNOWLEDGE GRAPH:
        {kg_data.get('kg_answer', 'No knowledge graph data available')}

        Your task:
        1. Identify what specific information the user is asking for.
        2. Determine if the combined information fully addresses all aspects of the query.
        3. List any MISSING INFORMATION that neither source provides.
        4. DO NOT repeat information already covered by the sources.

        Format your response as:
        QUERY INTENT: [Summarize what the user is specifically asking]
        COVERAGE ANALYSIS: [0-100% indicating how completely the sources answer the query]
        MISSING INFORMATION: [List specific aspects not covered by either source]

        If coverage is 100%, just write "COMPLETE".
        """

        # Get reflection analysis from LLM
        try:
            reflection_result = self.local_generate(reflection_prompt, max_tokens=800)
            print(f"🔍 Reflection result: {reflection_result}")

            # Extract missing information points
            missing_info = []
            if "MISSING INFORMATION:" in reflection_result:
                missing_section = reflection_result.split("MISSING INFORMATION:")[1].strip()
                if missing_section and "COMPLETE" not in missing_section:
                    # Extract missing points as a list
                    missing_raw = missing_section.split("\n")
                    for point in missing_raw:
                        clean_point = re.sub(r"^\d+\.\s*", "", point.strip())
                        if clean_point and len(clean_point) > 5:
                            missing_info.append(clean_point)

            # No longer generate supplementary information
            return None, missing_info

        except Exception as e:
            print(f"⚠️ Error in reflection process: {e}")
            return None, []


    def generate_response(self, user_input, user_type="User / Family"):
          """Generate response using RAG, KG, and adapt based on user type. Ask follow-up questions if needed."""
          if not user_input.strip():
              return "", []

          if self.qa_chain is None:
              success, message = self.initialize_qa_chain()
              if not success:
                  return message, []

          try:
              # Check if this is a response to a follow-up question
              if self.followup_context["asked"]:
                  print("🔍 User responded to follow-up question")

              # First, check if we need follow-up questions
              missing_info = self.identify_missing_info(user_input, self.chat_history)

              # If critical information is missing, ask follow-up questions
              if missing_info and len(missing_info) > 0:
                  follow_up_prompt = f"""
                  I need a bit more information to provide a helpful response. Could you please tell me:

                  {missing_info[0]}

                  This will help me give you a more accurate assessment.
                  """

                  # Add this to the chat history but mark it as a follow-up
                  self.chat_history.append((user_input, follow_up_prompt))

                  # Return follow-up question instead of regular response
                  return follow_up_prompt.strip(), []

              # Reset follow-up context for next query
              self.followup_context = {"asked": False, "round": 0}

              # Regular response generation below
              formatted_history = self.format_chat_history()

              # RAG
              rag_response = self.qa_chain.invoke({
                  "question": user_input,
                  "chat_history": formatted_history
              })

              rag_content = rag_response["answer"]
              if "Helpful Answer:" in rag_content:
                  rag_content = rag_content.split("Helpful Answer:")[-1]

              raw_rag_content = rag_content.strip()
              rag_sources = [doc.page_content[:200] + "..." for doc in rag_response["source_documents"][:3]]

              # KG
              kg_data = self.process_with_knowledge_graph(user_input)
              kg_content = kg_data.get("kg_answer", "")
              raw_kg_content = kg_content.strip() if kg_content else ""

              all_sources = rag_sources
              if kg_data.get("disease"):
                  all_sources.append(f"KG Disease: {kg_data['disease']}")

              # When constructing the final response prompt, include context from follow-up responses
              full_user_context = user_input
              if self.chat_history and len(self.chat_history) >= 2:
                  # Include the last exchange if it seems to be related
                  prev_user_msg, prev_bot_msg = self.chat_history[-1]
                  if "more information" in prev_bot_msg or "please tell me" in prev_bot_msg:
                      full_user_context = f"Context from previous message: {prev_user_msg}\nCurrent message: {user_input}"

              prompt = f"""
             You are a medical information assistant providing evidence-based answers from verified sources.

USER QUERY: {full_user_context}

AVAILABLE INFORMATION:
- Retrieved Documents: {raw_rag_content}
- Knowledge Graph Data: {raw_kg_content}

RESPONSE GUIDELINES:

1. SOURCE PRIORITY:
   - First: Use information from Retrieved Documents
   - Second: Use information from Knowledge Graph
   - Third: If neither are available, reference information from these trusted organizations for relevant conditions:
    * Hypertension: 
      - American Heart Association (https://www.heart.org)
      - NHLBI (https://www.nhlbi.nih.gov)
      - Indian Heart Association (https://indianheartassociation.org)
    * Cardiovascular Disease: 
          - American College of Cardiology (https://www.acc.org)
          - CDC Heart Disease (https://www.cdc.gov/heartdisease)
          - Heart Care Foundation of India (https://www.heartcarefoundation.org)
    * Obesity: 
          - CDC Obesity (https://www.cdc.gov/obesity)
          - NIDDK Weight Management (https://www.niddk.nih.gov/health-information/weight-management)
          - Obesity Foundation India (https://obesityfoundationindia.org)
    * Type 2 Diabetes: 
          - American Diabetes Association (https://www.diabetes.org)
          - CDC Diabetes (https://www.cdc.gov/diabetes)
          - Diabetes India (https://www.diabetesindia.org)
    * Respiratory Infections: 
          - CDC Respiratory Diseases (https://www.cdc.gov/respiratory)
          - American Lung Association (https://www.lung.org)
          - National Centre for Disease Control India (https://ncdc.gov.in)


2. ATTRIBUTION REQUIREMENTS:
   - For Retrieved Documents: [Document Source]
   - For Knowledge Graph: [Knowledge Graph]
   - When citing information from these trusted organizations, format as a clickable Markdown link:
    "[Source: Organization Name](URL)" - for example: "[Source: American Heart Association](https://www.heart.org)"

3. RESPONSE FORMAT:
   - Use conversational, clear language suitable for general public
   - Organize information in logical sections with bullet points where helpful
   - Include these sections when information is available:
     * Possible causes or explanation
     * Recommended approaches (if source-supported)
     * Self-care advice (if appropriate and source-supported)
     * When to seek medical attention
   - End with a brief medical disclaimer

4. IMPORTANT RULES:
   - If you cannot find reliable information from any of the sources above, respond: "I don't have enough reliable information to answer this medical question. Please consult with a healthcare professional for accurate guidance."
   - Do not generate unsourced medical content
   - Keep responses focused and concise
   - Be reassuring while honest about medical concerns

DISCLAIMER TEXT TO USE:
"This information is not a substitute for professional medical advice. If symptoms persist or worsen, please consult with a qualified healthcare provider."
              Answer:
              """

              final_response = self.local_generate(prompt, max_tokens=800)
              self.chat_history.append((user_input, final_response))

              return final_response, all_sources

          except Exception as e:
              return f"Error generating response: {str(e)}", []

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
        st.image(logo, width=600)  # Adjust width as needed
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
                    
                    # Add feedback buttons
                    col1, col2 = st.columns([1, 10])
                    with col1:
                        if st.button("👍", key=f"thumbs_up_{i}"):
                            feedback_result = vote_message(
                                [(st.session_state.messages[i-1][0], msg)] if i > 0 else [("", msg)],
                                "thumbs_up",
                                0,
                                user_type
                            )
                            st.toast(feedback_result)
                    with col1:
                        if st.button("👎", key=f"thumbs_down_{i}"):
                            feedback_result = vote_message(
                                [(st.session_state.messages[i-1][0], msg)] if i > 0 else [("", msg)],
                                "thumbs_down",
                                0,
                                user_type
                            )
                            st.toast(feedback_result)

        # Chat input at the bottom of the conversation
        st.container()
        # This creates space to push the input to the bottom
        st.write("  \n" * 2)  

        # Message input - now will appear at the bottom
        if prompt := st.chat_input("Ask your medical question..."):
            # First, check if this is a medical query
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
                st.session_state.messages.append((prompt, True))
                st.chat_message("user").write(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        bot_response, sources = st.session_state.chatbot.generate_response(prompt, user_type)
                        
                        # Format sources with improved citation
                        formatted_sources = []
                        for src in sources:
                            if "rawdata.pdf" in src or "dxbook" in src.lower():
                                # Extract page number if available
                                page_match = re.search(r'page[_\s]?(\d+)', src, re.IGNORECASE)
                                page_num = page_match.group(1) if page_match else "N/A"
                                # Format as internal data with page reference
                                formatted_sources.append(f"[Internal Data: DxBook, Page {page_num}] {src}")
                            else:
                                formatted_sources.append(src)
                        
                        # Add formatted sources to response
                        source_info = "\n\nReferences:\n" + "\n\n".join([f"- {src}" for src in formatted_sources]) if formatted_sources else ""
                        full_response = bot_response + source_info
                        
                        st.write(full_response)
                        st.session_state.messages.append((full_response, False))
                        st.session_state.chatbot.chat_history.append((prompt, full_response))
            
            # Force a rerun to update the UI immediately
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
