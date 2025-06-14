
Markdown
# DxAI-Agent

DxAI-Agent is an experimental, context-aware medical chat assistant designed to provide users—both healthcare professionals and laypeople—with helpful, accurate, and contextually relevant medical information.

## Features

- **Medical Q&A Chatbot:** Ask medical questions and get contextually relevant, easy-to-understand answers.
- **User-Type Adaptation:** Select between "User / Family" and "Physician" modes for tailored explanations and terminology.
- **Integrated Knowledge Sources:** Combines information from an internal knowledge graph, uploaded/simulated PDF documents, and the Gemini LLM.
- **Self-Reflection & Answer Refinement:** The system evaluates its own answers for completeness and may supplement missing information automatically.
- **Symptom Extraction & Confirmation:** Automatically detects symptoms in user queries, suggests additional possible symptoms, and can ask for confirmation to improve accuracy.
- **Triage Guidance (For Family Users):** Integrates advice on when to seek urgent, emergency, or routine medical attention.
- **Feedback Logging:** Users can vote on responses and submit detailed feedback for continuous improvement.
- **Session Memory:** Supports multi-turn conversations with context retention.

---

> **Disclaimer:**  
> DxAI-Agent does not provide medical diagnoses, nor is it a substitute for professional medical advice.  
> **Always consult a qualified healthcare provider for personal health concerns.**

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Shardz689/Dx_AI_NEW.git
cd Dx_AI_NEW

### 2. Install Dependencies
Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt

### Main Dependencies:

**streamlit**
**sentence-transformers**
**langchain**
**langchain_community**
**faiss-cpu**
**neo4j**
**google-generativeai**
**langchain_google_genai**
**python-dotenv**
**Pillow**
**numpy**
**torch**

> **If you encounter issues with torch or numpy, see their official installation instructions for your platform.**

### 3. Environment Variables
Create a .env file in your project root (or export these variables in your environment):

env
GEMINI_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY
NEO4J_URI=bolt+s://YOUR_NEO4J_HOST
NEO4J_USER=YOUR_NEO4J_USERNAME
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD

The application has default/test keys, but for production or real use, you must set your own credentials.

### 4. Data & Customization
**PDFs for RAG:**
Place your medical PDF files in the project root or update the HARDCODED_PDF_FILES list in app3.py with their paths.
Default example: rawdata.pdf

**Knowledge Graph**
Ensure your Neo4j instance is running and loaded with an appropriate medical schema (nodes for symptom, disease, treatment, etc.).

### 5. Running the App
```bash
streamlit run app3.py

The app will open in your browser at http://localhost:8501.

### Usage
**Select User Type:**
Use the sidebar to choose "User / Family" or "Physician" mode.
**Ask Questions:**
Enter your medical question in the chat input.
**ymptom Confirmation:**
If the system suggests more symptoms, confirm or add any relevant symptoms.
**Review Answers:**
Answers include source pathways and a disclaimer.
**Feedback:**
Use thumbs up/down and the feedback form to help improve the assistant.

###Project Structure
Code
├── app3.py                   # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # (You create) Environment variables
├── rawdata.pdf               # (Optional) Example PDF for document retrieval
├── feedback_log.csv          # (Auto-generated) User feedback
├── detailed_feedback_log.csv # (Auto-generated) Detailed feedback
├── Zoom My Life.jpg          # (Optional) Logo image
└── ...                       # Other files and cache folders

### Key Components
**DocumentChatBot**: Core class managing embeddings, vector DB, LLM, KG connection, and all answer logic.
**Knowledge Graph Agent**: Matches symptoms to diseases and treatments using Cypher queries.
**Retrieval-Augmented Generation (RAG)**: Finds relevant passages from PDFs via FAISS vector search.
**Reflection/Refinement**: The LLM checks its own answers for completeness and can fill gaps.
**Streamlit UI**: Handles chat interface, user-type selection, symptom checklists, and feedback.

### Troubleshooting
**Initialization Errors:**
Ensure all environment variables are set and dependencies installed.
**Neo4j Connection Issues:**
Check your Neo4j URI, credentials, and network access.
**PDFs Not Found:**
Place your PDFs in the correct path or update the filenames in the code.
**LLM/Embedding Failures:**
Make sure your API keys are valid and all model dependencies are installed.

### License
This project is for research, educational, and demonstration purposes only.

### Acknowledgments
**Google Gemini**
**LangChain**
**Sentence Transformers**
**Neo4j Graph Database**
**Streamlit**

> **Always consult a qualified healthcare professional for medical advice and emergencies.**

