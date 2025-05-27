import streamlit as st
import os
from dotenv import load_dotenv
import uuid # To generate unique session IDs (basic example)


# --- Google Cloud Imports ---
# Install: pip install google-cloud-documentai google-generativeai
from google.cloud import documentai
import google.generativeai as genai

# --- Configuration ---
# Replace with your Google Cloud Project ID and Document AI Processor ID

# Load environment variables from .env file
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("DOCUMENTAI_LOCATION")
PROCESSOR_ID = os.getenv("DOCUMENTAI_PROCESSOR_ID")

# Configure the Generative AI model
# You might use an API Key or authenticate via Google Cloud (recommended for production)
# Option 1: Using API Key (less secure for production)
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# model = genai.GenerativeModel('gemini-pro') # Choose your preferred model

# Option 2: Authenticating via Google Cloud (Recommended)
# Ensure your environment is authenticated (e.g., `gcloud auth application-default login`)
try:
    # If using Vertex AI, you'd use google.cloud.aiplatform
    # For this example, we'll use the general genai library which can often infer credentials
    model = genai.GenerativeModel('gemini-2.0-flash-lite') # Using a model available via the genai library
    st.sidebar.success("Generative AI model loaded.")
except Exception as e:
     st.sidebar.error(f"Generative AI client error: {e}")
     model = None # Set model to None if loading fails


# --- Initialize Document AI Client ---
try:    
    documentai_client = documentai.DocumentProcessorServiceClient()
    st.sidebar.success("Document AI client initialized.")
except Exception as e:
    st.sidebar.error(f"Document AI client error: {e}")
    documentai_client = None

# --- Streamlit App Setup ---
st.title("QNL - Talk to Google")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'document_text' not in st.session_state:
    st.session_state['document_text'] = None
if 'document_name' not in st.session_state:
    st.session_state['document_name'] = None
if 'session_id' not in st.session_state:
     st.session_state['session_id'] = str(uuid.uuid4()) # Basic session ID

# --- Functions ---

@st.spinner("Processing document...")
def process_document(uploaded_file):
    """Processes an uploaded file using Google Cloud Document AI."""
    if documentai_client is None:
        st.error("Document AI client is not initialized. Cannot process document.")
        return None

    mime_type = uploaded_file.type
    # Read the file content
    document_content = uploaded_file.getvalue()

    st.write(f"Processing file: {uploaded_file.name} ({mime_type})")

    # Online processing request
    request = documentai.ProcessRequest(
        name=documentai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID),
        raw_document=documentai.RawDocument(content=document_content, mime_type=mime_type)
    )

    try:
        result = documentai_client.process_document(request=request)
        document = result.document

        # Extract text
        # You can add more sophisticated parsing here based on document structure if needed
        text = document.text
        st.success(f"Document '{uploaded_file.name}' processed successfully.")
        return text

    except Exception as e:
        st.error(f"Error processing document with Document AI: {e}")
        return None

def get_generative_model_response(user_query, history, document_text=None):
    """Gets a response from the Generative AI model."""
    if model is None:
        return "Language model is not available."

    # Build the chat history for the Gemini API
    chat_history = []

    # Add system message
    system_message = (
        "You are a helpful assistant. "
        "If the question is not related to the document content, answer using your general knowledge. "
        "Keep your answers concise and relevant."
    )
    chat_history.append({"role": "user", "parts": [system_message]})

    # Add document context if available
    if document_text:
        doc_context = f"The following is text from a document:\n{document_text}\n"
        chat_history.append({"role": "user", "parts": [doc_context]})

    # Add previous conversation history
    for turn in history:
        chat_history.append({"role": turn["role"], "parts": [turn["content"]]})

    # Add the current user query
    chat_history.append({"role": "user", "parts": [user_query]})

    try:
        response = model.generate_content(chat_history)
        return response.text
    except Exception as e:
        st.error(f"Error getting response from Language Model: {e}")
        return "An error occurred while generating a response."


# --- Streamlit UI ---

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF or image file", type=["pdf", "png", "jpg", "jpeg", "tiff"])

if uploaded_file is not None and uploaded_file.name != st.session_state.get('last_uploaded_file'):
    # Only process if a new file is uploaded
    st.session_state['last_uploaded_file'] = uploaded_file.name
    st.session_state['document_text'] = None # Clear previous document
    st.session_state['document_name'] = None # Clear previous document name

    processed_text = process_document(uploaded_file)

    if processed_text:
        st.session_state['document_text'] = processed_text
        st.session_state['document_name'] = uploaded_file.name
        st.sidebar.success(f"Loaded text from '{uploaded_file.name}'")
        st.session_state['chat_history'] = [] # Clear history for new document context (optional)


# Display document info if loaded
if st.session_state.get('document_name'):
    st.info(f"Currently chatting about: **{st.session_state['document_name']}**")
    # Optional: Add a way to clear the document context
    if st.button("Clear Document Context"):
        st.session_state['document_text'] = None
        st.session_state['document_name'] = None
        st.session_state['chat_history'] = [] # Clear history too
        st.rerun() # Rerun the app to update the display

# Display chat history
for message in st.session_state['chat_history']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me a question...")

if user_input:
    # Add user message to history and display
    st.session_state['chat_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from the model
    # Pass the relevant document text if available
    response = get_generative_model_response(
        user_query=user_input,
        history=st.session_state['chat_history'][:-1], # Pass history excluding the current user turn
        document_text=st.session_state['document_text']
    )

    # Add model response to history and display
    st.session_state['chat_history'].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)