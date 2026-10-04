"""
AI Appointment Booking Assistant - Streamlit Frontend

Simple ChatGPT-like interface for booking appointments.

Features:
- Clean chat interface
- Session persistence
- Auto-wake for cold starts
- Error handling
"""
import time
import uuid
from datetime import datetime

import requests
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

# Page configuration
st.set_page_config(
    page_title="AI Appointment Assistant",
    page_icon="📅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Get backend URL from secrets or environment
try:
    BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")
except FileNotFoundError:
    BACKEND_URL = "http://localhost:8000"

# Optional API key (must match API_KEY on the backend; leave unset for local use)
try:
    API_KEY = st.secrets.get("API_KEY", "")
except FileNotFoundError:
    API_KEY = ""
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# Timeout settings
REQUEST_TIMEOUT = 30  # seconds
WARMUP_TIMEOUT = 60   # seconds for cold starts


# =============================================================================
# Fun & Colorful CSS - Kid Designer Style! 🎨
# =============================================================================

def load_custom_css():
    """Apply fun, colorful styling"""
    st.markdown("""
        <style>
        /* Fun gradient background */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Hide streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Cool title with shadow */
        h1 {
            text-align: center;
            color: #ffffff;
            font-weight: 800;
            padding: 2rem 0;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
            font-size: 3rem !important;
        }
        
        /* Chat input styling - make it pop! */
        .stChatInputContainer {
            background: white;
            border-radius: 30px;
            padding: 10px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        
        /* User message - cool blue bubble */
        .stChatMessage[data-testid="user-message"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 15px 20px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .stChatMessage[data-testid="user-message"] p {
            color: white !important;
            font-size: 1.1rem;
            font-weight: 500;
        }
        
        /* AI message - bright green bubble */
        .stChatMessage[data-testid="assistant-message"] {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            border-radius: 20px;
            padding: 15px 20px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .stChatMessage[data-testid="assistant-message"] p {
            color: white !important;
            font-size: 1.1rem;
            font-weight: 500;
        }
        
        /* Message icons - bigger and bolder */
        .stChatMessage svg {
            width: 35px !important;
            height: 35px !important;
        }
        
        /* Error box - fun red */
        .stAlert {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            border-radius: 15px;
            border: none;
            font-weight: 600;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Buttons - super colorful */
        .stButton button {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 700;
            font-size: 1.1rem;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }
        
        /* Spinner - colorful animation */
        .stSpinner > div {
            border-top-color: #f5576c !important;
        }
        
        /* Chat container - white box with shadow */
        [data-testid="stChatMessageContainer"] {
            background: white;
            border-radius: 25px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Session State Management
# =============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"user-{uuid.uuid4().hex[:12]}"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "backend_ready" not in st.session_state:
        st.session_state.backend_ready = None


# =============================================================================
# Backend Communication
# =============================================================================

def check_backend_health() -> bool:
    """Check if backend is ready"""
    try:
        with st.spinner("🔄 Connecting to server..."):
            response = requests.get(f"{BACKEND_URL}/health", timeout=WARMUP_TIMEOUT)
            return response.status_code == 200
    except:
        return False


def send_message(message: str) -> str:
    """Send message to backend and get response"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": message},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()["reply"]
        else:
            return "❌ Sorry, something went wrong. Please try again."
            
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. Please try again."
    except:
        return "❌ Cannot connect to server. Please check if backend is running."


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application - Simple ChatGPT-like interface"""
    
    # Initialize
    initialize_session_state()
    load_custom_css()
    
    # Fun title with emoji
    st.title("🎉 AI Appointment Assistant 🤖")
    
    # Check backend on first load
    if st.session_state.backend_ready is None:
        st.session_state.backend_ready = check_backend_health()
    
    # Show error if backend not ready
    if not st.session_state.backend_ready:
        st.error("❌ Cannot connect to backend. Please ensure the server is running.")
        if st.button("🔄 Retry"):
            st.session_state.backend_ready = None
            st.rerun()
        return
    
    # Display all messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input with fun placeholder
    if prompt := st.chat_input("✨ Type your message here... ✨"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response with fun spinner
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking... 💭"):
                response = send_message(prompt)
            st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
