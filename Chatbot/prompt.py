import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage


# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

# Initialize ChatGroq model
chat_model = ChatGroq(
    api_key=api_key,
    model="llama3-8b-8192"
)

st.set_page_config(page_title="Llama Chatbot", page_icon="💬", layout="centered")

# CSS for a clean, dark chat UI
st.markdown(
    """
    <style>
    .chat-container {
        max-width: 720px;
        margin: 0 auto;
        padding: 24px;
        font-family: 'Inter', sans-serif;
        background: #1e1e2f;
        color: #eee;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .chat-message {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 16px;
        max-width: 80%;
        line-height: 1.4;
        word-wrap: break-word;
    }
    .chat-user {
        background: #3a3a55;
        color: #d0d0ff;
        align-self: flex-end;
        text-align: right;
        margin-left: auto;
    }
    .chat-bot {
        background: #4c4cff;
        color: white;
        align-self: flex-start;
        text-align: left;
        margin-right: auto;
    }
    .chat-history {
        display: flex;
        flex-direction: column;
    }
    .input-area {
        margin-top: 24px;
        display: flex;
        gap: 12px;
        width: 100%;
    }
    .input-textbox {
        flex-grow: 1;
        padding: 12px;
        font-size: 16px;
        border-radius: 12px;
        border: none;
        outline: none;
    }
    .send-button {
        padding: 0 24px;
        background: linear-gradient(135deg, #6b46c1, #805ad5);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    .send-button:hover {
        background: linear-gradient(135deg, #805ad5, #6b46c1);
    }
    .send-button:disabled {
        background: #666;
        cursor: not-allowed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 Llama Chatbot")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Clear chat history button
if st.button("🔄 Clear Chat"):
    st.session_state.history = []
    st.rerun()

def generate_response(user_input):
    """Call the Groq Chat model to generate a response to user_input."""
    messages = []
    for role, content in st.session_state.history:
        if role == "user":
            messages.append(HumanMessage(content))
        elif role == "bot":
            messages.append(AIMessage(content))
    messages.append(HumanMessage(user_input))
    
    try:
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"Error: {e}")
        return "Sorry, something went wrong."
# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if st.session_state.history:
        for role, message in st.session_state.history:
            if role == "user":
                st.markdown(f'<div class="chat-message chat-user">{message}</div>', unsafe_allow_html=True)
            elif role == "bot":
                st.markdown(f'<div class="chat-message chat-bot">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="chat-message chat-bot">Hi! I am your Llama chatbot. How can I assist you today?</div>',
            unsafe_allow_html=True,
        )

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message", key="input_text", max_chars=512)
        submit_button = st.form_submit_button("Send")

        if submit_button and user_input.strip():
            st.session_state.history.append(("user", user_input.strip()))
            with st.spinner("Thinking..."):
                response = generate_response(user_input.strip())
            st.session_state.history.append(("bot", response))
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
