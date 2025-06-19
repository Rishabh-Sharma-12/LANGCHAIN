import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import re

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

code_model =ChatGroq(
    api_key=api_key,
    model="llama3-70b-8192"
)

st.set_page_config(page_title="Llama Chatbot", page_icon="😎", layout="centered")

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
        background: #87ceeb; /* Light blue */
        color: #333; /* Darker text color */
        align-self: flex-end;
        text-align: right;
        margin-left: auto;
    }
    .chat-user.code {
        background: #7A288A; /* Purple */
        color: #fff; /* White text color */
    }

    .chat-bot.code {
        background: #87ceeb; /* Light blue */
        color: #333; /* Darker text color */
    }
    .chat-bot {
        background: #7A288A; /* Purple */
        color: #fff; /* White text color */
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
        color: #fff; /* White text color */
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

st.title("😎 Llama Chatbot")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Clear chat history button
if st.button("🐽 Clear Chat"):
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

def generate_code_response(user_input):
    message=[]
    for role,content in st.session_state.history:
        if role=="user":
            message.append(HumanMessage(content))
        elif role=="bot":
            message.append(AIMessage(content))
    message.append(HumanMessage(user_input))
    
    try:
        response=code_model.invoke(message)
        return response.content
    except Exception as e:
        st.error (f"Error:{e}")
        return "#ERROR"
        
    
    
    # try:
    #     response=code_model.invoke(message)
    #     return response.content
    # except Exception as e:
    #     st.error (f"Error:{e}")
    #     return "#ERROR"

def render_message(role, message, model_class=""):
    if role == "user":
        st.markdown(f'<div class="chat-message chat-user {model_class}">{message}</div>', unsafe_allow_html=True)
    elif role == "bot":
        # Detect code block
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", message, re.DOTALL)
        if code_blocks:
            # Show message before code (if any)
            parts = re.split(r"```(?:\w+)?\n.*?```", message, flags=re.DOTALL)
            if parts[0].strip():
                st.markdown(f'<div class="chat-message chat-bot {model_class}">{parts[0]}</div>', unsafe_allow_html=True)
            for code in code_blocks:
                st.code(code.strip())
            if len(parts) > 1 and parts[1].strip():
                st.markdown(f'<div class="chat-message chat-bot {model_class}">{parts[1]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message chat-bot {model_class}">{message}</div>', unsafe_allow_html=True)
            
            
st.session_state.setdefault("model_mode","Chat")
model_mode=st.selectbox("Select model type",["Chat","Code"],index=["Chat","Code"].index(st.session_state["model_mode"]))
st.session_state["model_mode"]=model_mode
st.markdown(f"<p style='text-align:center; color:#ccc;'>🔍 Using <strong>{model_mode}</strong> model</p>", unsafe_allow_html=True)

# with st.container():
#     model_class = "code" if model_mode == "Code" else ""

#     if st.session_state.history:
#         for role, message in st.session_state.history:
#             if role == "user":
#                 st.markdown(f'<div class="chat-message chat-user {model_class}">{message}</div>', unsafe_allow_html=True)
#             elif role == "bot":
#                 st.markdown(f'<div class="chat-message chat-bot {model_class}">{message}</div>', unsafe_allow_html=True)
#     else:
#         st.markdown(
#             '<div class="chat-message chat-bot">Hi! I am your Llama chatbot. How can I assist you today?</div>',
#             unsafe_allow_html=True,
#         )

#     # Input form
#     with st.form(key="chat_form", clear_on_submit=True):
#         user_input = st.text_area("Type or paste your message/code here 👇", key="input_text", height=150)
#         submit_button = st.form_submit_button("Send")

#         if submit_button and user_input.strip():
#             st.session_state.history.append(("user", user_input.strip()))
#             with st.spinner("Thinking..."):
#                 if model_mode == "Chat":
#                     response = generate_response(user_input.strip())
#                 elif model_mode == "Code":
#                     response = generate_code_response(user_input.strip())
#             st.session_state.history.append(("bot", response))
#             st.rerun()

#     st.markdown("</div>", unsafe_allow_html=True)

model_class = "code" if model_mode == "Code" else ""

if st.session_state.history:
    for role, message in st.session_state.history:
        render_message(role, message, model_class)
else:
    st.markdown(
        '<div class="chat-message chat-bot">Hi! I am your Llama chatbot. How can I assist you today?</div>',
        unsafe_allow_html=True,
    )

# Add JavaScript to allow Shift+Enter or Enter to submit
st.markdown("""
    <script>
        const textarea = window.parent.document.querySelector('textarea');
        if (textarea) {
            textarea.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    window.parent.document.querySelector('button[kind="primary"]').click();
                }
            });
        }
    </script>
""", unsafe_allow_html=True)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Type or paste your message/code here 👇", key="input_text", height=150)

    # JavaScript: Shift+Enter to submit the form
    st.markdown("""
        <script>
        document.addEventListener("keydown", function(event) {
            const textarea = window.parent.document.querySelector('textarea');
            const sendButton = window.parent.document.querySelector('button[kind="primary"]');
            if (event.shiftKey && event.key === "Enter") {
                event.preventDefault();
                if (sendButton) sendButton.click();
            }
        });
        </script>
    """,
    unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 2])
    with col1:
        submit_button = st.form_submit_button("Send")
    with col2:
        clear_button = st.form_submit_button("Clear Chat")

if submit_button and user_input.strip():
   
    st.session_state.history.append(("user", user_input.strip()))
    with st.spinner("Thinking..."):
        if model_mode == "Chat":
            response = generate_response(user_input.strip())
        elif model_mode == "Code":
            response = generate_code_response(user_input.strip())
    st.session_state.history.append(("bot", response))
    st.rerun()

if clear_button:
    st.session_state.history = []
    st.rerun()



#----------------------------------------------------------------

