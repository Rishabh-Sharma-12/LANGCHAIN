# SECURITY WARNING: Your token is now public - regenerate it immediately!
# 1. Go to https://huggingface.co/settings/tokens
# 2. Delete the current token
# 3. Create a new one with 'Read' permissions

# Test the current token (replace with your NEW token)
from huggingface_hub import HfApi

# DO NOT use this token - it's compromised
test_token = "REMOVED_TOKEN"

try:
    api = HfApi()
    user_info = api.whoami(token=test_token)
    print(f"Token works for: {user_info}")
    
    # If token works, test the chatbot
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    
    # Try different models that are more likely to work
    models_to_try = [
        "microsoft/DialoGPT-small",
        "gpt2",
        "facebook/blenderbot-400M-distill"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"\nTesting {model_name}...")
            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=test_token
            )
            
            model = ChatHuggingFace(llm=llm)
            result = model.invoke("What is India?")
            print(f"✓ SUCCESS with {model_name}:")
            print(result.content[:200] + "..." if len(result.content) > 200 else result.content)
            break
            
        except Exception as e:
            print(f"✗ {model_name} failed: {str(e)[:100]}...")
            continue
    
except Exception as e:
    print(f"Token is invalid or expired: {e}")
    print("\nSteps to fix:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token (delete the old one)")
    print("3. Give it 'Read' permissions")
    print("4. Use environment variables for security")

# SECURE SETUP for future use:
print("\n" + "="*60)
print("SECURE SETUP (use this after getting a new token):")
print("="*60)

secure_code = '''
# Method 1: Using environment variable (RECOMMENDED)
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Set in terminal: export HF_TOKEN="your_new_token_here"
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("Set HF_TOKEN environment variable")

llm = HuggingFaceEndpoint(
    repo_id="microsoft/DialoGPT-small",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is India?")
print(result.content)

# Method 2: Local alternative (no token needed)
from transformers import pipeline

# This runs on your local machine
generator = pipeline('text-generation', model='gpt2')
response = generator("What is India?", max_length=100, do_sample=True)
print(response[0]['generated_text'])
'''

print(secure_code)