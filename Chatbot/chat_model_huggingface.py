import os
from huggingface_hub import HfApi

# Get the token from an environment variable
test_token = os.getenv("HF_TOKEN")
if not test_token:
    raise ValueError("Environment variable 'HF_TOKEN' is not set. Please set it before running the script.")

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