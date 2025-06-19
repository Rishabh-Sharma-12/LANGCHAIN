import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Invalid API key.")
        sys.exit(1)

    model_name = input("Enter the model name you want to check access for: ").strip()
    if not model_name:
        print("Invalid model name.")
        sys.exit(1)

    print(f"Checking access for model: {model_name}")

    try:
        chat_model = ChatGroq(api_key=api_key, model=model_name)
        test_message = [HumanMessage("Hello, this is a test to check model access.")]
        response = chat_model.invoke(test_message)
        print("Model access successful.")
        print("Response preview:")
        if isinstance(response.content, str):
            preview = response.content.strip()
            print(preview if len(preview) < 100 else preview[:10] + "...")
        else:
            print("Response format not supported.")
    except (ValueError, APIError) as e:
        print("Failed to access model.")
        print(f"Error: {e}")

if __name__ == "__main__":
    while True:
        main()