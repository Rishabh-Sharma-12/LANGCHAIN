from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from huggingface_hub import login
# from dotenv import load_dotenv
# import os
# load_dotenv()

api_token=input("HUGGINGFACEHUB_ACCESS_TOKEN : ")
login(token=api_token)
if not api_token:
    raise ValueError("HUgging face api not found")
try:
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        task="text-generation"
        )
        
except Exception as e:
    print("1 error_here{e}")
    
model=ChatHuggingFace(llm=llm)

while True:
  user_input=input("YOU:")
  if user_input=="`":
    break
  try:
    result=model.invoke(user_input)
    print(f"AI: {result.content}")
  except Exception as e:
      print(f"2 error here{e}")

  
