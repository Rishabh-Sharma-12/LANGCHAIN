from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage ,SystemMessage
from dotenv import load_dotenv
import os,sys
from typing import TypedDict

def main():
    load_dotenv()
    api_key=os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Invalid API KEY")
        sys.exit(1)
    
    llm=ChatGroq(
        api_key=api_key,
        model="meta-llama/Llama-3.1-8B-Instruct"
    )
    
    user_input="""
    I was genuinely excited when I first unboxed the new laptop — the sleek design and bright display immediately caught my attention. However, the excitement didn’t last long. Within an hour, it started overheating, and the fan noise was unbearable. I tried contacting customer support, and while they were polite, they couldn’t offer a clear solution. Despite its impressive looks, the performance issues and poor after-sales support left me frustrated and disappointed.
    """
    result=llm.invoke(user_input)
    print(f"Raw output:{result.content}")
    
    class Review(TypedDict):
        summary:str
        sentiment:str
    
    structured_model=llm.with_structured_output(user_input)
    result_2=structured_model.invoke(user_input)
    print(f"formated data:{result_2}")
    
    
if __name__=="__main__":
    main()
    



    