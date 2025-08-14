from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv("/Users/ssris/Desktop/RIMSAB/LANG/.env")
message=[]

@tool
def multiply(a:int,b:int)->int:
    """
    given two numbers a&b this tool returns there products
    """
    return a*b

# print(multiply.invoke({'a':3,'b':10}))
# print(multiply.description)
# print(multiply.args)

# print(multiply.name)

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="Llama3-8b-8192",
    temperature=0.3
)

llm_w_t=llm.bind_tools([multiply])
human_message=HumanMessage("can you show me the result if we multiply 20 with 100 ")
message.append(human_message)

ai_message=llm_w_t.invoke(message)
# print(ai_message.tool_calls[0]['args'])

tool_result=[]
for tool in ai_message.tool_calls:
    result=multiply.invoke(tool['args'])
    tool_result.append(result)
    message.append(ToolMessage(tool_call_id=tool["id"],content=str(result)))
    
# print(message)
final_res=llm.invoke(message)
print(final_res.content)


