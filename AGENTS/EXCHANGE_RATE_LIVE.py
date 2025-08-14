from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
import requests
import os 
from typing import Annotated
from langchain_core.tools import InjectedToolArg
from dotenv import load_dotenv
import json

load_dotenv("/Users/ssris/Desktop/RIMSAB/LANG/.env")
message=[]

@tool
def get_conversion_factor(base_currency:str,target_currency:str)->float:
    """
    given two currencies base and target this tool help to fetch the exchange rate of the mentioned currency
    """
    url=f"--apikey---"
    response=requests.get(url)
    return response.json()

@tool
def convert(base_currency_value:int,conversion_rate:Annotated[float,InjectedToolArg])->float:
    """
    given a currency conversion rate this calculate the target currecy value 
    """
    return base_currency_value*conversion_rate

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="Llama3-8b-8192",
    temperature=0.3
)

llm_with_tools=llm.bind_tools([get_conversion_factor,convert])
human_message=HumanMessage("What is the conversion factor between india and dubai and based on that can you convert 100 indian rupees to dubai currency")
message.append(human_message)
ai_res=llm_with_tools.invoke(message)

for tool in ai_res.tool_calls:
    if tool["name"]=="get_conversion_factor":
        tool_args=tool["args"]
        base_currency=tool_args["base_currency"]
        target_currency=tool_args["target_currency"]
        
        tool_message_1=get_conversion_factor.invoke(tool_args)
        conv_rate=tool_message_1["conversion_rate"]
        
        message.append(
            ToolMessage(
                tool_call_id=tool["id"],
                content=json.dumps({
                                       "tool":"get_conversion_factor",
                                       "base_currency":base_currency,
                                       "target_currency":target_currency,
                                       "conversion_rate":conv_rate
                                   })))
    if tool["name"] == "convert":
        tool["args"]["conversion_rate"] = conv_rate
        tool_message_2 = convert.invoke(tool["args"])
        base_currency_value = tool["args"]["base_currency_value"]

        message.append(
            ToolMessage(
                tool_call_id=tool["id"],
                content=json.dumps({
                    "tool": "convert",
                    "base_currency_value": base_currency_value,
                    "conversion_rate": conv_rate,
                    "converted_amount": tool_message_2
                })
            )
        )

final_result=llm.invoke(message)
print(final_result.content)