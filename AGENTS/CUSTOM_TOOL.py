from langchain_groq import ChatGroq
from langchain_core.tools import tool

@tool
def mult(a:int,b:int)->int:
    "multiply two numbers"
    return a*b
    
result=mult.invoke({'a':3,'b':4})
print(result)
print(result.description)
print(result.args)