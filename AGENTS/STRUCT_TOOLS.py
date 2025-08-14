from langchain.tools import StructuredTool
from pydantic import BaseModel,Field

class mult_class(BaseModel):
    a:int=Field(required=True,description="this is a number a")
    b:int=Field(required=True,description="this is a number b")
    
def mult(a:int,b:int)->int:
    return a*b

mult_tool=StructuredTool.from_function(
    func=mult,
    name="multiply",
    description="Multiply tool ",
    args_schema=mult_class
)

result=mult_tool.invoke({'a':6,'b':3})
print(result)