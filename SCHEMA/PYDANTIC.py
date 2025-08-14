#-------------------------------------

from lanchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model=ChatOpenAI
model.invoke("""
             Review contet
             """)
#schema-typedict
class Review(TypedDict):
    summary:str
    sentiment:str

structured_model=model.with_structured_output(Review)
result=structured_model.invoke(
    """
    review content
    """
)

print(result.content)
print(result["summary"])
print(result["sentiment"])

#------------------------------------------
from pydantic import BaseModel

class student(BaseModel):
    name:str
    
new_s={"name":"Rishabh Sharma"}
Student=student(**new_s)
print(Student)

#-------------------------------------------------------------

from pydantic import BaseModel
from typing import Optional

class student(BaseModel):
    name:str="Rishabh"
    age:Optional[int]=1112
    
new_s2={}#will return default values
new_s2={'name':"RAmesh",'age':33}
Student=student(**new_s2)
print(Student)
    
#-------------------------------------------------------------

from pydantic import EmailStr,BaseModel

class Student(BaseModel):
    email:EmailStr

s_2={"email":"rishabh_222@gmail.com"}
student=Student(**s_2)
print(s_2)

#-------------------------------------------------------------

from pydantic import BaseModel ,Field
from typing import Optional,Annotated

class Review(BaseModel):
    cgpa:float=Field(description="this is the value of sgpa",gt=1,lt=10)
    summary:str=Field(description="It is the summary of the content")
    sentiment:str=Field(description="This is the sentiment if the review i.e positive or negative")
    pros:Optional[list[str]]=Field(description="Pro added to review")
    cons:Optional[list[str]]=Field(description="cons added to review")
    name:Optional[str]=Field(description="Name of author of the review")

s_3={"cgpa":2,"summary":'This is my name',"sentiment":'hello',"pros":["a","b","c","d","e","f"],"cons":["a","b","c","d","e","f"],"name":"Rishabh"}
student=Review(**s_3)
print(student)