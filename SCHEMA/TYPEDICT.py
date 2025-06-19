from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int
new_person:Person={
    "Rishabh",23
}
print(new_person)

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

from lanchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict
from typing import Annotated

#schema-typedict
class Review(TypedDict):
    summary:Annotated[str,"It is the summary of the content"]
    sentiment:Annotated[str,"This is the sentiment if the review i.e positive or negative"]

#------------------------------------------

from lanchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Optional ,Annotated


#schema-typedict
class Review(TypedDict):
    summary:Annotated[str,"It is the summary of the content"]
    sentiment:Annotated[str,"This is the sentiment if the review i.e positive or negative"]
    pros:Annotated[Optional[list[str]],"Pro added to review"]
    cons:Annotated[Optional[list[str]],"negative ,con added to review"]
    name:Annotated[Optional[str],"Name of author of the review"]

#------------------------------------------

    
    