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

json_schema={
    "title":"Review",
    "type":"object",
    "properties":{
        "cgpa":{
            "title":"cgpa",
            "description":"this is the cgpa of value of the student",
            "type":"number",
            "exclusiveMinimum":1,
            "exclusiveMaximum":10
        },
        "summary": {
            "title": "Summary",
            "description": "It is the summary of the content",
            "type": "string"
        },
        "sentiment": {
            "title": "Sentiment",
            "description": "This is the sentiment if the review i.e positive or negative",
            "type": "string"
        },
        "pros": {
            "title": "Pros",
            "description": "Pro added to review",
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "cons": {
            "title": "Cons",
            "description": "cons added to review",
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "name": {
            "title": "Name",
            "description": "Name of author of the review",
            "type": "string"
        }
    },
     "required": ["cgpa", "summary", "sentiment"]
}
structured_model=model.with_structured_model(json_schema)
result=structured_model.invoke("""
    review
""")