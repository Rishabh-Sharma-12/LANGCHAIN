import os 
from itertools import chain
from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel,Field

try:
    class Person(BaseModel):
        name:str=Field(description="the name of the person ")
        age: int=Field(description="the age of the person ")
        city:str=Field(description="the name of the city of the person")
        
    parser=PydanticOutputParser(pydantic_object=Person)

    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        max_new_tokens=300
    )

    model=ChatHuggingFace(llm=llm)

    template=PromptTemplate(
        template="""
        give me name ,age and city of a fictional place
        {text}
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={
            "format_instructions":parser.get_format_instructions()
        }
    )

    # prompt=template.format(text="Delhi Dwarka sec-19")

    # result=model.invoke(prompt)

    # fr=parser.parse(result.content)

    # print(fr)

    chain=(
        template
        |model
        |parser
    )

    result=chain.invoke({"text":"Delhi Dwarka sec-19"})
    print(result)


except Exception as e:
    print('ERROR:',str(e))
    

