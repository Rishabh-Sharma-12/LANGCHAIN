from itertools import chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.output_parsers import StructuredOutputParser ,ResponseSchema
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
import os

try:
    # login(os.getenv("HUGGINGFACE_KEY"))

    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        max_new_tokens=300
    )

    model=ChatHuggingFace(llm=llm)

    schemas=[
        ResponseSchema(
            name="fact_1",
            description="The first fact about the topic"
        ),
        ResponseSchema(
            name="fact_2",
            description="The second fact about the topic"
        ),
        ResponseSchema(
            name="fact_3",
            description="The third fact about the topic"
        )
    ]

    parser=StructuredOutputParser.from_response_schemas(schemas)

    template=PromptTemplate(
        template="Give 3 facts about topic {topic} in the following format {format_instructions}",
        input_variables=['topic'],
        partial_variables={
            'format_instructions':parser.get_format_instructions()
        }
    )

    prompt=template.format(topic="Cars")
    result=model.invoke(prompt)
    # chain=(
    #     template
    #     |model
    #     |parser
    # )
    # result=chain.invoke({"topic":"Cars"})    
    fr=parser.parse(result.content)
    print(result)
except Exception as e:
    print("Error:", str(e))
