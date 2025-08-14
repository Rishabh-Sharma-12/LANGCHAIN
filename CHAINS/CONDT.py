# Importing necessary modules from LangChain, Pydantic, and typing
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import Literal

# Define a Pydantic model for feedback sentiment classification
class Feedback(BaseModel):
    # The Sentiment field can only be 'positive' or 'negative'
    Sentiment: Literal['positive', 'negative'] = Field(
        description='the sentiment of the text majorly the review of the client'
    )

# Create a parser that will parse the LLM output into the Feedback Pydantic model
parser_2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt template for sentiment classification
prompt_1 = PromptTemplate(
    template="""
    genrate the sentiment of the positive or negative over the fedback of the customer the following text provided to you 
    ->{text}
    {format_instructions}
    """,
    input_variables=["text"],
    # Partial variable to inject format instructions for the output parser
    partial_variables={
        "format_instructions": parser_2.get_format_instructions()
    }
)

# Prompt template for generating a response to positive feedback
prompt_2 = PromptTemplate(
    template="""
    write an appropriate responst to the POSITIVE feedback you recieved from the  
    ask for appropriate ratiing over the feedback offer discount and offers over refralls and make ask for rating
    -> {text}
    """,
    input_variables=["text"]
)

# Prompt template for generating a response to negative feedback
prompt_3 = PromptTemplate(
    template="""
    write an appropriate responst to the NEGATIVE feedback you recieved from the  
    offer assistance over customer care and make it feel like you are very much concerned about the exoerience
    ->{text}
    """,
    input_variables=["text"]
)

# Initialize the HuggingFace LLM endpoint with the specified model and parameters
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=300
)

# Wrap the LLM endpoint for chat-style interaction
model = ChatHuggingFace(llm=llm)

# Create a chain that classifies the sentiment of the input text
classify_chain = (
    prompt_1 | model | parser_2
)
# Example: class_result = classify_chain.invoke({"text": "The food was amazing! I loved the taste and the service was excellent."})

# Switch the parser to a string output parser for the response generation step
parser_2 = StrOutputParser()

# ----------------------------------------------------------------------------
# Branching logic: depending on the classified sentiment, select the appropriate response prompt
branch_chain = RunnableBranch(
    # If sentiment is positive, use prompt_2 to generate a positive response
    (lambda x: x.Sentiment == 'positive', prompt_2 | model | parser_2),
    # If sentiment is negative, use prompt_3 to generate a negative response
    (lambda x: x.Sentiment == 'negative', prompt_3 | model | parser_2),
    # If sentiment is not found, return a default message
    RunnableLambda(lambda x: "could not find the sentiment")
)

# Compose the full chain: classify sentiment, then branch to the appropriate response
chain = (classify_chain | branch_chain)

# Example invocation: analyze the sentiment and generate a response for the given text
result = chain.invoke({"text": "IT was PATHETICLY PERFECT TO BE GIVEN TO DOGS ,DO YOU THINK WE ARE DOGS ONLK."})

# Print the generated response
print(result)

# Print the ASCII representation of the chain's computation graph for debugging/visualization
chain.get_graph().print_ascii()