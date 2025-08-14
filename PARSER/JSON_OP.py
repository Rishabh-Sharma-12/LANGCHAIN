# -------------------- WORKFLOW --------------------
# 1. Authenticate to Hugging Face Hub using token from environment variable.
# 2. Initialize the LLM endpoint (Meta-Llama-3-8B-Instruct) for text generation.
# 3. Wrap the LLM with a chat interface.
# 4. Set up a JSON output parser to extract structured data.
# 5. Create a prompt template that asks for name, city, and age from a given text,
#    and includes format instructions for JSON output.
# 6. Format the prompt with the input text.
# 7. Invoke the model with the formatted prompt to get a response.
# 8. Parse the model's response into JSON using the output parser.
# 9. Print the extracted information.
# --------------------------------------------------

# from itertools import chain 
# from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from huggingface_hub import login

# import os

# login(token=os.getenv("HUGGINGFACE_TOKEN"))

# llm=HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation",
#     max_new_tokens=300
# )

# model=ChatHuggingFace(llm=llm)
# parser=JsonOutputParser()

# template_1=PromptTemplate(
#     template="""
#     give me name ,city,age of the person in the text given
#     text :{text}
#     {format_essential}
#     """,
#     input_variables=["text"],
#     partial_variables={
#         'format_essential':parser.get_format_instructions()
#     }
# )
# input_text="My name is Rishabh Sharma. I live in Jaipur and I'm 24 years old."
# prompt=template_1.format(text=input_text)
# result=model.invoke(prompt)
# fr=parser.parse(result.content)
# print(fr)


#--------------------------------------------------------------------

# -------------------- Workflow Overview --------------------
# 1. Authenticate to Hugging Face Hub using token from environment variable.
# 2. Initialize the LLM endpoint (Meta-Llama-3-8B-Instruct) for text generation.
# 3. Wrap the LLM with a chat interface.
# 4. Set up a JSON output parser to extract structured data.
# 5. Create a prompt template that asks for city, name, and age from a given text,
#    and includes format instructions for JSON output.
# 6. Format the prompt with the input text.
# 7. Build a chain: prompt -> model -> parser.
# 8. Invoke the chain with the input and print the extracted information.
# -----------------------------------------------------------

from itertools import chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from huggingface_hub import login

import os

# Step 1: Authenticate to Hugging Face Hub
login(token=os.getenv("HUGGING_TOKEN"))

# Step 2: Initialize the LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=300
)

# Step 3: Wrap the LLM with a chat interface
model = ChatHuggingFace(llm=llm)

# Step 4: Set up a JSON output parser
parser = JsonOutputParser()

# Step 5: Create a prompt template with format instructions
temp_1 = PromptTemplate(
    template="""
    give me city, name and age of the person in the text given
    {text}
    {format_essential}
    """,
    input_variables=["text"],
    partial_variables={
        'format_essential': parser.get_format_instructions()
    }
)

# Step 6: Format the prompt with the input text
input_text = "My name is Rishabh Sharma. I live in Jaipur and I'm 24 years old."

# Step 7: Build the chain (prompt -> model -> parser)
chain = (
    temp_1
    | model
    | parser
)

# Step 8: Invoke the chain and print the result
result = chain.invoke({'text': input_text})
print(result)