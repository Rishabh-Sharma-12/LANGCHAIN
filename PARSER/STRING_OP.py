from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from huggingface_hub import login
from dotenv import load_dotenv
import os

# --- 1. Setup and Authentication ---
load_dotenv()
# Use the standard Hugging Face environment variable
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# --- 2. Define the desired JSON output structure ---
class Summary(BaseModel):
    """Pydantic model for a summary."""
    detailed_summary: str = Field(description="A detailed summary of the topic.")
    five_line_summary: str = Field(description="A concise, five-line summary of the detailed summary.")

# --- 3. Provide Few-Shot Examples ---
# Here we define one or more examples to show the model what we want.
examples = [
    {
        "topic": "the sun",
        "output": """
{
    "detailed_summary": "The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, heated to incandescence by nuclear fusion reactions in its core. It is the most important source of energy for life on Earth.",
    "five_line_summary": "The Sun is our solar system's central star.\\nA hot ball of plasma powered by fusion.\\nIt radiates light and heat.\\nThis is the primary energy source for Earth.\\nLife as we know it depends on it."
}
"""
    },
]

# --- 4. Initialize Model and Parser ---
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=500  # Increased tokens for JSON output
)
model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser(pydantic_object=Summary)

# --- 5. Create a Prompt with Few-Shot Examples ---
# This prompt formats each example into a human message and an AI response.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Provide a detailed summary of {topic}, then create a five-line summary from that."),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# The final prompt combines system instructions, few-shot examples, and the user's question.
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that provides summaries and formats them as JSON. Follow the user's instructions and use the provided JSON schema.\n{format_instructions}"),
        few_shot_prompt,
        ("human", "Provide a detailed summary of {topic}, then create a five-line summary from that."),
    ]
)

# --- 6. Build and Invoke the Chain ---
try:
    chain = final_prompt | model | parser
    result = chain.invoke({
        "topic": "black holes",
        "format_instructions": parser.get_format_instructions()
    })
    
    print("--- Detailed Summary ---")
    print(result['detailed_summary'])
    print("\n--- Five-Line Summary ---")
    print(result['five_line_summary'])

except Exception as e:
    print("Error during model invocation:", str(e))