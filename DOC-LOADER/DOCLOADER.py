# ============================================================
# SECTION 1: CSV FILE LOADING
# ============================================================
# This section demonstrates how to load documents from a CSV file using LangChain's CSVLoader.
# Replace '----path----' with the actual path to your CSV file.
# The loader reads the CSV and loads its contents as documents.
# Uncomment and modify the code below to use.

from langchain_community.document_loaders import CSVLoader

csv_path = "----path----"  # Path to your CSV file
loader = CSVLoader(csv_path)
docs = loader.load()
print(docs[10])  # Print the 11th document loaded from the CSV


# ============================================================
# SECTION 2: PDF FILE LOADING (SINGLE FILE)
# ============================================================
# This section shows how to load a single PDF file using PyPDFLoader.
# Replace '----path----' with the path to your PDF file.
# Uncomment and modify the code below to use.

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("----path----")  # Path to your PDF file
doc = loader.load()
print(doc[0].metadata)  # Print metadata of the first page/document


# ============================================================
# SECTION 3: PDF FILE LOADING (MULTIPLE FILES IN DIRECTORY)
# ============================================================
# This section demonstrates loading multiple PDF files from a directory using DirectoryLoader and PyPDFLoader.
# All PDF files matching the glob pattern in the specified directory will be loaded.
# Replace '----path----' with the path to your directory containing PDF files.

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path="----path----",      # Directory containing PDF files
    glob="*.pdf",             # Pattern to match PDF files
    loader_cls=PyPDFLoader    # Loader class to use for each file
)

docs = loader.lazy_load()  # Use lazy_load for memory efficiency with large datasets
for i, doc in enumerate(docs):
    print(doc.page_content)  # Print the content of each loaded document
    if i == 50:              # Stop after 51 documents
        break


# ============================================================
# SECTION 4: TEXT FILE LOADING (SINGLE FILE)
# ============================================================
# This section shows how to load a single text file using TextLoader.
# Replace '----path----' with the path to your text file.

from langchain_community.document_loaders import TextLoader

loader = TextLoader('----path----', 'utf-8')  # Path to your text file and encoding
docs = loader.load()
for doc in docs:
    print(doc.page_content)  # Print the content of each loaded document


# ============================================================
# SECTION 5: TEXT FILE LOADING (MULTIPLE FILES IN DIRECTORY)
# ============================================================
# This section demonstrates loading multiple text files from a directory using DirectoryLoader and TextLoader.
# Replace '----path----' with the path to your directory containing text files.

from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    path="----path----",      # Directory containing text files
    glob="*.txt",             # Pattern to match text files
    loader_cls=TextLoader     # Loader class to use for each file
)

docs = loader.load()
print(docs[0].page_content)  # Print the content of the first loaded document


# ============================================================
# SECTION 6: LLM CHAIN EXAMPLE WITH TEXT FILE
# ============================================================
# This section demonstrates how to use a loaded text file as input to an LLM chain for summarization.
# It uses ChatGroq as the LLM, but you can swap in other LLMs as needed.
# Make sure to set your environment variables and update the path to your .env and text file.

from itertools import chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
import os

# Load environment variables from .env file (update path as needed)
load_dotenv("----path----/.env")

# Step 1: Authenticate to Hugging Face Hub (if using HuggingFace LLMs)
from huggingface_hub import login
login(token=os.getenv("HUGGING_TOKEN"))

# Step 2: Load a text file for summarization
loader = TextLoader('----path----/Req_AIMANTRA.txt')  # Path to your text file
docs = loader.load()

# Step 3: Combine text from all loaded documents into a single string
text_content = "\n".join(doc.page_content for doc in docs)

# Step 4: Set up the LLM (ChatGroq in this example; update API key and model as needed)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="Llama3-8b-8192"
)

# Step 5: Define a prompt template for summarization
prompt = PromptTemplate(
    template="""
    tell me a summary and what do you understand with this text-
    {text}
    """,
    input_variables=['text']
)

# Step 6: Set up the output parser
parser = StrOutputParser()

# Step 7: Create the chain (prompt -> LLM -> parser)
chain = prompt | llm | parser

# Step 8: Invoke the chain with the loaded text content
result = chain.invoke({'text': text_content})
print(result)

# Note: The following line may raise an error if 'docs' is a list and does not have a dtype() method.
# print(docs.dtype())


# ============================================================
# SECTION 7: WEB PAGE LOADING (STATIC AND DYNAMIC)
# ============================================================
# This section demonstrates how to load web pages using WebBaseLoader (for static pages)
# and SeleniumURLLoader (for dynamic pages requiring JavaScript rendering).
# Replace the URLs with your target web pages.

# --- Static Web Page Loading Example ---
from langchain_community.document_loaders import WebBaseLoader

url = "----url----"
loader = WebBaseLoader(url)
doc = loader.load()
print(len(doc))
print(doc[0].page_content)

# --- Dynamic Web Page Loading Example (with Selenium) ---
from langchain_community.document_loaders import SeleniumURLLoader

URLs = ["----url----"]  # List of URLs to load
loader = SeleniumURLLoader(URLs)
docs = loader.load()
print(docs)