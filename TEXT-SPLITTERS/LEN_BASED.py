import chunk
from lzma import CHECK_UNKNOWN
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.base_o365 import CHUNK_SIZE

text = """
A RAG architecture can be effectively tailored for tender document 
retrieval and analysis by customizing ingestion, retrieval, and generation
components to the unique characteristics of tender processes. This enables 
organizations to automate, accelerate, and enhance the quality of their tender 
management workflows.
"""

splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=5,
    separator=" "  # Correct spelling here
)

result = splitter.split_text(text)
print(result)

# ---------------------------------------

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("/Users/ssris/Desktop/RIMSAB/LANG/BANK_STM/STM/1751806669799.pdf")
docs=loader.load()

full_text="\n".join([doc.page_content for doc in docs])

splitter=CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    seprator=" "
)

result=splitter.split_text(full_text)

print(result[0])

# ------------------------------------------


from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("/Users/ssris/Desktop/RIMSAB/LANG/BANK_STM/STM/1751806669799.pdf")
docs=loader.load()

full_text="\n".join([doc.page_content for doc in docs])

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

result=splitter.split_text(full_text)

print(result[0])

# ---------------------------------------------------------

from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_core.documents import Document

doc=Document(
    page_content="""
    from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("/Users/ssris/Desktop/RIMSAB/LANG/BANK_STM/STM/1751806669799.pdf")
docs=loader.load()

full_text="\n".join([doc.page_content for doc in docs])

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

result=splitter.split_text(full_text)

print(result[0])
    """
)

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=10
)

result=splitter.split_documents([doc])

print(result)
