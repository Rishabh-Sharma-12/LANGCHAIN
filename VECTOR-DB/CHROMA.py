from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/Users/ssris/Desktop/RIMSAB/LANG/.env")

# Create some dummy documents
doc1 = Document(page_content="The sun rises in the east and sets in the west.", metadata={"id": 1})
doc2 = Document(page_content="Water boils at 100 degrees Celsius at sea level.", metadata={"id": 2})
doc3 = Document(page_content="The capital of France is Paris.", metadata={"id": 3})
doc4 = Document(page_content="The fuck you bahi ya you ris.", metadata={"id": 4})
documents = [doc1, doc2, doc3]

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(documents)

# Use OpenAI Embeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Create and persist Chroma vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)

vector_store.add_documents([doc4])

results = vector_store.similarity_search("sunset", k=3)

for r in results:
    print("Page Content:", r.page_content)
    print("Metadata:", r.metadata)

