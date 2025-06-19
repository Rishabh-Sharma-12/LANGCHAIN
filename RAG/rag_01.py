from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Load PDF
loader = PyPDFLoader("/Users/ssris/Desktop/DATA-STRUCT/Ai Mantra/tendors/2025_NHIDC_860902_1/RFP.pdf")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings with Ollama
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create vectorstore
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)


# 5. Create retriever
retriever = vectorstore.as_retriever()

def get_llm(
    model_name="LLaMA3-8b-8192",
    temperature=0.1,
    max_tokens=2048
):
    api_key=os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ API KEY NOT FOUND")
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
llm=get_llm()

# 7. Build RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
   
# 8. Ask question
query = "What are the key details mentioned in the NHAI document?"

prompt=f"""
            You are a laymen-freindly but keen observent expert of government office
            Extract minimum qualification requiremnet and format them in structrued Tables,
            Focus on Financial, techincal and experience required{query} <|assistant|>
            """
         
response = qa.invoke({"query": query,"Prompt":prompt})

# 9. Print result
print(response["result"])
