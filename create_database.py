from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv
import time

load_dotenv()

data = PyPDFLoader(r"D:\Downloads\Ai\RAG\document_loader\deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embeddings = MistralAIEmbeddings(model="mistral-embed")


vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

