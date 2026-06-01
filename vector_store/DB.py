from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    dimensions = 64
)

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db",
)

result = vector_store.similarity_search("What is used for data analysis?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vector_store.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)