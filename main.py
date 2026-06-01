from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()

embedding_model = MistralAIEmbeddings(model="mistral-embed")

model = ChatMistralAI(model="mistral-small-2603")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retrievers = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5  #diversity vs relevance
    }
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """ You are a helpful AI assistant.

        Use ONLY the provided context to answer the question.

        If the answer is not present in the context,
        say: "I could not find the answer in the document." """),

        ("human", """context: {context},
        question: {question}""")
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retrievers.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = model.invoke(final_prompt)

    print(f"\n AI: {response.content}")
    