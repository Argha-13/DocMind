from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()

# ---------------- MODELS ----------------
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

model = ChatMistralAI(
    model="mistral-small-2603"
)

# ---------------- VECTOR DB ----------------
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),

        ("human", """
Context:
{context}

Question:
{question}
""")
    ]
)


# ---------------- ADVANCED RAG ----------------
def rag_advanced(
    query,
    vectorstore,
    llm,
    top_k=3,
    min_score=0.2,
    return_context=False
):
    """
    Advanced RAG:
    - similarity threshold
    - confidence score
    - sources
    - optional context return
    """

    # Get docs with relevance score
    results = vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=top_k
    )

    # Filter using min_score
    filtered_results = [
        (doc, score)
        for doc, score in results
        if score >= min_score
    ]

    # No relevant docs found
    if not filtered_results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0,
            "context": ""
        }

    # Create context
    context = "\n\n".join([
        doc.page_content
        for doc, score in filtered_results
    ])

    # Create source info
    sources = [
        {
            "source": doc.metadata.get(
                "source",
                "unknown"
            ),
            "page": doc.metadata.get(
                "page",
                "unknown"
            ),
            "score": round(score, 3),

            "preview":
            doc.page_content[:120] + "..."
        }

        for doc, score
        in filtered_results
    ]

    # Highest score = confidence
    confidence = max(
        score
        for doc, score
        in filtered_results
    )

    # Prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # LLM response
    response = llm.invoke(final_prompt)

    output = {
        "answer": response.content,
        "sources": sources,
        "confidence": round(confidence, 3)
    }

    # Optional context
    if return_context:
        output["context"] = context

    return output


print("RAG system created")
print("Press 0 to exit")

while True:
    query = input("\nYou: ")

    if query == "0":
        break

    result = rag_advanced(
        query=query,
        vectorstore=vectorstore,
        llm=model,
        top_k=3,
        min_score=0.3,
        return_context=True
    )

    print("\nAI:", result["answer"])
    print("\nConfidence:", result["confidence"])
    print("\nSources:", result["sources"])

    if "context" in result:
        print(
            "\nContext Preview:",
            result["context"][:300]
        )