from langchain_community.retrievers import ArxivRetriever

# Create the retriever
retriever = ArxivRetriever(
    load_max_docs=2,
    get_full_documents=False  # Fixed typo here: added the second 'l' in full
)

# Query arxiv
docs = retriever.invoke("large language models")

# Print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content[:500])