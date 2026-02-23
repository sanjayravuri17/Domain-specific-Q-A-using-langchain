from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
def setup_rag_pipeline(query):
# --- Configuration ---
    chroma_persist_dir = "./chroma_db"  # Path where ChromaDB is persisted

    # Load existing ChromaDB
    vectorstore = Chroma(
        collection_name="rag-chroma",
        persist_directory=chroma_persist_dir,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )
    retriever = vectorstore.as_retriever()

    # Initialize Ollama chat model
    model_local = ChatOllama(model="llama3.2")

    # Define RAG prompt
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {query}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    # Combine retriever and model for RAG
    after_rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    # Example query
    answer = after_rag_chain.invoke(query)
    print("Answer:(Rag)", answer)
    return answer
