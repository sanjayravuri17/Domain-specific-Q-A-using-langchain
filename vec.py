from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
# The `embeddings` import is no longer needed if you use OllamaEmbeddings directly
# from langchain_community import embeddings 

# Initialize the models
# 1. Initialize the embedding model once and store it in a variable
ollama_embeddings = OllamaEmbeddings(model='nomic-embed-text')

# Load PDF and split into chunks
pdf_path = "PPT_Text analytics.pdf"
loader = PyPDFLoader(pdf_path)
docs_list = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

# Create / load ChromaDB
chroma_persist_dir = "./chroma_db"

print("Creating and persisting new ChromaDB.")
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    persist_directory=chroma_persist_dir,
        # 2. Use the variable here for creation
        embedding=ollama_embeddings,
    )
vectorstore.persist()
print ("ChromaDB created and persisted.")