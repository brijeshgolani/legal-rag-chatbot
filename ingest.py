from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest(pdf_path):
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("🔢 Creating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="text_vdb"
    )
    print(f"Successfully stored {len(chunks)} chunks in vector database!")

if __name__ == "__main__":
    pdf_path = r"C:\Users\Mohak Golani\OneDrive\Documents\bns_v2.pdf"
    ingest(pdf_path)