from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

def load_rag_chain():
    print("⚙️ Loading RAG chain...")

    # Load vector DB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory="text_vdb",
        embedding_function=embeddings
    )
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Load LLM
    llm = OllamaLLM(model="llama3.2:1b")

    # Create prompt
    system_template = """You are a helpful legal assistant. 
    Use the following context to answer the user's question clearly and accurately.
    If the answer is not in the context, say "I don't have enough information to answer this."
    
    Context: {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])

    # Build chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("✅ RAG chain ready!")
    return rag_chain