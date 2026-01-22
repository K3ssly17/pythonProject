from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

data_dir = 'data'
db_dir = 'vectorstore'

def load_documents():
    docs = []
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.endswith('.txt'):  # fixed extension and colon
            loader = TextLoader(path, autodetect_encoding=True)
            docs.extend(loader.load())
    print(f"Loaded {len(docs)} documents.")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(  # fixed typo
        chunk_size=2000,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} document chunks.")
    return chunks

def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    db.persist()
    print("Vector store created and persisted.")

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)
