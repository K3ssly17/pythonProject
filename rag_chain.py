from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

db_dir = 'vectorstore'


embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="llama3.1")

template = """
You are an AI assistant. Use only the provided content.

Context:
{context}

Question:
{question}

Answer clearly.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

def answer_question(query):

    docs = retriever.get_relevant_documents(query)  # <-- fixed method

    context_text = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(context=context_text, question=query)

    return llm.invoke(final_prompt)
