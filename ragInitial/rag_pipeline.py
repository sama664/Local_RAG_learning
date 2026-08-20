import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuration Constants
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"


def setup_sample_document(filename="sample_data.txt"):
    """Creates a local file to serve as our document source for RAG."""
    sample_text = """
    Team Topologies framework identifies four core team types:
    1. Stream-aligned team: Aligned to a single flow of work from a segment of the business.
    2. Enabling team: Helps stream-aligned teams acquire missing capabilities or technical skills.
    3. Complicated-subsystem team: Responsible for a specific subsystem requiring deep specialized knowledge.
    4. Platform team: Enables stream-aligned teams to deliver value autonomously without needing deep infrastructure knowledge.

    It also defines three interaction modes: Collaboration, X-as-a-Service, and Facilitating.
    """
    with open(filename, "w") as f:
        f.write(sample_text.strip())
    return filename


def build_or_load_vectorstore(doc_path):
    """Loads text, splits into chunks, embeds locally, and persists to ChromaDB."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        print("--> Loading existing ChromaDB from disk...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        print("--> Ingesting document and building vector store...")
        loader = TextLoader(doc_path)
        docs = loader.load()

        # Chunking: 500 characters with overlap to preserve context boundaries
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        print(f"--> Indexed {len(chunks)} chunks into ChromaDB.")

    return vectorstore


def format_docs(docs):
    """Formats retrieved context documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    # 1. Prepare Document
    doc_path = setup_sample_document()

    # 2. Build Vector Store
    vectorstore = build_or_load_vectorstore(doc_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. Define LLM & RAG Prompt
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    template = """
    You are an assistant for question-answering tasks. 
    Use ONLY the following pieces of retrieved context to answer the question. 
    If you do not know the answer based on the context, say that you don't know.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Construct LCEL RAG Chain
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 5. Execute Query
    user_query = "What are the four core team types and what does a Platform team do?"
    print(f"\nUser Query: {user_query}\n")
    print("Generating Answer...\n" + "-" * 40)

    response = rag_chain.invoke(user_query)
    print(response)


if __name__ == "__main__":
    main()