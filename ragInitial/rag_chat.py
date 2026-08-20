import os
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Attach to existing local vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. Local Model & Conversational Prompt
llm = ChatOllama(model="llama3.2", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant answering questions using the context provided below.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# 3. Memory store per session
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 4. Chain setup
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "history": lambda x: x.get("history", [])
        }
        | prompt
        | llm
)

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 5. Interactive Terminal Loop
if __name__ == "__main__":
    print("Local RAG Chat Active. Type 'exit' to quit.\n" + "-"*40)
    config = {"configurable": {"session_id": "session_1"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = conversational_chain.invoke({"question": user_input}, config=config)
        print(f"\nLlama3.2: {response.content}")