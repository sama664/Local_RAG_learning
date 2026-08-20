from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ollama_calculator import calculate

# 1. Setup Models & ChromaDB
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

llm = ChatOllama(model=LLM_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. RAG Chain for Team Queries
rag_prompt = ChatPromptTemplate.from_template("""
Answer using only the context below:
Context: {context}
Question: {question}
""")

def handle_team_query(query: str) -> str:
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    chain = rag_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})

# 3. Router Node (LLM decides route)
router_prompt = ChatPromptTemplate.from_template("""
Classify the user input into exactly one of two categories: 'CALCULATOR' or 'TEAM_INFO'.

Input: "Calculate 15% tip on 80 dollars" -> CALCULATOR
Input: "What does a Platform team do?" -> TEAM_INFO
Input: "What is 45 * 12?" -> CALCULATOR
Input: "Tell me about stream-aligned teams" -> TEAM_INFO

Input: {query}
Category (Output ONLY the word 'CALCULATOR' or 'TEAM_INFO'):
""")

router_chain = router_prompt | llm | StrOutputParser()

# 4. Master Execution Loop
def process_user_request(query: str):
    category = router_chain.invoke({"query": query}).strip().upper()
    print(f"\n[Router Choice]: Selected -> {category}")

    if "CALCULATOR" in category:
        result = calculate(query)
    else:
        result = handle_team_query(query)

    print(f"Response:\n{result}\n")

# if __name__ == "__main__":
#     print("--- Testing Unified Router Pipeline ---")
#
#     # Test 1: Math Search
#     process_user_request("What is 345 multiplied by 18?")
#
#     # Test 2: Vector Store Search
#     process_user_request("What are the main goals of a Stream-aligned team?")


if __name__ == "__main__":
    print("=" * 50)
    print("Interactive Router Active!")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        user_input = input("\nEnter your query: ").strip()

        # Exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        process_user_request(user_input)