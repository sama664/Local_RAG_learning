from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize local model
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. Strict prompt template for mathematical evaluation
calc_prompt = ChatPromptTemplate.from_template("""
You are a precise mathematical calculator.
Solve the following math expression or word problem step-by-step, and state the final result clearly.

Problem: {expression}

Answer:
""")

calc_chain = calc_prompt | llm | StrOutputParser()

def calculate(expression: str) -> str:
    return calc_chain.invoke({"expression": expression})

if __name__ == "__main__":
    test_query = "What is (125 * 8) + (450 / 5)?"
    print(f"Query: {test_query}\n" + "-" * 40)
    print(calculate(test_query))