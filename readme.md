#Goal is Create learn LangChain Framework by building local RAG database, Chat appplication and multiple models selection using Router
**_FLOW: User--> Enter prompt --> Router --> RAG --> LLM --> Output_**
#MCP INCLUSION FOLLOW
# 1. Create virtual environment named '.venv'
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install all dependencies from requirements.txt
pip install -r requirements.txt

# 4. Create sample  application using team detials and try to see if this answering question is working or not.
python rag_pipeline.py

# 5. Create sample ch aplication using team details and try to see if this answering question is working or not.
python rag_chat.py

# 6. added calculator functionality and enable rag to analyzed inout and select proper models
python router_pipeline.py