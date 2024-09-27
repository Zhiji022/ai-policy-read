from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# embedding model
MODEL_ID = "Snowflake/snowflake-arctic-embed-m"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=MODEL_ID)

# rag chat model
RAG_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)