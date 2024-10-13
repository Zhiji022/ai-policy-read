from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# embedding model

FINE_TUNE_MODEL_ID = 'snowflake-arctic-embed-xs'
FINE_TUNED_EMBEDDING = HuggingFaceEmbeddings(model_name=f"jimmydzj2006/{FINE_TUNE_MODEL_ID}_finetuned_aipolicy")

MODEL_ID = 'Snowflake/snowflake-arctic-embed-m-v1.5'
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=MODEL_ID)

# rag chat model
RAG_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)