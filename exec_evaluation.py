
import json

from langchain_core.prompts import ChatPromptTemplate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.evaluation import evaluate_rag
from utils.prompts import RAG_PROMPT
from utils.vector_store import get_default_documents, get_vector_store
from utils.models import EMBEDDING_MODEL, RAG_LLM
from utils.rag import RAGRunnables, create_rag_chain

# create rag chain
documents = get_default_documents()

rag_runnables = RAGRunnables(
                        rag_prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT),
                        vector_store = get_vector_store(documents, EMBEDDING_MODEL),
                        llm = RAG_LLM
                    )
rag_chain = create_rag_chain(rag_runnables.rag_prompt_template, 
                                 rag_runnables.vector_store, 
                                 rag_runnables.llm)

metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

results = evaluate_rag(rag_chain, metrics)
json.dump(results, open('eval_results.json', 'wb'))