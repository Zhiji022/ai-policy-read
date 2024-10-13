
import json
import argparse

from langchain_core.prompts import ChatPromptTemplate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.evaluation import evaluate_rag
from utils.prompts import RAG_PROMPT
from utils.vector_store import get_default_documents, get_vector_store
from utils.models import EMBEDDING_MODEL, RAG_LLM, FINE_TUNED_EMBEDDING
from utils.rag import RAGRunnables, create_rag_chain
from utils.advanced_chunking import get_enhanced_documents

# get CL arguments
parser = argparse.ArgumentParser()
parser.add_argument('chunking', type=str, help="Chunking strategy: chose between default or advanced")
parser.add_argument('model', type=str, help="Embedding model: chose between base or finetuned")

args = parser.parse_args()
chunking_strategy = args.chunking
embedding = args.model

if chunking_strategy == 'default':
    documents = get_default_documents()
elif chunking_strategy == 'advanced':
    documents = get_enhanced_documents(chunk_size=400, chunk_overlap=50)
else:
    raise ValueError('Invalid chunking type')
print(f'chunking strategy: {chunking_strategy}')

if embedding == 'base':
    model = EMBEDDING_MODEL
    emb_dim = 768
elif embedding == 'finetuned':
    model = FINE_TUNED_EMBEDDING
    emb_dim = 384
else:
    raise ValueError('Invalid model type')
print(f'model: {model}')

# create rag chain to be evaluated
rag_runnables = RAGRunnables(
                        rag_prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT),
                        vector_store = get_vector_store(documents, model, emb_dim=emb_dim),
                        llm = RAG_LLM
                    )
rag_chain = create_rag_chain(rag_runnables.rag_prompt_template, 
                                 rag_runnables.vector_store, 
                                 rag_runnables.llm)

metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

results = evaluate_rag(rag_chain, metrics)

with open(f'data/eval_results/{chunking_strategy}_chunking_{embedding}_model.json', 'w') as f:
    json.dump(results, f)
