from utils.advanced_chunking import get_enhanced_documents
from utils.models import RAG_LLM, FINE_TUNE_MODEL_ID
from utils.prompts import QA_PROMPT
from utils.finetuning import *

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from tqdm.autonotebook import tqdm, trange
from torch.utils.data import DataLoader, Dataset

from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


# Prepare data for finetuning
try:
    training_questions, training_relevant_contexts, training_corpus = load_finetuning_datasets("data/training_dataset.json")
    test_questions, test_relevant_contexts, test_corpus = load_finetuning_datasets("data/test_dataset.json")
    val_questions, val_relevant_contexts, val_corpus = load_finetuning_datasets("data/val_dataset.json")
    print("Finetuning datasets loaded!")
    
except:
    print('Generating dataset for finetuning...')

    documents = get_enhanced_documents(chunk_size=400, chunk_overlap=50)
    documents = set_documents_ids(documents) # assign a uuid for each document in metadata
    print(documents[0])
    training_split_documents, val_split_documents, test_split_documents = train_test_split_documents(documents)

    qa_chain = ChatPromptTemplate.from_template(QA_PROMPT) | RAG_LLM

    training_questions, training_relevant_contexts, training_corpus = generate_questions(training_split_documents, qa_chain, 10, "data/training_dataset.json")
    val_questions, val_relevant_contexts, val_corpus = generate_questions(val_split_documents, qa_chain, 4, "data/val_dataset.json")
    test_questions, test_relevant_contexts, test_corpus = generate_questions(test_split_documents, qa_chain, 4, "data/test_dataset.json")
    

# Finetuning
BATCH_SIZE = 16
EPOCHS = 10

## data loader
examples = []
for query_id, query in training_questions.items():
    doc_id = training_relevant_contexts[query_id][0]
    text = training_corpus[doc_id]
    example = InputExample(texts=[query, text])
    examples.append(example)

loader = DataLoader(examples, batch_size=BATCH_SIZE)

## Model
model = SentenceTransformer(f'Snowflake/{FINE_TUNE_MODEL_ID}')

## Loss function
matryoshka_dimensions = [284, 256, 128, 64, 32]
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dimensions)

## evaluator
evaluator = InformationRetrievalEvaluator(val_questions, val_corpus, val_relevant_contexts)

## model training
warmup_steps = int(len(loader) * EPOCHS * 0.1)
model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path='finetuned_arctic',
    show_progress_bar=True,
    evaluator=evaluator,
    evaluation_steps=50,
)

## save model

model.push_to_hub(f"{FINE_TUNE_MODEL_ID}_finetuned_aipolicy")
print('Model pushed to huggingface')