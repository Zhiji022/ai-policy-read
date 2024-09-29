
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import evaluate

from utils.vector_store import get_default_documents

from datasets import Dataset

def generate_ragas_testset(save_path='data/testset.csv', num_qa_pairs=5):
    documents = get_default_documents()
    
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo")
    critic_llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    distributions = {
        simple: 0.5,
        multi_context: 0.4,
        reasoning: 0.1
    }
    
    testset = generator.generate_with_langchain_docs(documents, num_qa_pairs, distributions, with_debugging_logs=True)
    testset_df = testset.to_pandas()
    testset_df.to_csv(save_path)
    
    return testset_df


def get_evaluation_dataset(rag_chain, csv_path='data/testset.csv', overwrite=False):
    
    if overwrite:
        test_df = generate_ragas_testset(csv_path)
    else:
        try:
            test_df = pd.read_csv(csv_path)
        except:
            test_df = generate_ragas_testset(csv_path)
    
    test_questions = test_df["question"].values.tolist()
    test_groundtruths = test_df["ground_truth"].values.tolist()
    
    answers = []
    contexts = []

    for question in test_questions:
        response = rag_chain.invoke({"question" : question})
        answers.append(response["response"])
        contexts.append([context.page_content for context in response["context"]])
        
    evaluation_dataset = Dataset.from_dict({
        "question" : test_questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truth" : test_groundtruths
        })
    
    return evaluation_dataset

def evaluate_rag(rag_chain, metrics, csv_path='data/testset.csv'):
    ds = get_evaluation_dataset(rag_chain, csv_path)
    results = evaluate(ds, metrics)
    print(results)
    return results