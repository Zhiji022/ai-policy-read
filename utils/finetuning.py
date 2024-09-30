import uuid
from typing import List
import random
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict
import json

def train_test_split_documents(documents: List, ratios: List= [6, 2, 2]):
    
    """
    Randomize and split documents into train/test/validation sets
    """
    
    doc_length = len(documents)
    splits = [int(i*doc_length/sum(ratios)) for i in ratios]
    documents = random.shuffle(documents)
    
    return  documents[:splits[0]], documents[splits[0]:splits[1]], documents[splits[1]:]

def set_documents_ids(documents):
    id_set = set()
    for document in documents:
        id = str(uuid.uuid4())
        while id in id_set:
            id = uuid.uuid4()
        id_set.add(id)
        document.metadata["id"] = id
    return documents

def load_finetuning_datasets(path):
    
    ds = json.load(open(path, 'r'))
    questions = ds["questions"]
    contexts = ds["relevant_contexts"]
    corpus = ds["corpus"]
    return questions, contexts, corpus

def generate_questions(documents, chain, n_questions, file_name):
    questions = {}
    relevant_docs = defaultdict(list)
    
    # generate question ids
    ids = set([d.metadata["id"] for d in documents])
    qids = []
    for i in range(len(documents)*n_questions):
        id = str(uuid.uuid4())
        while id in ids:
            id = uuid.uuid4()
        qids.append(id)
    assert len(qids) == len(documents)*n_questions
    
    for document in tqdm(documents, desc='Generating questions...'):
        results = chain.invoke({'context': document.page_content, 'n_questions': n_questions}).content
    
        results = literal_eval(results)
        
        if len(results) != n_questions:
            print(results)
            raise Exception('Wrong number of questions!')
        for q in results:
            qid = qids.pop()
            questions[qid] = q
            relevant_docs[qid].append(document.metadata['id'])

    # save to jsonl
    corpus = {item.metadata["id"] : item.page_content for item in documents}

    data = {
        "questions" : questions,
        "relevant_contexts" : relevant_docs,
        "corpus" : corpus
    }

    with open(file_name, "w") as f:
        json.dump(data, f)
    
    return questions, relevant_docs, corpus