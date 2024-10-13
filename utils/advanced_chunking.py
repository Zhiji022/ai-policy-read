import os

import tiktoken
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import CSVLoader

import pymupdf4llm
import pdfplumber

import re
from collections import Counter
import pandas as pd

######Load documents by markdown########

def replace_newlines(text):
    # Replace consecutive newlines (two or more) with the same number of <br>
    text = re.sub(r'\n{2,}', '\n\n', text)
    # Replace single newlines with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Ensure there is a blank line before headings
    text = re.sub(r'([^\n])\n(#+)', r'\1\n\n\2', text)
    text = re.sub(r'([^\n|#])(#+)', r'\1\n\n\2', text)
    # Remove page breakers
    text = re.sub(r'\n\n-----\n\n', ' ', text)
    
    return text

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(text)
    return len(tokens)

def get_markdown_documents(path, pages, margins, **kwargs):
    md = pymupdf4llm.to_markdown(path, pages=pages, margins=margins, force_text=True)
    md = replace_newlines(md)
    
    chunk_size = kwargs.get('chunk_size')
    chunk_overlap = kwargs.get('chunk_overlap')
    
    markdown_splitter = MarkdownTextSplitter(chunk_size = chunk_size,
                                        chunk_overlap = chunk_overlap,
                                        length_function = tiktoken_len,
                                        )
    documents = markdown_splitter.create_documents([md])
    return documents

#####Load tables##########

def get_pages(path):
    text = pymupdf4llm.to_markdown(path, page_chunks=True, margins=(10,70), force_text=True)
    text_pages = [d['metadata']['page']-1 for d in text if not d['tables']]
    table_pages = [d['metadata']['page']-1 for d in text if d['tables']]
    print(f'text pages: {text_pages}')
    print(f'table pages: {table_pages}')
    return text_pages, table_pages

def clean_up_table(table):
    table = [[i for i in r if i is not None] for r in table]
    rows_cnt = Counter([len(r) for r in table])
    if rows_cnt[1]>2 or rows_cnt[3]==0:
        return None, None, None
    
    gov_id = []
    action = []
    if len(table[-1]) == 1:
        action.append(table.pop()[0])
    if len(table[0]) == 1:
        gov_id.append(table.pop(0)[0])
        try:
            df = pd.DataFrame(table[1:], columns=['Action ID', 'Suggested Action', 'GAI Risks'])
        except:
            df = None
            pass
    else:
        df = pd.DataFrame(table, columns=['Action ID', 'Suggested Action', 'GAI Risks'])
    return df, gov_id, action

def extract_and_process_tables(path, table_pages):
    pdf = pdfplumber.open(path)
    
    table_settings = {"vertical_strategy": "lines", 
                        "horizontal_strategy": "lines",
                        "snap_y_tolerance": 20}
    
    tables = []
    dfs = []
    gov_id = []
    actions = []
    for p in table_pages:
        table = pdf.pages[p].extract_tables(table_settings)
        tables.extend(table)
        
    for t in tables:
        df, gid, action = clean_up_table(t)
        dfs.append(df)
        if gid:
            gov_id.extend(gid)
            
        if action:
            actions.extend(action)          
    
    df = pd.concat(dfs)
    dsc = pd.DataFrame(list(zip(gov_id, actions)))    
    
    df.to_csv('data/actions.csv', header=True, index=False)
    dsc.to_csv('data/tasks.csv', header=False, index=False)
    
    return True

def get_table_documents(path, field_names=None):
       
    csv_loader = CSVLoader(file_path=path,
                            csv_args={'delimiter': ',',
                                        'quotechar': '"',
                                        'fieldnames': field_names
                            })
    documents = csv_loader.load()
    os.remove(path)
    return documents


######Final call#########
    
def get_enhanced_documents(**kwargs):
    doc1_path = 'data/Blueprint-for-an-AI-Bill-of-Rights.pdf'
    md_documents1 = get_markdown_documents(doc1_path, pages=list(range(1,73)), margins=(10,40), **kwargs)

    doc2_path = 'data/NIST.AI.600-1.pdf'
    text_pages, table_pages = get_pages(doc2_path)
    extract_and_process_tables(doc2_path, table_pages)
    table_documents1 = get_table_documents('data/actions.csv', ['Action ID', 'Suggested Action', 'GAI Risks'])
    table_documents2 = get_table_documents('data/tasks.csv')
    md_documents2 = get_markdown_documents(doc2_path, text_pages, margins=(10, 70), **kwargs)
    return [*md_documents1, *md_documents2, *table_documents1, *table_documents2]