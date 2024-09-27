from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters.base import TextSplitter

from langchain_community.document_loaders import PyMuPDFLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_huggingface import HuggingFaceEmbeddings
from chainlit.types import AskFileResponse

from typing import List


def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(text)
    return len(tokens)

class ChunkDocument:
    '''
    Choose your document loader and text splitter and chunk the document
    '''
    def __init__(self, file_path: str, loader: BaseLoader, splitter: TextSplitter):
        self.file_path = file_path
        self.loader = loader
        self.splitter = splitter        
    
    def process_documents(self, **kwargs):
        '''
        Read a single document and chunk it
        '''
        docs = self.loader(self.file_path).load()
        chunks = self.splitter(**kwargs).split_documents(docs)
        print(len(chunks))
        return chunks
    
def get_default_documents():
    '''
    Process default documents under data folder
    '''
    chunking = ChunkDocument(file_path = 'data/Blueprint-for-an-AI-Bill-of-Rights.pdf', 
                        loader = PyMuPDFLoader, 
                        splitter = RecursiveCharacterTextSplitter
                        )
    chunks1 = chunking.process_documents(chunk_size = 300,
                                    chunk_overlap = 0,
                                    length_function = tiktoken_len
                                    )
    
    chunking = ChunkDocument(file_path = 'data/NIST.AI.600-1.pdf', 
                        loader = PDFPlumberLoader, 
                        splitter = RecursiveCharacterTextSplitter
                        )
    chunks2 = chunking.process_documents(chunk_size = 300,
                                    chunk_overlap = 0,
                                    length_function = tiktoken_len
                                    )
    
    return [*chunks1, *chunks2]
    

def process_uploaded_file(file: AskFileResponse):
    '''
    Process upleaded file using PyMuPDFLoader
    '''
    chunking = ChunkDocument(file_path = file.path, 
                        loader = PyMuPDFLoader, 
                        splitter = RecursiveCharacterTextSplitter
                        )
    return chunking.process_documents(chunk_size = 300,
                                    chunk_overlap = 0,
                                    length_function = tiktoken_len
                                    )

def get_vector_store(documents: List, embedding_model: HuggingFaceEmbeddings, emb_dim=768) -> QdrantVectorStore:
    '''
    Return a qdrant vector score retriever
    '''
    
    qdrant_client = QdrantClient(":memory:")

    qdrant_client.create_collection(
        collection_name="ai-policy",
        vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE)
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="ai-policy",
        embedding=embedding_model
        )

    vector_store.add_documents(documents)

    return vector_store