from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4




def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def upsert_to_pinecone(text_chunks,docsearch):
    uuids = [str(uuid4()) for _ in range(len(text_chunks))]
    docsearch.add_documents(documents=text_chunks, ids=uuids)