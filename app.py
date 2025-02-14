from flask import Flask, request, jsonify, render_template
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone as pt, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
import os
from dotenv import load_dotenv
from langchain import PromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv('PINCECONE_API_KEYS')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INDEX_NAME = "applabqatar"

# Initialize Pinecone
pc = pt(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

docsearch = PineconeVectorStore(embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY), index=index)

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def upsert_to_pinecone(text_chunks):
    uuids = [str(uuid4()) for _ in range(len(text_chunks))]
    docsearch.add_documents(documents=text_chunks, ids=uuids)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    text_chunks = process_pdf(file_path)
    upsert_to_pinecone(text_chunks)
    return jsonify({"message": "File processed and upserted successfully"})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question")

    retriever = docsearch.as_retriever(search_kwargs={'k': 10})

    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8, max_tokens=4096, api_key=OPENAI_API_KEY)
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

    response = qa({"query": question})
    
    return jsonify({"answer": response['result']})

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
