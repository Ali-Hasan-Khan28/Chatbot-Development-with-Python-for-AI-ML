
from flask import Flask, request, jsonify, render_template
from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
from pinecone import Pinecone as pt, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from utils.Prompt_preprocessing import preprocess
from utils.Upload_PDF import process_pdf, upsert_to_pinecone
from utils.conversation import converse



# Load environment variables
load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.getenv('PINCECONE_API_KEYS')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INDEX_NAME = "applabqatar"

# Initialize Pinecone
pc = pt(api_key=PINECONE_API_KEY)

pc.delete_index("applabqatar")

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME, 
        dimension=1536, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # or your specific region
        )
    )
index = pc.Index(INDEX_NAME)
docsearch = PineconeVectorStore(embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY), index=index)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
    upsert_to_pinecone(text_chunks,docsearch)
    return jsonify({"message": "File processed and upserted successfully"})

@app.route('/ask', methods=['POST'])
def ask_question():
    # Check if the index contains any data
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)

    if total_vectors == 0:
        return jsonify({"answer": "No document uploaded. Please upload a PDF file first."}), 400

    data = request.json
    question = data.get("question")

    temp_val = preprocess(question)
    response = converse(docsearch,temp_val,OPENAI_API_KEY,memory,question)
    
    return jsonify({"answer": response['result']})

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
