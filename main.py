from fastapi import FastAPI, UploadFile, File
import fitz  # PyMuPDF
from pinecone import Pinecone
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

nltk.download("punkt")
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

app = FastAPI()
@app.get("/")
def home():
    return {"message": "Chatbot is running!"}
YOUR_PINECONE_API_KEY = os.getenv("PINCECONE_API_KEYS")
print(YOUR_PINECONE_API_KEY)

pc = Pinecone(api_key="pcsk_2Wf51V_3db8TfcJj91FkyhKNPhDwainW3yWC2ErC3z4T8hoLx8fowaFYfhAbLS5TK22n1q")
index = pc.Index("applabqatar")
# index = pinecone.Index("applabqatar")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def preprocess_text(text):
    """Apply NLP preprocessing (Tokenization, Stopword Removal, Lemmatization)"""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)


def extract_named_entities(text):
    """Extract key Named Entities from text using Spacy"""
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities


def extract_text_from_pdf(pdf_path):
    """Extract text from the PDF and apply summarization"""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])

    # Summarize long text
    if len(text) > 1000:
        text = summarizer(text[:1024], max_length=300, min_length=100, do_sample=False)[0]["summary_text"]

    return preprocess_text(text)

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = f"temp_{file.filename}"
    with open(pdf_path, "wb") as buffer:
        buffer.write(await file.read())

    text = extract_text_from_pdf(pdf_path)

    # Chunk and embed text
    sentences = sent_tokenize(text)
    doc_embeddings = [embeddings.embed_query(sent) for sent in sentences]

    # Store in Pinecone
    for i, vector in enumerate(doc_embeddings):
        index.upsert([(f"{file.filename}_{i}", vector, {"text": sentences[i]})])

    return {"message": "PDF uploaded and processed successfully!", "entities": extract_named_entities(text)}

def rank_text_with_tfidf(query, texts):
    """Rank retrieved text chunks based on TF-IDF relevance"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query] + texts)
    scores = tfidf_matrix[0].dot(tfidf_matrix.T).toarray()[0][1:]
    ranked_texts = [texts[i] for i in sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)]
    return " ".join(ranked_texts[:3])  # Top 3 ranked chunks


@app.post("/chat/")
async def chat(query: str):
    results = index.query(query, top_k=5, include_metadata=True)
    retrieved_texts = [r["metadata"]["text"] for r in results["matches"]]

    # Rank retrieved texts using TF-IDF
    context = rank_text_with_tfidf(query, retrieved_texts)

    chat_model = ChatOpenAI(model="gpt-4", temperature=0)
    response = chat_model.predict(f"Context: {context} \n\nAnswer the question: {query}")

    return {"response": response}
