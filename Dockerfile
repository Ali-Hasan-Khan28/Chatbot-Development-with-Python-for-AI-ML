FROM python:3.10-alpine
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m textblob.download_corpora
EXPOSE 5000
CMD ["python", "app.py"]
