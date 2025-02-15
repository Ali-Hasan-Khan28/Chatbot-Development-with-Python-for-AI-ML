from textblob import TextBlob



def preprocess(question):
    blob = TextBlob(question)
    temp_val = 0
    confidence_score = 0.0
    for sentence in blob.sentences:
        confidence_score+=sentence.sentiment.polarity
    confidence_score = confidence_score/len(blob.sentences)

    if confidence_score < 0:
        temp_val = 0.2
    elif confidence_score < 0.5:
        temp_val = 0.5
    else:
        temp_val = 0.8
    return temp_val

