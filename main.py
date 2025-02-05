from transformers import pipeline

# Load a model that supports Positive, Neutral, and Negative
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]  # Get first result
    return(result['label'])

if __name__ == "__main__":
    while(True):
        print("Enter a sentence: ")
        text = input()
        sentiment = analyze_sentiment(text)
        print(f"Sentiment: {sentiment}")
