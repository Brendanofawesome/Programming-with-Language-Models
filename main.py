from transformers import pipeline

# Load Hugging Face sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]  # Get first result
    return result["label"]

if __name__ == "__main__":
    text = input("Enter a sentence: ")
    sentiment = analyze_sentiment(text)
    print(f"Sentiment: {sentiment}")
