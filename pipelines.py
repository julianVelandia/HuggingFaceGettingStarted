from transformers import pipeline


def sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    return classifier("I want to learn more")
