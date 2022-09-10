from transformers import pipeline


def sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    return classifier("I want to learn more")


def text_generator():
    generator = pipeline("text2text-generation", model="distilgpt2")

    return generator(
        "i will learn math, because",
        max_length=20,
        num_return_sequences=1,
    )


def text_classifier():
    classifier = pipeline("zero-shot-classification")
    return classifier(
        "I want to learn more",
        candidate_labels=["education", "politics", "business"]
    )


def text_summarization():
    summarizer = pipeline("summarization")

    return summarizer(
        "Eureka is a Rest-API project for Web Scraping, data cleaning and organization, based on FastAPI and "
        "following a hexagonal architecture. "
    )
