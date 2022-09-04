from transformers import pipeline


def sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    return classifier("I want to learn more")


def text_generator():
    generator = pipeline("text2text-generation", model = "distilgpt2")

    return generator(
        "i will learn math, because",
        max_length= 20,
        num_return_sequences = 1,
    )