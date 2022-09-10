from pipelines import sentiment_analysis
from pipelines import text_generator
from pipelines import text_classifier
from pipelines import text_summarization
from tokenizer import tokenizer_distilbert

if __name__ == '__main__':
    # print(text_summarization())
    # print(sentiment_analysis())
    # print(text_generator())
    # print(text_classifier())
    tokenizer_distilbert()
