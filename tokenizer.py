from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel


def tokenizer_distilbert():
    classifier = pipeline("sentiment-analysis")
    print(classifier("i want this"))

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    autoTokenizer = AutoTokenizer.from_pretrained(model_name)

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=autoTokenizer)
    print(autoTokenizer("Using a Transformer network is simple"))

    tokens = autoTokenizer.tokenize("Using a Transformer network is simple")
    print(tokens)

    ids = autoTokenizer.convert_tokens_to_ids(tokens)
    print(ids)

    print(autoTokenizer.decode(ids))
