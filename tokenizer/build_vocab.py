from .tokenizer import CharTokenizer


def build_vocab(text):
    return CharTokenizer.from_text(text).stoi
