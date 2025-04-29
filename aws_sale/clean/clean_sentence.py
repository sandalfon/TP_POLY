import re
from typing import List

import tensorflow as tf
from keras.src.legacy.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from spacy import Language


def sentence_to_tokens_nlk(sentence: str, to_lower: bool = False) -> List[str]:
    if to_lower:
        sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    return tokens


def sentence_to_tokens_spacy(sentence: str, nlp: Language, to_lower: bool = False) -> List[str]:
    if to_lower:
        sentence = sentence.lower()
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    return tokens


class SimpleTokenizer(Tokenizer):
    def tokenize(self, input):
        return tf.strings.split(input)


def sentence_to_tokens_keras(sentence: str, to_lower: bool = False) -> List[str]:
    if to_lower:
        sentence = sentence.lower()
    tokens = SimpleTokenizer().tokenize([sentence])
    tokens = [token.decode("utf-8") for token in tokens.to_list()[0]]
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    stopwords_ = set(stopwords.words("english"))
    clean_tokens = [t for t in tokens if not t in stopwords_]
    return clean_tokens


def remove_punctuation(sentence: str) -> str:
    res = re.sub(r'[^\w\s]', ' ', sentence)
    return res


def remove_small_words(tokens: List[str], min_word_len: int) -> List[str]:
    return [token for token in tokens if len(token) >= min_word_len]
