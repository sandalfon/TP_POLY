from typing import List

from nltk import WordNetLemmatizer
from spacy import Language
from spacy.lang.en import English


def spacy_lemmer(sentence: str, nlp: Language) -> List[str]:
    doc = nlp(sentence)
    return [token.lemma_ for token in doc]


def nltk_lemmer(tokens: List[str]):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token, pos="v") for token in tokens]


def tokens_to_lemmed_tokens(tokens: List[str], lem_name: str = "nltk", nlp: Language = None) -> List[str]:
    if lem_name == "nltk":
        return nltk_lemmer(tokens)
    elif lem_name == "spacy" and type(nlp) == English:
        return spacy_lemmer(' '.join(tokens), nlp)
    else:
        raise ValueError(lem_name, type(nlp))
