import re
from typing import List, Callable

import spacy
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from pandas import DataFrame, to_numeric


def remove_na_and_duplicate(df: DataFrame) -> DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def reformat_df(df: DataFrame) -> DataFrame:
    df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)

    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)

    df['rating'] = to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')

    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(int)

    return df


def sentence_to_stemmed_tokens(sentence: str, stem_name: str = "porter") -> List[str]:
    if stem_name == "snowball":
        sb = SnowballStemmer(language='english')
        stemmer = sb.stem
    else:
        ps = PorterStemmer()
        stemmer = ps.stem
    tokens = word_tokenize(sentence)
    return [stemmer(token) for token in tokens]


def sentence_to_lemmatized_tokens(sentence: str, lem_name: str = "nltk") -> List[str]:
    if lem_name == "spacy":
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        doc = nlp(sentence)
        return [token.lemma_ for token in doc]
    else:
        tokens = word_tokenize(sentence)
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token, pos="v") for token in tokens]


def clean_sentence_to_tokens(sentence: str, cleaner: Callable, name: str):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
    tokens = cleaner(sentence, name)
    return tokens


def clean_tokens(tokens: List[str], min_word_len: int = 3) -> List[str]:
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in tokens if word not in stop_words]
    words = [word for word in words if len(word) >= min_word_len]
    return words


def clean_sentence_to_str(sentence: str, cleaner: Callable, name: str, min_word_len: int = 3):
    tokens = clean_tokens(clean_sentence_to_tokens(sentence, cleaner, name), min_word_len)
    text = ' '.join(tokens)
    return text


def df_apply_cleaner_on_column(df: DataFrame, column: str, cleaner: Callable, name: str) -> DataFrame:
    df[column + '_clean'] = df[column].apply(lambda s: clean_sentence_to_str(s, cleaner, name))
    return df


def df_apply_cleaner_on_columns(df: DataFrame, columns: str, cleaner: Callable, name: str) -> DataFrame:
    for column in columns:
        df[column + '_clean'] = df[column].apply(lambda s: clean_sentence_to_str(s, cleaner, name))
    return df
