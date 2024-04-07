from typing import Callable

from pandas import DataFrame, to_numeric

from aws_sale.clean.clean_sentence import remove_punctuation, remove_stopwords, sentence_to_tokens_nlk
from aws_sale.lem.lemmer import tokens_to_lemmed_tokens


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

    df['review_content'] = df['review_content'].astype(str).apply(
        lambda s: s.encode('ascii', errors='ignore').decode('utf-8'))

    return df


def get_clean_data(df: DataFrame) -> DataFrame:
    df = remove_na_and_duplicate(df)
    df = reformat_df(df)
    return df


def nltk_cleaner(sentence: str) -> str:
    no_punct_sentence = remove_punctuation(sentence)
    tokens = remove_stopwords(sentence_to_tokens_nlk(no_punct_sentence, True))
    lem_tokens = tokens_to_lemmed_tokens(tokens, "nltk")
    return ' '.join(lem_tokens)


def df_apply_cleaner_on_column(df: DataFrame, column: str, cleaner: Callable, tag: str) -> DataFrame:
    df[column + '_' + tag] = df[column].apply(lambda s: cleaner(s))
    return df
