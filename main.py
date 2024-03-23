from typing import Callable

from pandas import read_csv, DataFrame

from aws_sale.cleaning import remove_na_and_duplicate, reformat_df, sentence_to_stemmed_tokens, \
    sentence_to_lemmatized_tokens
from aws_sale.cleaning.clean_data import df_apply_cleaner_on_columns, clean_sentence_to_tokens, \
    df_apply_cleaner_on_column
from aws_sale.semantic import word2vec
from aws_sale.semantic.word2vec import make_world_clusters, tsne_plot_similar_words, get_models_similarity
from aws_sale.sentiment import frequencies, sentiment_intensity_analysis, plot_sentiments


def lem_stem(df: DataFrame):
    sentence = """Your time is limited, so don't waste it living someone else's life.
         The future belongs to those who believe in the beauty of their dreams"""

    stem_tokens_porter = sentence_to_stemmed_tokens(sentence, "porter")
    stem_tokens_sb = sentence_to_stemmed_tokens(sentence, "snowball")

    print(stem_tokens_porter)
    print(stem_tokens_sb)

    lem_tokens_spacy = sentence_to_lemmatized_tokens(sentence, "spacy")
    lem_tokens_nltk = sentence_to_lemmatized_tokens(sentence, "nltk")
    print(lem_tokens_spacy)
    print(lem_tokens_nltk)

    columns = ['product_name', 'about_product', 'review_content', 'category']

    print("spacy ")
    df_spacy = df_apply_cleaner_on_columns(df, columns, sentence_to_lemmatized_tokens, 'spacy')
    df_spacy.to_csv('data/output/spacy_amazon.csv')

    print("nltk ")
    df_nltk = df_apply_cleaner_on_columns(df, columns, sentence_to_lemmatized_tokens, 'nltk')
    df_nltk.to_csv('data/output/nltk_amazon.csv')

    print("porter ")
    df_porter = df_apply_cleaner_on_columns(df, columns, sentence_to_stemmed_tokens, 'porter')
    df_porter.to_csv('data/output/porter_amazon.csv')

    print("snowball ")
    df_snowball = df_apply_cleaner_on_columns(df, columns, sentence_to_stemmed_tokens, 'snowball')
    df_snowball.to_csv('data/output/snowball_amazon.csv')


def prepare(df: DataFrame, column: str, cleaner: Callable, callable_name: str, w2v_name):
    sentences = df[column].apply(lambda s: clean_sentence_to_tokens(s, cleaner, callable_name))
    return word2vec(sentences, w2v_name)


def w2v(df: DataFrame):
    words = ['phone', 'good', 'photo']
    w2v_skip = prepare(df, 'review_content', sentence_to_lemmatized_tokens, 'nltk', 'skipgramm')
    w2v_cbow = prepare(df, 'review_content', sentence_to_lemmatized_tokens, 'nltk', 'cbow')
    w2v_fasttext = prepare(df, 'review_content', sentence_to_lemmatized_tokens, 'nltk', 'fasttext')
    df = get_models_similarity(words, [w2v_skip, w2v_cbow, w2v_fasttext], 10)
    print(df)
    embedding_clusters, word_clusters = make_world_clusters(words, w2v_cbow)
    tsne_plot_similar_words(words, embedding_clusters, word_clusters)


def sentiment(df: DataFrame):
    fd = frequencies(df, sentence_to_lemmatized_tokens, 'nltk')
    print(fd.most_common(3))
    df = df_apply_cleaner_on_column(df, 'review_content', sentence_to_lemmatized_tokens, 'nltk')
    results = sentiment_intensity_analysis(df['review_content_clean'].tolist())
    plot_sentiments(results)


df = read_csv("data/input/amazon.csv")
df = remove_na_and_duplicate(df)
df = reformat_df(df)
print(df.head())
# lem_stem(df)
# w2v(df)
sentiment(df)
