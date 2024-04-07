from random import random
from typing import List

import spacy
from pandas import DataFrame, read_csv

from aws_sale.clean.clean_data import df_apply_cleaner_on_column, nltk_cleaner, get_clean_data
from aws_sale.clean.clean_sentence import remove_stopwords, sentence_to_tokens_nlk, sentence_to_tokens_spacy, \
    sentence_to_tokens_keras, remove_punctuation
from aws_sale.compute.recommendation import get_avg_similarity, get_avg_rating, get_avg_sentiment, get_recommendation
from aws_sale.lem.lemmer import tokens_to_lemmed_tokens
from aws_sale.sentiment.sentiment import sentiment_intensity_analysis, plot_sentiments
from aws_sale.similarity.nearest_neighbors import compute_nn, sentences_to_nlp_doc, sentences_to_vector, \
    get_query_sim_spacy, get_query_sim_bag, sentences_to_vector_tf_idf, get_query_sim_tf_idf
from aws_sale.similarity.x2vec.doc import sentences_to_doc, doc2vec_train, get_query_sim_doc2vec
from aws_sale.similarity.x2vec.word import word2vec, get_models_similarity, tsne_plot_similar_words, make_world_clusters
from aws_sale.stem.stemmers import tokens_to_stemmed_tokens

NLP = spacy.load("en_core_web_sm")

tweet = """Publication of a code snippet for concurrent reading of French legal codes via the
@huggingface
’s #Datasets library 🤗

You can now access all the French legal codes on my #HF account: @louisbrulenaudet

#Data #GenAI #LLM #NLP #Legal #Python
"""


def demo_token(nltk_tokens: List[str], spacy_tokens: List[str], keras_tokens: List[str]):
    print(" from \"\"\"{}\"\"\" ".format(tweet))
    print("to")
    print("nltk")
    print(nltk_tokens)
    print("spacy")
    print(spacy_tokens)
    print("keras")
    print(keras_tokens)


def demo_stem(nltk_tokens: List[str], spacy_tokens: List[str], keras_tokens: List[str]):
    for stem_name in ["snowball", "porter"]:
        print("stemmer: {}".format(stem_name))
        print("nltk")
        print(tokens_to_stemmed_tokens(nltk_tokens, stem_name))
        print("spacy")
        print(tokens_to_stemmed_tokens(spacy_tokens, stem_name))
        print("keras")
        print(tokens_to_stemmed_tokens(keras_tokens, stem_name))


def demo_lem(nltk_tokens: List[str], spacy_tokens: List[str], keras_tokens: List[str]):
    for lem_name in ["nltk", "spacy"]:
        print("lemmer: {}".format(lem_name))
        print("nltk")
        print(tokens_to_lemmed_tokens(nltk_tokens, lem_name, NLP))
        print("spacy")
        print(tokens_to_lemmed_tokens(spacy_tokens, lem_name, NLP))
        print("keras")
        print(tokens_to_lemmed_tokens(keras_tokens, lem_name, NLP))


def demo_prepare_date_frame(df: DataFrame, show: int = 3) -> DataFrame:
    df = get_clean_data(df)
    df = df_apply_cleaner_on_column(df, 'review_content', nltk_cleaner, 'nltk')
    for i in range(show):
        print('-----')
        print(df['review_content'][i])
        print('<<<<<<>>>>>>')
        print(df['review_content_nltk'][i])
        print('*******')
    return df


def demo_w2v(df: DataFrame):
    words = ['phone', 'good', 'cable']
    word2vec_fasttext = word2vec(df, 'review_content_nltk', "fasttext")
    word2vec_skipgram = word2vec(df, 'review_content_nltk', "skipgram")
    word2vec_cbow = word2vec(df, 'review_content_nltk', "cbow")
    df = get_models_similarity(words, [word2vec_skipgram, word2vec_cbow, word2vec_fasttext], 10)
    print(df)

    embedding_clusters, word_clusters = make_world_clusters(words, word2vec_fasttext)
    tsne_plot_similar_words(words, embedding_clusters, word_clusters, "fasttext")

    embedding_clusters, word_clusters = make_world_clusters(words, word2vec_skipgram)
    tsne_plot_similar_words(words, embedding_clusters, word_clusters, "skipgram")

    embedding_clusters, word_clusters = make_world_clusters(words, word2vec_cbow)
    tsne_plot_similar_words(words, embedding_clusters, word_clusters, "cbow")


def demo_similarity(df: DataFrame, method: str, n: int = 2):
    sentences = df['review_content_nltk'].tolist()
    query_index = int(random() * 1000)
    query = sentences[query_index]
    if method == "spacy":
        sentence_embeddings = sentences_to_nlp_doc(sentences, NLP)
        nn_model = compute_nn(sentence_embeddings, 'cosine', 'auto', n)
        distances, indices = get_query_sim_spacy(nn_model, query, NLP)
    elif method == "bag":
        sentence_embeddings, vectorizer = sentences_to_vector(sentences)
        nn_model = compute_nn(sentence_embeddings, 'euclidean', 'auto', n)
        distances, indices = get_query_sim_bag(nn_model, query, vectorizer)
    elif method == "tf-idf":
        sentence_embeddings, vectorizer = sentences_to_vector_tf_idf(sentences)
        nn_model = compute_nn(sentence_embeddings, 'cosine', 'brute', n)
        distances, indices = get_query_sim_tf_idf(nn_model, query, vectorizer)
    elif method == "doc2vec":
        sentence_embeddings = sentences_to_doc(sentences)
        d2v_model = doc2vec_train(sentence_embeddings)
        distances, indices = get_query_sim_doc2vec(d2v_model, query, len(sentences), n)
    else:
        raise ValueError(method)
    print("Query:")
    print(df['review_content'].tolist()[query_index])
    print("Nearest neighbors:")
    for i, index in enumerate(indices[0][1:]):
        print(df['review_content'].tolist()[index], "- Distance:", distances[0][i])
        print()


def demo_sentiment(df: DataFrame):
    sentences = df['review_content'].tolist()
    results = sentiment_intensity_analysis(sentences)
    print(results)
    plot_sentiments(results)


def demo_recommendation(df: DataFrame, query_index: int):
    df = get_clean_data(df)
    df = df_apply_cleaner_on_column(df, 'review_content', nltk_cleaner, 'nltk')
    sentences = df['review_content_nltk'].tolist()
    sentences_raw = df['review_content'].tolist()
    query = sentences[query_index]
    sentence_embeddings = sentences_to_doc(sentences)
    d2v_model = doc2vec_train(sentence_embeddings)
    distances, indices = get_query_sim_doc2vec(d2v_model, query, len(sentences), len(sentences))
    df_sim = get_avg_similarity(df, distances)
    df_rating = get_avg_rating(df)
    results = sentiment_intensity_analysis(sentences_raw)
    df_sentiments = get_avg_sentiment(df, results)
    print(df_sim.head())
    print(df_rating.head())
    print(df_sentiments.head())
    df_avg = get_recommendation(df_sim, 3.0, df_rating, 0.5, df_sentiments, 1.5)
    print(df_avg)
    product_id = df_avg['score'].idxmax()
    print('query')
    print(df.iloc[[query_index]][['product_id', 'product_name', 'category']])
    print('best reco')
    print(df[df['product_id'] == product_id][['product_id', 'product_name', 'category']].drop_duplicates())


def demo_recommendation_product(df: DataFrame, query_index: int):
    df = get_clean_data(df)
    df = df_apply_cleaner_on_column(df, 'product_name', nltk_cleaner, 'nltk')
    sentences = df['product_name_nltk'].tolist()
    sentences_raw = df['product_name'].tolist()
    query = sentences[query_index]
    sentence_embeddings = sentences_to_doc(sentences)
    d2v_model = doc2vec_train(sentence_embeddings)
    distances, indices = get_query_sim_doc2vec(d2v_model, query, len(sentences), len(sentences))
    df_sim = get_avg_similarity(df, distances)
    df_rating = get_avg_rating(df)
    results = sentiment_intensity_analysis(sentences_raw)
    df_sentiments = get_avg_sentiment(df, results)
    print(df_sim.head())
    print(df_rating.head())
    print(df_sentiments.head())
    df_avg = get_recommendation(df_sim, 3.0, df_rating, 0.5, df_sentiments, 1.5)
    print(df_avg)
    product_id = df_avg['score'].idxmax()
    print('query')
    print(df.iloc[[query_index]][['product_id', 'product_name', 'category']])
    print('best reco')
    print(df[df['product_id'] == product_id][['product_id', 'product_name', 'category']].drop_duplicates())


no_punct_tweet = remove_punctuation(tweet)
nltk_tokens = remove_stopwords(sentence_to_tokens_nlk(no_punct_tweet))
spacy_tokens = remove_stopwords(sentence_to_tokens_spacy(no_punct_tweet, NLP))
keras_tokens = remove_stopwords(sentence_to_tokens_keras(no_punct_tweet))
# demo_token(nltk_tokens, spacy_tokens, keras_tokens)
# demo_stem(nltk_tokens, spacy_tokens, keras_tokens)
# demo_lem(nltk_tokens, spacy_tokens, keras_tokens)

df = read_csv("data/input/amazon.csv")
# df = demo_prepare_date_frame(df, 0)
# demo_w2v(df)
# demo_similarity(df, "doc2vec", 5)
# demo_sentiment(df)

demo_recommendation(df, int(random() * df.shape[0]))
demo_recommendation_product(df, int(random() * df.shape[0]))
