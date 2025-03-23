from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec, FastText
from pandas import DataFrame, concat
from sklearn.manifold import TSNE


def word2vec_skipgram(sentences: List[List[str]]) -> Word2Vec:
    model_skipgram = Word2Vec(min_count=1, sg=1, vector_size=100, window=10, workers=4)
    model_skipgram.build_vocab(sentences)
    model_skipgram.train(sentences, total_examples=model_skipgram.corpus_count, epochs=100)
    return model_skipgram


def word2vec_fasttext(sentences: List[List[str]]) -> Word2Vec:
    return FastText(sentences, vector_size=4, window=3, min_count=1, sample=1e-2, sg=0)


def word2vec_cbow(sentences: List[List[str]]) -> Word2Vec:
    model_cbow = Word2Vec(min_count=1, sg=0, vector_size=100, window=10, workers=4)
    model_cbow.build_vocab(sentences)
    model_cbow.train(sentences, total_examples=model_cbow.corpus_count, epochs=100)
    return model_cbow


def word2vec(sentences: List[List[str]], model_name: str = "skipgram") -> Word2Vec:
    if model_name == "skipgram":
        return word2vec_skipgram(sentences)
    else:
        if model_name == "fasttext":
            return word2vec_fasttext(sentences)
    return word2vec_cbow(sentences)


def get_models_similarity(words: List[str], models: List[Word2Vec], nb: int) -> DataFrame:
    df = DataFrame()
    for model in models:
        top_similar = model.wv.most_similar(positive=words, topn=nb)
        df = concat([df, DataFrame(top_similar, columns=["word", "cosine_sim"])])
    return df


def make_world_clusters(input_words: List[str], model: Word2Vec) -> Tuple[List[List],List[List]]:
    embedding_clusters = []
    word_clusters = []
    for word in input_words:
        embeddings = []
        words = []
        for similar_word, _ in model.wv.most_similar(word, topn=10):
            words.append(similar_word)
            embeddings.append(model.wv[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
    return embedding_clusters, word_clusters


def tsne_plot_similar_words(words, embedding_clusters, word_clusters, a=0.7):
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init="pca", max_iter=3500, random_state=32)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    plt.figure(figsize=(9, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(words)))
    for label, embeddings, words, color in zip(words, embeddings_en_2d, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color.reshape(1, -1), alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(
                word,
                alpha=0.5,
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                size=8,
            )
    plt.legend()
    plt.grid(True)
    plt.savefig("i.png", format="png", dpi=150, bbox_inches="tight")
    plt.show()
