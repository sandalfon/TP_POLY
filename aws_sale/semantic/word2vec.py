from typing import List

from gensim.models import Word2Vec


def word2vec_skipgram(sentences: List[List[str]]) -> Word2Vec:
    model_skipgram = Word2Vec(min_count=1, sg=1, vector_size=100, window=10, workers=4)
    model_skipgram.build_vocab(sentences)
    model_skipgram.train(sentences, total_examples=model_skipgram.corpus_count, epochs=100)
    return model_skipgram


def word2vec_cbow(sentences: List[List[str]]) -> Word2Vec:
    model_cbow = Word2Vec(min_count=1, sg=0, vector_size=100, window=10, workers=4)
    model_cbow.build_vocab(sentences)
    model_cbow.train(sentences, total_examples=model_cbow.corpus_count, epochs=100)
    return model_cbow
