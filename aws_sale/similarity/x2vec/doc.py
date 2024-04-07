from typing import List

import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity


def sentences_to_doc(sentences: List[str], min_len_word: int = 4) -> List[TaggedDocument]:
    tagged_sentences = [TaggedDocument(words=sentence.split(' '), tags=[i]) for i, sentence in enumerate(sentences)]
    return tagged_sentences


def doc2vec_train(tagged_sentences: List[TaggedDocument]) -> Doc2Vec:
    max_epochs = 50
    vec_size = 50
    alpha = 0.025
    min_alpha = 0.00025
    min_count = 1
    dm = 1
    model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=min_alpha, min_count=min_count, dm=dm)

    model.build_vocab(tagged_sentences)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    return model


def get_query_sim_doc2vec(model: Doc2Vec, query: str, sentences_len: int, nb: int):
    tokenized_query = query.split(' ')
    query_vec = model.infer_vector(tokenized_query)
    sentence_vecs = [model.docvecs[i] for i in range(sentences_len)]
    similarities = cosine_similarity([query_vec], sentence_vecs)
    most_similar_idx = np.argsort(similarities)[0][:nb]
    return similarities, [most_similar_idx]
