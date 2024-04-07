from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from spacy.lang.en import English


def sentences_to_vector(sentences: List[str]) -> np.ndarray:
    vectorizer = CountVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    return sentence_vectors, vectorizer


def sentences_to_nlp_doc(sentences: List[str], nlp: English) -> np.ndarray:
    sentence_embeddings = [nlp(sentence).vector for sentence in sentences]
    sentence_embeddings = np.array(sentence_embeddings)
    return sentence_embeddings


def sentences_to_vector_tf_idf(sentences: List[str]):
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(sentences)
    return tfidf_vectors.toarray(), vectorizer


def compute_nn(sentence_embeddings: np.ndarray, metric: str, algorithm: str = 'auto',
               nb_neighbor: int = 2) -> NearestNeighbors:
    nn_model = NearestNeighbors(n_neighbors=nb_neighbor, metric=metric, algorithm=algorithm)
    nn_model.fit(sentence_embeddings)
    return nn_model


def get_query_sim_spacy(nn_model: NearestNeighbors, query: str, nlp: English) -> Tuple[np.ndarray, np.ndarray]:
    query_embedding = nlp(query).vector.reshape(1, -1)
    distances, indices = nn_model.kneighbors(query_embedding)
    return distances, indices


def get_query_sim_bag(nn_model: NearestNeighbors, query: str, vectorizer: CountVectorizer) -> Tuple[
    np.ndarray, np.ndarray]:
    query_vector = vectorizer.transform([query])
    distances, indices = nn_model.kneighbors(query_vector)
    return distances, indices


def get_query_sim_tf_idf(nn_model: NearestNeighbors, query: str, vectorizer: CountVectorizer) -> Tuple[
    np.ndarray, np.ndarray]:
    query_vec = vectorizer.transform([query])
    distances, indices = nn_model.kneighbors(query_vec.toarray(), return_distance=True)
    return distances, indices
