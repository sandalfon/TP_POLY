from typing import List

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from pandas import DataFrame


def create_document(df: DataFrame) -> List[TaggedDocument]:
    texts = df['review_content_clean'].tolist()
    return [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]


def train_doc2vec_model(documents: List[TaggedDocument]) -> Doc2Vec:
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def get_index_similarity(tokens: List[str], model: Doc2Vec, max_result: int) -> List[int]:
    inferred_vector = model.infer_vector(tokens)
    sims = model.dv.most_similar([inferred_vector], topn=max_result)
    return [res[0] for res in sims]
