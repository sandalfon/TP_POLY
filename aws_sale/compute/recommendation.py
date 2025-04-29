from typing import List

import numpy as np
from pandas import DataFrame


def get_avg_similarity(df: DataFrame, similarities: np.ndarray) -> DataFrame:
    df_sim = DataFrame(
        {
            'product_id': df['product_id'].tolist(),
            'similarities': similarities[0]
        }
    )
    df_sim['similarities'] = 1 - df_sim['similarities']
    df_pivot = df_sim.pivot_table(index='product_id', values='similarities', aggfunc='mean')
    return df_pivot.fillna(df_pivot.mean())


def get_avg_rating(df: DataFrame) -> DataFrame:
    df_pivot = df.pivot_table(index='product_id', values='rating', aggfunc='mean')
    df_pivot['rating'] = df_pivot['rating'] / 5.0
    return df_pivot.fillna(df_pivot.mean())


def get_avg_sentiment(df: DataFrame, results: List[dict]):
    pos_values = [sub['pos'] for sub in results]
    df_sim = DataFrame(
        {
            'product_id': df['product_id'].tolist(),
            'sentiments': pos_values
        }
    )
    df_pivot = df_sim.pivot_table(index='product_id', values='sentiments', aggfunc='mean')
    return df_pivot.fillna(df_pivot.mean())


def get_recommendation(df_sim: DataFrame, w_sim: float, df_rating: DataFrame, w_rating: float, df_sentiments: DataFrame,
                       w_sentiments: float) -> DataFrame:
    df_avg = df_sim.join(df_rating, on='product_id').join(df_sentiments, on='product_id')
    df_avg['score'] = df_avg['similarities'] * w_sim + df_avg['rating'] * w_rating + df_avg['sentiments'] * w_sentiments
    return df_avg
