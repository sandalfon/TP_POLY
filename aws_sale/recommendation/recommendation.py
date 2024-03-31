from typing import Callable, List

from numpy import ndarray
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from aws_sale.cleaning.clean_data import df_apply_cleaner_on_columns
from aws_sale.sentiment.sentiment import get_sentiment, sentiment_intensity_analysis


def prepare_df(df: DataFrame, cleaner: Callable, name: str) -> DataFrame:
    df = df_apply_cleaner_on_columns(df, ['review_content'], cleaner, name)
    df['combined_text'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product'] + ' ' + df[
        'review_content_clean']
    df['combined_text'] = df['combined_text'].fillna('')
    sentiments = sentiment_intensity_analysis(df['review_content_clean'].tolist())
    df['sentiment'] = get_sentiment(sentiments)
    label_encoder = LabelEncoder()
    df['encoded_sentiment'] = label_encoder.fit_transform(df['sentiment'])
    return df[
        ['product_id', 'product_name', 'category', 'about_product', 'review_content_clean', 'combined_text', 'rating',
         'sentiment']]


def get_sim_from_tfidf(df: DataFrame) -> ndarray:
    vectorized = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1, 1))
    tfidfs = vectorized.fit_transform(df['combined_text'])
    return cosine_similarity(tfidfs)


def get_avg_rating(df: DataFrame) -> DataFrame:
    df_pivot = df.pivot_table(index='product_id', values='rating', aggfunc='mean')
    return df_pivot.fillna(df_pivot.mean())


def _get_nth_sorted_content_sim(df: DataFrame, product_id: str, content_sim: ndarray, max_result: int) -> List[str]:
    index = df.index[df['product_id'] == product_id][0]
    sim_scores = list(enumerate(content_sim[index]))
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [i[0] for i in sorted_sim_scores[1:max_result + 1]]


def _get_nth_avg_rating_product(df: DataFrame, product_id: str, avg_rating_df: DataFrame, max_result: int) -> List[str]:
    if product_id in avg_rating_df.index:
        current_product_rating = avg_rating_df.loc[product_id].values[0]
        ids = avg_rating_df.iloc[
            (avg_rating_df['rating'] - current_product_rating).abs().argsort()[:max_result]].index.values
        return df[df['product_id'].isin(ids)].index.tolist()
    return []


def get_product_recommendation(df: DataFrame, product_id: str, content_sim: ndarray, avg_rating_df: DataFrame,
                               max_result: int) -> DataFrame:
    content_recommendations_index = _get_nth_sorted_content_sim(df, product_id, content_sim, max_result)
    rating_recommendation_index = _get_nth_avg_rating_product(df, product_id, avg_rating_df, max_result)

    recommendation_index = list(set(content_recommendations_index + rating_recommendation_index))
    return df.iloc[recommendation_index][['product_id', 'product_name', 'rating']]
