from typing import Callable, List

from matplotlib import pyplot as plt
from nltk import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from pandas import DataFrame

from aws_sale.cleaning.clean_data import clean_sentence_to_tokens, clean_tokens


def _flatten_concatenation(matrix: List[List[str]]) -> List[str]:
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


def frequencies(
    df: DataFrame, cleaner: Callable, callable_name: str, remove_stop_words: bool = True, word_min_len: int = 3
) -> FreqDist:
    words = df["review_content"].apply(lambda s: clean_sentence_to_tokens(s, cleaner, callable_name)).tolist()
    words = _flatten_concatenation(words)
    if remove_stop_words:
        words = clean_tokens(words, word_min_len)
    return FreqDist(words)


def sentiment_intensity_analysis(sentences: List[str]) -> List[dict]:
    sia = SentimentIntensityAnalyzer()
    results = []
    for sentence in sentences:
        results.append(sia.polarity_scores(sentence))
    return results


def _list_of_sent_to_df(list_of_sentiments: List[dict]) -> DataFrame:
    sentiments = []
    for sentiment in list_of_sentiments:
        compound = sentiment["compound"]
        if compound > 0.1:
            sentiments.append("Positive")
        elif compound < -0.1:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return DataFrame(sentiments, columns=["review_sentiment"])


def plot_sentiments(list_of_sentiments: List[dict]):
    df = _list_of_sent_to_df(list_of_sentiments)

    sentiment_counts = df.value_counts()

    sentiment_counts.plot(kind="bar", color=["green", "blue", "red"], title="Sentiment Analysis of Review Content")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig("j.png", format="png", dpi=150, bbox_inches="tight")
    plt.show()
