from typing import List

from matplotlib import pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from pandas import DataFrame


def sentiment_intensity_analysis(sentences: List[str]) -> List[dict]:
    sia = SentimentIntensityAnalyzer()
    results = []
    for sentence in sentences:
        results.append(sia.polarity_scores(sentence))
    return results


def get_sentiment(list_of_sentiments: List[dict]) -> DataFrame:
    sentiments = []
    for sentiment in list_of_sentiments:
        compound = sentiment["compound"]
        if compound > 0.1:
            sentiments.append('Positive')
        elif compound < -0.1:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')
    return DataFrame(sentiments, columns=['review_sentiment'])


def plot_sentiments(list_of_sentiments: List[dict]):
    df = get_sentiment(list_of_sentiments)

    sentiment_counts = df.value_counts()
    print(sentiment_counts)

    sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'], title='Sentiment Analysis of Review Content')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig("data/output/sentiments.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()
