from pandas import read_csv

from aws_sale.cleaning import remove_na_and_duplicate, reformat_df, sentence_to_stemmed_tokens, \
    sentence_to_lemmatized_tokens
from aws_sale.cleaning.clean_data import df_apply_cleaner_on_columns

df = read_csv("data/input/amazon.csv")
df = remove_na_and_duplicate(df)
df = reformat_df(df)
print(df.head())

sentence = """Your time is limited, so don't waste it living someone else's life.
     The future belongs to those who believe in the beauty of their dreams"""

stem_tokens_porter = sentence_to_stemmed_tokens(sentence, "porter")
stem_tokens_sb = sentence_to_stemmed_tokens(sentence, "snowball")

print(stem_tokens_porter)
print(stem_tokens_sb)

lem_tokens_spacy = sentence_to_lemmatized_tokens(sentence, "spacy")
lem_tokens_nltk = sentence_to_lemmatized_tokens(sentence, "nltk")
print(lem_tokens_spacy)
print(lem_tokens_nltk)

columns = ['product_name', 'about_product', 'review_content', 'category']

print("spacy ")
df_spacy = df_apply_cleaner_on_columns(df, columns, sentence_to_lemmatized_tokens, 'spacy')
df_spacy.to_csv('data/output/spacy_amazon.csv')

print("nltk ")
df_nltk = df_apply_cleaner_on_columns(df, columns, sentence_to_lemmatized_tokens, 'nltk')
df_nltk.to_csv('data/output/nltk_amazon.csv')

print("porter ")
df_porter = df_apply_cleaner_on_columns(df, columns, sentence_to_stemmed_tokens, 'porter')
df_porter.to_csv('data/output/porter_amazon.csv')

print("snowball ")
df_snowball = df_apply_cleaner_on_columns(df, columns, sentence_to_stemmed_tokens, 'snowball')
df_snowball.to_csv('data/output/snowball_amazon.csv')
