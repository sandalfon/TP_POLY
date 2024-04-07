from typing import List

from nltk.stem import PorterStemmer, SnowballStemmer


def tokens_to_stemmed_tokens(tokens: List[str], stem_name: str = "porter") -> List[str]:
    if stem_name == "snowball":
        sb = SnowballStemmer(language='english')
        stemmer = sb.stem
    elif stem_name == "porter":
        ps = PorterStemmer()
        stemmer = ps.stem
    else:
        raise ValueError(stem_name)
    return [stemmer(token) for token in tokens]
