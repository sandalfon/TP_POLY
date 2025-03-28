# TP_POLY

## Système de recommandation sur un extrait de données d'amazon

### Consigne:

Créer un système de recommandation de produits à partir de données déjà existantes et
disponible [ici](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset).

### Objectif

1. Designer le système
2. Préparer des données brutes
3. Décomposer en différentes sous parties
4. Utiliser des notions comme lemmatisation, stemmatisation, word2vec, tf-idf etc...
5. Rendre le code modulable
6. Evaluer l'impact des choix
7. Rendre un code lisible et compréhensible

## Install

### Poetry
See poetry installation [page](https://python-poetry.org/docs/#installation)

### Ruff
See ruff installation [page](https://docs.astral.sh/ruff/installation/)


## Install projext
```bash
poetry install
poetry shell
```

## Install models
```bash
python setup/nltk_setup
python setup/spacy_setup
```

#### linting
`ruff format`

## Dataset

[amazon sale dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset?resource=download)

# Compare

[medium](https://medium.com/@aneesha161994/exploring-diverse-techniques-for-sentence-similarity-bc62058c7972)

# Chatbot

[doc](https://spacy.io/universe/project/Chatterbot)
[medium](https://medium.com/@guandika8/on-your-local-pc-a-local-chatbot-that-is-completely-offline-and-private-26b298dc4076)

# Ollama

[site](https://ollama.com/download)

[doc llm](https://docs.mistral.ai/deployment/self-deployment/overview/)

# Voice Cloning

[repo](https://github.com/CorentinJ/Real-Time-Voice-Cloning?tab=readme-ov-file)

[dataset librespeech](https://www.openslr.org/12)