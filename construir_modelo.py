import pandas as pd
import numpy as np
import regex
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import os

# Cargar modelo de spaCy para inglés
nlp = spacy.load('en_core_web_sm')

def ConvertirAcentos(texto):
    return texto

def CrearCorpusDesdeCSV(path_csv):
    df = pd.read_csv(path_csv)
    corpus = []
    doc_id = []

    for idx, row in df.iterrows():
        texto = row['advice']
        texto = ConvertirAcentos(texto)
        corpus.append(texto)
        doc_id.append(f"doc_{idx}")

    return corpus, doc_id

def PreProcesar(textos, remove_stopwords=True, use_lemmatization=True):
    texto_limpio = []
    for texto in textos:
        texto = texto.lower()
        if remove_stopwords:
            texto = EliminarStopwords(texto)
        if use_lemmatization:
            texto = Lematizar(texto)
        texto = EliminaNumeroYPuntuacion(texto)
        if len(texto) != 0:
            texto = regex.sub(' +', ' ', texto)
            texto_limpio.append(texto)
    return texto_limpio

def Lematizar(oracion):
    doc = nlp(oracion)
    lemas = [token.lemma_ for token in doc if not token.is_punct]
    return ' '.join(lemas)

def EliminarStopwords(oracion):
    doc = nlp(oracion)
    tokens = [token.text for token in doc if token.text not in STOP_WORDS and not token.is_punct]
    return ' '.join(tokens)

def EliminaNumeroYPuntuacion(oracion):
    string_numeros = regex.sub(r'\d+', '', oracion)
    return ''.join(c for c in string_numeros if c not in punctuation)

def ConstruirYGuardarModelo(path_csv, nombre_modelo, remove_stopwords=True, use_lemmatization=True, min_df=5, max_df=0.8, ngram_range=(1,1)):
    corpus, doc_id = CrearCorpusDesdeCSV(path_csv)
    corpus_preprocesado = PreProcesar(corpus, remove_stopwords, use_lemmatization)

    # Crear un diccionario que mapee doc_id al contenido original del documento
    doc_contents = dict(zip(doc_id, corpus))

    # Crear carpeta si no existe
    if not os.path.exists(nombre_modelo):
        os.makedirs(nombre_modelo)

    # Ajustar parámetros del vectorizador
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words='english' if remove_stopwords else None
    )

    tfidf_matrix = vectorizer.fit_transform(corpus_preprocesado)

    # Guardar el vectorizador completo y la matriz TF-IDF
    joblib.dump(vectorizer, f"{nombre_modelo}/vectorizer.pkl")
    joblib.dump(tfidf_matrix, f"{nombre_modelo}/tfidf_matrix.pkl")
    joblib.dump(doc_id, f"{nombre_modelo}/doc_id.pkl")
    # Guardar el contenido de los documentos
    joblib.dump(doc_contents, f"{nombre_modelo}/doc_contents.pkl")

    print(f"Modelo guardado en la carpeta '{nombre_modelo}'.")

if __name__ == "__main__":
    path_csv = "ofertas_procesadas.csv"

    # Versión 1: Sin stopwords ni lematización
    ConstruirYGuardarModelo(
        path_csv,
        nombre_modelo="Modelo_V1",
        remove_stopwords=False,
        use_lemmatization=False,
        min_df=1,
        max_df=1.0,
        ngram_range=(1,1)
    )

    # Versión 2: Sin stopwords, con lematización
    ConstruirYGuardarModelo(
        path_csv,
        nombre_modelo="Modelo_V2",
        remove_stopwords=False,
        use_lemmatization=True,
        min_df=1,
        max_df=1.0,
        ngram_range=(1,1)
    )

    # Versión 3: Con stopwords y lematización 
    ConstruirYGuardarModelo(
        path_csv,
        nombre_modelo="Modelo_V3",
        remove_stopwords=True,
        use_lemmatization=True,
        min_df=5,
        max_df=0.8,
        ngram_range=(1,2)
    )
