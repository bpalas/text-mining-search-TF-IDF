import numpy as np
import regex
import joblib
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
import sys

nlp = spacy.load('en_core_web_sm')

def PreProcesarConsulta(texto, remove_stopwords=True, use_lemmatization=True):
    texto = texto.lower()
    if remove_stopwords:
        texto = EliminarStopwords(texto)
    if use_lemmatization:
        texto = Lematizar(texto)
    texto = EliminaNumeroYPuntuacion(texto)
    texto = regex.sub(' +', ' ', texto)
    return texto

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

def CargarModelo(nombre_modelo):
    vectorizer = joblib.load(f"{nombre_modelo}/vectorizer.pkl")
    tfidf_matrix = joblib.load(f"{nombre_modelo}/tfidf_matrix.pkl")
    doc_id = joblib.load(f"{nombre_modelo}/doc_id.pkl")
    # Cargar el contenido de los documentos
    doc_contents = joblib.load(f"{nombre_modelo}/doc_contents.pkl")
    return vectorizer, tfidf_matrix, doc_id, doc_contents

def RecuperarDocumentosRelevantes(query, vectorizer, tfidf_matrix, doc_id, remove_stopwords=True, use_lemmatization=True):
    query_preprocesada = PreProcesarConsulta(query, remove_stopwords, use_lemmatization)
    query_vector = vectorizer.transform([query_preprocesada])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1]
    RelDocs = [(cosine_similarities[idx], doc_id[idx]) for idx in top_indices[:10]]
    return RelDocs

def MostrarDocumentos(Docs, doc_contents=None, mostrar_contenido=False):
    print("Lista de documentos relevantes a la query:\n")
    for (sim, d) in Docs:
        print(f"Doc: {d} (Similitud: {sim:.4f})")
        if mostrar_contenido and doc_contents is not None:
            # Obtener el contenido del documento usando el doc_id
            doc_text = doc_contents.get(d, "Documento no encontrado.")
            print(f"Contenido del documento:\n{doc_text}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        nombre_modelo = sys.argv[1]
    else:
        nombre_modelo = "Modelo_V3"  

    if nombre_modelo == "Modelo_V1":
        remove_stopwords = False
        use_lemmatization = False
    elif nombre_modelo == "Modelo_V2":
        remove_stopwords = False
        use_lemmatization = True
    elif nombre_modelo == "Modelo_V3":
        remove_stopwords = True
        use_lemmatization = True
    else:
        remove_stopwords = True
        use_lemmatization = True

    vectorizer, tfidf_matrix, doc_id, doc_contents = CargarModelo(nombre_modelo)

    print("*********************************************")
    print(f"        Bienvenido al {nombre_modelo}!")
    print("*********************************************")

    terms = input("Ingrese query: ")
    DocsRelevantes = RecuperarDocumentosRelevantes(
        terms,
        vectorizer,
        tfidf_matrix,
        doc_id,
        remove_stopwords=remove_stopwords,
        use_lemmatization=use_lemmatization
    )
    # Puedes elegir si mostrar el contenido o no
    MostrarDocumentos(DocsRelevantes, doc_contents, mostrar_contenido=False)
