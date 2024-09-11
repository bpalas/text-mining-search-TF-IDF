
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import os
import regex
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
import joblib
import es_core_news_sm
from string import punctuation
import spacy

def ConvertirAcentos(texto):
    texto=texto.replace("\xc3\xa1","á")
    texto=texto.replace("\xc3\xa9","é")   
    texto=texto.replace("\xc3\xad","í")
    texto=texto.replace("\xc3\xb3","ó")
    texto=texto.replace("\xc3\xba","ú")
    texto=texto.replace("\xc3\x81","Á")
    texto=texto.replace("\xc3\x89","É")
    texto=texto.replace("\xc3\x8d","Í")
    texto=texto.replace("\xc3\x93","Ó")
    texto=texto.replace("\xc3\x9a","Ú")
    texto=texto.replace("\xc3±","ñ")
    return(texto)

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
     
def PreProcesar(textos):
    texto_limpio = []
    for texto in textos:  
        texto = EliminarStopwords(texto.lower())    
        texto = Lematizar(texto)     
        texto = EliminaNumeroYPuntuacion(texto)      
        if len(texto)!=0:
          texto = regex.sub(' +', ' ', texto)
          texto_limpio.append(texto)
    return(texto_limpio)

def Lematizar(oracion):
   doc = nlp(oracion)
   lemas = [token.lemma_ for token in doc]
   return(Lista_a_Oracion(lemas))  

def Lista_a_Oracion(Lista):
   return(" ".join(Lista))          

def EliminarStopwords(oracion):
    Tokens = Tokenizar(oracion)
    oracion_filtrada =[] 
    for palabra in Tokens:
       if palabra not in STOP_WORDS:
           palabra_limpia = palabra.rstrip()
           if len(palabra_limpia)!=0:
              oracion_filtrada.append(palabra_limpia) 
    return(Lista_a_Oracion(oracion_filtrada))

def Tokenizar(oracion):
    doc = nlp(oracion)
    tokens = [palabra.text for palabra in doc]
    return(tokens)

def EliminaNumeroYPuntuacion(oracion):
    string_numeros = regex.sub(r'[\”\“\¿\°\d+]','', oracion)
    return ''.join(c for c in string_numeros if c not in punctuation)

path_csv = "ofertas_procesadas.csv"  
corpus, doc_id = CrearCorpusDesdeCSV(path_csv)


def CargarModelo(NombreModelo):
    modelo = joblib.load(NombreModelo+"/"+'tfidf.pkl')
    idf   = joblib.load(NombreModelo+"/"+'idf.pkl')
    vocab  = joblib.load(NombreModelo+"/"+'vocab.pkl')
    return(modelo,idf,vocab)    

def crearQuery(terms,idf,vocabulario):
    query = np.zeros(len(vocabulario))
    listaTerminos = Tokenizar(Lematizar(terms))
    for t in listaTerminos:      
       try:
           indice = vocabulario[t]
           query[indice] = 1
       except KeyError:
           indice = -1
    if (np.count_nonzero(query) != 0):
              query = query * idf
              return(query)
    return([])
def RecuperarDocumentosRelevantes(query, modelo, doc_id):
    RelDocs = []
    for ind_doc in range(len(doc_id)):
        filename = doc_id[ind_doc]  
        similitud = 1 - cosine(query, modelo[ind_doc, :])
        RelDocs.append((similitud, filename))  
    
    RelDocs = sorted(RelDocs, reverse=True)
    
    return RelDocs[:10]
def MostrarDocumentos(Docs):
    print("Lista de documentos relevantes a la query:\n")
    for (sim,d) in Docs:
        print("Doc: "+d+" ("+str(sim)+")\n")



(tfidf, idf, vocabulario) = CargarModelo("Linkedin 2.0")


print("*********************************************")
print("        Bienvenido al Linkedin 2.0!")
print("*********************************************")

terms = input("Ingrese query: ")
vector_query = crearQuery(terms,idf,vocabulario)

if len(vector_query)==0:
    print("ERROR en vector de consulta, no se pueden recuperar documentos!..")
else:
    DocsRelevantes = RecuperarDocumentosRelevantes(vector_query,tfidf,doc_id)
    MostrarDocumentos(DocsRelevantes)