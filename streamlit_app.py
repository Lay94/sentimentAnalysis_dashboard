import pandas as pd
import wordcloud
import spacy
import re
import plotly.express as px
import matplotlib.pyplot as plt
from sentiment_analysis_spanish import sentiment_analysis
import streamlit as st
from PIL import Image

# Variables globales
nlp = spacy.load('es_core_news_md')
dir = "./data/radio_transcript.csv"
init_stopword = [
        'méxico', 'jefa', 'gobierno', 'Claudia'
    ]
init_name = 'Claudia'


def sentences(text):
    """ Separando las oraciones y las preguntas. """
    text = re.split('[.?]', text)
    clean_sent = []
    for sent in text:
        clean_sent.append(sent)
    return clean_sent

def data_preprocessing():
    """ Limpieza de los datos """
    
    # Importando y leyendo datos
    dtypes = {'city_code':'string', 'station':'string', 'time':'str', 'transcript':'string'}
    
    df = pd.read_csv(
        dir,
        encoding ='utf-8',
        dtype= dtypes,
        parse_dates = ['time']
        )
    
    # Eliminando valores perdidos
    df.dropna(inplace=True)

    # Agregando columna con las listas de oraciones
    df['sentences'] = df.transcript.apply(sentences)

    # Creando dataframe donde se almacenarán las oraciones por cada fila
    df_sentences = pd.DataFrame(columns=['city_code', 'station', 'time','sentences'])

    row_list = []

    # Separando las listas de oraciones
    for i in range(len(df)):
        for sentence in df.iloc[i,4]:
        
            city_code = df.iloc[i,0]
            station = df.iloc[i,1]
            time = df.iloc[i,2]

            row_dict = {'city_code':city_code,'station':station,'time':time, 'sentences':sentence}
            row_list.append(row_dict)
        
    df_sentences = pd.DataFrame(row_list)

    return df_sentences

def mentions(name = init_name):
    """ Buscando menciones """
    df_sentences = data_preprocessing()
    return df_sentences[df_sentences['sentences'].str.contains(name, na=False, regex=True)]

def word_cloud(name = init_name, stop_word=init_stopword):
    """ Creando nube de palabras """
    palabras_paro = nlp.Defaults.stop_words

    # Agregando palabras a ser ignoradas
    palabras_paro.update(stop_word)

    df_mentions = mentions(name)

    text = '\n'.join(df_mentions.sentences.str.lower().values)

    # Genera la nube de palabras
    wc = wordcloud.WordCloud(
        stopwords=palabras_paro
    ).generate(text)
    
    return wc

def get_sentiment(score):
        """ Clasificando sentimientos de acuerdo al score """
        if score < 0.00009:
            return 'Negativo'
        elif score > 0.6:
            return 'Positivo'
        else:
            return 'Neutral'


# Ajuste de páginas
st.set_page_config(layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Row A
a1, a2, a3 = st.columns([5, 2, 15])
with a2:
    a2.image(Image.open('logo.png'))
with a3:
    a3.title("Análisis de sentimiento")

name = st.text_input('Inserte el nombre de la persona', 'Claudia')

# Data
df_mentions = mentions(name)

# Creando modelo de análisis de sentimiento
sentiment = sentiment_analysis.SentimentAnalysisSpanish()
    
# La funcion sentiment.sentiment devuelve un valor entre 0 y 1. Mientra más cercano este a cero, más negativo será la mención
df_mentions['score_sp_model']=[sentiment.sentiment(text) for text in df_mentions.sentences]

# Clasificando cada mención de acuerdo al score
df_mentions['sentiment_sp_model']=df_mentions.score_sp_model.apply(get_sentiment)

text = '\n'.join(df_mentions.sentences.str.lower().values)

# Row B
st.write('--- El análisis de sentimiento global es', get_sentiment(sentiment.sentiment(text)), ' ---')

# Gráficos
### Global
fig1 = px.bar(df_mentions.groupby('sentiment_sp_model').count().sort_values(by='score_sp_model', ascending=False ).reset_index(), 
             x='sentiment_sp_model', y='score_sp_model', height=500)

### Menciones por estación de radio
fig2 = px.bar(df_mentions.groupby('station').count().sort_values(by='score_sp_model', ascending=False).reset_index(),
             x='station', y='score_sp_model')

### Sentimiento por estación de radio
df_sa_station = df_mentions.groupby(['station', 'sentiment_sp_model']).count()[['score_sp_model']]
fig3 = px.bar(df_sa_station.reset_index(), x="station", y="score_sp_model",
             color='sentiment_sp_model', barmode='group')

### Menciones por horas
df_time = df_mentions.groupby('time').count()[['score_sp_model']]
#Agregando horas q faltan, con 0 menciones
df_time_res = df_time.resample("60min").sum()
# Sumando menciones en periodos de una hora
periodo = df_time_res.index.to_period("H")
df_periodo = df_time_res.groupby([ periodo]).sum()
df_periodo.index = df_periodo.index.to_timestamp()
fig4 = px.line(df_periodo.reset_index(), x='time', y='score_sp_model', width=1050)


### Nube de palabras
wc = word_cloud(name)
fig_cloud, ax = plt.subplots()
ax.imshow(wc, interpolation='bilinear')
ax.axis("off")

# Row C
c1, c2 = st.columns((5,5))
with c1:
    st.markdown('### Nube de palabras')
    st.pyplot(fig_cloud)
with c2:
    st.markdown('### Análisis de sentimiento general')
    st.plotly_chart(fig1, use_container_width=True)
    
# Row D
d1, d2 = st.columns((5,5))
with d1:
    st.markdown('### Análisis de sentimiento por estación')
    st.plotly_chart(fig3, use_container_width=True)
with d2:
    st.markdown('### Cantidad de menciones por estación')
    st.plotly_chart(fig2, use_container_width=True)

# Row E
e1, e2, e3 = st.columns((15,20,5))
with e2:
    st.markdown('### Cantidad de menciones por hora')

# Row F
f1, f2, f3 = st.columns((5,15,5))
with f2:
    st.plotly_chart(fig4)