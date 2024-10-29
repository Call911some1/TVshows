# Импорт основных библиотек
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm

# Функция предобработки текста
def preprocess_text(text):
    # Удаление HTML-тегов
    text = re.sub(r'<[^>]+>', '', text)
    # Удаление небуквенных символов и цифр
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Предобработка без лемматизации
def preprocess_and_lemmatize(text):
    return preprocess_text(text)

# Функция для семантического поиска
def semantic_search(query, model, embeddings, data, top_k=5):
    # Предобработка запроса
    query_clean = preprocess_and_lemmatize(query)
    # Вычисление эмбеддинга для запроса
    query_embedding = model.encode([query_clean])
    # Вычисление косинусного сходства между запросом и эмбеддингами описаний
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    # Получение индексов топ-k наиболее похожих описаний
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    # Получение соответствующих записей из данных
    results = data.iloc[top_k_indices]
    return results

# Загрузка данных и модели
def load_data():
    data = pd.read_csv('processed_tvshows_data.csv')
    embeddings = np.load('embeddings.npy')
    return data, embeddings

def load_model():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model

# Инициализация данных и модели
data, embeddings = load_data()
model = load_model()

# Интерфейс Streamlit
st.title("Поиск сериалов по описанию")
st.write("Введите описание или ключевые слова, чтобы найти похожие сериалы.")

# Поле ввода для запроса
query = st.text_input("Введите описание или ключевые слова:")

# Кнопка для выполнения поиска
if st.button("Поиск"):
    if query:
        # Показ сообщения о загрузке
        with st.spinner("Идет поиск похожих сериалов..."):
            results = semantic_search(query, model, embeddings, data)

    # Вывод результатов
    st.write("## Результаты поиска")
    for index, row in results.iterrows():
        # Увеличение размера и выделение названия
        st.markdown(f"<h3 style='text-align: center; '>{row['tvshow_title']}</h3>", unsafe_allow_html=True)
        
        # Центрирование изображения
        st.markdown(
            f"<div style='display: flex; justify-content: center;'>"
            f"<img src='{row['image_url']}' width='400'></div>", 
            unsafe_allow_html=True
        )
        
        # Остальная информация о сериале
        st.write(f"**Жанры:** {row['genres']}")
        st.write(f"**Год:** {row['year']}")
        st.write(f"**Страна:** {row['country']}")
        st.write(f"**Описание:** {row['description']}")
        st.write(f"[Подробнее]({row['page_url']})")
        st.write("---")
    else:
        st.warning("Пожалуйста, введите запрос для поиска.")
