# import streamlit as st
# import gensim
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import umap
# import plotly.express as px
# import os

# # Функция для загрузки обученной модели векторных представлений
# @st.cache_resource
# def load_model(model_path):
#     try:
#         model = None
#         model_type = None
        
#         # Определяем тип модели по имени файла
#         if 'fasttext' in model_path.lower():
#             model = gensim.models.FastText.load(model_path)
#             model_type = 'fasttext'
#         elif 'word2vec' in model_path.lower():
#             model = gensim.models.Word2Vec.load(model_path)
#             model_type = 'word2vec'
#         elif 'doc2vec' in model_path.lower() or 'd2v' in model_path.lower():
#             model = gensim.models.Doc2Vec.load(model_path)
#             model_type = 'doc2vec'
#         else:
#             st.error(f"Неизвестный тип модели для пути: {model_path}")
#             return None, None

#         # Для Doc2Vec моделей предоставляем выбор между векторами слов и документов
#         if model_type == 'doc2vec':
#             if hasattr(model, 'wv') and len(model.wv) > 0:
#                 return model.wv, 'doc2vec_words'
#             elif hasattr(model, 'dv') and len(model.dv) > 0:
#                 return model.dv, 'doc2vec_docs'
#             else:
#                 st.error("Doc2Vec модель не содержит ни векторов слов, ни векторов документов")
#                 return None, None
#         elif model and hasattr(model, 'wv'):
#             return model.wv, model_type
#         else:
#             st.error(f"Загруженный объект модели не имеет атрибута '.wv'. Проверьте тип загруженного файла.")
#             return None, None

#     except FileNotFoundError:
#         st.error(f"Файл модели не найден по пути: {model_path}")
#         return None, None
#     except Exception as e:
#         st.error(f"Ошибка загрузки модели {model_path}: {e}")
#         st.exception(e)
#         return None, None

# # Функция для проверки наличия токена в модели
# def token_in_model(model, token, model_type):
#     if model_type == 'doc2vec_docs':
#         # Для документов проверяем наличие тега
#         return token in model
#     else:
#         # Для слов используем стандартную проверку
#         return token in model

# # Функция для получения вектора
# def get_vector(model, token, model_type):
#     if model_type == 'doc2vec_docs':
#         # Для документов получаем вектор документа
#         return model[token]
#     else:
#         # Для слов используем стандартный метод
#         return model[token]

# # --- Streamlit приложение ---
# st.title("Интерактивный анализ векторных пространств")

# st.sidebar.header("Настройки модели")

# # Поиск доступных моделей в целевой директории
# path_to_models = os.path.join("Текст Анализ", "models")
# model_files = [f for f in os.listdir(path_to_models) if f.endswith('.model')]
# selected_model_path = st.sidebar.selectbox("Выберите модель:", model_files)

# model_wv = None
# model_type = None

# if selected_model_path:
#     full_model_path = os.path.join(path_to_models, selected_model_path)
#     model_wv, model_type = load_model(full_model_path)

# if model_wv is None:
#     st.warning("Пожалуйста, выберите и загрузите модель для продолжения.")
# else:
#     st.success(f"Модель '{selected_model_path}' успешно загружена. Тип: {model_type}. Размер словаря: {len(model_wv)}")

#     # Показываем предупреждение для Doc2Vec моделей с документами
#     if model_type == 'doc2vec_docs':
#         st.info("⚠️ Загружена Doc2Vec модель с векторами документов. Работайте с тегами документов вместо отдельных слов.")

#     # --- 1. Интерактивная векторная арифметика ---
#     st.header("1. Интерактивная векторная арифметика")
    
#     if model_type == 'doc2vec_docs':
#         st.write("Введите выражение с тегами документов в формате 'doc1 - doc2 + doc3'")
#     else:
#         st.write("Введите выражение в формате 'слово1 - слово2 + слово3' (используйте токены из словаря модели).")

#     default_example = "путин - мужчин + женщин" if model_type != 'doc2vec_docs' else "DOC_1 - DOC_2 + DOC_3"
#     arithmetic_input = st.text_input("Введите выражение:", default_example)

#     if st.button("Вычислить векторную арифметику"):
#         if model_wv:
#             try:
#                 # Парсинг входной строки
#                 parts = arithmetic_input.split()
#                 positive = []
#                 negative = []
#                 current_op = '+'
#                 valid_input = True

#                 for part in parts:
#                     if part == '+':
#                         current_op = '+'
#                     elif part == '-':
#                         current_op = '-'
#                     elif token_in_model(model_wv, part, model_type):
#                         if current_op == '+':
#                             positive.append(part)
#                         else:
#                             negative.append(part)
#                     else:
#                         st.warning(f"Токен '{part}' не найден в модели или формат неверен.")
#                         valid_input = False
#                         break

#                 if valid_input and (positive or negative):
#                     st.write(f"Вычисление: {' + '.join(positive)} - {' - '.join(negative)}")
#                     try:
#                         # Вычисление результирующего вектора
#                         if positive:
#                             result_vector = get_vector(model_wv, positive[0], model_type).copy()
#                         else:
#                             result_vector = np.zeros(model_wv.vector_size)
                        
#                         for token in positive[1:]:
#                             result_vector += get_vector(model_wv, token, model_type).copy()
#                         for token in negative:
#                             result_vector -= get_vector(model_wv, token, model_type).copy()

#                         st.write("Ближайшие соседи для результирующего вектора:")
                        
#                         # Поиск ближайших соседей
#                         if model_type == 'doc2vec_docs':
#                             # Для документов используем свой подход
#                             similarities = []
#                             for doc_tag in list(model_wv.key_to_index.keys())[:100]:  # Ограничиваем поиск
#                                 doc_vector = get_vector(model_wv, doc_tag, model_type)
#                                 similarity = cosine_similarity([result_vector], [doc_vector])[0][0]
#                                 similarities.append((doc_tag, similarity))
                            
#                             # Сортируем по убыванию сходства
#                             similarities.sort(key=lambda x: x[1], reverse=True)
#                             for token, similarity in similarities[:10]:
#                                 st.write(f"- {token} (Сходство: {similarity:.4f})")
#                         else:
#                             # Для слов используем встроенный метод
#                             most_similar_results = model_wv.most_similar(positive=positive, negative=negative, topn=10)
#                             for word, similarity in most_similar_results:
#                                 st.write(f"- {word} (Сходство: {similarity:.4f})")

#                     except Exception as e:
#                         st.error(f"Ошибка при вычислении или поиске соседей: {e}")

#                 elif valid_input:
#                     st.warning("Введите токены для векторной арифметики.")

#             except Exception as e:
#                 st.error(f"Ошибка парсинга ввода: {e}")
#         else:
#             st.warning("Модель не загружена.")

#     # --- 2. Эксперименты с семантическим сходством ---
#     st.header("2. Эксперименты с семантическим сходством")

#     label_1 = "Тег документа 1:" if model_type == 'doc2vec_docs' else "Слово 1:"
#     label_2 = "Тег документа 2:" if model_type == 'doc2vec_docs' else "Слово 2:"
    
#     default_1 = "DOC_1" if model_type == 'doc2vec_docs' else "путин"
#     default_2 = "DOC_2" if model_type == 'doc2vec_docs' else "президент"
    
#     token1_sim = st.text_input(label_1, default_1)
#     token2_sim = st.text_input(label_2, default_2)

#     if st.button("Рассчитать косинусное сходство"):
#         if model_wv:
#             if (token_in_model(model_wv, token1_sim, model_type) and 
#                 token_in_model(model_wv, token2_sim, model_type)):
#                 try:
#                     if model_type == 'doc2vec_docs':
#                         # Для документов вычисляем сходство вручную
#                         vec1 = get_vector(model_wv, token1_sim, model_type)
#                         vec2 = get_vector(model_wv, token2_sim, model_type)
#                         similarity = cosine_similarity([vec1], [vec2])[0][0]
#                     else:
#                         # Для слов используем встроенный метод
#                         similarity = model_wv.similarity(token1_sim, token2_sim)
                    
#                     entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
#                     st.write(f"Косинусное сходство между '{token1_sim}' и '{token2_sim}': {similarity:.4f}")
#                 except Exception as e:
#                     st.error(f"Ошибка при расчете сходства: {e}")
#             else:
#                 oov_tokens = [t for t in [token1_sim, token2_sim] if not token_in_model(model_wv, t, model_type)]
#                 entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
#                 st.warning(f"Один или оба {entity_type} не найдены в модели: {', '.join(oov_tokens)}")
#         else:
#             st.warning("Модель не загружена.")

#     # Остальные разделы (визуализация осей, UMAP) также нужно адаптировать для Doc2Vec
#     # Для краткости оставлю их без изменений, но в реальном приложении их тоже нужно доработать

#     # --- 3. Визуализация семантических осей ---
#     st.header("3. Визуализация семантических осей")
    
#     if model_type == 'doc2vec_docs':
#         st.write("Выберите два тега документа для определения семантической оси.")
#         pole1_label = "Полюс 1 (тег документа):"
#         pole2_label = "Полюс 2 (тег документа):"
#         default_pole1 = "DOC_1"
#         default_pole2 = "DOC_2"
#     else:
#         st.write("Выберите два слова для определения семантической оси и визуализируйте проекции других слов.")
#         pole1_label = "Полюс 1:"
#         pole2_label = "Полюс 2:"
#         default_pole1 = "мужчин"
#         default_pole2 = "женщин"

#     axis_token1 = st.text_input(pole1_label, default_pole1)
#     axis_token2 = st.text_input(pole2_label, default_pole2)
#     num_tokens_on_axis = st.slider("Количество токенов для проекции:", 10, 100, 30)

#     if st.button("Визуализировать ось"):
#         if model_wv:
#             if (token_in_model(model_wv, axis_token1, model_type) and 
#                 token_in_model(model_wv, axis_token2, model_type)):
#                 try:
#                     # Получаем вектор оси
#                     axis_vector = (get_vector(model_wv, axis_token2, model_type) - 
#                                  get_vector(model_wv, axis_token1, model_type))

#                     # Получаем список токенов для проекции
#                     all_tokens = list(model_wv.key_to_index.keys())
#                     # Пропускаем первые несколько токенов для разнообразия
#                     start_idx = min(50, len(all_tokens) - num_tokens_on_axis)
#                     tokens_to_project = all_tokens[start_idx:start_idx + num_tokens_on_axis]

#                     # Вычисляем проекции
#                     projections = []
#                     token_labels = []
#                     for token in tokens_to_project:
#                         if token_in_model(model_wv, token, model_type):
#                             token_vector = get_vector(model_wv, token, model_type)
#                             # Нормализуем вектор оси
#                             norm_axis_vector = axis_vector / np.linalg.norm(axis_vector)
#                             projection_value = np.dot(token_vector, norm_axis_vector)
#                             projections.append(projection_value)
#                             token_labels.append(token)

#                     # Создаем DataFrame для построения графика
#                     projection_df = pd.DataFrame({'Token': token_labels, 'Projection': projections})
#                     projection_df = projection_df.sort_values('Projection')

#                     # Построение графика
#                     entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
#                     fig = px.bar(projection_df, x='Projection', y='Token', orientation='h',
#                                  title=f'Проекция {entity_type} на ось "{axis_token1}" - "{axis_token2}"')
#                     st.plotly_chart(fig)

#                 except Exception as e:
#                     st.error(f"Ошибка при визуализации оси: {e}")
#             else:
#                 oov_tokens = [t for t in [axis_token1, axis_token2] if not token_in_model(model_wv, t, model_type)]
#                 entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
#                 st.warning(f"Один или оба полюса оси не найдены в модели: {', '.join(oov_tokens)}")
#         else:
#             st.warning("Модель не загружена.")

#     # --- 4. Визуализация 2D/3D проекций ---
#     st.header("4. Визуализация 2D/3D проекций")
#     st.write("Визуализируйте токены в 2D пространстве с использованием UMAP.")

#     num_tokens_for_viz = st.slider("Количество токенов для визуализации:", 50, 500, 200)

#     if st.button("Построить 2D проекцию (UMAP)"):
#         if model_wv:
#             try:
#                 # Получаем векторы для выборки токенов
#                 all_tokens = list(model_wv.key_to_index.keys())
#                 tokens_for_viz = all_tokens[:num_tokens_for_viz]
#                 vectors_for_viz = np.array([get_vector(model_wv, token, model_type) for token in tokens_for_viz])

#                 # Применяем UMAP
#                 reducer = umap.UMAP(n_components=2, random_state=42)
#                 embedding_2d = reducer.fit_transform(vectors_for_viz)

#                 # Создаем DataFrame для визуализации
#                 viz_df = pd.DataFrame(embedding_2d, columns=['UMAP 1', 'UMAP 2'])
#                 viz_df['Token'] = tokens_for_viz

#                 # Построение графика
#                 entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
#                 fig = px.scatter(viz_df, x='UMAP 1', y='UMAP 2', text='Token',
#                                  title=f'2D UMAP проекция {num_tokens_for_viz} {entity_type}',
#                                  hover_name='Token')
#                 fig.update_traces(textposition='top center')
#                 st.plotly_chart(fig)

#             except Exception as e:
#                 st.error(f"Ошибка при построении UMAP проекции: {e}")
#         else:
#             st.warning("Модель не загружена.")

import streamlit as st
import gensim
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import umap
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Функция для загрузки обученной модели векторных представлений
@st.cache_resource
def load_model(model_path):
    try:
        model = None
        model_type = None
        
        # Определяем тип модели по имени файла
        if 'fasttext' in model_path.lower():
            model = gensim.models.FastText.load(model_path)
            model_type = 'fasttext'
        elif 'word2vec' in model_path.lower():
            model = gensim.models.Word2Vec.load(model_path)
            model_type = 'word2vec'
        elif 'doc2vec' in model_path.lower() or 'd2v' in model_path.lower():
            model = gensim.models.Doc2Vec.load(model_path)
            model_type = 'doc2vec'
        else:
            st.error(f"Неизвестный тип модели для пути: {model_path}")
            return None, None

        # Для Doc2Vec моделей предоставляем выбор между векторами слов и документов
        if model_type == 'doc2vec':
            if hasattr(model, 'wv') and len(model.wv) > 0:
                return model.wv, 'doc2vec_words'
            elif hasattr(model, 'dv') and len(model.dv) > 0:
                return model.dv, 'doc2vec_docs'
            else:
                st.error("Doc2Vec модель не содержит ни векторов слов, ни векторов документов")
                return None, None
        elif model and hasattr(model, 'wv'):
            return model.wv, model_type
        else:
            st.error(f"Загруженный объект модели не имеет атрибута '.wv'. Проверьте тип загруженного файла.")
            return None, None

    except FileNotFoundError:
        st.error(f"Файл модели не найден по пути: {model_path}")
        return None, None
    except Exception as e:
        st.error(f"Ошибка загрузки модели {model_path}: {e}")
        st.exception(e)
        return None, None

# Функция для проверки наличия токена в модели
def token_in_model(model, token, model_type):
    if model_type == 'doc2vec_docs':
        return token in model
    else:
        return token in model

# Функция для получения вектора
def get_vector(model, token, model_type):
    if model_type == 'doc2vec_docs':
        return model[token]
    else:
        return model[token]

# --- Streamlit приложение ---
st.title("Интерактивный анализ векторных пространств")

st.sidebar.header("Настройки модели")

# Поиск доступных моделей в целевой директории
path_to_models = os.path.join("Текст Анализ", "models")
model_files = [f for f in os.listdir(path_to_models) if f.endswith('.model')]
selected_model_path = st.sidebar.selectbox("Выберите модель:", model_files)

model_wv = None
model_type = None

if selected_model_path:
    full_model_path = os.path.join(path_to_models, selected_model_path)
    model_wv, model_type = load_model(full_model_path)

if model_wv is None:
    st.warning("Пожалуйста, выберите и загрузите модель для продолжения.")
else:
    st.success(f"Модель '{selected_model_path}' успешно загружена. Тип: {model_type}. Размер словаря: {len(model_wv)}")

    # Показываем предупреждение для Doc2Vec моделей с документами
    if model_type == 'doc2vec_docs':
        st.info("⚠️ Загружена Doc2Vec модель с векторами документов. Работайте с тегами документов вместо отдельных слов.")

    # --- 1. Интерактивная векторная арифметика ---
    st.header("1. Интерактивная векторная арифметика")
    
    if model_type == 'doc2vec_docs':
        st.write("Введите выражение с тегами документов в формате 'doc1 - doc2 + doc3'")
    else:
        st.write("Введите выражение в формате 'слово1 - слово2 + слово3' (используйте токены из словаря модели).")

    default_example = "путин - мужчин + женщин" if model_type != 'doc2vec_docs' else "DOC_1 - DOC_2 + DOC_3"
    arithmetic_input = st.text_input("Введите выражение:", default_example)

    # Сохраняем результаты векторной арифметики для отчёта
    arithmetic_results = None

    if st.button("Вычислить векторную арифметику"):
        if model_wv:
            try:
                parts = arithmetic_input.split()
                positive = []
                negative = []
                current_op = '+'
                valid_input = True

                for part in parts:
                    if part == '+':
                        current_op = '+'
                    elif part == '-':
                        current_op = '-'
                    elif token_in_model(model_wv, part, model_type):
                        if current_op == '+':
                            positive.append(part)
                        else:
                            negative.append(part)
                    else:
                        st.warning(f"Токен '{part}' не найден в модели или формат неверен.")
                        valid_input = False
                        break

                if valid_input and (positive or negative):
                    st.write(f"Вычисление: {' + '.join(positive)} - {' - '.join(negative)}")
                    try:
                        if positive:
                            result_vector = get_vector(model_wv, positive[0], model_type).copy()
                        else:
                            result_vector = np.zeros(model_wv.vector_size)
                        
                        for token in positive[1:]:
                            result_vector += get_vector(model_wv, token, model_type).copy()
                        for token in negative:
                            result_vector -= get_vector(model_wv, token, model_type).copy()

                        st.write("Ближайшие соседи для результирующего вектора:")
                        
                        if model_type == 'doc2vec_docs':
                            similarities = []
                            for doc_tag in list(model_wv.key_to_index.keys())[:100]:
                                doc_vector = get_vector(model_wv, doc_tag, model_type)
                                similarity = cosine_similarity([result_vector], [doc_vector])[0][0]
                                similarities.append((doc_tag, similarity))
                            
                            similarities.sort(key=lambda x: x[1], reverse=True)
                            arithmetic_results = similarities[:10]
                            for token, similarity in arithmetic_results:
                                st.write(f"- {token} (Сходство: {similarity:.4f})")
                        else:
                            arithmetic_results = model_wv.most_similar(positive=positive, negative=negative, topn=10)
                            for word, similarity in arithmetic_results:
                                st.write(f"- {word} (Сходство: {similarity:.4f})")

                    except Exception as e:
                        st.error(f"Ошибка при вычислении или поиске соседей: {e}")

                elif valid_input:
                    st.warning("Введите токены для векторной арифметики.")

            except Exception as e:
                st.error(f"Ошибка парсинга ввода: {e}")
        else:
            st.warning("Модель не загружена.")

    # --- 2. Эксперименты с семантическим сходством ---
    st.header("2. Эксперименты с семантическим сходством")

    label_1 = "Тег документа 1:" if model_type == 'doc2vec_docs' else "Слово 1:"
    label_2 = "Тег документа 2:" if model_type == 'doc2vec_docs' else "Слово 2:"
    
    default_1 = "DOC_1" if model_type == 'doc2vec_docs' else "путин"
    default_2 = "DOC_2" if model_type == 'doc2vec_docs' else "президент"
    
    token1_sim = st.text_input(label_1, default_1)
    token2_sim = st.text_input(label_2, default_2)

    similarity_result = None

    if st.button("Рассчитать косинусное сходство"):
        if model_wv:
            if (token_in_model(model_wv, token1_sim, model_type) and 
                token_in_model(model_wv, token2_sim, model_type)):
                try:
                    if model_type == 'doc2vec_docs':
                        vec1 = get_vector(model_wv, token1_sim, model_type)
                        vec2 = get_vector(model_wv, token2_sim, model_type)
                        similarity_result = cosine_similarity([vec1], [vec2])[0][0]
                    else:
                        similarity_result = model_wv.similarity(token1_sim, token2_sim)
                    
                    entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
                    st.write(f"Косинусное сходство между '{token1_sim}' и '{token2_sim}': {similarity_result:.4f}")
                except Exception as e:
                    st.error(f"Ошибка при расчете сходства: {e}")
            else:
                oov_tokens = [t for t in [token1_sim, token2_sim] if not token_in_model(model_wv, t, model_type)]
                entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
                st.warning(f"Один или оба {entity_type} не найдены в модели: {', '.join(oov_tokens)}")
        else:
            st.warning("Модель не загружена.")

    # --- 3. Визуализация семантических осей ---
    st.header("3. Визуализация семантических осей")
    
    if model_type == 'doc2vec_docs':
        st.write("Выберите два тега документа для определения семантической оси.")
        pole1_label = "Полюс 1 (тег документа):"
        pole2_label = "Полюс 2 (тег документа):"
        default_pole1 = "DOC_1"
        default_pole2 = "DOC_2"
    else:
        st.write("Выберите два слова для определения семантической оси и визуализируйте проекции других слов.")
        pole1_label = "Полюс 1:"
        pole2_label = "Полюс 2:"
        default_pole1 = "мужчин"
        default_pole2 = "женщин"

    axis_token1 = st.text_input(pole1_label, default_pole1)
    axis_token2 = st.text_input(pole2_label, default_pole2)
    num_tokens_on_axis = st.slider("Количество токенов для проекции:", 10, 100, 30)

    if st.button("Визуализировать ось"):
        if model_wv:
            if (token_in_model(model_wv, axis_token1, model_type) and 
                token_in_model(model_wv, axis_token2, model_type)):
                try:
                    axis_vector = (get_vector(model_wv, axis_token2, model_type) - 
                                 get_vector(model_wv, axis_token1, model_type))

                    all_tokens = list(model_wv.key_to_index.keys())
                    start_idx = min(50, len(all_tokens) - num_tokens_on_axis)
                    tokens_to_project = all_tokens[start_idx:start_idx + num_tokens_on_axis]

                    projections = []
                    token_labels = []
                    for token in tokens_to_project:
                        if token_in_model(model_wv, token, model_type):
                            token_vector = get_vector(model_wv, token, model_type)
                            norm_axis_vector = axis_vector / np.linalg.norm(axis_vector)
                            projection_value = np.dot(token_vector, norm_axis_vector)
                            projections.append(projection_value)
                            token_labels.append(token)

                    projection_df = pd.DataFrame({'Token': token_labels, 'Projection': projections})
                    projection_df = projection_df.sort_values('Projection')

                    entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
                    fig = px.bar(projection_df, x='Projection', y='Token', orientation='h',
                                 title=f'Проекция {entity_type} на ось "{axis_token1}" - "{axis_token2}"')
                    st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Ошибка при визуализации оси: {e}")
            else:
                oov_tokens = [t for t in [axis_token1, axis_token2] if not token_in_model(model_wv, t, model_type)]
                entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
                st.warning(f"Один или оба полюса оси не найдены в модели: {', '.join(oov_tokens)}")
        else:
            st.warning("Модель не загружена.")

    # --- 4. Визуализация 2D/3D проекций ---
    st.header("4. Визуализация 2D/3D проекций")
    st.write("Визуализируйте токены в 2D пространстве с использованием UMAP.")

    num_tokens_for_viz = st.slider("Количество токенов для визуализации:", 50, 500, 200)

    umap_results = None

    if st.button("Построить 2D проекцию (UMAP)"):
        if model_wv:
            try:
                all_tokens = list(model_wv.key_to_index.keys())
                tokens_for_viz = all_tokens[:num_tokens_for_viz]
                vectors_for_viz = np.array([get_vector(model_wv, token, model_type) for token in tokens_for_viz])

                reducer = umap.UMAP(n_components=2, random_state=42)
                embedding_2d = reducer.fit_transform(vectors_for_viz)

                viz_df = pd.DataFrame(embedding_2d, columns=['UMAP 1', 'UMAP 2'])
                viz_df['Token'] = tokens_for_viz
                
                # Сохраняем результаты для отчёта
                umap_results = {
                    'dataframe': viz_df,
                    'tokens': tokens_for_viz,
                    'vectors': vectors_for_viz
                }

                entity_type = "документов" if model_type == 'doc2vec_docs' else "слов"
                fig = px.scatter(viz_df, x='UMAP 1', y='UMAP 2', text='Token',
                                 title=f'2D UMAP проекция {num_tokens_for_viz} {entity_type}',
                                 hover_name='Token')
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Ошибка при построении UMAP проекции: {e}")
        else:
            st.warning("Модель не загружена.")

    # --- 5. Генерация динамического отчёта ---
    st.header("5. Генерация динамического отчёта")
    
    # Создаем вкладки для разных разделов отчёта
    report_tabs = st.tabs([
        "📊 Сводная статистика", 
        "🧮 Векторная арифметика", 
        "📈 Точность аналогий",
        "🔥 Heatmap близостей",
        "🔍 Кластерный анализ"
    ])
    
    with report_tabs[0]:
        st.subheader("Сводная статистика модели")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Размер словаря", len(model_wv))
        with col2:
            st.metric("Размерность векторов", model_wv.vector_size)
        with col3:
            st.metric("Тип модели", model_type)
        
        # Примеры токенов
        st.subheader("Примеры токенов")
        example_tokens = list(model_wv.key_to_index.keys())[:20]
        st.write(", ".join(example_tokens))
        
        # Распределение частот токенов (если доступно)
        if hasattr(model_wv, 'key_to_index'):
            st.subheader("Распределение частот токенов")
            token_counts = pd.DataFrame({
                'Token': list(model_wv.key_to_index.keys())[:50],
                'Index': list(range(50))
            })
            fig = px.bar(token_counts, x='Index', y='Index', 
                        title='Первые 50 токенов в словаре (условная частота)')
            st.plotly_chart(fig)
    
    with report_tabs[1]:
        st.subheader("Результаты векторной арифметики")
        
        if arithmetic_results:
            # Визуализация результатов
            if isinstance(arithmetic_results[0], tuple):
                words = [item[0] for item in arithmetic_results]
                similarities = [item[1] for item in arithmetic_results]
            else:
                words = [item[0] for item in arithmetic_results]
                similarities = [item[1] for item in arithmetic_results]
            
            results_df = pd.DataFrame({
                'Токен': words,
                'Сходство': similarities
            })
            
            fig = px.bar(results_df, x='Токен', y='Сходство',
                        title='Результаты векторной арифметики',
                        color='Сходство',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig)
            
            # Таблица результатов
            st.dataframe(results_df.style.format({'Сходство': '{:.4f}'}))
        else:
            st.info("Выполните векторную арифметику в разделе 1, чтобы увидеть результаты здесь.")
    
    with report_tabs[2]:
        st.subheader("Статистика по точности аналогий")
        
        # Предопределённые тесты аналогий
        analogy_tests = [
            ["мужчин", "женщин", "президент", "чемпионат"],
            ["париж", "франция", "москва", "россия"],
            ["холодный", "холоднее", "горячий", "горячее"],
            ["собака", "щенок", "кошка", "котенок"]
        ]
        
        # Пользовательские аналогии
        st.write("Добавьте свои тесты аналогий:")
        custom_analogy = st.text_input("Введите аналогию (формат: слово1 слово2 слово3 слово4):", 
                                      "мужчина женщина король королева")
        
        if st.button("Добавить тест"):
            parts = custom_analogy.split()
            if len(parts) == 4:
                analogy_tests.append(parts)
                st.success("Тест добавлен!")
        
        # Выполнение тестов
        correct = 0
        total = 0
        results = []
        
        for test in analogy_tests:
            if all(token_in_model(model_wv, word, model_type) for word in test):
                total += 1
                try:
                    # Вычисляем аналогию: word1 - word2 + word3 ≈ word4
                    positive = [test[0], test[2]]  # word1 и word3
                    negative = [test[1]]           # word2
                    
                    # Ищем ближайшие соседи
                    similar_words = model_wv.most_similar(
                        positive=positive, 
                        negative=negative, 
                        topn=5
                    )
                    
                    # Проверяем, есть ли целевое слово в топ-5 результатах
                    target_word = test[3]
                    found = any(target_word in word for word, score in similar_words)
                    
                    if found:
                        correct += 1
                        results.append({
                            'Аналогия': f"{test[0]} - {test[1]} + {test[2]} ≈ {test[3]}",
                            'Результат': '✅ Правильно',
                            'Топ-5 результатов': [word for word, score in similar_words[:5]]
                        })
                    else:
                        results.append({
                            'Аналогия': f"{test[0]} - {test[1]} + {test[2]} ≈ {test[3]}",
                            'Результат': '❌ Ошибка',
                            'Топ-5 результатов': [word for word, score in similar_words[:5]]
                        })
                        
                except Exception as e:
                    results.append({
                        'Аналогия': f"{test[0]} - {test[1]} + {test[2]} ≈ {test[3]}",
                        'Результат': f'⚠️ Ошибка: {str(e)}',
                        'Топ-5 результатов': []
                    })
        
        # Визуализация результатов
        if total > 0:
            accuracy = correct / total
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Точность", f"{accuracy:.2%}")
            with col2:
                st.metric("Пройдено тестов", f"{correct}/{total}")
            
            # График точности
            fig = px.pie(
                values=[correct, total - correct],
                names=['Правильно', 'Неправильно'],
                title='Распределение результатов тестов аналогий'
            )
            st.plotly_chart(fig)
            
            # Детальная таблица результатов
            st.subheader("Детальные результаты")
            for result in results:
                with st.expander(result['Аналогия']):
                    st.write(f"**Результат:** {result['Результат']}")
                    if result['Топ-5 результатов']:
                        st.write("**Топ-5 результатов:**")
                        for i, word in enumerate(result['Топ-5 результатов'], 1):
                            st.write(f"{i}. {word}")
        else:
            st.warning("Не удалось выполнить ни один тест аналогии. Проверьте наличие слов в словаре модели.")
    
    with report_tabs[3]:
        st.subheader("Heatmap семантических близостей")
        
        # Выбор слов для heatmap
        default_words = "путин, медведев, навальный, зеркало, собака, кошка, машина, дом, работа, деньги"
        selected_words = st.text_area(
            "Введите слова для heatmap (через запятую):", 
            default_words,
            height=100
        )
        
        words_list = [word.strip() for word in selected_words.split(',') if word.strip()]
        valid_words = [word for word in words_list if token_in_model(model_wv, word, model_type)]
        
        if len(valid_words) >= 2:
            # Создаем матрицу сходств
            similarity_matrix = np.zeros((len(valid_words), len(valid_words)))
            
            for i, word1 in enumerate(valid_words):
                for j, word2 in enumerate(valid_words):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        try:
                            if model_type == 'doc2vec_docs':
                                vec1 = get_vector(model_wv, word1, model_type)
                                vec2 = get_vector(model_wv, word2, model_type)
                                similarity_matrix[i, j] = cosine_similarity([vec1], [vec2])[0][0]
                            else:
                                similarity_matrix[i, j] = model_wv.similarity(word1, word2)
                        except:
                            similarity_matrix[i, j] = 0.0
            
            # Создаем heatmap
            fig = px.imshow(
                similarity_matrix,
                x=valid_words,
                y=valid_words,
                color_continuous_scale='RdBu_r',
                title='Heatmap семантических близостей',
                aspect="auto"
            )
            
            fig.update_layout(
                xaxis_title="Слова",
                yaxis_title="Слова"
            )
            
            st.plotly_chart(fig)
            
            # Дополнительная информация
            st.subheader("Статистика сходств")
            flat_similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Макс. сходство", f"{np.max(flat_similarities):.4f}")
            with col2:
                st.metric("Мин. сходство", f"{np.min(flat_similarities):.4f}")
            with col3:
                st.metric("Среднее сходство", f"{np.mean(flat_similarities):.4f}")
                
        else:
            st.warning("Введите как минимум 2 слова, присутствующих в модели.")
    
    with report_tabs[4]:
        st.subheader("Кластерный анализ 2D проекций")
        
        if umap_results is not None:
            # Настройки кластеризации
            n_clusters = st.slider("Количество кластеров:", 2, 10, 4)
            
            # Выполняем кластеризацию
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(umap_results['vectors'])
            
            # Добавляем информацию о кластерах в DataFrame
            cluster_df = umap_results['dataframe'].copy()
            cluster_df['Кластер'] = clusters.astype(str)
            cluster_df['Размер_точки'] = 5  # Для визуализации
            
            # Визуализация с кластерами
            fig = px.scatter(
                cluster_df, 
                x='UMAP 1', 
                y='UMAP 2', 
                color='Кластер',
                hover_name='Token',
                title=f'2D UMAP проекция с кластеризацией (K-means, k={n_clusters})',
                size='Размер_точки',
                size_max=8
            )
            
            st.plotly_chart(fig)
            
            # Анализ кластеров
            st.subheader("Анализ кластеров")
            
            for cluster_id in range(n_clusters):
                cluster_tokens = cluster_df[cluster_df['Кластер'] == str(cluster_id)]['Token'].tolist()
                with st.expander(f"Кластер {cluster_id} ({len(cluster_tokens)} токенов)"):
                    st.write(", ".join(cluster_tokens[:20]))  # Показываем первые 20 токенов
                    if len(cluster_tokens) > 20:
                        st.write(f"... и еще {len(cluster_tokens) - 20} токенов")
            
            # Метрики качества кластеризации
            from sklearn.metrics import silhouette_score
            
            try:
                silhouette_avg = silhouette_score(umap_results['vectors'], clusters)
                st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
                
                if silhouette_avg > 0.5:
                    st.success("Хорошее качество кластеризации")
                elif silhouette_avg > 0.25:
                    st.warning("Умеренное качество кластеризации")
                else:
                    st.error("Плохое качество кластеризации")
                    
            except Exception as e:
                st.warning(f"Не удалось вычислить метрики качества: {e}")
                
        else:
            st.info("Сначала постройте 2D проекцию в разделе 4, чтобы выполнить кластерный анализ.")
    
    # Кнопка для генерации полного отчёта
    st.divider()
    if st.button("📄 Сгенерировать полный отчёт PDF"):
        st.info("Функция генерации PDF отчёта в разработке. Все визуализации доступны выше.")