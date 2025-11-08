import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from scipy.stats import boxcox, boxcox_normmax, probplot, shapiro
from scipy.special import inv_boxcox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import json
import io


# Заголовок приложения
st.title("Интерактивное прогнозирование временных рядов")

# Инициализация session_state для хранения результатов моделей
if 'model_results' not in st.session_state:
    st.session_state.model_results = []

# --- Секция загрузки данных ---
st.header("Загрузка данных")
uploaded_file = st.file_uploader("Загрузите файл CSV или Parquet", type=["csv", "parquet"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)

        # Попытка преобразовать первый столбец в datetime, если он не называется 'timestamp'
        # или если есть столбец с именем 'timestamp'
        if 'timestamp' in df.columns:
             df['timestamp'] = pd.to_datetime(df['timestamp'])
             df.set_index('timestamp', inplace=True)
        elif isinstance(df.columns[0], str) and ('дата' in df.columns[0].lower() or 'date' in df.columns[0].lower()):
             try:
                 df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                 df.set_index(df.columns[0], inplace=True)
             except Exception as e:
                 st.warning(f"Не удалось преобразовать первый столбец '{df.columns[0]}' в datetime. Используйте столбец 'timestamp' или убедитесь, что формат даты корректен. Ошибка: {e}")
        else:
             st.warning("Не удалось определить столбец с меткой времени. Убедитесь, что есть столбец 'timestamp' или первый столбец содержит даты/время.")

        st.success("Файл успешно загружен.")
        st.write("Предварительный просмотр данных:")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Ошибка при загрузке или обработке файла: {e}")

# --- Секция выбора параметров ---
if df is not None:
    st.header("Настройки прогнозирования")

    # Выбор целевой переменной
    # Фильтруем нечисловые столбцы для выбора целевой переменной
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target_column = st.selectbox("Выберите целевую переменную", numeric_cols)

    if target_column:
        # Убедимся, что выбранный целевой столбец не содержит все NaN
        if df[target_column].dropna().empty:
            st.warning(f"Выбранный целевой столбец '{target_column}' содержит только пропущенные значения. Пожалуйста, выберите другой столбец или загрузите другие данные.")
            target_column = None # Сбрасываем выбор, если столбец пуст
        else:
             time_series = df[target_column].dropna() # Рабочий ряд без NaN в целевой переменной
             st.write(f"Выбран временной ряд: '{target_column}' с {len(time_series)} наблюдениями.")

             # Выбор горизонта прогноза
             h = st.slider("Выберите горизонт прогноза (количество шагов вперед)", min_value=7, max_value=365, value=30, step=7)
             st.write(f"Горизонт прогноза (h): {h}")


             # --- Секция декомпозиции (опционально) ---
             st.header("Декомпозиция временного ряда")
             perform_decomposition = st.checkbox("Выполнить декомпозицию?")

             if perform_decomposition:
                 decomposition_model = st.selectbox("Выберите модель декомпозиции", ['аддитивная', 'мультипликативная'])
                 seasonal_period = st.number_input("Введите период сезонности", min_value=1, value=12, step=1)

                 if decomposition_model == 'мультипликативная' and (time_series <= 0).any():
                      st.warning("Мультипликативная декомпозиция требует положительных значений. Будет использовано абсолютное значение ряда для декомпозиции.")
                      ts_for_decomp = time_series.abs()
                 else:
                      ts_for_decomp = time_series

                 try:
                     decomposition = seasonal_decompose(ts_for_decomp, model=decomposition_model, period=seasonal_period)

                     st.write(f"Результаты {decomposition_model} декомпозиции:")
                     
                     # Создаем subplots для декомпозиции
                     from plotly.subplots import make_subplots
                     
                     fig_decomp = make_subplots(
                         rows=4, cols=1,
                         subplot_titles=['Наблюдаемый ряд', 'Тренд', 'Сезонность', 'Остатки'],
                         vertical_spacing=0.08
                     )
                     
                     fig_decomp.add_trace(
                         go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Наблюдаемый'),
                         row=1, col=1
                     )
                     fig_decomp.add_trace(
                         go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Тренд'),
                         row=2, col=1
                     )
                     fig_decomp.add_trace(
                         go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Сезонность'),
                         row=3, col=1
                     )
                     fig_decomp.add_trace(
                         go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Остатки'),
                         row=4, col=1
                     )
                     
                     fig_decomp.update_layout(
                         height=800,
                         title_text=f'{decomposition_model.capitalize()} декомпозиция временного ряда',
                         showlegend=False
                     )
                     
                     st.plotly_chart(fig_decomp)

                 except Exception as e:
                      st.error(f"Ошибка при выполнении декомпозиции: {e}")

             # --- Секция преобразований (опционально) ---
             st.header("Преобразования ряда")
             apply_transformations = st.checkbox("Применить преобразования?")

             transformed_series = time_series.copy()
             transformation_params = {}
             transformation_type = 'Нет'

             if apply_transformations:
                 transformation_type = st.selectbox("Выберите тип преобразования", ['Нет', 'Лог-трансформация', 'Преобразование Бокса–Кокса', 'Дифференцирование'])

                 if transformation_type == 'Лог-трансформация':
                      if (transformed_series > 0).all():
                          transformed_series = np.log(transformed_series)
                          st.write("Применена лог-трансформация.")
                      else:
                          st.warning("Лог-трансформация требует положительных значений. Пропускаем.")

                 elif transformation_type == 'Преобразование Бокса–Кокса':
                      min_value = transformed_series.min()
                      if min_value <= 0:
                           shift_value = abs(min_value) + 1
                           transformed_series_shifted = transformed_series + shift_value
                           st.write(f"Ряд сдвинут на {shift_value} для Бокса–Кокса.")
                           transformation_params['shift'] = shift_value
                      else:
                           transformed_series_shifted = transformed_series
                           transformation_params['shift'] = 0

                      if (transformed_series_shifted.dropna() > 0).all():
                           try:
                               lambda_boxcox = boxcox_normmax(transformed_series_shifted.dropna())
                               # ИСПРАВЛЕНИЕ: boxcox возвращает только один массив
                               transformed_series = boxcox(transformed_series_shifted.dropna(), lmbda=lambda_boxcox)
                               transformed_series = pd.Series(transformed_series, index=transformed_series_shifted.dropna().index)
                               transformation_params['lambda'] = lambda_boxcox
                               st.write(f"Применено преобразование Бокса–Кокса с lambda = {lambda_boxcox:.4f}.")
                           except Exception as e:
                               st.error(f"Ошибка при выполнении преобразования Бокса–Кокса: {e}")
                               transformed_series = time_series.copy() # Возвращаемся к исходному ряду при ошибке
                               transformation_params = {} # Сбрасываем параметры
                      else:
                           st.warning("Ряд содержит неположительные значения даже после сдвига. Преобразование Бокса–Кокса невозможно.")
                           transformed_series = time_series.copy() # Возвращаемся к исходному ряду
                           transformation_params = {} # Сбрасываем параметры


                 elif transformation_type == 'Дифференцирование':
                      diff_order = st.number_input("Порядок обычного дифференцирования (d)", min_value=0, max_value=5, value=1, step=1)
                      seasonal_diff_order = st.number_input("Порядок сезонного дифференцирования (D)", min_value=0, max_value=5, value=0, step=1)
                      seasonal_period_diff = st.number_input("Период сезонного дифференцирования (m)", min_value=1, value=12, step=1)

                      if diff_order > 0:
                           # Сохраняем последние значения для обратного преобразования
                           transformation_params['last_values'] = []
                           for i in range(diff_order):
                               transformation_params['last_values'].append(transformed_series.iloc[-(i+1)])
                           
                           transformed_series = transformed_series.diff(diff_order).dropna()
                           st.write(f"Применено обычное дифференцирование порядка {diff_order}.")
                           transformation_params['diff_order'] = diff_order

                      if seasonal_diff_order > 0 and seasonal_period_diff > 0:
                           # Сохраняем последние сезонные значения для обратного преобразования
                           transformation_params['last_seasonal_values'] = transformed_series.iloc[-seasonal_period_diff:].tolist()
                           
                           transformed_series = transformed_series.diff(seasonal_period_diff).dropna()
                           st.write(f"Применено сезонное дифференцирование порядка {seasonal_diff_order} с периодом {seasonal_period_diff}.")
                           transformation_params['seasonal_diff_order'] = seasonal_diff_order
                           transformation_params['seasonal_period'] = seasonal_period_diff

             # Визуализация преобразованного ряда (если он отличается от исходного)
             if not transformed_series.equals(time_series):
                 st.write("Визуализация преобразованного ряда:")
                 fig_transformed = px.line(transformed_series)
                 fig_transformed.update_layout(
                     title='Преобразованный временной ряд', 
                     xaxis_title='Время', 
                     yaxis_title='Значение'
                 )
                 st.plotly_chart(fig_transformed)


             # --- Секция построения прогноза ---
             st.header("Построение прогноза")

             # Выбор модели экспоненциального сглаживания
             model_type = st.selectbox("Выберите модель экспоненциального сглаживания", ['SES', 'Хольта (аддитивная)', 'Хольта (мультипликативная)'])

             # Проверка применимости мультипликативной модели после преобразований
             if model_type == 'Хольта (мультипликативная)' and (transformed_series <= 0).any():
                 st.warning("Мультипликативная модель Хольта требует положительных значений после преобразований. Пожалуйста, выберите другую модель или измените преобразования.")
                 model_type = None # Сбрасываем выбор модели

             if model_type:
                 # Разделение на обучающую и тестовую выборки (простое разделение для прогнозирования)
                 train_size = int(len(transformed_series) * 0.8) # Пример: 80% для обучения
                 train_series = transformed_series.iloc[:train_size]
                 test_series = transformed_series.iloc[train_size:]

                 st.write(f"Размер обучающей выборки: {len(train_series)}")
                 st.write(f"Размер тестовой выборки (для оценки): {len(test_series)}")


                 # Инициализация и обучение выбранной модели
                 try:
                     if model_type == 'SES':
                         model = ExponentialSmoothing(train_series, trend=None, seasonal=None, initialization_method='estimated')
                         model_fit = model.fit(optimized=True)
                         st.write("Обучена модель SES.")
                         # Прогнозирование
                         forecast = model_fit.forecast(h)
                         forecast_lower = None
                         forecast_upper = None

                     elif model_type == 'Хольта (аддитивная)':
                         model = ExponentialSmoothing(train_series, trend='add', seasonal=None, initialization_method='estimated')
                         model_fit = model.fit(optimized=True)
                         st.write("Обучена модель Хольта (аддитивная).")
                         # Прогнозирование
                         forecast = model_fit.forecast(h)
                         forecast_lower = None
                         forecast_upper = None

                     elif model_type == 'Хольта (мультипликативная)':
                         # Убедились ранее, что данные положительны
                         model = ExponentialSmoothing(train_series, trend='mul', seasonal=None, initialization_method='estimated')
                         model_fit = model.fit(optimized=True)
                         st.write("Обучена модель Хольта (мультипликативная).")
                         # Прогнозирование
                         forecast = model_fit.forecast(h)
                         forecast_lower = None
                         forecast_upper = None

                     st.write(f"Получен прогноз на следующие {h} шагов.")

                     # --- Обратное преобразование прогнозов и фактических значений ---
                     forecast_to_evaluate = forecast.copy()
                     forecast_to_plot = forecast.copy()
                     actual_for_evaluation = test_series.iloc[:h] if len(test_series) >= h else test_series
                     actual_to_plot = time_series
                     forecast_lower_to_plot = forecast_lower
                     forecast_upper_to_plot = forecast_upper

                     if apply_transformations:
                         st.write("Применение обратного преобразования...")
                         
                         if transformation_type == 'Лог-трансформация':
                             try:
                                 forecast_original_scale = np.exp(forecast)
                                 test_series_original_scale = time_series.loc[test_series.index]
                                 
                                 forecast_to_evaluate = forecast_original_scale
                                 forecast_to_plot = forecast_original_scale
                                 actual_for_evaluation = test_series_original_scale.iloc[:h] if len(test_series_original_scale) >= h else test_series_original_scale
                                 
                                 st.success("Применена обратная лог-трансформация.")
                             except Exception as inverse_e:
                                 st.warning(f"Ошибка при обратной лог-трансформации: {inverse_e}")

                         elif transformation_type == 'Преобразование Бокса–Кокса':
                             try:
                                 lambda_val = transformation_params.get('lambda')
                                 shift_val = transformation_params.get('shift', 0)
                                 
                                 if lambda_val is not None:
                                     forecast_original_scale = inv_boxcox(forecast, lambda_val) - shift_val
                                     test_series_original_scale = time_series.loc[test_series.index]
                                     
                                     forecast_to_evaluate = forecast_original_scale
                                     forecast_to_plot = forecast_original_scale
                                     actual_for_evaluation = test_series_original_scale.iloc[:h] if len(test_series_original_scale) >= h else test_series_original_scale
                                     
                                     st.success("Применено обратное преобразование Бокса-Кокса.")
                                 else:
                                     st.warning("Не найден параметр lambda для обратного преобразования Бокса-Кокса.")
                             except Exception as inverse_e:
                                 st.warning(f"Ошибка при обратном преобразовании Бокса-Кокса: {inverse_e}")

                         elif transformation_type == 'Дифференцирование':
                             try:
                                 # Обратное дифференцирование для обычного дифференцирования
                                 diff_order = transformation_params.get('diff_order', 0)
                                 last_values = transformation_params.get('last_values', [])
                                 
                                 if diff_order > 0 and len(last_values) == diff_order:
                                     # Начинаем с последних значений обучающей выборки
                                     last_train_value = train_series.iloc[-1]
                                     forecast_undiff = [last_train_value]
                                     
                                     # Восстанавливаем ряд путем кумулятивного суммирования
                                     for i in range(len(forecast)):
                                         if i == 0:
                                             new_value = last_train_value + forecast.iloc[i]
                                         else:
                                             new_value = forecast_undiff[-1] + forecast.iloc[i]
                                         forecast_undiff.append(new_value)
                                     
                                     forecast_undiff = forecast_undiff[1:]  # Убираем начальное значение
                                     forecast_original_scale = pd.Series(forecast_undiff, index=forecast.index)
                                     
                                     # Для тестовых данных также нужно применить обратное преобразование
                                     # (упрощенная версия - в реальном приложении нужна более сложная логика)
                                     test_undiff = []
                                     last_actual_value = train_series.iloc[-1]
                                     
                                     for i in range(len(test_series)):
                                         if i == 0:
                                             new_val = last_actual_value + test_series.iloc[i]
                                         else:
                                             new_val = test_undiff[-1] + test_series.iloc[i] if i > 0 else last_actual_value + test_series.iloc[i]
                                         test_undiff.append(new_val)
                                     
                                     test_series_original_scale = pd.Series(test_undiff, index=test_series.index)
                                     
                                     forecast_to_evaluate = forecast_original_scale
                                     forecast_to_plot = forecast_original_scale
                                     actual_for_evaluation = test_series_original_scale.iloc[:h] if len(test_series_original_scale) >= h else test_series_original_scale
                                     
                                     st.success("Применено обратное дифференцирование.")
                                 else:
                                     st.warning("Обратное дифференцирование не может быть применено (недостаточно данных или неправильные параметры).")
                             except Exception as inverse_e:
                                 st.warning(f"Ошибка при обратном дифференцировании: {inverse_e}")

                     # --- Оценка качества прогноза (на первых h шагах тестовой выборки) ---
                     # Убедимся, что фактических значений в тестовой выборке достаточно для оценки
                     if len(actual_for_evaluation) >= min(h, len(actual_for_evaluation)):
                         eval_horizon = min(h, len(actual_for_evaluation))
                         mae = mean_absolute_error(actual_for_evaluation.iloc[:eval_horizon], forecast_to_evaluate.iloc[:eval_horizon])
                         rmse = np.sqrt(mean_squared_error(actual_for_evaluation.iloc[:eval_horizon], forecast_to_evaluate.iloc[:eval_horizon]))
                         
                         # MAPE (Mean Absolute Percentage Error) - избегать деления на 0
                         with np.errstate(divide='ignore', invalid='ignore'):
                             mape_values = np.abs((actual_for_evaluation.iloc[:eval_horizon] - forecast_to_evaluate.iloc[:eval_horizon]) / 
                                               actual_for_evaluation.iloc[:eval_horizon])
                             mape_values = mape_values.replace([np.inf, -np.inf], np.nan)
                             mape = np.nanmean(mape_values) * 100
                         
                         mape = np.nan_to_num(mape, nan=0.0)  # Заменяем NaN на 0

                         st.subheader("Оценка качества прогноза")
                         st.write(f"Метрики на первых {eval_horizon} шагах тестовой выборки:")
                         st.write(f"  MAE: {mae:.4f}")
                         st.write(f"  RMSE: {rmse:.4f}")
                         st.write(f"  MAPE: {mape:.4f}%")

                         # Сохраняем результаты модели
                         model_result = {
                             'model_type': model_type,
                             'mae': mae,
                             'rmse': rmse,
                             'mape': mape,
                             'forecast': forecast_to_plot,
                             'params': {
                                 'transformation_params': transformation_params,
                                 'transformation_type': transformation_type,
                                 'horizon': h,
                                 'target_column': target_column
                             }
                         }
                         
                         # Кнопка для сохранения результатов модели
                         if st.button("Сохранить результаты модели для сравнения"):
                             st.session_state.model_results.append(model_result)
                             st.success(f"Результаты модели {model_type} сохранены!")

                     else:
                          st.warning(f"В тестовой выборке недостаточно фактических значений ({len(actual_for_evaluation)}) для оценки прогноза на горизонт {h}.")


                     # --- Визуализация прогноза ---
                     st.subheader("Визуализация прогноза")
                     fig_forecast = go.Figure()
                     
                     # Добавляем обучающие данные
                     fig_forecast.add_trace(go.Scatter(
                         x=actual_to_plot.index, 
                         y=actual_to_plot, 
                         mode='lines', 
                         name='Фактические значения',
                         line=dict(color='blue')
                     ))
                     
                     # Добавляем прогноз
                     fig_forecast.add_trace(go.Scatter(
                         x=forecast_to_plot.index, 
                         y=forecast_to_plot, 
                         mode='lines', 
                         name='Прогноз',
                         line=dict(color='red', dash='dash')
                     ))

                     # Добавление доверительных интервалов, если они рассчитаны
                     if forecast_lower_to_plot is not None and forecast_upper_to_plot is not None:
                          fig_forecast.add_trace(go.Scatter(
                              x=forecast_upper_to_plot.index, 
                              y=forecast_upper_to_plot, 
                              fill=None, 
                              mode='lines', 
                              line_color='rgba(255,0,0,0.3)', 
                              showlegend=False
                          ))
                          fig_forecast.add_trace(go.Scatter(
                              x=forecast_lower_to_plot.index, 
                              y=forecast_lower_to_plot, 
                              fill='tonexty', 
                              mode='lines', 
                              line_color='rgba(255,0,0,0.3)', 
                              name='Доверительный интервал (95%)'
                          ))

                     fig_forecast.update_layout(
                         title=f'Прогноз с помощью {model_type}', 
                         xaxis_title='Время', 
                         yaxis_title=target_column
                     )
                     st.plotly_chart(fig_forecast)


                     # --- Диагностика остатков ---
                     st.subheader("Диагностика остатков (на обучающей выборке)")
                     if hasattr(model_fit, 'resid'):
                          residuals = model_fit.resid.dropna()

                          if not residuals.empty:
                               # Визуализация остатков vs время
                               st.write("Остатки vs время:")
                               fig_resid_time = px.line(residuals)
                               fig_resid_time.update_layout(
                                   title='Остатки на обучающей выборке', 
                                   xaxis_title='Время', 
                                   yaxis_title='Остатки'
                               )
                               fig_resid_time.add_shape(
                                   type='line', 
                                   xref='paper', 
                                   yref='y', 
                                   x0=0, x1=1, 
                                   y0=0, y1=0, 
                                   line=dict(color='red', dash='dash')
                               )
                               st.plotly_chart(fig_resid_time)

                               # Q-Q plot с использованием plotly
                               st.write("Q-Q plot остатков:")
                               try:
                                   # Используем scipy.stats.probplot для получения данных для Q-Q plot
                                   qq_data = probplot(residuals, dist="norm")
                                   theoretical_quantiles = qq_data[0][0]
                                   sample_quantiles = qq_data[0][1]
                                   
                                   fig_qq = go.Figure()
                                   fig_qq.add_trace(go.Scatter(
                                       x=theoretical_quantiles, 
                                       y=sample_quantiles, 
                                       mode='markers', 
                                       name='Квантили остатков'
                                   ))
                                   
                                   # Добавляем линию для идеального нормального распределения
                                   slope, intercept, r = qq_data[1]
                                   line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
                                   line_y = intercept + slope * line_x
                                   
                                   fig_qq.add_trace(go.Scatter(
                                       x=line_x, 
                                       y=line_y, 
                                       mode='lines', 
                                       name='Теоретическая линия',
                                       line=dict(color='red', dash='dash')
                                   ))
                                   
                                   fig_qq.update_layout(
                                       title='Q-Q plot остатков',
                                       xaxis_title='Теоретические квантили',
                                       yaxis_title='Выборочные квантили'
                                   )
                                   st.plotly_chart(fig_qq)
                                   
                               except Exception as qq_e:
                                    st.warning(f"Не удалось построить Q-Q plot: {qq_e}")


                               # Тест Льюнга–Бокса
                               st.write("Тест Льюнга–Бокса (автокорреляция остатков):")
                               try:
                                   # lags = min(40, len(residuals)//5)
                                   # Choose a reasonable number of lags, e.g., 20 or min(10, N/5)
                                   lags_to_test = np.arange(1, min(20, len(residuals)//5) + 1)
                                   if len(lags_to_test) > 0:
                                        ljung_box_results = acorr_ljungbox(residuals, lags=lags_to_test, return_df=True)
                                        st.dataframe(ljung_box_results)
                                        st.write("Интерпретация: Низкое p-значение (обычно < 0.05) указывает на наличие значимой автокорреляции в остатках.")
                                   else:
                                        st.write("Недостаточно данных для теста Льюнга–Бокса.")
                               except Exception as lb_e:
                                    st.warning(f"Ошибка при выполнении теста Льюнга–Бокса: {lb_e}")


                               # Тест Шапиро–Уилка (для нормальности)
                               st.write("Тест Шапиро–Уилка (нормальность остатков):")
                               # Shapiro-Wilk test is good for small samples (< 5000).
                               # For larger samples, it might be too sensitive or fail.
                               if len(residuals) <= 5000:
                                   try:
                                       shapiro_test = shapiro(residuals)
                                       st.write(f"Статистика Шапиро–Уилка: {shapiro_test[0]:.4f}")
                                       st.write(f"p-value Шапиро–Уилка: {shapiro_test[1]:.4f}")
                                       st.write("Интерпретация: Низкое p-значение (обычно < 0.05) указывает на отклонение от нормального распределения.")
                                   except Exception as shapiro_e:
                                       st.warning(f"Ошибка при выполнении теста Шапиро–Уилка: {shapiro_e}")
                               else:
                                   st.write(f"Тест Шапиро–Уилка не рекомендуется для выборок > 5000 наблюдений (текущий размер: {len(residuals)}).")
                                   st.write("Пожалуйста, используйте визуальный анализ (гистограмма, Q-Q plot) для оценки нормальности.")


                          else:
                              st.info("Остатки отсутствуют или содержат только NaN значения.")
                     else:
                          st.info("Модель не предоставляет остатков для анализа.")


                 except Exception as e:
                     st.error(f"Ошибка при обучении или прогнозировании модели: {e}")


# --- Секция сравнения метрик ---
st.header("Сравнение метрик моделей")

if st.session_state.model_results:
    # Создаем таблицу сравнения
    comparison_data = []
    for i, result in enumerate(st.session_state.model_results):
        comparison_data.append({
            'Модель': result['model_type'],
            'MAE': f"{result['mae']:.4f}",
            'RMSE': f"{result['rmse']:.4f}",
            'MAPE': f"{result['mape']:.4f}%",
            'Горизонт': result['params']['horizon']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Визуализация сравнения метрик
    if len(st.session_state.model_results) > 1:
        st.subheader("Визуализация сравнения моделей")
        
        models = [result['model_type'] for result in st.session_state.model_results]
        maes = [result['mae'] for result in st.session_state.model_results]
        rmses = [result['rmse'] for result in st.session_state.model_results]
        mapes = [result['mape'] for result in st.session_state.model_results]
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(name='MAE', x=models, y=maes))
        fig_comparison.add_trace(go.Bar(name='RMSE', x=models, y=rmses))
        fig_comparison.add_trace(go.Bar(name='MAPE', x=models, y=mapes))
        
        fig_comparison.update_layout(
            title='Сравнение метрик моделей',
            xaxis_title='Модели',
            yaxis_title='Значения метрик',
            barmode='group'
        )
        st.plotly_chart(fig_comparison)
    
    # Кнопка для очистки результатов
    if st.button("Очистить все результаты"):
        st.session_state.model_results = []
        st.success("Все результаты очищены!")
        st.rerun()
        
else:
    st.info("Нет сохраненных результатов моделей для сравнения. Постройте и сохраните несколько моделей, чтобы увидеть их сравнение здесь.")


# --- Секция экспорта ---
st.header("Экспорт результатов")

if st.session_state.model_results:
    # Экспорт последней модели
    latest_result = st.session_state.model_results[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Экспорт прогнозов в CSV
        forecast_df = pd.DataFrame({
            'timestamp': latest_result['forecast'].index,
            'forecast': latest_result['forecast'].values
        })
        
        csv_forecast = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Скачать прогнозы (CSV)",
            data=csv_forecast,
            file_name=f"forecast_{latest_result['model_type']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Экспорт параметров модели в JSON
        params_json = json.dumps(latest_result['params'], indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Скачать параметры (JSON)",
            data=params_json,
            file_name=f"model_params_{latest_result['model_type']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Экспорт таблицы сравнения
    if len(st.session_state.model_results) > 1:
        comparison_csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📊 Скачать таблицу сравнения (CSV)",
            data=comparison_csv,
            file_name=f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
else:
    st.info("Нет результатов для экспорта. Постройте модель, чтобы получить возможность экспортировать результаты.")


# --- Информация о возможных ошибках для других моделей ---
with st.expander("ℹ️ Информация о возможных ошибках для различных моделей"):
    st.write("""
    **Возможные ошибки для различных типов моделей:**
    
    **Для всех моделей экспоненциального сглаживания:**
    - `get_forecast()` метод не поддерживается - используем простой `forecast()`
    - Доверительные интервалы недоступны через стандартные методы
    - Мультипликативные модели требуют строго положительных значений
    
    **Для ARIMA/SARIMA моделей (если будут добавлены):**
    - Проблемы со стационарностью ряда
    - Сложности с подбором параметров (p, d, q)
    - Длительное время обучения для больших рядов
    
    **Для моделей машинного обучения:**
    - Проблемы с пропущенными значениями в фичах
    - Необходимость масштабирования данных
    - Переобучение на временных рядах
    
    **Для нейросетевых моделей (LSTM, GRU):**
    - Требовательность к вычислительным ресурсам
    - Сложности с настройкой гиперпараметров
    - Проблемы с интерпретируемостью результатов
    """)


# --- Конец приложения ---
# streamlit run "Timeseries\lab2_app.py"