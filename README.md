# Анализ портретного фотоархива (DeepFace & Data Science)

## Описание проекта

Данный проект посвящён автоматизации обработки неструктурированных визуальных данных. В рамках исследования был проанализирован портретный фотоархив с целью извлечения структурированной информации о людях на снимках.

С помощью нейросетевой библиотеки **DeepFace** (Python) была проведена обработка нескольких сотен изображений (датасет Kaggle). Для каждого снимка система автоматически определяла:

- Пол;
    
- Возраст;
    
- Этническую принадлежность;
    
- Эмоциональное состояние.
    

На основе полученных данных с помощью библиотек **Pandas, Matplotlib и Seaborn** проведен статистический анализ, выявлены закономерности (например, корреляция улыбок с этнической принадлежностью, разница в эмоциональном фоне между мужчинами и женщинами) и построены визуализации.

## Используемые технологии

- **Python**
    
- **DeepFace** (анализ лиц)
    
- **Pandas** (обработка данных)
    
- **Matplotlib / Seaborn** (визуализация)
    
- **Jupyter Notebook**
    

## Структура проекта

Архитектура проекта включает последовательный процесс обработки: от загрузки изображений до статистического анализа полученных данных.

|**Компонент**|**Назначение**|
|---|---|
|`DeepFace Analysis`|Извлечение признаков из изображений|
|`Data Cleaning`|Очистка и структурирование данных (Pandas)|
|`Visualization`|Построение графиков и отчетов|
|`Analysis_Logs`|Логирование результатов обработки|

## Визуализации

Ниже представлены основные аналитические срезы, полученные в ходе исследования фото-архива.

<img width="1152" height="582" alt="Фото архива" src="https://github.com/user-attachments/assets/a1724fec-4219-4311-a8e9-04cd84021a7a" />

### Возраст и Этнические группы

**Распределение по возрасту:**

<img width="562" height="455" alt="изображение" src="https://github.com/user-attachments/assets/a90f68eb-d590-4983-b812-6d144255f44b" />

**Средний возраст по этническим группам:**

<img width="562" height="545" alt="изображение" src="https://github.com/user-attachments/assets/5fffe18f-6d0d-4fcb-afa4-9cdfa260443f" />

**Распределение по этническим группам:**

<img width="571" height="545" alt="изображение" src="https://github.com/user-attachments/assets/daf02a2e-ae8a-44f5-82d7-b89ce5f73953" />

### Гендерный анализ

**Распределение по гендеру:**

<img width="389" height="411" alt="изображение" src="https://github.com/user-attachments/assets/82b2b679-fec8-415b-9cb9-ed3b00a6a0da" />

### Эмоциональный анализ

**Общее распределение эмоций:**

<img width="562" height="496" alt="изображение" src="https://github.com/user-attachments/assets/56bd0b7b-1253-4150-bb48-766bc6883cee" />

**Эмоции по гендерному признаку:**

<img width="571" height="493" alt="изображение" src="https://github.com/user-attachments/assets/32744408-fca0-4b49-8291-6293484bf51d" />

**Эмоции в зависимости от этнической группы:**

<img width="571" height="545" alt="изображение" src="https://github.com/user-attachments/assets/826794e2-8993-4ed8-ba36-d28ad123b06e" />

Преобладающие эмоции:

<img width="562" height="496" alt="изображение" src="https://github.com/user-attachments/assets/ad5bb74f-c9a2-4fa3-94ad-58547c463911" />

## Методология анализа

Для получения результатов были применены следующие подходы:

1. **Анализ данных**: Группировка данных по категориальным признакам (пол, этническая принадлежность).
    
2. **Сравнение**: Сравнение среднего возраста и эмоционального спектра между различными группами.
    
3. **Выявление зависимостей**: Оценка частоты встречаемости конкретных эмоций (например, "улыбка" или "нейтральное выражение") в разных выборках.
    
## Структура кода

## 1. Установка библиотек
Первая ячейка устанавливает необходимые пакеты:
```python
!pip install deepface opencv-python pandas matplotlib seaborn
```

## 2. Импорт модулей
```python
from deepface import DeepFace
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
Подключаются основные библиотеки: DeepFace для анализа лиц, pandas для таблиц, matplotlib и seaborn для графиков.

## 3. Задание пути к папке с изображениями
```python
folder = "/content/faces"
```
В этой директории лежат фотографии лиц (например, `000097.jpg`, `000127.jpg` и т.д.).

## 4. Цикл по изображениям и вызов DeepFace.analyze
```python
data = []

for img_name in os.listdir(folder):
    img_path = os.path.join(folder, img_name)

    try:
        result = DeepFace.analyze(
            img_path,
            actions=['emotion', 'gender', 'age', 'race'],
            enforce_detection=False
        )[0]

        gender = result['dominant_gender']
        age = result['age']
        race = result['dominant_race']
        emotion = result['dominant_emotion']
        emotions = result['emotion']

        data.append([
            img_name, gender, age, race, emotion,
            emotions['happy'], emotions['neutral'], emotions['sad'],
            emotions['angry'], emotions['fear'], emotions['surprise'],
            emotions['disgust']
        ])

    except:
        continue
```
- Перебираются все файлы в папке.
- `DeepFace.analyze` анализирует эмоции, пол, возраст и расу. `enforce_detection=False` позволяет не падать при отсутствии лица.
- Из результата берутся доминирующие значения и вероятности каждой эмоции.
- Данные накапливаются в списке `data`. Если возникает ошибка (например, лицо не найдено), файл пропускается.

## 5. Формирование DataFrame
```python
columns = [
    "image", "gender", "age", "race", "dominant_emotion",
    "happy", "neutral", "sad", "angry", "fear", "surprise", "disgust"
]

df = pd.DataFrame(data, columns=columns)
df.head()
```
Создаётся таблица с 12 колонками. `df.head()` показывает первые 5 строк, например:
```
      image gender  age            race dominant_emotion      happy  ...
0  000097.jpg  Woman   31           white             fear   0.016909
...
```

## 6. Сохранение результатов в CSV
```python
df.to_csv("analysis_results.csv", index=False)
```
Файл `analysis_results.csv` будет содержать все собранные данные.

## 7. Визуализация распределения рас
```python
plt.figure()
df["race"].value_counts().plot(kind="bar")
plt.title("Ethnic Group Distribution")
plt.xlabel("Ethnic Group")
plt.ylabel("Count")
plt.show()
```
Строится столбчатая диаграмма количества представителей разных рас.

## 8. Визуализация распределения полов (круговая)
```python
plt.figure()
df["gender"].value_counts().plot(kind="pie", autopct="%1.0f%%")
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()
```
Круговая диаграмма с процентным соотношением мужчин и женщин.

## 9. Гистограмма возраста
```python
plt.figure()
df["age"].plot(kind="hist", bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

## 10. Распределение доминирующих эмоций
```python
plt.figure()
df["dominant_emotion"].value_counts().plot(kind="bar")
plt.title("Dominant Emotions")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()
```

## 11. Средние вероятности эмоций
```python
emotion_cols = ["happy","neutral","sad","angry","fear","surprise","disgust"]
avg_emotions = df[emotion_cols].mean()

plt.figure()
avg_emotions.plot(kind="bar")
plt.title("Average Emotion Probabilities")
plt.xlabel("Emotion")
plt.ylabel("Average Probability")
plt.show()
```
Считается среднее по каждому столбцу эмоций и строится столбчатый график.

## 12. Связь улыбки и расы
```python
df["smiling"] = df["dominant_emotion"].apply(lambda x: "smiling" if x=="happy" else "not_smiling")
smile_by_race = pd.crosstab(df["race"], df["smiling"])
smile_by_race.plot(kind="bar", stacked=True)
plt.title("Smiling by Ethnic Group")
plt.xlabel("Ethnic Group")
plt.ylabel("Count")
plt.show()
```
Создаётся новый столбец `smiling`, затем строится стековая столбчатая диаграмма, показывающая, сколько людей улыбается / не улыбается в каждой расовой группе.

## 13. Средний возраст по расам
```python
avg_age = df.groupby("race")["age"].mean()
plt.figure()
avg_age.plot(kind="bar")
plt.title("Average Age by Ethnic Group")
plt.xlabel("Ethnic Group")
plt.ylabel("Average Age")
plt.show()
```

## 14. Эмоции в разрезе пола
```python
gender_emotion = pd.crosstab(df["gender"], df["dominant_emotion"])
gender_emotion.plot(kind="bar", stacked=True)
plt.title("Emotions by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
```
Сравнивается, какие эмоции чаще встречаются у мужчин и у женщин.

Таким образом, код выполняет полный пайплайн: загрузка изображений → анализ DeepFace → сохранение → множество графиков для исследовательского анализа.

## Как запустить проект локально

Для работы с проектом необходимо настроить окружение Python:

1. **Установка зависимостей:**

```Bash
 pip install deepface pandas matplotlib seaborn
```
    
2. **Запуск анализа:**
    
    - Откройте `main.ipynb` в Jupyter Notebook или Google Volab.
        
    - Убедитесь, что папка с изображениями доступна для скрипта.
        
    - Запустите ячейки последовательно для извлечения признаков.
        
3. **Анализ данных:**
    
    - После генерации CSV-файла с результатами (в коде) запустите блок визуализации для построения графиков.
        
