"""
##  Анализ медицинских страховых расходов

**Цель:**  Проанализировать датасет, чтобы выявить факторы, влияющие на стоимость медицинской страховки.

**Описание данных:**

Датасет содержит информацию о расходах на медицинскую страховку для различных лиц.

**Атрибуты:**

* **age:** Возраст основного выгодоприобретателя
* **sex:** Пол страхового агента (женщина, мужчина)
* **bmi:** Индекс массы тела (кг/м^2)
* **children:** Количество детей, охваченных медицинским страхованием
* **smoker:** Курение (да/нет)
* **region:** Регион проживания бенефициара в США (северо-восток, юго-восток, юго-запад, северо-запад)
* **charges:** Индивидуальные медицинские расходы, выставленные медицинским страхованием

"""
# Импорт необходимых библиотек

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency


# Функция для загрузки и предобработки данных
def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)
    print("Размеры датасета:", df.shape)
    df.drop_duplicates(inplace=True)
    print("Размеры датасета после удаления дубликатов:", df.shape)
    df.dropna(inplace=True)
    print("Размеры датасета после удаления пропущенных значений:", df.shape)
    return df


# Основной код
if __name__ == '__main__':
    df = load_and_preprocess_data('insurance.csv')

    # ----------------------------------------
    # 2. Подготовка данных
    # ----------------------------------------

    # Проверка размеров таблицы и пропусков
    print("Размеры датасета:", df.shape)
    df.drop_duplicates(inplace=True)
    print("Размеры датасета после удаления дубликатов:", df.shape)

    # Информация о типе данных и пропущенных значениях
    print(df.info())

    # --- Гистограммы для числовых признаков ---
    df.hist(bins=100, figsize=(8, 8))
    plt.suptitle("Гистограммы числовых признаков", y=1.02)
    plt.show()

    # ----------------------------------------
    # 3. Однофакторный анализ
    # ----------------------------------------

    # --- Дети ---
    plt.figure(figsize=(5, 4))
    plt.hist(df.children, bins=range(0, 9))
    plt.title('Распределение количества детей')
    plt.xlabel('Количество детей')
    plt.ylabel('Частота')
    plt.show()

    # Доля людей без детей
    print("Доля людей без детей:", len(df[df.children == 0]) / len(df) * 100, "%")

    # Зависимость расходов от количества детей
    plt.figure(figsize=(6, 4))
    plt.scatter(df.children, df.charges)
    plt.title('Зависимость расходов от количества детей')
    plt.xlabel('Количество детей')
    plt.ylabel('Расходы')
    plt.show()

    # Корреляция между количеством детей и расходами
    print("Корреляция между количеством детей и расходами:", np.corrcoef(df.children, df.charges)[0][1])

    # --- Индекс массы тела (ИМТ) ---
    plt.figure(figsize=(5, 4))
    plt.hist(df.bmi, bins=20)
    plt.title('Распределение ИМТ')
    plt.xlabel('ИМТ')
    plt.ylabel('Частота')
    plt.show()

    # Минимальный и максимальный ИМТ
    print("Минимальный ИМТ:", df.bmi.min())
    print("Максимальный ИМТ:", df.bmi.max())

    # Количество людей с разными категориями ИМТ
    print("Количество людей с нормальным ИМТ:", len(df[(df.bmi >= 18.5) & (df.bmi < 25)]))
    print("Доля людей с нормальным ИМТ:", len(df[(df.bmi >= 18.5) & (df.bmi < 25)]) / len(df) * 100, "%")

    print("Количество людей с ожирением:", len(df[df.bmi >= 30]))
    print("Доля людей с ожирением:", len(df[df.bmi >= 30]) / len(df) * 100, "%")

    print("Количество людей с недостаточным весом:", len(df[df.bmi < 18.5]))
    print("Доля людей с недостаточным весом:", len(df[df.bmi < 18.5]) / len(df) * 100, "%")

    # Зависимость расходов от ИМТ
    plt.figure(figsize=(6, 4))
    plt.scatter(df.bmi, df.charges)
    plt.title('Зависимость расходов от ИМТ')
    plt.xlabel('ИМТ')
    plt.ylabel('Расходы')
    plt.show()

    # Корреляция между ИМТ и расходами
    print("Корреляция между ИМТ и расходами:", np.corrcoef(df.bmi, df.charges)[0][1])

    # --- Возраст ---
    plt.figure(figsize=(5, 4))
    plt.hist(df.age, bins=20)
    plt.title('Распределение возраста')
    plt.xlabel('Возраст')
    plt.ylabel('Частота')
    plt.show()

    # Минимальный и максимальный возраст
    print("Минимальный возраст:", df.age.min())
    print("Максимальный возраст:", df.age.max())

    # Зависимость расходов от возраста
    plt.figure(figsize=(6, 4))
    plt.scatter(df.age, df.charges)
    plt.title('Зависимость расходов от возраста')
    plt.xlabel('Возраст')
    plt.ylabel('Расходы')
    plt.show()

    # Корреляция между возрастом и расходами
    print("Корреляция между возрастом и расходами:", np.corrcoef(df.age, df.charges)[0][1])

    # Медианные расходы по возрастам
    print("Медианные расходы по возрастам:")
    print(df.groupby('age').agg(ChargesMedian=('charges', 'median')).sort_values(by='ChargesMedian', ascending=False))

    # --- Курильщики ---
    smoker_counts = df['smoker'].value_counts()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=smoker_counts.index, y=smoker_counts.values, hue=smoker_counts.index, palette='summer')
    plt.title('Количество курильщиков и не курильщиков')
    plt.xlabel('Курильщик')
    plt.ylabel('Количество')
    plt.show()

    # Разделение курильщиков по полу
    smokers_by_sex = df['sex'][df['smoker'] == "yes"].value_counts()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=smokers_by_sex.index, y=smokers_by_sex.values, hue=smokers_by_sex.index, palette='summer')
    plt.title('Разделение курильщиков по полу')
    plt.xlabel('Пол')
    plt.ylabel('Количество')
    plt.show()

    # Зависимость расходов от курения
    plt.figure(figsize=(6, 4))
    plt.scatter(df.smoker, df.charges)
    plt.title('Зависимость расходов от курения')
    plt.xlabel('Курильщик')
    plt.ylabel('Расходы')
    plt.show()

    # Медианные расходы для курильщиков и некурящих
    print("Медианные расходы для курильщиков и некурящих:")
    print(
        df.groupby('smoker').agg(ChargesMedian=('charges', 'median')).sort_values(by='ChargesMedian', ascending=False))

    # --- Пол ---
    sex_counts = df['sex'].value_counts()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=sex_counts.index, y=sex_counts.values, hue=sex_counts.index, palette='summer')
    plt.title('Разделение по полу')
    plt.xlabel('Пол')
    plt.ylabel('Количество')
    plt.show()

    # Зависимость расходов от пола
    plt.figure(figsize=(6, 4))
    plt.scatter(df.sex, df.charges)
    plt.title('Зависимость расходов от пола')
    plt.xlabel('Пол')
    plt.ylabel('Расходы')
    plt.show()

    # Медианные расходы для мужчин и женщин (с учетом курения)
    print("Медианные расходы для мужчин и женщин (с учетом курения):")
    print(
        df.groupby(['smoker', 'sex']).agg(ChargesMedian=('charges', 'median')).sort_values(by='ChargesMedian',
                                                                                           ascending=False))

    # --- Регион ---
    region_counts = df['region'].value_counts()
    region_counts = region_counts.sort_index()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=region_counts.index, y=region_counts.values, hue=region_counts.index, palette='summer')
    plt.title('Разделение по регионам')
    plt.xlabel('Регион')
    plt.ylabel('Количество')
    plt.show()

    # Средние расходы по регионам
    mean_charges_by_region = df.groupby('region')['charges'].mean().reset_index()

    plt.figure(figsize=(6, 4))
    sns.barplot(x='region', y='charges', hue='region', data=mean_charges_by_region, palette='summer')
    plt.title('Средние затраты в регионе')
    plt.xlabel('Регион')
    plt.ylabel('Средние затраты')
    plt.show()

    # Медианные расходы по регионам
    print("Медианные расходы по регионам:")
    print(
        df.groupby('region').agg(ChargesMedian=('charges', 'median')).sort_values(by='ChargesMedian', ascending=False))

    # ----------------------------------------
    # 4. Корреляционный анализ
    # ----------------------------------------

    # --- Корреляция Пирсона ---
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap='BuPu', annot=True)
    plt.title('Корреляция Пирсона')
    plt.show()

    # --- Корреляция Спирмена ---
    corr = df.corr(numeric_only=True, method='spearman')
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="BuPu", annot=True)
    plt.title('Корреляция Спирмена')
    plt.show()

    # --- V-мера Крамера (χ2) ---
    df_cat = df.apply(lambda x: x.astype("category") if x.dtype == "object" else x)


    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        min_dim = min(confusion_matrix.shape) - 1
        return np.sqrt(phi2 / min_dim)


    # Создаем матрицу V-меры Крамера
    cols = df_cat.columns
    n = len(cols)
    cramersv_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cramersv_matrix[i, j] = cramers_v(df_cat.iloc[:, i], df_cat.iloc[:, j])

    # Преобразуем матрицу в DataFrame для красивого отображения
    cramersv_df = pd.DataFrame(cramersv_matrix, index=cols, columns=cols)

    # Строим heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cramersv_df, annot=True, cmap='BuPu')
    plt.title('V-мера Крамера')
    plt.show()

    # --- ANOVA ---
    Data = []
    for c1 in df.columns:
        for c2 in df.columns:
            if df[c1].dtype == 'object' and df[c2].dtype != 'object':
                CategoryGroupLists = df.groupby(c1)[c2].apply(list)
                AnovaResults = f_oneway(*CategoryGroupLists)

                if AnovaResults[1] >= 0.05:
                    Data.append({'Category': c1, 'Numerical': c2, 'Is correlated': 'No'})
                else:
                    Data.append({'Category': c1, 'Numerical': c2, 'Is correlated': 'Yes'})

    AnovaRes = pd.DataFrame.from_dict(Data)
    print("Результаты ANOVA:")
    print(AnovaRes)

    # ----------------------------------------
    # 5. Анализ целевой переменной
    # ----------------------------------------

    # --- Гистограмма целевой переменной ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], kde=True)
    plt.title('Распределение целевой переменной (до логарифмирования)')
    plt.xlabel('Расходы')
    plt.show()

    df.to_csv('preprocessed_insurance.csv', index=False)
