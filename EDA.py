import pandas as pd
import matplotlib.pyplot as plt

# Список файлов
files = ['Wine_TEST.csv', 'Wine_TRAIN.csv', 'yoga_TEST.csv', 'yoga_TRAIN.csv']

# Функция для выполнения EDA и отображения графиков
def perform_eda(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path, header=None)

    # Разделение данных по значениям в первом столбце
    data_1 = data[data[0] == 1].drop(columns=[0])
    data_2 = data[data[0] == 2].drop(columns=[0])

    # Вычисление средних значений для каждого столбца
    mean_data_1 = data_1.mean()
    mean_data_2 = data_2.mean()

    # Графики временных рядов для средних значений
    plt.figure(figsize=(15, 8))
    plt.plot(mean_data_1, label='Mean Values for Class 1', marker='o')
    plt.plot(mean_data_2, label='Mean Values for Class 2', marker='x')
    plt.title(f'Mean Values Time Series for {file_path}')
    plt.xlabel('Column Index')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.show()

# Выполнение EDA для всех файлов
for file in files:
    perform_eda(file)
