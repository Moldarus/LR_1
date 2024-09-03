import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns

# Список файлов
files = ['Wine_TEST.csv', 'Wine_TRAIN.csv', 'yoga_TEST.csv', 'yoga_TRAIN.csv']

# Функция для выполнения классификации и оценки модели
def perform_classification(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path, header=None)

    # Разделение данных на признаки (X) и метки (y)
    X = data.drop(columns=[0])
    y = data[0]

    # Разделение данных на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Создание и обучение модели
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Предсказание на тестовом наборе
    y_pred = model.predict(X_test)

    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    print(f'Accuracy for {file_path}: {accuracy}')
    print(f'F1 Score for {file_path}: {f1}')
    print(f'Classification Report for {file_path}:\n{report}')

    # Перекрёстная проверка
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
    print(f'Cross-Validation F1 Scores for {file_path}: {cv_scores}')
    print(f'Mean Cross-Validation F1 Score for {file_path}: {cv_scores.mean()}')

    # График точности и F1-score
    plt.figure(figsize=(8, 6))
    plt.bar(['Test Accuracy', 'Test F1 Score', 'Mean CV F1 Score'], [accuracy, f1, cv_scores.mean()])
    plt.ylim(0, 1)
    plt.title(f'Accuracy and F1 Score for {file_path}')
    plt.ylabel('Score')
    plt.show()

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {file_path}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Выполнение классификации для всех файлов
for file in files:
    perform_classification(file)
