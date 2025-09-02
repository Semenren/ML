# Метод k ближайших соседей
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
# 1. Загружаем набор данных Iris
iris = load_iris()
# 2. Сохраняем данные (признаки) в X, а метки (классы) в y
X = iris.data
y = iris.target
# 3. Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Обучаем модель
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
print(accuracy)
# если n_neighbors = k:
# for k in range(1, 11):
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(predictions, y_test)
#     print(k, accuracy)