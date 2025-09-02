# Обучение модели логичстической регрессией
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Создаем столбец X
X = np.random.randint(1, 101, 100)
# Создаем столбец y с бинарными метками
# Условие: если X > 50, то y = 1, иначе y = 0
y = (X > 50).astype(int)
# Создаем DataFrame
df = pd.DataFrame({'X': X, 'y': y})
# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(df[['X']], df['y'], test_size=0.2, random_state=42)
# Создаем и обучаем модель
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accurace = accuracy_score(y_test, predictions)
print(accurace)