# Обучение модели с помощью Random Forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Создаем столбец X
X = np.random.randint(1, 101, 100)
# Создаем столбец y с бинарными метками
y = (X > 50).astype(int)
# Создаем DataFrame
df = pd.DataFrame({'X': X, 'y': y})
# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(df[['X']], df['y'], test_size=0.2, random_state=42)
# Обучение модели
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accurace = accuracy_score(y_test, predictions)
print(accurace)