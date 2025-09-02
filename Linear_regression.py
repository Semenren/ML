# Обучение модели с помощью линейной регресии
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
# Создаем DataFrame
np.random.seed(42)
X = np.random.randint(1, 101, 1000)
noise = np.random.randn(1000) * 10
Y = X * 2 + 5 + noise
df = pd.DataFrame({'X': X, 'Y': Y})
# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(df[['X']], df['Y'], test_size=0.2, random_state=42)
# Создаем и обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)
# Делаем предсказывания
predictions = model.predict(X_test)
# МАЕ
mae = mean_absolute_error(y_test, predictions)
print(f"MAE нашей модели: {mae}")
# сохранение модели в файл
joblib.dump(model, 'my_first_model.joblib')
# загрузка модели
loaded_model = joblib.load('my_first_model.joblib')
# предсказания
predictions_2 = loaded_model.predict(X_test)
print(predictions_2)