# K-means
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
# Создание Data Frame
np.random.seed(42)
cluster_1 = np.random.normal(loc=10, scale=1, size=(100, 2))
cluster_2 = np.random.normal(loc=50, scale=1, size=(100, 2))
data = np.vstack((cluster_1, cluster_2))
df = pd.DataFrame(data)
# Обучение модели
model = KMeans(2)
model.fit(df)
print(model.cluster_centers_)