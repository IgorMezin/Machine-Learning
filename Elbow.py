import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = datasets.load_iris()
X = iris.data

# Метод локтя для определения оптимального количества кластеров
k_ = range(2, 9)
inertia = []
silhouette_scores = [] #оценка силуэтов

for k in k_:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

plt.figure(figsize=(12, 5))

# Метод локтя
plt.subplot(1, 2, 1)
plt.plot(k_, inertia, 'bo-')
plt.xlabel('Количество кластеров(K)')
plt.ylabel('Сумма квадратов расстояний (Inertia)')
plt.title('Метод локтя')

plt.show()