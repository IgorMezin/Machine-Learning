import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


# Загрузка данных
iris = load_iris()
features = iris.data  # Используем признаки из датасета Ирисов


class CustomKMeans:
    def __init__(self, num_clusters=3, max_iter=100, visualize_steps=True):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.visualize_steps = visualize_steps

    def fit(self, data):
        # Инициализация центроидов
        random_indices = random.sample(range(data.shape[0]), self.num_clusters)
        self.centroids = data[random_indices]

        for iteration in range(self.max_iter):
            # Назначение точек кластерам
            cluster_assignments = self._assign_clusters(data)

            # Пересчёт центроидов
            new_centroids = self._update_centroids(data, cluster_assignments)

            # Визуализация шага
            if self.visualize_steps:
                self._plot_clusters(data, cluster_assignments, iteration)

            # Проверка на сходимость
            if np.array_equal(new_centroids, self.centroids):
                print(f"Сошлось на {iteration + 1}-й итерации")
                break

            self.centroids = new_centroids

    def _assign_clusters(self, points):
        # Вычисление расстояний от точек до центроидов
        distances = np.linalg.norm(points[:, np.newaxis] - self.centroids, axis=2)
        # Определение ближайшего кластера
        return np.argmin(distances, axis=1)

    def _update_centroids(self, points, labels):
        # Обновление центроидов как среднего значений точек каждого кластера
        new_centroids = np.zeros((self.num_clusters, points.shape[1]))
        for cluster_idx in range(self.num_clusters):
            cluster_points = points[labels == cluster_idx]
            if cluster_points.size > 0:  # Проверка, что кластер не пустой
                new_centroids[cluster_idx] = cluster_points.mean(axis=0)
        return new_centroids

    def _plot_clusters(self, points, labels, step):
        plt.figure(figsize=(8, 6))
        for cluster_idx in range(self.num_clusters):
            cluster_points = points[labels == cluster_idx]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Кластер {cluster_idx + 1}")
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='red', marker='X', s=200, label='Центры')
        plt.title(f"Итерация {step + 1}")
        plt.xlabel("Признак 1")
        plt.ylabel("Признак 2")
        plt.legend()
        plt.grid()
        plt.show()


# Пример использования
kmeans = CustomKMeans(num_clusters=3, visualize_steps=True)
kmeans.fit(features)
