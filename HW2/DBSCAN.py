import pygame
import numpy as np
import random
from sklearn.datasets import make_blobs

# Функция для вычисления евклидового расстояния между двумя точками
def calculate_distance(point_a, point_b):
    return np.linalg.norm(np.array(point_a) - np.array(point_b))

# Функция нахождения соседей для точки в заданном радиусе
def find_neighbors(center, points, radius):
    return [point for point in points if 0 < calculate_distance(center, point) < radius]

# Алгоритм DBSCAN
def dbscan(points, radius, min_neighbors):
    core_points = []
    border_points = []
    noise_points = []
    clusters = {}
    cluster_id = 0
    visited_points = set()

    for point in points:
        if point in visited_points:
            continue
        visited_points.add(point)
        neighbors = find_neighbors(point, points, radius)

        if len(neighbors) >= min_neighbors:
            cluster_id += 1
            clusters[cluster_id] = [point]
            core_points.append(point)

            queue = neighbors[:]
            while queue:
                neighbor = queue.pop(0)
                if neighbor not in visited_points:
                    visited_points.add(neighbor)
                    sub_neighbors = find_neighbors(neighbor, points, radius)
                    if len(sub_neighbors) >= min_neighbors:
                        core_points.append(neighbor)
                        queue.extend(sub_neighbors)
                    else:
                        border_points.append(neighbor)
                if neighbor not in clusters[cluster_id]:
                    clusters[cluster_id].append(neighbor)
        else:
            noise_points.append(point)

    return clusters, core_points, border_points, noise_points

# Функция отображения точек по их типам (core, border, noise)
def display_points_by_type(screen, core_points, border_points, noise_points):
    screen.fill((255, 255, 255))
    for point in core_points:
        pygame.draw.circle(screen, (0, 255, 0), point, 5)
    for point in border_points:
        pygame.draw.circle(screen, (255, 255, 0), point, 5)
    for point in noise_points:
        pygame.draw.circle(screen, (255, 0, 0), point, 5)
    pygame.display.update()

# Функция отображения точек по кластерам
def display_points_by_cluster(screen, clusters):
    screen.fill((255, 255, 255))
    cluster_colors = {}
    for cluster_id, cluster_points in clusters.items():
        if cluster_id not in cluster_colors:
            cluster_colors[cluster_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        for point in cluster_points:
            pygame.draw.circle(screen, cluster_colors[cluster_id], point, 5)
    pygame.display.update()

# Основная функция для рисования точек и запуска DBSCAN
def draw_and_cluster():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("DBSCAN Clustering")
    screen.fill((255, 255, 255))
    points = []
    is_drawing = False
    running = True
    clusters = {}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                is_drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                is_drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    clusters, core_points, border_points, noise_points = dbscan(points, 30, 2)
                    display_points_by_type(screen, core_points, border_points, noise_points)
                elif event.key == pygame.K_a:
                    display_points_by_cluster(screen, clusters)
                elif event.key == pygame.K_ESCAPE:
                    screen.fill((255, 255, 255))
                    points = []

        if is_drawing:
            position = pygame.mouse.get_pos()
            if not points or calculate_distance(position, points[-1]) > 5:
                points.append(position)
                pygame.draw.circle(screen, (0, 0, 255), position, 5)
                pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    draw_and_cluster()
