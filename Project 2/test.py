import random

from math import dist, inf
import matplotlib.pyplot as plt

points = [
    (random.random(), random.random()),
    (random.random(), random.random()),
    (random.random(), random.random()),
    (random.random(), random.random()),
    (random.random(), random.random()),
    (random.random(), random.random()),
    (random.random(), random.random()),
]

random.shuffle(points)


sorted_points = []

current_point_id = random.randrange(len(points))
# current_point_id = 0
sorted_points.append(current_point_id)

while len(sorted_points) != len(points):
    closest_point_id = current_point_id
    closest_point_dist = inf

    for point_id in range(len(points)):
        if point_id in sorted_points:
            continue

        if dist(points[current_point_id], points[point_id]) < closest_point_dist:
            closest_point_id = point_id
            closest_point_dist = dist(points[current_point_id], points[point_id])

    current_point_id = closest_point_id
    sorted_points.append(closest_point_id)

print(sorted_points)

X = []
Y = []
L = []

for id in sorted_points:
    X.append(points[id][0])
    Y.append(points[id][1])
    L.append(id)

plt.plot(X, Y, "-o")

for i, txt in enumerate(L):
    plt.annotate(txt, (X[i], Y[i]))

plt.show()
