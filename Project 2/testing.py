from math import *
import numpy as np

points = [
    (1,1), # 0    1º
    (-1,-1), # 1  5º
    (1,-1), # 2   7º
    (-1,1), # 3   3º
    (-1,0), # 4   4º
    (0,-1), # 5   6º
    (1,0), # 6    8º
    (0,1) # 7     2º
]

# SOLUTION = [0,7,3,4,1,5,2,6]

centroid= (0,0)

def findangles(centroid, list_of_points):

    angles = []

    for point in list_of_points[1:]:
        vec_a = (centroid[0] - list_of_points[0][0],centroid[1] - list_of_points[0][1] )
        vec_b = (centroid[0] - point[0],centroid[1] - point[1])

        vec_a_angle = atan2(vec_a[1], vec_a[0])

        vec_b_angle = atan2(vec_b[1], vec_b[0])

        angles.append( degrees(vec_b_angle - vec_a_angle))

    return angles

def counter_clock_wise_heuristic(list_of_points):

    # list_of_points has to contain WareHouse as well


    # Calculate centroid to have 2 common points everytime
    centroid_x = centroid_y = 0

    for index, point in enumerate(list_of_points):
        if index == 0:
            continue
        centroid_x += point[0]
        centroid_y += point[1]

    centroid = (centroid_x/(len(list_of_points)-1),centroid_y/(len(list_of_points)-1) )
    #centroid = (-2,-2)

    # List of points no including the warehouse
    angles = [0.0] + findangles(centroid, list_of_points)
    print(angles)
    print()

    # Order it
    angles = np.array(angles)

    angles_sorted = np.argsort(angles)

    # Return our selection
    return angles_sorted

print(findangles(centroid, points))
print("")
print(counter_clock_wise_heuristic(points))